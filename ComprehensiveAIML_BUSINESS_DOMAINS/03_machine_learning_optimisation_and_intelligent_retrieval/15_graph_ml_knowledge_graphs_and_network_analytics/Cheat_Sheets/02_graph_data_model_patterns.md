---
title: "Graph data-model patterns"
topic: "Graph data-model patterns: typed multigraphs, edge tables, RDF triples, adjacency/CSR layouts, bitemporal facts and projection"
collection: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Graph data-model patterns

## 1. File Header

> **Purpose:** Choose, justify and cost the physical and logical representation of a relationship dataset — which facts become edges, which identifiers become nodes, which derived structure (adjacency list, CSR array, triple table, projection) is materialised — so that a downstream graph feature or traversal can be computed reproducibly at a stated decision time.
> **Prerequisites:** Set and multiset reasoning, the handshake lemma, row-versus-column storage intuition, event-time versus availability-time discipline, and the distinction between a primary key and a resolved entity.
> **Key outcome:** Given a candidate dataset and a decision latency budget, you can state the fact grain, pick a layout, compute its storage and traversal cost from first principles, name the information the layout destroys, and reject a layout whose loss breaks the decision.
> **This file does NOT cover:** Cypher, Gremlin or SPARQL syntax; vendor feature lists; graph neural network architecture (see the GNN decision guide in this collection); centrality ranking (see the centrality sheet); ontology design at depth (see the knowledge-graph sheet).
> **Level:** Core. Foundation material is the definitions in Section 3; advanced material is the counterexample analysis in Section 4.4 and the loss theorems in Section 10.
> **Version / reviewed:** Draft 1.0, 2026-11-10. Independent review: not yet performed. Review trigger: any change to the fact grain, the time semantics, or the storage engine.
> **Evidence:** All algebraic identities are DERIVED and re-checked in Section 5. All node names, amounts, timestamps and byte sizes are ILLUSTRATIVE constructions; no vendor benchmark, no production measurement and no organisation-specific schema is claimed. Byte counts assume documented primitive widths (int32 = 4 B, int64/float64 = 8 B) and are labelled as estimates, not measurements.
> **Retrieval cue:** If a graph feature is cheap offline and expensive or wrong in production, read Section 4.4 first, then the failure entries on projection loss, supernode fan-out, and availability-time backdating.

A data model is a promise about what you will be able to ask later. Every promise has a price: bytes on disk, milliseconds per hop, and — the part that is usually unaccounted — information that the representation cannot carry at all. The three cheapest decisions to make early are also the three most expensive to reverse: the grain of a fact, the identity of a node, and whether time is a column or an afterthought. This sheet treats those three as the primary design surface and everything else as a consequence.

The recurring trap is to treat "the graph" as a single object. In practice there are at least four distinct artifacts. There is the **base fact store**: an append-only table of typed, timestamped relationships with provenance. There is the **identity layer**: the mapping from raw source identifiers to resolved nodes, versioned and reversible. There is the **serving structure**: an adjacency array, an index, or a triple table optimised for a specific query shape at a specific latency. And there is the **training snapshot**: a frozen projection of the first three at one decision time, used to build features and labels. These four have different correctness rules, different lifecycles and different owners, and collapsing them into one is the root cause of most "the graph says X but the report says Y" incidents.

This file therefore proceeds in the order a designer should: define the fact, define the identity, define the time, then choose a layout whose cost you can compute. Where a layout loses information, the loss is stated as a theorem with a validity condition rather than as an opinion. Where a number appears, the arithmetic is shown; where a number is constructed for teaching, it says ILLUSTRATIVE.

**Takeaway:** A graph model is a set of lossy projections over an append-only typed fact table; the design work is choosing which losses the decision can survive.

## 2. Concept Map / Mental Model

```text
                 RAW SOURCE RECORDS (immutable, per system)
        payments | logins | device sightings | ownership declarations
                            |
                            | 1. FACT DESIGN
                            |    choose grain, direction, relation type,
                            |    value columns, event & availability time
                            v
        +-------------------------------------------------------+
        |  BASE FACT TABLE  (append-only, typed, bitemporal)     |
        |  edge_id | src_raw | rel | dst_raw | value | t_event | |
        |          |         |     |         |       | t_avail | |
        +-------------------------------------------------------+
               |                                   |
               | 2. IDENTITY RESOLUTION            | 4. AS-OF SELECT
               |    raw id -> node id, versioned   |    t_avail <= t_decision
               v                                   v
        +---------------------+          +--------------------------+
        | IDENTITY MAP        |          | SNAPSHOT at t_decision   |
        | version, evidence,  |          | (immutable, addressable) |
        | reversible merges   |          +--------------------------+
        +---------------------+                     |
               |                                    |
               | 3. LAYOUT CHOICE                   | 5. PROJECTION
               v                                    v
   edge table | CSR/CSC array | triple table |   node-level table,
   property graph store | hypergraph         bipartite 2-section,
                                              motif counts
               |                                    |
               +----------------+-------------------+
                                v
                    feature / traversal / rank
                                |
                                v
                    decision at t_decision
```

Read the diagram as five separate transformation steps, not one pipeline. Each arrow is a place where information can be dropped, and each drop is only acceptable if the decision downstream is invariant to it. Step 1 decides what a fact is. Step 2 decides what an entity is. Step 3 decides how it is laid out. Step 4 decides what is visible. Step 5 decides what is aggregated. A defect at any step produces a symptom at the last step, which is why debugging graph features from the metric backwards usually misattributes the cause.

The **central invariant** is reconstructibility: every derived structure must be reproducible from the base fact table plus the identity map, with no undocumented input. An adjacency array is legitimate because you can rebuild it. A hand-edited edge added to satisfy a query is not, because the rebuild will silently drop it and two reports will diverge. Reconstructibility is what makes the third and fifth steps safe to be lossy: any loss can be re-derived with a different setting when a decision changes.

The **second invariant** is that time is a property of the fact, stored as two distinct columns, not a metadata tag on the table. Event time answers "when did it happen"; availability time answers "when could we have known". Any structure built without availability time cannot answer the only question a backtest actually asks: what did the serving path see at 09:00 on the third of March? A dataset with only event time can produce a beautiful offline graph that no production system could have assembled, and the resulting model will look excellent until it is deployed.

What varies across the layout families is only *which query shape is cheap*. A row-per-fact table makes arbitrary filtered scans cheap and multi-hop traversal expensive. A CSR adjacency array makes one-hop expansion cache-friendly and per-edge filtering impossible. A triple table makes predicate-specific patterns cheap and property-joining expensive. A property-graph store makes pointer-chasing natural and bulk analytics awkward. Nothing about the *facts* differs; the choice is a bet on which queries dominate, and that bet should be written down because it will be re-litigated.

The boundary that matters most for machine learning is between the **base graph and the projected feature table**. The base graph is high-cardinality, multivalued and time-indexed: nine edges over six nodes in the worked example. The feature table is one row per entity per decision time with scalar columns: degree, distinct-neighbour count, weighted sum. Projection is where multiplicity becomes a statistic, and it is also where the single largest class of silent bias enters — an aggregate computed over all edges ever seen is not the aggregate the serving path could compute.

```text
Invariant ladder (each rung is a question a reviewer must be able to answer):

  rung 1  what is one row of the fact table?          -> grain statement
  rung 2  which identifiers are nodes, in what name?  -> namespace statement
  rung 3  what two timestamps does each fact carry?   -> time statement
  rung 4  which structures are materialised?          -> layout statement
  rung 5  which of those are rebuilt, never edited?   -> rebuild statement
  rung 6  what does each structure fail to represent? -> loss statement
```

A model that cannot answer rung 6 is not wrong, only unaudited — and unaudited graph features tend to fail in the direction of overstating evidence, because a projection that merges multiplicity into a single boolean edge makes two accounts on one shared device look identical to two accounts on one device used once each.

**Takeaway:** Separate fact, identity, layout, snapshot and projection; keep the first four reconstructible from the fact table and write down what the last one loses.

## 3. Foundational Definitions

### 3.1 The fact layer

**Fact (edge record).** An immutable row asserting that at a stated event time a typed relationship held between a subject identifier and an object identifier, optionally carrying a value and a provenance reference. A fact is not a state; it is an event of assertion. "A1 transferred 250.00 USD to M1 at 09:00:00Z" is a fact. "A1 is a customer of M1" is a derived state that requires an aggregation rule over facts.

**Relation type (predicate).** The label that fixes the semantics of the edge: `TRANSFER`, `USED_DEVICE`, `OWNS_ACCOUNT`, `SHARES_ADDRESS`. Two edges with different types are not interchangeable even when they join the same pair of nodes, and the type must be in the primary key of the fact table rather than a nullable column, otherwise the same pair cannot hold two kinds of relationship simultaneously.

**Grain.** The answer to "what does one row mean". Three different grains can look alike: one row per event (multigraph), one row per distinct pair aggregated over a window (simple graph with a weight), one row per pair per relation (typed simple graph). Choosing the aggregated grain before deciding the feature destroys the ability to compute anything time-weighted later, and it destroys the ability to answer "how many transfers in the last seven days" at any cut-off other than the aggregation boundary.

**Direction.** Whether the fact is ordered (subject → object) or unordered. Direction is semantic, not cosmetic: an ownership declaration points from person to account, while a co-occurrence on a shared device is symmetric. Storing a symmetric relation as one directed row forces every traversal to query both directions, which is correct but twice the work; storing a directed relation as one undirected row destroys the ability to ask who transferred to whom.

**Value columns.** Numeric or categorical payload attached to the fact: amount and currency, session duration, channel, confidence score for an inferred edge. Values belong to the fact, not to the node, because the same pair of nodes can carry different amounts at different times. A model that pushes the value onto the edge type (`TRANSFER_SMALL`, `TRANSFER_LARGE`) has binned the data irreversibly and chosen the bin edges before seeing the distribution.

**Provenance.** The reference that says where the fact came from: source system, extraction batch, record identifier, and the rule version that produced it if the edge is derived. Provenance is what lets an auditor distinguish an observed ownership declaration from an inferred one, and it is the only defence against the class of leak where a derived edge encodes a post-outcome decision.

### 3.2 The identity layer

**Node.** A typed entity within a declared namespace: `account:A1`, `device:D1`, `person:P1`. Types are part of the identity, so an account numbered 4471 and a merchant numbered 4471 are distinct nodes; joining them on the numeric coincidence is a defect that survives review because the joined graph still looks plausible.

**Raw identifier.** The string a source system emitted. Raw identifiers are unstable across systems (leading zeros, case, internal versus external numbering) and must never be used as join keys without a normalisation rule that is versioned alongside the identity map.

**Entity resolution.** The mapping from raw identifiers to resolved nodes, with evidence and version. Reversible means the pre-merge identifiers and the merge rule are retained, so a bad merge can be undone and its downstream features invalidated. Non-reversible resolution is the most expensive kind of technical debt in a graph system, because a merge error propagates into every aggregate that ever touched the merged node.

**Supernode (hub).** A node whose degree is orders of magnitude above the median, typically a shared device token, a corporate address, a payment processor or a sentinel value such as `UNKNOWN`. Supernodes are usually instrumentation artifacts rather than genuine structure: a device identifier that fails to populate collapses to a constant, and that constant acquires tens of thousands of edges with no relationship behind them.

**Namespace collision.** Two source systems using the same identifier space for different entities, or one system reusing identifiers after recycling. It is invisible in the graph and visible only as an implausible edge between entities that should be far apart, so it is detected by sampling edges and checking them against source records rather than by any structural statistic.

### 3.3 The time layer

**Event time.** When the source asserts the relationship occurred, in the source's own clock. Clocks skew, exporters batch, and time zones get applied twice; a device sighting stamped 23:50 local can be the same wall-clock instant as a payment stamped 22:50 UTC in a neighbouring system.

**Availability time.** When the record became readable by the serving path. For an authorisation this is the commit that followed the ledger write; for a batch extract it is the load completion. Availability is always at or after event time, and the gap is the ingestion latency, which is heavy-tailed rather than constant.

**Decision time.** The instant at which a score, an alert or a queue ordering must exist. It fixes the cut-off: a fact may enter a feature only if its availability time precedes the decision time. Decision time is a property of the workflow (a nightly batch boundary, a payment round trip), not of the data.

**Bitemporal fact.** A fact carrying both event time and availability time. Adding system-validity (when the row was written, when it was superseded) turns it into a four-timestamp fact. The four timestamps answer four different questions and substituting one for another is the single most common cause of an unreproducible backtest.

**Snapshot.** The set of facts visible at a decision time under a stated identity-map version and a stated relation filter. A snapshot is an addressable artifact: it has an identifier, and every feature in a training set should record which snapshot produced it. Without that identifier, "reproduce last quarter's model" is a request nobody can satisfy.

**Late arrival and backfill.** A late arrival is a fact with an early event time and a late availability time; a backfill is a bulk load of such facts. Both are legitimate and both are poisonous to naive replay, because a naive replay keyed on event time will show the past gaining information it never had. The repair is not to discard them but to keep them with their true availability time and let the snapshot rule exclude them from earlier decisions.

### 3.4 The layout layer

**Adjacency list.** For each node, the list of its neighbours. Natural for pointer-chasing traversal, compact when neighbour lists are short, and pathological when one node holds a million neighbours because the list is a single contiguous allocation that must be scanned.

**CSR (compressed sparse row).** Two arrays: an offset array of length n+1 and a target array of length m, where the neighbours of node i occupy positions `offset[i]` to `offset[i+1]-1`. Optional parallel arrays carry weights, relation codes or timestamps. CSR is the default in-memory layout for analytics because one-hop expansion is a contiguous scan and the whole structure is two integers per position.

**CSC (compressed sparse column).** The transpose of CSR. Needed when the query direction is reversed — in-neighbours rather than out-neighbours — and building it is the cheapest way to serve "who sent to this account" without scanning every edge.

**Triple table.** A three-column relation `(subject, predicate, object)` with optional extra columns, the natural form for RDF and for sparse, schema-flexible relationship stores. Cheap to extend with a new predicate, expensive to traverse because each hop is an indexed join rather than a pointer read.

**Reification.** Turning a statement into a resource so that the statement itself can carry properties: `(edge:E1, hasSubject, A1)`, `(edge:E1, hasObject, M1)`, `(edge:E1, hasAmount, 250.00)`. Reification buys full expressiveness at a cost of roughly five rows per original statement; it is the right tool when edges carry provenance that must be queried, and the wrong tool when the graph is only ever traversed.

**Projection (2-section).** Replacing a set of facts by a node-level derived table or by a co-occurrence graph over one node type: all accounts that ever used any of the devices that A1 used. Projection is where the multiplicity of the fact layer is destroyed and where pair counts explode quadratically in the size of the shared node's neighbourhood.

**Hyperedge.** A fact relating more than two identifiers, such as a joint account, a three-party settlement, or a group booking. Most storage layouts are binary, so a hyperedge is decomposed into a node plus binary edges (the star expansion) or replaced by a clique over its participants (the 2-section). Both are lossy in opposite directions: the star keeps the incident structure but adds artificial hubs, and the clique keeps pairwise adjacency but invents pairwise relationships that were never asserted.

**Degree (two meanings).** *Occurrence degree* counts edges incident to a node; *distinct-neighbour degree* counts distinct adjacent nodes. For a simple graph they coincide. For a multigraph they diverge, and every aggregate must state which it uses, because averaging an occurrence count over a node set whose multiplicity distribution is skewed measures behaviour intensity rather than connectivity.

### 3.5 The consumption layer

The four definitions below are the ones a feature owner signs, and they are the ones most often left implicit. Each has a testable consequence.

**Feature contract.** A written statement of numerator, denominator, population filter, time window, null policy, unit and refresh cadence for one feature column. It is a contract because a consumer may compute a business rule against it; changing the denominator without a version bump is a breaking change that no type system catches. The toy graph supplies an instance: numerator = sum of `value_usd` over eligible `TRANSFER` facts; denominator = count of those facts; null policy = exclude accounts with none, never emit 0.00.

**Snapshot identifier.** An opaque handle that names a frozen set of facts, an identity-map version and a relation filter. Features and labels are joined on it. The testable consequence: two people running the same training job with the same snapshot identifier must produce byte-identical features, or the snapshot is not frozen. A snapshot identified only by a date string fails this test whenever a late batch lands, because the date now resolves to different bytes.

**Scope (tenant, purpose, jurisdiction).** The declared subset of facts a computation may read. Scope is not a nicety: a shared graph across tenants can disclose that one tenant's customer transacts with another's, which is a legal problem before it is a modelling problem. The testable consequence is that every structure carries a scope key, and a query without a scope predicate is rejected at the interface rather than filtered later.

**Retention and deletion.** The rule that decides when a fact stops being visible, and how a deletion propagates into every derived structure. Deletion is the hard case, because a derived degree or centrality encodes a fact without storing it; a deletion that removes the row but not the aggregate leaves a residue that no row-level audit can find. The remedy is to record the source fact identifiers in the lineage of any published aggregate, so a deletion can invalidate the aggregates that touched it.

**Cardinality constraint.** A declared bound on how many edges a node may hold, per relation type, used to catch instrumentation collapse. A device token with 50,000 accounts violates a sensible bound, so the constraint converts an invisible modelling problem into a rejected load. The constraint must be per relation type, because a merchant legitimately has many counterparties while a person does not legitimately hold thousands of accounts.

**Serving replica versus store of record.** Two different obligations on two different copies of the same facts. The store of record is append-only, complete, bitemporal and slow. The replica is derived, filtered, fast and disposable. Confusing them produces the two classic failures: a replica promoted to truth (history is lost) and a store of record used for scoring (latency is lost).

### 3.6 Layout vocabulary that is often conflated

**Property graph versus triple store.** A property graph attaches key–value pairs to nodes and edges and is traversed by pointers; a triple store makes every statement a row and is queried by pattern. They can hold identical facts, and the choice is a query-language and storage-engine decision rather than a semantic one. The semantic difference appears only when a property itself needs properties, at which point the property graph either nests a map or reifies, and the triple store reifies naturally.

**Index versus materialised view.** An index is a redundant structure maintained by the engine on writes; a materialised view is a redundant structure refreshed on a schedule. Both accelerate reads and both must be invalidated on deletion. The distinction matters because an index is usually transactional and a materialised view usually is not, so the latter can be stale in a way that a latency budget must tolerate explicitly.

**Vertex-centric versus edge-centric storage.** Vertex-centric stores group edges by source so that one-hop expansion is local; edge-centric stores group edges by relation or by time so that filtered scans are local. A design that needs both must either duplicate or accept one slow path; there is no layout in which both a per-node expansion and a per-relation time range scan are simultaneously contiguous.

**Static versus dynamic graph.** A static graph is built once and read many times, so CSR is ideal. A dynamic graph accepts insertions, and every insertion into the middle of a CSR offset array is O(n). Dynamic graphs therefore use adjacency lists with per-node capacity, or a small mutable delta plus periodic merge. Choosing CSR for a graph with intraday inserts is the most common layout-level performance defect in this area, and it manifests as a rebuild that grows without bound.

**Takeaway:** The vocabulary that prevents most graph defects is small: grain, direction, two timestamps, two degree definitions, and the name of the loss each layout introduces.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: incidence identity and per-edge storage

```text
FORMULA
  sum over nodes of occurrence degree  =  2 * m
  where m is the number of undirected edge occurrences, counted with multiplicity.

WHERE
  d_v   = occurrence degree of node v, dimensionless count, integer >= 0
  m     = number of edge occurrences, integer >= 0
  The identity holds for any multigraph with no self-loops.

CONSTRAINT
  Each undirected edge occurrence contributes exactly one to the degree of each of
  its two endpoints. Summing degrees therefore counts every occurrence twice. The
  identity holds per snapshot, so it is a cheap integrity assertion: recompute the
  degree column and compare against 2m for the same as-of filter.
  With self-loops the count per loop is 2 rather than 1, so the identity becomes
  sum d_v = 2m_external + 2*m_loop and a naive check will report a false alarm.

NUMERICAL EXAMPLE (ILLUSTRATIVE toy graph of Section 5, m = 9)
  degrees: A1=5, A2=4, M1=5, D1=2, D2=1, P1=1
  sum   = 5 + 4 + 5 + 2 + 1 + 1 = 18
  2 * m = 2 * 9 = 18
  match, so no edge is double-counted or orphaned in the fact table.

INTERPRETATION
  The identity converts a modelling question into an assertion. If the degree column
  disagrees with the fact count, the disagreement is a bug in the projection or in
  the as-of filter, not a property of the network, and it should block publication
  of the feature table rather than be explained away.
```

The same identity is the basis of the storage estimate, because per-edge bytes can be attributed exactly once the layout is named. A row-store table with seven columns costs a row header plus the payload; a CSR array costs offsets amortised over edges plus fixed-width arrays per edge. Neither number is a benchmark; both are computable before any code exists, and computing them is what stops a design that needs 176 GB from being discovered in production.

```text
FORMULA
  B_CSR(m, n) = 4*(n+1) + m*(4 + w + 1 + t)
  B_row(m)    = m * r

WHERE
  n = node count            (integer, >= 1)
  m = edge occurrences      (integer, >= 0)
  w = bytes per value       (8 for float64, 0 if no weight)
  t = bytes per timestamp   (8 if a timestamp is stored per edge, 0 if not)
  1 = bytes per relation code (uint8 suffices for <= 256 relation types)
  r = bytes per stored row  (ILLUSTRATIVE 176 for a seven-column heap row plus
                             two secondary indexes, page overhead included)

CONSTRAINT
  Assumes 32-bit node identifiers: valid only while n < 2^31 - 1. Above that the
  offset and target arrays must widen to 8 bytes and the estimate changes by
  roughly (8-4)*(n+1) + (8-4)*m bytes. Assumes the arrays are dense and exactly
  sized; a growable vector with 1.5x over-allocation costs up to 50% more.

NUMERICAL EXAMPLE (toy graph: n = 6, m = 9, w = 8, t = 8)
  offsets      = 4 * 7              =  28 B
  targets      = 9 * 4              =  36 B
  weights      = 9 * 8              =  72 B
  rel codes    = 9 * 1              =   9 B
  timestamps   = 9 * 8              =  72 B
  total        = 28 + 36 + 72 + 9 + 72 = 217 B
  per edge     = 217 / 9            = 24.11 B
  row store    = 9 * 176            = 1584 B
  ratio        = 1584 / 217         = 7.30x
  without the per-edge timestamp: 145 B total, 16.11 B per edge, ratio 10.92x

INTERPRETATION
  At one billion edge occurrences the same arithmetic gives 24.1 GB for CSR and
  176 GB for the row store. The ratio is not a constant of nature: it narrows as
  the value payload grows, because a row can hold many columns for one header,
  while CSR pays per parallel array. The comparison is therefore valid only for
  the stated column set, and a design review must restate the column set when it
  restates the estimate.
```

### 4.2 General case: multigraph degree, distinct neighbours and projection size

```text
FORMULA
  d_occ(v)   = number of incident edge occurrences
  d_dist(v)  = |{ u : an edge occurrence exists between v and u }|
  d_dist(v) <= d_occ(v),  with equality iff no pair is repeated

WHERE
  d_occ, d_dist integers >= 0, per node per snapshot.
  The gap g(v) = d_occ(v) - d_dist(v) is the number of repeated pair occurrences
  incident to v, i.e. the multiplicity surplus.

CONSTRAINT
  The inequality is the definition, not an empirical finding, so it cannot be
  violated by data; it can only be destroyed by a layout. A layout that stores one
  row per distinct pair has already collapsed d_occ into presence and cannot
  report d_occ at all. Any feature that divides by degree must therefore declare
  which degree it uses, and the two produce different normalisations on identical
  data.

NUMERICAL EXAMPLE (toy graph)
  A1: d_occ = 5 (E1,E2,E3 to M1; E6 to D1; E9 from P1)
      d_dist = 3 (M1, D1, P1)          g = 2
  A2: d_occ = 4 (E4,E5 to M1; E7 to D1; E8 to D2)
      d_dist = 3 (M1, D1, D2)          g = 1
  M1: d_occ = 5, d_dist = 2            g = 3
  total d_occ = 18, total d_dist = 12, total surplus = 6
  cross-check: repeated pairs are A1-M1 (3 occurrences, surplus 2) and
  A2-M1 (2 occurrences, surplus 1); each surplus is counted at both endpoints,
  so 2*(2+1) = 6, matching the degree surplus.

INTERPRETATION
  On this graph a weighted mean over neighbours and an unweighted mean over
  neighbours differ for every node that touches M1. If the feature is "mean
  neighbour risk" and M1 is a merchant with many transfers, the weighted variant
  is dominated by transfer count while the unweighted variant is dominated by
  merchant identity. Both are defensible; only one is documented, and the
  undocumented switch is undetectable downstream because both produce a number in
  the same range.
```

Projection is the second place where the two degrees diverge, and it is where cost explodes. Projecting a bipartite structure onto one side manufactures a clique per shared node, so the projected edge count is quadratic in neighbourhood size and independent of how many pairs are genuinely interesting.

```text
FORMULA
  pairs produced by projecting a shared node of degree k   = k*(k-1)/2
  total projected pairs over a node set S                  = sum_{s in S} k_s*(k_s-1)/2

WHERE
  k_s = number of neighbours of shared node s (integer >= 0)
  S   = shared nodes used for the projection (devices, addresses, merchants)
  Output counts unordered pairs; use k*(k-1) if ordered pairs are materialised.

CONSTRAINT
  Assumes the projection is a full clique. Any pruning rule (only pairs sharing at
  least two attributes, only pairs above a similarity threshold) changes the count
  and must be stated, because a pruned projection is no longer a function of the
  graph alone. Assumes pairs are materialised rather than computed on demand; an
  on-demand join has the same output cardinality and a different cost profile.

NUMERICAL EXAMPLE
  toy graph: M1 has k = 2 -> 1 pair; D1 has k = 2 -> 1 pair; total 2 pairs.
  a hub device with k = 50,000 -> 50,000*49,999/2 = 1,249,975,000 pairs.
  at 100 ns per pair emitted (ILLUSTRATIVE constant), 1.25e9 pairs = 125 s of
  pure emission for one hub, before any downstream aggregation.
  k = 100,000 gives 4,999,950,000 pairs, a 4x increase for a 2x degree increase.

INTERPRETATION
  Quadratic growth in hub degree is the mechanical reason supernodes must be
  handled by a policy, not by a bigger machine. The cost of a projection is set by
  the largest k in S, not by the mean, which is why a mean-based capacity estimate
  for a projection pipeline is systematically optimistic.
```

### 4.3 Variants

**Variant A — reified facts.** Expressing each edge as a node with property statements multiplies row count by the number of properties carried plus two structural statements.

```text
FORMULA
  rows_reified(edge) = 2 + p
  where p = number of properties stored on the edge (value, currency, timestamps,
  provenance reference). Structural statements are hasSubject and hasObject.

CONSTRAINT
  Each statement is a row; a triple store may pack several statements per row, in
  which case the row multiplier is lower but the logical count is unchanged.
  Cost comparison is only valid between layouts storing the same property set.

NUMERICAL EXAMPLE (p = 3: value, event time, availability time)
  rows per edge = 5; at 56 B per triple row (ILLUSTRATIVE), 280 B per edge
  versus 176 B for the flattened row: ratio 280/176 = 1.59x
  for m = 9: 2520 B versus 1584 B.

INTERPRETATION
  Reification is priced at roughly 1.6x on this property set, and that price buys
  the ability to attach a new property to an existing edge without migrating the
  table. Buy it when edges are audited individually; skip it when edges are only
  ever aggregated, because the aggregate cannot see the difference and the
  traversal pays for the extra hops.
```

**Variant B — bitemporal widening.** Adding system-validity timestamps to event time and availability time gives four temporal columns.

```text
FORMULA
  B_bitemporal(m) = B_layout(m) + m * 4 * b_ts
  where b_ts = 8 bytes per int64 timestamp and 4 timestamps are stored
  (event, availability, row-insert, row-supersede; the last is nullable but
  still occupies its slot in a fixed-width row).

CONSTRAINT
  Assumes fixed-width columns. A columnar store with dictionary or delta encoding
  can compress monotone timestamps heavily, so the 32 B per edge is an upper bound
  for a row-oriented layout and an overestimate for a sorted columnar one.

NUMERICAL EXAMPLE (toy graph, m = 9, layout = CSR without per-edge timestamps)
  145 B + 9*32 = 145 + 288 = 433 B total, 48.11 B per edge.
  at 1e9 edges: 48.1 GB versus 16.1 GB for a single-timestamp layout, 3.0x.

INTERPRETATION
  The four-timestamp fact is the only form in which "what did we believe on
  Tuesday about what happened on Monday" is answerable. Paying 3x storage for it
  is usually correct for the base fact table and usually wrong for the serving
  structure, which is why the recommended pattern is bitemporal base, monotemporal
  snapshot, and no timestamps at all in the CSR used for scoring.
```

**Variant C — directed storage of a symmetric relation.** A symmetric relation stored once must be traversed in both directions; stored twice it occupies two rows and can diverge.

```text
FORMULA
  storage factor = 1 row per fact (single-direction storage, bidirectional query)
                 = 2 rows per fact (both directions materialised, one query path)

CONSTRAINT
  Materialising both directions is safe only if the write path is atomic across
  the two rows. A partial write leaves one direction present and the other absent,
  producing an asymmetric graph that fails the symmetry assertion
  A[i][j] == A[j][i] for exactly the interrupted writes.

NUMERICAL EXAMPLE (toy graph)
  symmetric facts: USED_DEVICE (E6, E7, E8) = 3, co-occurrence none.
  directed facts: TRANSFER (E1..E5) = 5, OWNS_ACCOUNT (E9) = 1.
  single-direction storage: 9 rows, every undirected query needs an OR on both
  endpoints; double-direction storage: 9 + 3 = 12 rows, a 33% growth that buys a
  single-direction query path for the symmetric relation only.

INTERPRETATION
  The choice is a query-shape bet with a correctness clause attached: doubling is
  fine, partial doubling is not. If the write path cannot be atomic, store once
  and pay the OR in the query, because a silently asymmetric graph corrupts every
  symmetric aggregate and is invisible in row counts.
```

**Variant D — delta plus base layout.** A mutable graph is served as a small adjacency-list delta merged into an immutable CSR base on a schedule.

```text
FORMULA
  query cost = O(d_base(v) + d_delta(v))
  merge cost = O(n + m_base + m_delta) amortised over k merges
  memory     = base arrays + delta structures + one spare copy during merge

CONSTRAINT
  Correct only if every fact appears in exactly one of base or delta, which requires
  the merge to be atomic with respect to reads. A read that observes the base after
  it has been rewritten but the delta before it has been cleared sees each delta
  fact twice; a read in the opposite order sees none.

NUMERICAL EXAMPLE (m_base = 1e9, m_delta = 1e6, merges every 6 hours)
  query overhead: delta lists add at most the delta degree, which for a hub-free
  graph is a handful of positions, so the p99 query cost is essentially unchanged.
  merge cost: (1e8 + 1e9 + 1e6) operations ≈ 1.1e9, at 1e8 ops/s = 11 s per merge,
  44 s per day, 4.4e-4 of a day.
  memory: base 13.4 GB + delta ~13 MB + spare 13.4 GB during merge = 26.8 GB peak.

INTERPRETATION
  The delta pattern is what makes CSR compatible with a graph that changes, and its
  hidden cost is the spare copy: peak memory doubles during a merge, so a host sized
  to hold the base exactly will fail at the first merge. Either size for 2x or merge
  into a partitioned base one shard at a time.
```

**Variant E — partitioned layout by tenant or component.** Splitting the graph into independent sub-structures changes the cost from "one big build" to "many small builds plus a boundary".

```text
FORMULA
  B_partitioned = sum over partitions p of B_CSR(m_p, n_p) + boundary arrays
  boundary arrays cost O(m_cross) for the edges whose endpoints are in different
  partitions, which must be stored separately because neither partition owns both.

CONSTRAINT
  Correct only when the query never needs an unbounded path across a boundary. A
  two-hop query that crosses the boundary needs the cross arrays plus a join, and a
  k-hop query with k large degenerates to the full graph.

NUMERICAL EXAMPLE (m = 1e9 split into 20 tenant partitions of 5e7 edges each,
n = 1e8 so 5e6 nodes each, cross edges 2% = 2e7)
  per partition: 4*(5e6+1) + 5e7*13 = 20 MB + 650 MB = 670 MB
  twenty partitions: 13.4 GB, identical to the unpartitioned estimate, as expected
  boundary: 2e7 edges * 13 B = 260 MB, plus an index on endpoint pairs
  peak during a single-partition rebuild: 670 MB, not 13.4 GB

INTERPRETATION
  Partitioning does not reduce total storage; it reduces the peak working set and
  enables per-tenant rebuilds. That is the actual benefit, and it is worth paying a
  2% storage overhead in boundary arrays to get a 20x smaller rebuild unit.
```

### 4.4 Limits and counterexamples

The formulas above are exact under their constraints. Each breaks in a specific, predictable way, and the break is more informative than the identity.

**Counterexample 1 — the dense limit of an adjacency matrix.** An adjacency matrix costs n² cells regardless of m, so its storage is independent of sparsity and becomes absurd immediately. At n = 1,000,000 with float64, the matrix is 1e12 cells × 8 B = 8 TB even if the graph has a thousand edges. The formula's validity condition is density, not size: the matrix is the right structure only when m is a non-trivial fraction of n², which for real relationship data almost never holds. The counterexample matters because dense matrix operations are the fastest available primitive, so the temptation to densify a sparse graph is strong and the failure is a memory wall rather than a wrong answer.

**Counterexample 2 — supernode fan-out in a two-hop traversal.** The identity "two-hop cost ≈ mean degree squared" assumes independence between the degree of a node and the degree of its neighbours, which is false in every real network with hubs. A node with 50 neighbours whose neighbours average 200 each yields 10,000 second-hop visits; if one of those neighbours is a hub with 50,000 edges, that single branch contributes 50,000 visits and dominates the cost. The mean-based estimate is not merely imprecise, it is structurally biased low, because the quantity that governs cost is the maximum neighbour degree on the frontier, not the average.

**Counterexample 3 — projection loss under a threshold.** A projection that keeps only pairs sharing two or more attributes is no longer the 2-section of the graph; it is a different graph whose edges depend on the threshold. Two facts follow. First, the projected edge count is no longer computable from k alone, so the cost formula does not apply. Second, and worse, the projection is no longer monotone in the data: adding a true edge can *remove* a projected pair if the pruning compares a ratio, which means a snapshot cannot be incrementally updated and the feature is not reproducible from the base table without re-running the pruning rule at its original threshold.

**Counterexample 4 — bitemporal storage without a supersede rule.** Adding four timestamp columns does not by itself produce bitemporal semantics; it produces four columns. If a correction is written as a new row without marking the superseded row, both rows remain visible at every decision time and the aggregate double-counts. The invariant that must hold is: for any (edge_id, system-time range), at most one row is visible. That is a constraint enforced by the write path, not a consequence of the schema, and it is the reason bitemporal tables need a uniqueness assertion rather than a convention.

**Counterexample 5 — the null-model ceiling for motif counts.** A motif count is a property of the graph, not evidence of association, because a star of five accounts on one device is guaranteed by the degree sequence alone if the device has degree five. Any claim of the form "these accounts form an unusual structure" requires a comparison against a null model preserving the degrees; without that comparison, motif counts measure the degree distribution and nothing else. The formula boundary is therefore the claim boundary: a count can be computed exactly and still be uninformative.

**Takeaway:** Every layout identity is exact only inside its density, independence and monotonicity assumptions; the counterexamples mark where a correct formula yields a useless answer.

## 5. Worked Numerical Example

The example is a six-node typed multigraph built to be small enough to check by hand. All identifiers, amounts and timestamps are ILLUSTRATIVE.

### 5.1 The fact table

| edge_id | src  | rel          | dst  | value_usd | t_event            | t_avail            | source    |
|---------|------|--------------|------|-----------|--------------------|--------------------|-----------|
| E1      | A1   | TRANSFER     | M1   | 250.00    | 2026-03-01T09:00Z  | 2026-03-01T09:00:02Z | ledger_1 |
| E2      | A1   | TRANSFER     | M1   | 90.00     | 2026-03-02T11:30Z  | 2026-03-02T11:30:03Z | ledger_1 |
| E3      | A1   | TRANSFER     | M1   | 40.00     | 2026-03-04T08:15Z  | 2026-03-04T08:15:02Z | ledger_1 |
| E4      | A2   | TRANSFER     | M1   | 600.00    | 2026-03-01T09:05Z  | 2026-03-01T09:05:02Z | ledger_1 |
| E5      | A2   | TRANSFER     | M1   | 15.00     | 2026-03-03T19:45Z  | 2026-03-03T19:45:09Z | ledger_1 |
| E6      | A1   | USED_DEVICE  | D1   | null      | 2026-02-20T07:00Z  | 2026-02-20T07:00:31Z | sdk_telemetry |
| E7      | A2   | USED_DEVICE  | D1   | null      | 2026-02-21T22:10Z  | 2026-02-21T22:10:30Z | sdk_telemetry |
| E8      | A2   | USED_DEVICE  | D2   | null      | 2026-03-05T06:00Z  | 2026-03-05T06:00:04Z | sdk_telemetry |
| E9      | P1   | OWNS_ACCOUNT | A1   | null      | 2025-11-02T00:00Z  | 2025-11-02T00:00:01Z | onboard_batch |

Nine rows, six nodes, three relation types, two value-bearing column families and one null-bearing one. The nulls are informative: a device sighting has no amount, and a schema that forces 0.00 into `value_usd` for it has invented a transfer of zero that any average will silently absorb. This is the first modelling decision in the example and it is made by choosing nullable columns plus a relation discriminator rather than a typed table per relation.

### 5.2 Step 1 — degrees under both definitions

| node | occurrence degree | distinct-neighbour degree | surplus | neighbours   |
|------|-------------------|---------------------------|---------|--------------|
| A1   | 5                 | 3                         | 2       | M1, D1, P1   |
| A2   | 4                 | 3                         | 1       | M1, D1, D2   |
| M1   | 5                 | 2                         | 3       | A1, A2       |
| D1   | 2                 | 2                         | 0       | A1, A2       |
| D2   | 1                 | 1                         | 0       | A2           |
| P1   | 1                 | 1                         | 0       | A1           |
| sum  | 18                | 12                        | 6       |              |

Check one: 2m = 18, matching the occurrence sum. Check two: the surplus column sums to 6, and the repeated pairs A1-M1 and A2-M1 contribute 2 and 1 at each endpoint, so 2(2)+2(1) = 6. Check three: the distinct-neighbour sum 12 equals twice the number of distinct unordered pairs, which is 6 — and 6 pairs are visible in the neighbour column: A1-M1, A1-D1, A1-P1, A2-M1, A2-D1, A2-D2.

### 5.3 Step 2 — the as-of snapshot at a decision time

Take decision time t = 2026-03-05T00:00:00Z. The eligible set is every fact whose availability time precedes t. E8 becomes available at 06:00Z on the fifth of March, four hours after the decision, so it is excluded even though its event time is also after t. A naive event-time filter would give the same answer here and a different one a day later, which is precisely why the two filters must be tested on a case where they disagree.

| edge | t_avail        | eligible at t | reason                     |
|------|----------------|---------------|----------------------------|
| E1   | 03-01T09:00:02 | yes           | before t                   |
| E2   | 03-02T11:30:03 | yes           | before t                   |
| E3   | 03-04T08:15:02 | yes           | before t                   |
| E4   | 03-01T09:05:02 | yes           | before t                   |
| E5   | 03-03T19:45:09 | yes           | before t                   |
| E6   | 02-20T07:00:31 | yes           | before t                   |
| E7   | 02-21T22:10:30 | yes           | before t                   |
| E8   | 03-05T06:00:04 | no            | availability after t       |
| E9   | 2025-11-02T00:00:01 | yes       | before t                   |

Eight of nine facts are eligible: selectivity 8/9 = 0.8889. Degrees recomputed on the snapshot: A1 = 5, A2 = 3, M1 = 5, D1 = 2, D2 = 0, P1 = 1, summing to 16 = 2 × 8. Note that D2 is present as a node with degree zero — it exists because the identity layer knows about the device even though no eligible fact references it. Whether zero-degree nodes appear in the serving structure is a schema decision (keep the row, drop the entry) and it changes the denominator of any per-node average.

### 5.4 Step 3 — CSR construction for the snapshot

Order the nodes A1, A2, M1, D1, D2, P1 (indices 0–5). Each undirected fact contributes twice, once per endpoint; the directed facts are stored in the direction asserted. For a purely analytic adjacency view all eight edges are stored as undirected, giving 16 positions.

```text
adjacency (undirected view, as-of snapshot), sorted by (relation, target)
  index 0  A1  ->  M1(E1), M1(E2), M1(E3), D1(E6), P1(E9)          count 5
  index 1  A2  ->  M1(E4), M1(E5), D1(E7)                          count 3
  index 2  M1  ->  A1(E1), A1(E2), A1(E3), A2(E4), A2(E5)          count 5
  index 3  D1  ->  A1(E6), A2(E7)                                  count 2
  index 4  D2  ->  (none)                                          count 0
  index 5  P1  ->  A1(E9)                                          count 1
  total positions 16

CSR arrays
  offset = [0, 5, 8, 13, 15, 15, 16]          length n+1 = 7
  target = [2,2,2,3,5, 2,2,3, 0,0,0,1,1, 0,1, 0]   length 16
  weight = [250,90,40,0,0, 600,15,0, 250,90,40,600,15, 0,0, 0]
```

Three checks on the arrays. First, `offset` is non-decreasing and ends at 16, the position count. Second, the difference vector is `[5,3,5,2,0,1]`, which matches the snapshot degree column exactly. Third, the target array is symmetric under the offset ranges: A1's list contains M1 and M1's list contains A1 the same number of times, which is the assertion that the undirected view was built consistently.

### 5.5 Step 4 — cost of the snapshot in each layout

```text
per-edge storage, snapshot m = 8, n = 6, w = 8 (float64), t = 8 (int64)
  offsets      4 * 7            =  28 B
  targets      8 * 4            =  32 B
  weights      8 * 8            =  64 B
  rel codes    8 * 1            =   8 B
  timestamps   8 * 8            =  64 B
  total                           196 B     -> 24.50 B per edge
  without timestamps              132 B     -> 16.50 B per edge

row store, 8 rows * 176 B        = 1408 B     -> ratio 7.18x
reified (p = 3), 5 rows * 56 B   = 2240 B     -> 11.43x
adjacency matrix 6*6*8 B         =  288 B     -> denser than CSR at this size
```

The matrix is cheaper here, and that is the honest result: at six nodes the dense encoding wins because the offset array is overhead. The crossover is a function of density, not of principle. With n = 6 the matrix has 36 cells for 16 stored positions; with n = 1,000,000 it has 1e12 cells for maybe 5e6 positions and loses by five orders of magnitude. A design note that says "CSR is faster" without the node count is not a finding.

### 5.6 Step 5 — projection and one feature

Project the bipartite device structure onto accounts. M1 and P1 are not devices, so the projection universe is {D1, D2}. D1 has k = 2 (A1, A2) and produces 1 pair; D2 has k = 1 and produces none. The projected graph therefore has one account pair (A1, A2), which is a much weaker statement than "A1 and A2 are related": one shared device, observed twice within 24 hours, in a population where a device token can be a kiosk.

Compute a weighted mean transfer value per account on the snapshot, using occurrence multiplicity:

```text
A1 transfers: E1 250.00, E2 90.00, E3 40.00  -> sum 380.00 over 3 occurrences = 126.67
A2 transfers: E4 600.00, E5 15.00            -> sum 615.00 over 2 occurrences = 307.50
unweighted-by-occurrence mean over distinct transfer partners:
A1 partner {M1} -> 250+90+40 = 380.00 / 1 partner? no: per-partner mean is 380.00
   the two answers differ because one divides by occurrence count and the other
   by distinct-partner count; both are labelled and neither is "the" mean.
```

The arithmetic is trivial; the point is that the same three numbers produce 126.67 or 380.00 depending on a denominator choice made in a feature definition, and no downstream metric can tell which was used. The feature contract must state: numerator = sum of `value_usd` over eligible TRANSFER facts incident to the account in the window; denominator = count of eligible TRANSFER facts; null policy = exclude accounts with zero eligible transfers rather than emitting 0.00.

### 5.7 Independent re-check and sensitivity

Six independent checks: (i) 2m = 16 equals the snapshot degree sum; (ii) CSR offset differences equal the degree vector; (iii) the target array is symmetric in count; (iv) the distinct-pair count 6 equals half the distinct-neighbour sum 12; (v) the projected pair count 1 equals C(2,2) for D1; (vi) the row-store ratio 7.18x is consistent with 176/24.50 = 7.18. Each check is arithmetic on the same table, so agreement does not prove the table is right — it proves the transforms are internally consistent, which is a different and still necessary claim.

Sensitivity. Move the decision time forward four hours to 2026-03-05T06:00:01Z. E8 becomes eligible and the edge count rises from 8 to 9, a 12.5% change in m. A2's degree rises from 3 to 4 and D2 from 0 to 1. Any feature normalised by degree or by total edge count shifts for reasons that have nothing to do with behaviour, which is why features must be compared at a fixed snapshot identifier rather than at "the latest data". Move the decision time back to 2026-03-04T08:00:00Z and E3 drops out, changing A1's mean transfer value from 126.67 to (250+90)/2 = 170.00 — a 34% swing from a four-hour cut-off change on the same account.

### 5.8 Production-scale reading of the same arithmetic

At one billion edge occurrences and one hundred million nodes the numbers from Section 4.1 dominate: 24.1 GB for the CSR without timestamps and 32.8 GB with them, against 176 GB for the row store. Two consequences follow that a toy example cannot show. First, the snapshot cannot be rebuilt per request; it must be materialised and versioned, because rebuilding 24 GB of arrays inside a latency budget is not a thing that happens. Second, the hub policy determines whether the build completes at all: one node with 50,000 edges contributes 1.25e9 projected pairs, which is larger than the entire edge count and would be emitted as a projection before any pruning.

### 5.9 Identity-version sensitivity

The example so far has used one identity map. Change it: suppose version 7 merged P1 and A1 into a single person node `P1` with the account as an attribute, while version 6 kept them separate. The facts do not change; the node count does, and three derived numbers change with it.

```text
version 6: nodes = 6, A1 degree 5, P1 degree 1, m = 9, sum degree 18
version 7: nodes = 5 (A1 absorbed into P1), P1 degree 6,
           m = 9, sum degree 18 (unchanged, because merging does not move edges)
           E9 becomes a self-loop if the account is retained as a node and P1 also
           owns it; if the account is absorbed, E9 disappears and m = 8,
           sum degree = 16, distinct pairs = 5
```

Two lessons are arithmetic and one is procedural. The arithmetic lesson is that a merge can create or destroy self-loops depending on whether the merged node keeps the account as an entity, so the edge count is not merge-invariant even though the fact table is untouched. The procedural lesson is that any feature table must record the identity version, because the same row key `P1` now refers to different entities and a label joined across versions attaches to the wrong population. The check is a join from the feature table to the identity map on the recorded version; a missing node proves skew.

### 5.10 Rebuilding the snapshot for a second decision time

For a training set, many snapshots are needed. Rebuilding from scratch per decision time costs O(n + m) each; with weekly snapshots over two years that is 104 rebuilds of a 24 GB structure, or 2.5 TB of transient work. The alternative is one base build plus incremental filters, which works only if the filter is a monotone time predicate and the graph is append-only. Concretely:

```text
full rebuild per snapshot    104 * (n + m) = 104 * 1.1e9 = 1.14e11 ops
                              at 1e8 ops/s = 1140 s = 19 minutes per snapshot
                              times 104 = 32.9 hours of build time per training run
incremental (append-only)    one build + 104 * delta where delta = facts per week
                              at 1e7 facts/week the delta work is 104 * 1e7 = 1.04e9
                              ops, about 10 s in total, provided offsets can be
                              patched rather than rebuilt
```

The incrementality condition is what makes the difference, and it fails in two common ways. If the identity map changes between snapshots, the edge endpoint identifiers change and the CSR must be rebuilt rather than patched. If facts are corrected in place rather than superseded, the delta cannot express the removal. Both are reasons the bitemporal base is a prerequisite for incremental snapshots and not an optional refinement.

### 5.11 What the example deliberately does not model

The toy graph has no parallel edges between the same pair in different relations, no self-loops, no hyperedges, no deleted facts and no node with degree above 5. Each omission hides a class of defect that the failure catalogue covers but the arithmetic does not exercise. A teaching example that included all of them would be unreadable; a production schema review that assumed the clean case is how the defects get shipped. The practical response is to keep the toy graph for reasoning about identities and to keep a deliberately adversarial fixture — one hub, one self-loop, one repeated pair, one late fact, one unresolved identifier — for testing the load path.

**Takeaway:** The same nine facts support three different degree numbers, two different mean values and four different storage estimates; each must be named at the point of use, not inferred later.

## 6. Visual / Geometric Intuition

### 6.1 The adjacency matrix as a picture of the model

Write the snapshot as a 6×6 matrix with multiplicity in the cells, ordered A1, A2, M1, D1, D2, P1:

```text
        A1   A2   M1   D1   D2   P1
   A1 [  0    0    3    1    0    1 ]
   A2 [  0    0    2    1    1    0 ]
   M1 [  3    2    0    0    0    0 ]
   D1 [  1    1    0    0    0    0 ]
   D2 [  0    1    0    0    0    0 ]
   P1 [  1    0    0    0    0    0 ]
```

Non-zero cells: 18 of 36 = 0.50 density, symmetric across the diagonal. Three visual facts are immediate and each corresponds to a modelling statement. The multiplicity 3 and 2 in the M1 column is the repeated-pair structure that the simple-graph projection would erase. The block containing M1, D1, D2 has no edges inside it, because the graph is bipartite between accounts and non-account entities — a fact invisible in a node list and obvious in the matrix. And P1 has a single entry, marking it as peripheral in this snapshot regardless of how important the person is in reality.

The matrix also shows why the dense encoding is competitive at this size: 36 cells versus 16 positions is only a 2.25x overhead, while at a million nodes it is a 200,000x overhead. Density, not node count, is the quantity to reason with.

### 6.2 Incidence view: facts as the primary objects

Transposing the perspective gives a 6×9 incidence matrix whose columns are the edges. Each column has exactly two non-zeros (or one for a self-loop), so the column sums are all 2 and the row sums are the occurrence degrees. This is the correct mental picture for the fact layer: edges are first-class, and nodes are an aggregation of the columns that mention them. It is also why the fact table is the right source of truth — a node-centric view has already lost the columns.

```text
        E1 E2 E3 E4 E5 E6 E7 E8 E9
   A1 [  1  1  1  0  0  1  0  0  0 ]
   A2 [  0  0  0  1  1  0  1  1  0 ]
   M1 [  1  1  1  1  1  0  0  0  0 ]
   D1 [  0  0  0  0  0  1  1  0  0 ]
   D2 [  0  0  0  0  0  0  0  1  0 ]
   P1 [  0  0  0  0  0  0  0  0  1 ]
   column sums all 2; row sums = [5,4,5,2,1,1]
```

### 6.3 The projection as a geometric quotient

Projecting the bipartite account–device structure is a quotient operation: two accounts are identified if they share a device. Geometrically it collapses each device into a clique among the accounts that touch it. With D1 of degree 2 the clique is a single line; with a hub of degree 50,000 the clique is a near-complete graph with 1.25e9 edges, which is why projection is the operation that turns a sparse dataset into a dense one and why the resulting dense object is almost never materialised in full.

A useful picture for reviewers: the projection's edge set is the union of cliques, one per device, and the union is where the two failure directions live. Too few devices retained and real pairs are missed; every device retained and kiosks create cliques over unrelated people. Neither failure is visible in an aggregate count, because both produce a plausible number of projected pairs.

### 6.4 The temporal axis

Draw the fact table as a plane with event time on the x-axis and availability time on the y-axis. Facts sit at or above the diagonal (availability never precedes event time). A decision time t is a horizontal line; the eligible set is everything below it. A naive event-time filter is a vertical line, and the region between the two lines — early event time, late availability — is exactly the late-arrival region where backtests leak.

```text
   t_avail
     ^
     |                    * E8   <- late, excluded at t = 03-05T00:00
     |        ------------------------------ t (decision)
     |   *E1 *E2 *E4 *E3 *E5 *E6 *E7 *E9
     |________________________________> t_event
        region above the decision line but left of it in event time
        = facts the vertical filter would wrongly include
```

E8 is the concrete instance: event time 03-05T06:00Z, availability 03-05T06:00:04Z. At a decision time of midnight, an event-time filter excludes it and so does an availability filter, so this fact does not discriminate. A discriminating fact is one with an early event time and a late availability time; the example deliberately includes a latency of 31 s (E6) and 30 s (E7) so that a decision time inside that window separates the two filters. Teaching examples that never exercise the discriminating case give false confidence.

### 6.5 The hub as a geometric distortion

Draw the one-hop neighbourhood of an ordinary account and of a device hub at the same scale. The ordinary account's neighbourhood is a small disc; the hub's is a ring of 50,000 points, and the ring is wider than the entire rest of the graph. Any two-hop traversal that starts anywhere inside the hub's disc instantly gains the whole ring as its frontier, which is why a cost estimate built from the mean degree is wrong by orders of magnitude on exactly the queries that matter.

```text
mean-degree estimate for a 2-hop frontier
  start node degree 50, neighbour degrees averaging 200 -> 50 * 200 = 10,000 nodes

actual frontier when the start node is adjacent to one hub of degree 50,000
  49 ordinary neighbours * 200 = 9,800
  1 hub neighbour * 50,000     = 50,000
  total                        = 59,800 nodes, 5.98x the estimate

if the start node is itself the hub
  frontier = sum of neighbour degrees = at least 50,000 * median
```

The distortion is not a variance problem that more samples fix; it is a maximum-versus-mean problem, and the correct planning statistic is a high quantile of the neighbour-degree distribution, not its mean. A useful rule: size the traversal budget by the 99.9th percentile of neighbour degree, and cap the frontier explicitly with a stated selection rule so the cost bound is real rather than hoped for.

### 6.6 Sparse pattern in the incidence view

The incidence matrix of Section 6.2 has exactly two non-zeros per column. That is a stronger statement than "the matrix is sparse": it says every fact has exactly two endpoints, so the structure is a 2-uniform hypergraph. Hyperedges of size 3 or 4 break the column-sum invariant, and any code that assumes two endpoints per fact will either drop the extra participants or duplicate the row. The geometric picture is that the fact layer lives on a simplex of dimension two, and a size-4 hyperedge leaves that simplex. Recognising this is what tells a designer that a size-4 group event cannot be stored in a two-column table without a decomposition choice.

### 6.7 The projection lattice

Projections compose. Projecting accounts onto devices, then devices onto addresses, gives an account–address relation that is not the same as projecting accounts directly onto addresses. The composition loses the intermediate: after the first step you cannot tell which device produced an address link, so a later correction to a device merge cannot be propagated. The geometric statement is that projections form a lattice under composition, and each composition step reduces the rank of the recoverable lineage. A design that needs reversible projections must store the intermediate artifacts, which is a storage cost, not a query cost.

**Takeaway:** The matrix shows multiplicity and bipartiteness, the incidence view shows that facts are primary, the quotient view shows where projection explodes, and the time plane shows which filter leaks.

## 7. Comparison / Selection Table

| Pattern | Grain | Cheapest query | Expensive query | Information lost | Use when | Avoid when |
|---|---|---|---|---|---|---|
| Edge table (row store) | one fact per row, typed, bitemporal | filtered scan by time, type, node | multi-hop traversal | nothing, if all columns kept | base store of record; audit; corrections | latency-critical scoring at scale |
| Edge table with composite index | one fact per row, indexed pairs | one-hop by either endpoint | three-hop paths | nothing | serving a small snapshot per request | graphs above roughly 1e7 edges |
| CSR / CSC arrays | one position per (node, occurrence) | one-hop expansion, batch analytics | per-edge filtering, updates | relation-type and time filters unless parallel arrays added | nightly feature build, GNN batching | transactional writes, ad hoc predicates |
| Adjacency list of lists | one list per node | pointer-chasing traversal, iterative sampling | vectorised scans, shared-memory parallelism | nothing, but allocator-bound | small graphs, rapid prototyping | hub-heavy graphs, memory-constrained hosts |
| Adjacency matrix (dense) | n² cells | matrix products, spectral methods | anything sparse | nothing, wastes memory | n below a few thousand, spectral work | n above ~1e4 |
| Triple table (RDF-style) | one statement per row | predicate-first pattern matching | multi-hop joins, aggregates | values need extra triples or reification | schema evolution, heterogeneous sources | analytics over homogeneous facts |
| Reified property graph | edge as node with statements | per-edge provenance queries | bulk traversal | nothing | audited edges, per-edge annotations | high-volume aggregation |
| Hypergraph (star expansion) | node per hyperedge | incidence queries | pairwise adjacency | pairwise co-membership | group events, joint accounts | pairwise-only analysis |
| Hypergraph (2-section) | clique per hyperedge | pairwise adjacency | incidence, group identity | group identity, hyperedge count | pairwise similarity features | large hyperedges (quadratic) |
| Materialised projection | one row per projected pair | downstream pair features | provenance of the pair | the fact layer entirely | fixed pair-level model with no audit need | any need to explain or revise pairs |

Read the table as a set of bets, not a ranking. Three rows deserve explicit comment because they are commonly chosen for the wrong reason. The **adjacency matrix** row is chosen for spectral work, where the eigen-decomposition needs the dense form; choosing it to "simplify code" converts a sparse problem into a memory problem, and the failure appears only when the node count grows. The **materialised projection** row is chosen when a downstream model wants one row per pair; it is a legitimate choice, but it deletes the fact layer from the lineage, so an audit can no longer ask which device caused the pair. The **triple table** row is chosen for schema flexibility; the price is that every aggregate is a join, and joins on a flexible schema with an unconstrained predicate column are where query planners produce their worst plans.

Two derived selection rules follow from the cost formulas and are worth stating as heuristics with their conditions attached. If the dominant query is one-hop expansion over a frozen snapshot and the graph has more than roughly a million edges, CSR wins on both memory and scan locality. If the dominant query is "which edges involve this node, filtered by type and time", the row store wins despite the byte overhead, because the filter is applied before the join rather than after the expansion — and the 7x storage penalty buys an index that prunes 90% of the scan.

### 7.1 Selection procedure for a new dataset

The table above is a lookup; the procedure below is what produces a defensible choice when two rows look acceptable.

1. Write the decision the graph serves, with its latency budget and its refresh cadence. A nightly queue and an authorisation decision differ by five orders of magnitude in latency and cannot share a serving layout.
2. Write the fact grain explicitly. If the answer contains the word "per" twice, the grain is under-specified; for example "one row per transfer per account" is ambiguous about whether a transfer appears once or twice.
3. Enumerate the queries that will run, with their filters. The dominant query determines the layout; the second and third determine whether a second copy is needed.
4. Check every query against the loss column. If a query needs information the layout deletes, the layout is wrong regardless of speed.
5. Compute the storage from the formula, then multiply by the number of retained snapshots and the replication factor. Both multipliers are commonly forgotten and each is typically 2 to 3.
6. Name the rebuild path and its trigger. If there is no rebuild path, the layout is a store of record and must be append-only.
7. Write the assertion set: degree identity, offset monotonicity, relation registry coverage, time monotonicity. These are the checks that fail loudly when someone changes the loader.

Steps 1 and 3 are the ones that require talking to a consumer, and they are the steps most often skipped in favour of a benchmark. A benchmark answers "which is faster for the query I measured", which is a narrower question than "which is faster for the query that will dominate".

### 7.2 Two-layout architectures and when they are justified

The dominant pattern in practice is two layouts: a row store as the store of record and a CSR as the analytic replica. It is justified by three conditions, all of which must hold. The first is that the analytic workload can tolerate staleness bounded by the rebuild interval. The second is that the replica's rebuild is deterministic, so a rerun produces identical bytes. The third is that deletion propagates, meaning a deleted fact invalidates the replica within the same bound rather than surviving in it indefinitely.

When the third condition fails, the two-layout architecture becomes a correctness liability: the store of record and the replica disagree about whether a fact exists, and the replica is the one used for scoring. The failure is not detectable by row-count reconciliation unless deletions are counted separately, which is why deletion propagation deserves an explicit test and not a paragraph in a design document.

**Takeaway:** Choose the layout by naming the dominant query and the acceptable loss; a table that starts from the storage engine rather than the query has already skipped the design step.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, the causes that compete with it, and a check that separates them. The checks are the deliverable; the mechanisms are the vocabulary for asking.

### F-01 Multiplicity collapsed to presence

**Mechanism.** A pair occurring three times is stored once, usually because a `UNIQUE(src, dst)` constraint was added to make a join safe. Every downstream count then reports presence, and any frequency feature becomes a constant.

**Competing causes.** (a) multiplicity collapse at write time; (b) an as-of filter that is too aggressive and drops repeat occurrences; (c) deduplication in a SQL `SELECT DISTINCT` upstream of the feature. All three produce the same symptom: frequency features flat at 1.

**Discriminating check.** Count rows and count distinct pairs on the same snapshot for the same relation. If rows > pairs and the feature max is 1, the collapse is downstream of the fact table; if rows = pairs, the collapse is at write time. In the toy graph the ratio is 9/6 = 1.5, so a collapse would be detectable as a drop to 6.

### F-02 Availability time backdated to event time

**Mechanism.** A batch load sets `t_avail = t_event` because the pipeline has no load timestamp, making every historical snapshot appear to know facts it could not have known. Offline metrics rise; production metrics do not.

**Competing causes.** (a) missing availability column filled from event time; (b) a correction process that updates the availability of old rows; (c) a timezone shift that moves availability earlier than event time. All three raise offline metrics relative to production.

**Discriminating check.** Compute `min(t_avail - t_event)` over the fact table. A negative minimum proves a timezone or clock defect. A distribution with no mass above a few seconds proves the column is synthetic, because real ingestion has a tail. Then re-run one feature at a decision time inside the tail window: if the feature value changes when the availability rule is restored, the metric depended on the backdating.

### F-03 Supernode fan-out in traversal

**Mechanism.** A shared token acquires 50,000 edges, and any two-hop expansion that touches it visits the entire neighbourhood. The traversal does not merely slow down; it changes the composition of the result, because the hub contributes 50,000 neighbours out of an otherwise small frontier.

**Competing causes.** (a) genuine hub; (b) sentinel collapse such as `device = "unknown"`; (c) identifier recycling where one token was reused across years. All three look identical in degree.

**Discriminating check.** Sample twenty edges at the hub and inspect the source payload. If the device identifier is a constant string or a null, the hub is an artifact. If the edges are real but span years with disjoint account sets, it is recycling. If they are real and contemporaneous, it is a genuine hub and needs a policy (cap, downweight, exclude) rather than a fix.

### F-04 Projection threshold breaks monotonicity

**Mechanism.** A pruning rule keeps pairs sharing at least two attributes, or with similarity above 0.5. Adding a true edge changes the denominator of the similarity, so a pair can vanish when evidence is added. Incremental updates then produce a different graph from a full rebuild.

**Competing causes.** (a) non-monotone pruning; (b) unstable tie-breaking in the top-k of a similarity list; (c) non-deterministic iteration order in the projection job. All three produce rebuild-versus-incremental divergence.

**Discriminating check.** Build the projection twice: once from the full fact table, once by applying the incremental delta to a previous build. Compare pair sets exactly. Any symmetric difference is a reproducibility defect; then bisect with the pruning disabled — if the difference disappears, the pruning rule is non-monotone.

### F-05 Entity overmerge

**Mechanism.** Two people share a surname, a postcode and a birth year, and the resolution rule merges them. Their graphs union, and each appears to be connected to the other's counterparties.

**Competing causes.** (a) overmerge from weak attributes; (b) genuine shared account (joint account, power of attorney); (c) a merge caused by a data-entry error in one source. All three create edges between parties that do not co-operate.

**Discriminating check.** Take merged pairs and measure attribute agreement on a held-out adjudicated sample; report pair precision at the threshold actually used. Then check whether the merged node's edge set spans incompatible time ranges or geographies — a joint account does not typically produce transactions in two countries in the same minute.

### F-06 Entity undersplit

**Mechanism.** The same person appears as three nodes because three systems emit three identifier formats. The graph is fragmented and every degree is understated, so ring detection misses the connection that a human would see instantly.

**Competing causes.** (a) missing normalisation rule; (b) a threshold set too conservatively after an overmerge incident; (c) a source that genuinely uses separate identifiers with no shared attribute. All three lower apparent connectivity.

**Discriminating check.** Search for pairs of nodes with identical normalised attribute tuples but different identifiers; count them. A non-zero count with distinct identifiers means the rule is under-applying. If zero, the sources genuinely do not overlap on attributes and the remedy is a new attribute, not a lower threshold.

### F-07 Relation type used as a filter after projection

**Mechanism.** A projection pools `TRANSFER` and `REFUND` because both connect an account to a merchant. Aggregate features then mix money in with money out, and the mean value becomes meaningless without any error appearing.

**Competing causes.** (a) projection lost the relation discriminator; (b) a nullable relation column defaulted to `TRANSFER`; (c) an upstream join dropped the type to fit a two-column key. All three give a mixture distribution.

**Discriminating check.** Compute the feature separately per relation type on the same snapshot and compare distributions. If the pooled version is bimodal or has a variance far above either component, the discriminator was lost. Then inspect the projected table's schema: absence of the column is conclusive.

### F-08 Null value coerced to zero

**Mechanism.** Device sightings have no amount, and a `NOT NULL DEFAULT 0` converts absence into a measurement. A mean transfer value now divides by a count that includes non-transfers, and the mean falls without any behaviour change.

**Competing causes.** (a) default-zero coercion; (b) an outer join that produces nulls then summed without a null check; (c) a legitimate zero-value transfer. All three lower the mean.

**Discriminating check.** Count facts by relation type with `value IS NULL` before the coercion. In the toy graph four facts carry no value (E6–E9), so an average over all nine has denominator 9 while a correct average has denominator 5. Comparing 16.67 versus 30.00-style discrepancies exposes the coercion immediately.

### F-09 Correction overwritten in place

**Mechanism.** A mis-posted transfer is updated, not superseded. The prior fact disappears from history and any snapshot before the correction now contains the corrected value, so a backtest sees the future.

**Competing causes.** (a) in-place update; (b) a snapshot rebuilt from current state rather than from a bitemporal table; (c) a cache that was refreshed after the correction. All three make historical snapshots non-reproducible.

**Discriminating check.** Rebuild the snapshot for a past decision time from the base table and diff against the stored snapshot identifier. If the diff is non-empty and the base table has no supersede rows, the base itself is not bitemporal. The check requires the snapshot to have been stored, which is the argument for storing them.

### F-10 Direction dropped for symmetric convenience

**Mechanism.** A directed `TRANSFER` is stored with an unordered pair key so both directions can be queried with one index. In-degree and out-degree become the same number, and any flow-based feature silently becomes a volume feature.

**Competing causes.** (a) unordered key; (b) a normalisation that sorts the endpoints; (c) a view that presents the underlying table symmetrically. All three make in-degree equal out-degree.

**Discriminating check.** Assert that the count of rows with `in_degree != out_degree` is non-zero for a directed relation. In the toy graph A1 out-edges 3 and in-edges 1 (E9), so any snapshot where they are equal for every account has lost direction. The assertion is cheap and it fails loudly.

### F-11 Timestamp granularity mismatch

**Mechanism.** One source emits seconds and another milliseconds; a loader casts both to seconds by dropping digits, or to milliseconds by multiplying. The two directions differ by a factor of 1000, and as-of filters near a boundary include or exclude the wrong facts.

**Competing causes.** (a) unit mismatch; (b) timezone offset applied twice; (c) daylight-saving double counting in local time. All three move facts across a boundary.

**Discriminating check.** Compute the maximum gap between a fact's event time and its availability time. A gap near 1000 s, or a cluster at exactly ±3600 s, indicates the specific defect. Then check the fraction of facts with a negative gap: any non-zero fraction is conclusive for a zone error.

### F-12 Self-loop counted once

**Mechanism.** A transfer from an account to itself is stored as a self-loop and the degree aggregate adds one instead of two, so the handshake identity fails by the number of self-loops. A pipeline that treats this as corruption deletes real facts.

**Competing causes.** (a) genuine self-transfer; (b) an entity-resolution merge that creates a self-loop from a previously cross-account transfer; (c) a duplicate key collision. All three produce self-loops.

**Discriminating check.** Count self-loops and recompute `sum(degree) - 2m`; the residual should equal the self-loop count. If the residual equals the count, the loop handling is the only defect. If the residual also changes after a merge, the resolution step is synthesising loops and the merge rule needs a guard.

### F-13 Composite key collision across relation types

**Mechanism.** The fact table is keyed on `(src, dst)` without the relation, so a `USED_DEVICE` and a `OWNS` fact on the same pair cannot coexist; the second insert is rejected or overwrites. Relation types vanish from the graph silently.

**Competing causes.** (a) key missing the predicate; (b) upsert semantics on a shared key; (c) an ETL dedup that keeps the first row per pair. All three reduce the relation-type count.

**Discriminating check.** Compare the set of relation types in the source extract against the set present in the fact table for the same period. A missing type with a non-zero source count is conclusive. In the toy schema, keying without `rel` would have forced E6 and any `OWNS_DEVICE` fact to collide.

### F-14 Snapshot rebuilt from the wrong identity version

**Mechanism.** Features are computed on a snapshot built with identity version 7 while the training labels came from cases keyed on version 6. Two nodes that were split then are merged now, so the label attaches to a different entity.

**Competing causes.** (a) version skew; (b) a merge applied without invalidating stored features; (c) a label store keyed on raw identifiers. All three misalign labels and features.

**Discriminating check.** Join the feature table to the identity map on the version recorded in the snapshot metadata. Any node in the feature table absent from the stated version indicates skew. Then recompute one aggregate under both versions and compare: the difference bounds the contamination.

### F-15 Hub pruning applied at query time

**Mechanism.** A traversal caps the frontier at 1,000 neighbours, choosing them in storage order. Results become order-dependent and non-reproducible, and the cap silently excludes whichever neighbours happen to sit at the end of the array.

**Competing causes.** (a) arbitrary cap; (b) top-k by weight with unstable ties; (c) shuffled iteration order in a distributed job. All three produce run-to-run variation.

**Discriminating check.** Run the same query twice with a fixed seed and once with a different storage order; identical results under a seed but different results across orders isolates the ordering as the cause. A principled cap must be defined by a rule (highest weight, most recent) and the dropped fraction reported.

### F-16 Schema evolution that breaks the rebuild

**Mechanism.** A new relation type is added with a different value semantic (net rather than gross). The rebuild script does not know about it, so aggregates either exclude it silently or include it with the wrong meaning.

**Competing causes.** (a) unregistered relation type; (b) a value column reused for a different unit; (c) a nullable column repurposed. All three change feature semantics without changing code.

**Discriminating check.** Maintain a relation registry with units and null policy; assert that every distinct `rel` value in the fact table appears in the registry. Compare the distinct-value count per build and fail the build on an unregistered value rather than logging it.

### F-17 Identifier recycling across entity lifetimes

**Mechanism.** A device token or account number is reassigned after closure, so two unrelated entities occupy one node. The graph joins their histories and every aggregate that spans both is meaningless while remaining numerically plausible.

**Competing causes.** (a) recycling after closure; (b) an entity-resolution merge that keyed on the identifier alone; (c) a test fixture loaded into a production namespace by mistake. All three produce a node with disjoint, non-overlapping activity bursts.

**Discriminating check.** For each node, compute the maximum gap between consecutive incident facts and the overlap of attribute values across the bursts. A node whose activity is two widely separated bursts with no shared attributes is either recycled or misloaded; a node merged on a genuinely shared attribute shows attribute agreement across the gap. Confirm with the source's assignment dates, which are the only authority on recycling.

### F-18 Namespace collision on numeric identifiers

**Mechanism.** Two systems emit eight-digit identifiers that look alike, and a join on the identifier silently mixes an account with a merchant. The resulting edges are structurally ordinary and semantically wrong, so every statistical check passes.

**Competing causes.** (a) missing namespace prefix; (b) a union of two extracts with a compatible schema; (c) a manual reconciliation spreadsheet that assumed one space. All three produce cross-type edges that are rare but not improbable.

**Discriminating check.** Assert that every edge's source and target belong to the relation type's declared endpoint types. For `TRANSFER` the assertion is `type(src) = account AND type(dst) = merchant`. Any violation is a collision, and the count gives the blast radius. This is a schema assertion, not a statistical test, which is why it should run in the load rather than in a review.

### F-19 Derived edge recomputed from a post-outcome signal

**Mechanism.** An edge of type `SUSPECTED_LINK` is created by a rule that reads investigator notes, and the notes are written after the case is confirmed. The edge is available immediately, correctly stamped, and contaminated by provenance: it encodes the outcome the model is trying to predict.

**Competing causes.** (a) post-outcome derivation; (b) a legitimate adjudicated link that predates the outcome; (c) a backfill of a link whose evidence was discovered earlier. All three produce a link whose availability time is earlier than the label but whose information content is not.

**Discriminating check.** Timestamp filtering cannot catch this, so the check must be lineage-based: for every derived edge type, list the input artifacts and test whether any input is written by a process that runs only after adjudication. Then run the model with the derived type excluded; a large metric drop with the type included is evidence that the type carried outcome information. The discriminating step is the exclusion experiment, not the timestamp audit.

### F-20 Write path bypassed by a bulk correction

**Mechanism.** A monthly correction loads a CSV straight into the table, skipping the supersede logic, so both the wrong and the right value remain visible with the same event time. Aggregates double-count for every affected identifier.

**Competing causes.** (a) bypassed write path; (b) a duplicate load of the same batch; (c) a retry after a partial failure that reinserted already-committed rows. All three raise counts for a bounded set of identifiers.

**Discriminating check.** Group by the natural fact key and count rows; any group with more rows than the number of legitimate versions is a violation. If the duplicates share a load identifier, the cause is a retried batch; if they differ in load identifier but share the event key, the cause is a bypassed supersede.

### F-21 Schema default masking a missing join

**Mechanism.** A left join to the identity map fails for unresolvable identifiers, and a `COALESCE(node_id, 0)` puts every unresolved raw identifier into node 0. Node 0 becomes a supernode whose edges are unrelated by construction.

**Competing causes.** (a) coalesced null; (b) a sentinel value from the source; (c) a legitimate shared account. All three create one node with an implausibly large degree.

**Discriminating check.** Count facts whose resolved node equals the sentinel. A non-zero count is conclusive, because no genuine entity is identified by the sentinel. Then correlate the sentinel degree with the count of unresolved identifiers per load: equality proves the coalesce is the mechanism.

### F-22 Timezone-naive comparison across sources

**Mechanism.** One source stores local time without an offset and another stores UTC, so facts from the first shift by the offset. Edges that were simultaneous appear hours apart, and any co-occurrence window feature under-counts.

**Competing causes.** (a) missing offset; (b) offset applied twice; (c) daylight-saving transitions. All three shift a subset of facts by a round number of hours.

**Discriminating check.** Histogram `t_avail - t_event` by source. A source with a mode at exactly the local offset, or a bimodal split around 23:00 and 01:00 on transition dates, is conclusive. A single-source check cannot detect this; the histogram must be per source system, because the defect lives in one exporter.

### F-23 Co-occurrence window defined on event time after the fact

**Mechanism.** A co-occurrence edge is created for any pair of accounts on one device within 24 hours. The rule is applied at query time over all history, so a pair becomes adjacent retroactively; snapshots rebuilt at an earlier decision time therefore contain edges that did not exist then.

**Competing causes.** (a) retroactive window; (b) a window defined on availability time in one implementation and event time in another; (c) a boundary rule that counts an exactly-24-hour gap inconsistently. All three shift which pairs are adjacent near the boundary.

**Discriminating check.** Build the co-occurrence structure at two decision times with and without the derived edges, and check whether the earlier snapshot gained edges. Any addition to a past snapshot is conclusive for retroactivity. Then test the boundary explicitly with a pair exactly at the window edge.

### F-24 Cardinality explosion in the value column

**Mechanism.** A free-text `channel` value becomes part of the edge key, so the same relationship appears once per distinct string, including typos. Edge count inflates, degree inflates, and the multiplicity is an artifact of poor typing.

**Competing causes.** (a) unconstrained categorical column; (b) a schema change that moved a comment field into a key; (c) a vendor adding new enum values. All three raise the distinct-value count for one column.

**Discriminating check.** Count distinct values per column against a declared domain. A column with thousands of values where the registry declares twelve is conclusive, and the ratio of counts gives the inflation factor directly.

### F-25 Storage estimate that ignores the secondary indexes

**Mechanism.** A capacity plan counts only the heap rows and omits the indexes, so the deployment is sized at roughly 60% of the real footprint and the first month's growth triggers paging.

**Competing causes.** (a) omitted indexes; (b) omitted replication factor; (c) omitted snapshot copies kept for reproducibility. All three make the estimate low by a multiplicative factor.

**Discriminating check.** Recompute the estimate from the physical bytes on disk divided by the row count, and compare with the plan's per-row figure. A ratio above the heap-row size proves the plan omitted structures; the ratio itself tells you how much.

### F-26 Lossy integer widening during a migration

**Mechanism.** Node identifiers are renumbered into 32 bits, and a hash collision merges two nodes. The collision rate is low enough to pass review and high enough to matter at scale, and collisions are permanent.

**Competing causes.** (a) hash truncation; (b) a modulo shard assignment reused as an identifier; (c) a legitimate merge. All three create a small number of merged nodes.

**Discriminating check.** Recompute the identifier mapping twice, once with the original width, and diff. Any node whose member set differs is a collision. Because the check is exact rather than statistical, it should run as a one-off migration gate rather than as a monitor.

**Takeaway:** Each entry separates a symptom that three mechanisms share, so the check must change the observation rather than re-examine the same number.

## 9. Complexity & Cost Analysis

Costs below are stated in operation counts with ILLUSTRATIVE hardware constants (1e8 simple operations per second, 100 ns per random memory access, 10 ns per contiguous access). They are for comparing designs, not for predicting a deployment.

| Operation | Pattern | Time | Space | Notes |
|---|---|---|---|---|
| Ingest one fact | row store append | O(1) amortised | O(1) | plus index maintenance, typically O(log m) |
| Ingest one fact | CSR append | O(1) amortised | O(1) | only if the node is the last one; otherwise rebuild |
| One-hop expansion | CSR | O(d_v) contiguous | O(1) extra | d_v occurrence degree |
| One-hop expansion | row store with index | O(log m + d_v) | O(1) | index seek dominates for small d_v |
| Two-hop expansion | CSR | O(sum of d over the frontier) | O(frontier) | dominated by max degree on the frontier |
| k-hop bounded traversal | CSR + visited set | O(sum of d over visited) | O(visited) | visited set is the memory bound |
| As-of filter | any, with timestamp array | O(m) full scan or O(log m) per index range | O(1) | the filter is the cost, not the layout |
| Full sort by degree | any | O(n log n) | O(n) | avoid; a heap gives O(n log k) |
| Projection | any | O(sum C(k_s,2)) | O(pairs) | quadratic in hub degree |
| Rebuild snapshot | CSR | O(n + m) | O(n + m) | 24.1 GB at m = 1e9 |
| Triangle count | CSR, node iterator | O(m^1.5) bound, often far less | O(m) | AGM bound; degree ordering helps |
| Materialise CSC | transpose of CSR | O(n + m) | O(n + m) | one extra copy of the position array |

Three readings of the table. First, **as-of filtering is a full-graph operation when it is not indexed**, so a design that filters at query time on a 1e9-edge table pays 1e9 predicate evaluations per request; the correct pattern is a materialised snapshot with the filter applied once. Second, **CSR ingest is append-only in the favourable case and a rebuild otherwise**: inserting a position for node 3 shifts every offset after it, so a mutable graph needs a different structure (adjacency lists, or a delta array merged periodically). Third, **projection cost is set by the largest k**, so a capacity estimate must use the maximum hub degree, not the mean; the mean is the statistic that makes pipelines look affordable and then time out.

Worked cost comparison for one nightly feature build at m = 1e9, n = 1e8 (ILLUSTRATIVE constants):

```text
CSR build
  allocate      (n+1)*4 + m*(4+8+1)  = 400 MB + 13 GB   = 13.4 GB
  scan + write  m predicate evaluations + m stores        = 1e9 + 1e9 ops
  at 1e8 ops/s                                            = 20 s
  snapshot copy for immutability (one extra array set)    = +13.4 GB

row store build
  scan          m rows * 176 B read                       = 176 GB of I/O
  at 2 GB/s sequential read                               = 88 s
  index build   m * O(log m) comparisons                  = 1e9 * 30 = 3e10
  at 1e8 comparisons/s                                    = 300 s
  total dominated by index build                          ~390 s

projection
  sum C(k_s,2) with one hub k = 50,000 and 1e5 other hubs of k = 100
  hub contribution   1.25e9 pairs
  other contributions 1e5 * 4,950 = 4.95e8 pairs
  total ~1.75e9 pairs, i.e. 1.75x the edge count, emitted once per build
  at 1e7 pairs/s (emission plus dedup)                    = 175 s
```

The comparison is the useful part: the index build, not the scan, dominates the row-store path, and the projection, not the traversal, dominates the feature path. A design review that budgets only storage will approve a pipeline whose cost is 90% elsewhere.

Memory pressure has a second-order effect that belongs in the cost section because it changes asymptotics in practice. A CSR that does not fit in RAM is not merely slower; each frontier expansion becomes a random read per position, so an in-memory O(d_v) contiguous scan degrades to O(d_v) random accesses at 100 ns each, a 10x constant. The design consequence is that the snapshot size should be chosen to fit the memory of the serving host, and if it does not, the correct move is to partition the graph (by tenant, by region, by connected component) rather than to accept thrash.

### 9.1 Where the cost actually lands at three scales

The same design has different bottlenecks at different scales, and a cost analysis that ignores the transition produces a plan that is correct for the wrong size.

| Scale | Edges | Bytes as CSR | Dominant cost | Design consequence |
|---|---|---|---|---|
| Prototype | 1e4 | 130 KB | nothing measurable; developer time | use whatever is expressible; do not optimise |
| Single host analytic | 1e7 | 130 MB | build time, memory for intermediates | one process, in memory, no partitioning |
| Single host at limit | 1e8 | 1.3 GB | random access on hub expansion | hub policy becomes mandatory |
| Multi-host | 1e9 | 13 GB | I/O and shuffle; partition boundary | partition by tenant or component, accept boundary arrays |
| Beyond memory | 1e10 | 130 GB | paging or distributed storage latency | rethink the problem; aggregate first, traverse second |

The table's usefulness is in the fourth row, where the cost stops being arithmetic and becomes a distribution question. Up to roughly 1e8 edges a single host with a hub policy works because the working set fits and the random accesses hit cache. Past that, the same code degrades by a factor of ten per frontier expansion because every offset read misses, and the remedy is structural rather than procedural.

### 9.2 Amortisation of a rebuild

A nightly rebuild of a 24 GB structure costs the same every night regardless of how much changed. If facts accumulate at 1e7 per day, the rebuild processes 1e9 edges to absorb 1e7, a ratio of 100:1 wasted work.

```text
full rebuild      cost 1e9 edge operations + 1e9 writes per night
incremental       1e7 edge operations + a merge of the delta arrays
ratio             100x less work per night, at the price of:
                  (a) a merge step that doubles peak memory;
                  (b) a correctness dependency on append-only facts;
                  (c) a snapshot identity that must record which delta was applied.
```

The condition in (b) is the one that decides the design in practice. If corrections arrive as new versions rather than in-place updates, the delta is append-only and the incremental path is safe. If corrections overwrite, the delta cannot express the change and the rebuild must be full — which is a second, independent argument for bitemporal facts, beyond reproducibility.

### 9.3 Cost of the assertions

The integrity checks are cheap relative to the build and should never be skipped for cost. The degree identity costs one pass over the edge column, O(m). The offset monotonicity check costs O(n). The relation registry check costs a distinct count, O(m) or O(1) with an index. Together they add under 1% to a build and they localise defects that would otherwise surface as a metric anomaly weeks later.

The expensive check is the pair-level audit of resolved identity, which is O(pairs sampled × evidence lookup) and is the one that deserves a budget line. Sampling 1,000 resolved pairs against source records at 10 ms per lookup is 10 s, which is negligible; auditing all pairs is a full manual review and is not feasible. The sampling rate is therefore the design variable, and a rate derived from the acceptable miss probability is more defensible than a round number.

**Takeaway:** Budget the index build and the projection emission, not just the bytes; those two dominate the real cost of a nightly graph build.

## 10. Key Theorems / Results Reference

**Handshake lemma.** For any finite multigraph without self-loops, the sum of degrees equals twice the edge count. *Consequence:* degree sum is a checkable invariant per snapshot, and a mismatch localises the defect to the projection or the filter. *Limit:* loops contribute 2 each, so the naive assertion fails on self-loops.

**Euler's degree condition.** A connected undirected graph has an Eulerian circuit iff every vertex has even degree, and an Eulerian trail iff exactly zero or two vertices have odd degree. *Consequence:* trail-based traversal plans have an exact feasibility test. *Limit:* the condition concerns edge coverage, not the traversal you actually want, so it is a novelty check rather than a design tool.

**Sparsity of real relationship graphs.** Empirically, relationship graphs have m = O(n) rather than O(n²). *Consequence:* an adjacency matrix wastes a factor of n; a sparse layout is mandatory above a few thousand nodes. *Limit:* it is an empirical regularity, not a guarantee — dense subgraphs such as projection cliques violate it locally, and a projection is exactly where a designer is tempted to densify.

**Quadratic projection bound.** A bipartite 2-section over a shared node of degree k contains C(k,2) pairs. *Consequence:* projection cost is bounded by the square of the maximum hub degree, so hub policy is a cost decision as well as a modelling decision. *Limit:* the bound is on the emitted pair count; a pruned projection has no closed-form count because the pruning depends on the data.

**AGM bound for triangle counting.** The maximum number of triangles in a graph with m edges is O(m^{3/2}), attained by a complete graph. *Consequence:* exact triangle counting is feasible at 1e9 edges only with the bound exploited (degree ordering, orientation), not with a naive triple loop. *Limit:* the bound is worst-case; real graphs with skewed degrees often do far better with a node-iterator plus degree ordering, so the bound is a planning ceiling rather than a runtime prediction.

**Bitemporal closure requirement.** Reproducible as-of queries require, for every fact identifier and system-time range, at most one visible version. *Consequence:* the write path must enforce a supersede rule and the schema must carry a uniqueness assertion; four timestamp columns alone do not provide bitemporal semantics. *Limit:* enforcement is a write-path property, so a bulk load that bypasses the write path breaks the invariant even when the schema is correct.

**Reification cost identity.** Expressing an edge as a resource costs 2 + p statement rows for p properties. *Consequence:* the storage multiplier is computable before migration, and 1.6x on a three-property edge is a typical figure. *Limit:* statement-per-row stores may pack several statements into one physical row, so the logical count overstates the physical cost.

**Loss under 2-section expansion of a hypergraph.** A hyperedge of size s replaced by a clique preserves pairwise adjacency and loses group identity; replaced by a star it preserves incidence and adds a hub of degree s. *Consequence:* neither binary encoding can answer both "are these two co-members" and "which group were they in", so the choice follows the query. *Limit:* for s = 2 the two encodings coincide, which is why the defect is invisible in small tests built from pairwise data.

**Diameter and expansion under projection.** A projection can only decrease the hop distance between two nodes of the retained type, because every projected pair is adjacent by construction. *Consequence:* any feature that uses hop distance becomes systematically smaller after projection, so a "two hops away" rule means different things on the base graph and on the projection. *Limit:* the statement is about pairs that survive; a pruned projection can remove a pair entirely and leave a distance that is infinite rather than smaller.

**Monotonicity requirement for incremental snapshots.** Incremental maintenance of a snapshot is sound iff every change is an append (no deletions, no in-place updates) and every derived structure is a monotone function of the fact set. *Consequence:* the delta-plus-base layout of Variant D is valid exactly under these two conditions, and its validity is a property of the data lifecycle rather than of the code. *Limit:* most derived features are monotone in the fact set (counts, sums), but percentile features and top-k selections are not, because an appended fact can push a value out of the top k.

**Compression ceiling for sparse structures.** A structure with m positions and n rows carries at least log₂(C(m, n)) bits of information about which rows hold which positions, so no encoding can be smaller than that bound. *Consequence:* CSR's 4 bytes per position is far above the ceiling for graphs with strong locality, which is why reordered and gap-compressed variants exist, and why a designer should not assume 13 bytes per edge is the floor. *Limit:* the bound is informational and ignores access cost; a heavily compressed layout that cannot answer a one-hop query in O(d_v) has traded a cost the designer needed.

**Takeaway:** The usable results are the ones with a stated limit; each theorem above converts a design argument into an assertion with a known failure boundary.

## 11. Interview / Exam Questions

**Q1. A graph has 9 edge occurrences over 6 nodes; the degree column sums to 17. What does that tell you, and what is the first check?**
Answer: The handshake identity requires the sum to be 18, so one incidences is missing or one edge was counted once. The first check is whether any self-loop exists (it would change the expected value, not the sum) and then whether the degree aggregate was computed on a different as-of filter from the edge count. Reasoning: the identity is exact, so a mismatch is mechanical — a filter divergence, a deduplication, or an outer join producing a null degree. Follow-up: how would you localise it to a single node? Answer: recompute degree from the fact table grouped by node and diff the two columns; the differing nodes are the localisation. Red flag: "the graph is approximate". Scoring: 2 points for the identity, 2 for the filter hypothesis, 1 for the localisation method.

**Q2. Compute the distinct-neighbour degree of M1 in the toy graph and explain why it differs from 5.**
Answer: M1 has five occurrences (E1–E5) but only two distinct neighbours (A1, A2), so d_dist = 2 and the surplus is 3. It differs because multiplicity is real: three transfers from A1 and two from A2. Reasoning: occurrence degree measures activity, distinct degree measures reach. Follow-up: which one belongs in a normalisation denominator for a propagation score? Answer: distinct degree, because the propagation divides mass among neighbours, not among events; using occurrence degree would let one busy counterparty absorb mass repeatedly. Red flag: treating the difference as a data-quality problem. Scoring: 2 for the numbers, 2 for the reason, 1 for the normalisation choice.

**Q3. Estimate the storage of a 1e9-edge snapshot as CSR with int32 targets, float64 weights and uint8 relation codes, no timestamps, and compare with a 176-byte row store.**
Answer: offsets 4(n+1); for n = 1e8 that is 0.4 GB. Per edge 4 + 8 + 1 = 13 B, so 13 GB, total 13.4 GB. Row store: 176 GB. Ratio 13.1x. Reasoning: the comparison holds only for the same column set; adding an int64 timestamp per edge adds 8 GB and narrows the ratio to 8.1x. Follow-up: when does the row store win? Answer: when the dominant query filters by type and time before traversal, because the index prunes the scan and the byte overhead is paid only on the surviving rows. Red flag: quoting a vendor benchmark for a different column set. Scoring: 2 for the arithmetic, 2 for the ratio, 1 for the condition.

**Q4. A projection over devices produces 1.25e9 pairs from one hub. Name two distinct repairs and the loss each introduces.**
Answer: (a) exclude hubs above a degree threshold — loses genuine co-occurrence for all accounts on that device and biases against populations served by shared hardware; (b) cap pairs per hub by sampling — makes the projection non-deterministic and breaks rebuild equality unless the sample is seeded and stored. Reasoning: the pair count is C(k,2), so any repair either removes the hub or reduces its clique. Follow-up: what threshold would you set and how would you defend it? Answer: a threshold justified by the count of accounts per device in the population (kiosk, corporate handset pool) and checked against an adjudicated pair sample, not a round number. Red flag: raising memory as the fix. Scoring: 2 for two repairs, 2 for the losses, 1 for the threshold defence.

**Q5. Why must a fact table store availability time when it already stores event time?**
Answer: Because the decision rule is "what did we know at t", and only availability time answers it. Event time alone permits a backtest that includes facts that had not landed, which inflates offline performance. Reasoning: the two timestamps coincide in no real pipeline, and the gap is heavy-tailed, so any backtest with a cut-off inside the tail is affected. Follow-up: give a discriminating test. Answer: restrict to facts with `t_avail - t_event > 60 s` and compare feature values at a cut-off inside that window under both filters; a difference proves the filter matters and quantifies the leak. Red flag: "we can reconstruct availability from the load log" — that is a source, not a column, and it is usually rotated. Scoring: 2 for the reason, 2 for the test, 1 for the follow-up mechanism.

**Q6. You find four timestamp columns in a table and are told it is bitemporal. What single assertion would you run first?**
Answer: For every edge identifier and system-time range, assert at most one visible version: `COUNT(*) GROUP BY edge_id, valid_from HAVING COUNT(*) > 1` at the same instant. Reasoning: four columns are a schema; bitemporal semantics are a constraint on writes. Follow-up: what does a violation imply about historical snapshots? Answer: they double-count, because both the superseded and the correcting row are visible at times when only one should be. Red flag: accepting the column list as the proof. Scoring: 2 for the assertion, 2 for the implication, 1 for precision about visible versions.

**Q7. A feature computed from edge counts drops 12% overnight with no behaviour change. Give two competing causes and a check that separates them.**
Answer: (a) a late batch changed the snapshot boundary, so m fell from 9 to 8 or the reverse; (b) a deduplication was added upstream, collapsing multiplicity. Check: compare `m` and `distinct pairs` for the same snapshot identifier. If m changed, the boundary moved; if m is the same but pairs fell, deduplication was introduced. Reasoning: the two causes move different counts. Follow-up: what would make the check inconclusive? Answer: if both happened, since the two effects can cancel; then diff the fact table row sets directly. Red flag: attributing the drop to the model. Scoring: 2 for two causes, 2 for the discriminating counts, 1 for the cancellation caveat.

**Q8. When is a dense adjacency matrix the correct choice, and what is the numeric boundary?**
Answer: It is correct for spectral work (eigendecomposition, matrix products) at small n. The boundary is where n² cells × 8 B exceeds the working set you can afford: n = 10,000 gives 0.8 GB, n = 1,000,000 gives 8 TB. Reasoning: memory grows quadratically while m typically grows linearly. Follow-up: at n = 10,000 with 20,000 edges, what is the overhead ratio? 8e8 B for the matrix versus about 10,000×4 + 20,000×13 ≈ 0.30 MB for CSR, so roughly 2,600x. Red flag: "dense is faster" without n. Scoring: 2 for the use case, 2 for the boundary numbers, 1 for the ratio.

**Q9. Explain how entity overmerge and supernode fan-out can produce the same symptom of implausibly high connectivity, and how to separate them.**
Answer: Overmerge unions two nodes' neighbourhoods, raising degree for both; supernode collapse gives one node thousands of edges which then leaks into every neighbour's second hop. Both raise apparent connectivity in aggregates. Separation: inspect the degree distribution — overmerge shifts a small number of nodes up by roughly the sum of two moderate degrees, while a sentinel hub produces a single node orders of magnitude above the 99th percentile. Reasoning: the shapes differ even though the means move the same way. Follow-up: confirm the hub by sampling source payloads. Red flag: lowering the resolution threshold to "clean up". Scoring: 2 for the mechanism, 2 for the shape distinction, 1 for confirmation.

**Q10. Two snapshots of the same graph differ by one edge; the projection differs by 49 pairs. What does that tell you?**
Answer: The new edge either created or removed a shared node of degree 10 in the projection, because C(10,2) − C(9,2) = 45 − 36 = 9, and to move 49 the degree must be higher: C(k,2) differences are odd numbers 1, 3, 5, …, so 49 = 2k−1 gives k = 25. Reasoning: a one-edge change in the base graph can change 2k−1 pairs when it connects an account to a shared node of degree k, so 49 pairs implies k = 25. Follow-up: what if the projection also lost pairs? Then two shared nodes changed, or a pruning rule fired. Red flag: assuming the projection is linear in edges. Scoring: 2 for the quadratic insight, 3 for solving 2k−1 = 49, 1 for the two-node caveat.

**Q11. What is lost when a hyperedge of size 4 (a joint account with four holders) is encoded as a clique, and what is gained?**
Answer: The group identity is lost — the clique says every pair co-holds, and also that two of them might form a pair independently, which is false. The incidence structure (four members of one group) is not recoverable. Gained: pairwise adjacency, so pair-level models run unchanged. Reasoning: a 4-clique has 6 edges versus 4 incidence edges, so the encoding also inflates edge count by 6/4 = 1.5x. Follow-up: when would you choose the star expansion instead? Answer: when the group is queryable as an entity (all holders of account X) at the cost of adding a node of degree 4 that is not a person. Red flag: claiming the clique preserves equivalence. Scoring: 2 for the loss, 2 for the edge inflation, 1 for the star condition.

**Q12. Your rebuild takes 400 s of which 300 s is index maintenance. What layout change addresses that, and what new cost appears?**
Answer: Drop the secondary indexes on the analytic replica and use CSR for the batch build; the new cost is that any per-edge filtered query becomes a full scan, so the replica can serve the batch and not the point queries. Reasoning: the index is paid per row at build time and redeemed per query; a nightly analytics workload does not redeem it. Follow-up: how would you serve the point queries? Answer: keep a small row-store serving table for recent facts, which is bounded by the retention window rather than by total history. Red flag: removing indexes from the transactional store. Scoring: 2 for the change, 2 for the new cost, 1 for the two-store pattern.

**Q13. A motif count of 400 three-cycles is reported as evidence of a coordinated ring. What is missing?**
Answer: A null model preserving the degree sequence. A graph with the same degrees generates many 3-cycles without coordination, so 400 is a count, not a finding. Reasoning: cycle counts are driven by degree distribution and clustering; without the null, the number measures the degree sequence. Follow-up: what would a defensible comparison look like? Answer: configuration-model samples or a degree-preserving rewiring ensemble, reporting the observed count against the ensemble's quantiles and the fraction of cycles that touch a hub. Red flag: "400 is a lot". Scoring: 2 for the null model, 2 for the reason, 1 for the ensemble method.

**Q14. Given a fact table with a nullable value column and four of nine facts null, what is the correct mean-transfer feature and what does the wrong one produce?**
Answer: Correct: sum over TRANSFER facts / count of TRANSFER facts, with a null policy that drops non-transfers from the denominator. In the toy graph that is 380 + 615 = 995 over 5 transfers = 199.00. Wrong: coercing nulls to zero and dividing by 9 gives 995/9 = 110.56. Reasoning: the wrong version mixes two populations and its value depends on how many device sightings were joined in, so it drifts with logging volume. Follow-up: how do you detect the coercion post hoc? Answer: the denominator equals the total fact count rather than the transfer count; recompute both and compare. Red flag: "zero is a reasonable default". Scoring: 2 for the correct feature, 2 for the arithmetic, 1 for the detection.

**Q15. You must serve one-hop expansion at p99 under 50 ms over 1e8 edges. Which layout, and what does the requirement forbid?**
Answer: A materialised CSR (or CSC for reverse direction) snapshot in memory, partitioned to fit the host, with the as-of filter applied at build time. It forbids query-time as-of filtering over the base table, because that is a 1e8-predicate scan per request, and it forbids rebuilding the snapshot per request for the same reason at a larger constant. Reasoning: the latency budget permits an O(d_v) contiguous scan and nothing that touches m. Follow-up: what if the graph changes intraday? Answer: a delta array merged on a schedule, with the merge cost amortised, and the staleness bounded and stated. Red flag: "we will just cache it". Scoring: 2 for the layout, 2 for the forbiddance, 1 for the staleness plan.

**Takeaway:** The questions that matter ask for a number or a discriminating check; answers that describe concepts without either are not answers.

## 12. Quick Reference Card

| Element | Statement | Condition |
|---|---|---|
| Handshake check | sum of occurrence degrees = 2m | no self-loops; self-loops add 2 each |
| CSR storage | 4(n+1) + m(4 + w + 1 + t) bytes | int32 ids; widen above 2^31 nodes |
| Row-versus-CSR ratio | 7.3x for 176 B rows vs 24.1 B/edge | same column set; narrows as payload grows |
| Degree pair | d_occ >= d_dist, surplus = repeats at both ends | per snapshot, per node |
| Projection size | C(k,2) pairs per shared node of degree k | full clique; pruning breaks the closed form |
| Dense matrix bound | n² × 8 B; 0.8 GB at 1e4, 8 TB at 1e6 | correct only for small n or spectral work |
| Time rule | eligible iff t_avail <= t_decision | never substitute t_event |
| Bitemporal invariant | at most one visible version per (id, system time) | enforced by the write path, not the schema |
| Reification cost | 2 + p rows per edge, ~1.6x at p = 3 | logical count; packing reduces physical rows |
| Rebuild rule | every derived structure reproducible from base + identity version | any hand-edited edge breaks it |
