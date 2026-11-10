---
title: "Graph data-model patterns"
topic: "Graph data-model patterns: typed multigraphs, edge tables, triple tables, CSR layouts, bitemporal facts, partitioning and lossy projection"
collection: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Graph data-model patterns

> **Evidence:** every algebraic identity below is DERIVED and re-checked in Section 5. All node names,
> amounts, timestamps and byte sizes are ILLUSTRATIVE constructions. Byte counts use documented
> primitive widths (int32 = 4 B, int64/float64 = 8 B, uint8 = 1 B, UUID = 16 B) and are labelled
> estimates. No vendor benchmark, production measurement or organisation-specific schema is claimed.

A data model is a promise about what you will be able to ask later, and every promise is paid for three
times: in bytes on disk, in milliseconds per hop, and in the information the representation cannot carry
at all. The first two can be estimated before a single row is written. The third usually cannot be
discovered until a decision is questioned, which is why it is the expensive one. This sheet organises the
design space around that asymmetry: it derives the costs that are computable in advance, states the losses
as bounded claims with explicit validity conditions, and treats every layout choice as a bet on which
query shape will dominate.

The recurring mistake is to treat "the graph" as one object with one schema. In any system that survives
long enough to be audited there are at least four distinct artifacts. There is the **base fact store**: an
append-only table of typed, timestamped relationships carrying provenance. There is the **identity layer**:
a versioned, reversible map from source identifiers to resolved nodes. There is the **serving structure**:
an adjacency array, an index, or a triple table optimised for one query shape at one latency. And there is
the **training snapshot**: a frozen projection of the first three at a stated decision time. These four
have different correctness rules, different lifecycles and different owners. Collapsing them into a single
table is the root cause of most defects described in Section 8, because it forces one structure to be
simultaneously complete, fast, reproducible and cheap.

The order of sections follows the order a designer should work. Section 3 defines the fact, then the
identity, then the time semantics, then the layout. Section 4 derives the four formulas that decide most
arguments: the incidence identity that makes integrity checkable, the degree pair that distinguishes
activity from reach, the storage formula whose terms name the trade, and the partition cost that predicts
what a distributed deployment adds. Section 5 runs one small graph through all four so the arithmetic can
be checked end to end. Section 8 catalogues sixteen failure mechanisms, each with the competing causes a
symptom can have and the observation that separates them.

Two conventions are worth fixing before anything is measured. First, **event time and availability time
are distinct columns, not metadata**: event time answers when a thing happened, availability time answers
when the system could first have known it. Any structure built without availability time cannot answer the
only question a backtest actually poses, which is what the serving path saw at a given instant. Second,
**every derived structure must be reconstructible** from the base facts plus a named identity version,
with no undocumented input. A hand-edited edge added to satisfy a query breaks reconstructibility, because
the rebuild path will silently drop it and two reports will thereafter disagree with no visible cause.

The consequence of those two conventions is that a graph model is best understood as a stack of
deliberately lossy projections over an append-only fact table. Choosing a layout is choosing which losses
the decision downstream can survive. An adjacency array that discards relation types is legitimate if no
query filters by type and no aggregate depends on it. The same array is a defect the moment a regulator
asks whether a particular transfer edge existed, because the array cannot answer the question and the
answer is not recoverable without a rebuild.

**Takeaway:** A graph model is a set of lossy projections over an append-only typed fact table; the design
work is naming the dominant query, computing its cost, and stating which loss the decision can survive.

## 1. File Header

> **Purpose:** Given a candidate relationship dataset and a decision latency, state the fact grain, choose
> the logical and physical representation, compute the storage and traversal cost from first principles,
> name the information each structure destroys, and reject any layout whose loss breaks the decision.
> **Prerequisites:** set and multiset reasoning, the handshake lemma, row-versus-column storage intuition,
> event-time versus availability-time discipline, and the distinction between a primary key and a resolved
> identity.
> **Key outcome:** you can size three candidate layouts for the same fact set, show the ratio between them,
> predict which query each makes expensive, and write the assertions that fail loudly when a loader change
> breaks the model.
> **This file does NOT cover:** Cypher, Gremlin or SPARQL syntax, vendor feature lists, graph neural
> network architecture, centrality and community detection, ontology design at depth, and entity
> resolution algorithm detail. Those are separate sheets in this collection.
> **Level:** Core. The foundation layer is Section 3; the advanced layer is the loss analysis in Section
> 4.4, the partition arithmetic in Section 4.3, and the scale transitions in Section 9.1.
> **Version / reviewed:** Draft 1.0, 2026-11-10. Independent review: not yet performed. Review trigger:
> any change to the fact grain, the time semantics, the identity rule, or the storage engine.
> **Evidence:** DERIVED identities re-checked in Section 5; ILLUSTRATIVE inputs throughout. Byte counts
> are arithmetic on declared primitive widths, not measurements of any engine's on-disk format, which
> includes page headers, fill factors and compression that this sheet deliberately excludes so that the
> ratios stay interpretable.
> **Retrieval cue:** if a graph feature is cheap offline and slow or wrong in production, read Section 4.4
> first, then the failure entries on projection loss (F-04), supernode fan-out (F-06) and availability-time
> backdating (F-02).

Three numbers frame the whole problem and are worth memorising before the derivations. A million-edge
graph is small. A billion-edge graph is not, and it is the size at which the row store and the compressed
array differ by a factor of eight in bytes and by a factor of twenty in build time. Between those two
points, the binding constraint silently changes from total work to the largest single node's degree, which
is why a design that was correct at the smaller size can fail at the larger one with no algorithmic change.

The second framing number is the ratio of the two degree definitions. On the toy graph of Section 5 the
occurrence degree sums to 18 and the distinct-neighbour degree sums to 12, a surplus of 6 that exists
because five transfers between two accounts and one merchant repeat the same pairs. Every aggregate that
does not declare which definition it uses is ambiguous by that surplus, and the ambiguity grows with
transaction volume since multiplicity is exactly what volume adds.

The third is the projection multiplier. A bipartite structure over a hub of degree k yields C(k,2) pairs,
which for k = 50,000 is 1.25 billion — more than the entire edge count of the graph it came from. A
projection is therefore not a view in the database sense; it is an aggregation whose output can exceed its
input by an unbounded factor, and it must be planned with a peak-size estimate rather than discovered.

**Takeaway:** Fix the grain, fix the two timestamps, fix the identity version, then choose a layout whose
loss you can name — in that order, because each decision constrains the next.

## 2. Concept Map / Mental Model

```text
   RAW SOURCE RECORDS  (immutable; identifiers are local to each source system)
     payments | logins | device sightings | ownership declarations | case notes
                              |
        1. FACT DESIGN       grain, direction, relation type, value column,
        |                    event time, availability time, provenance
        v
   +--------------------------------------------------------------------+
   | BASE FACT TABLE   append-only, typed multigraph, bitemporal         |
   |  edge_id | src_raw | rel | dst_raw | value | ccy | t_event | t_avail |
   +--------------------------------------------------------------------+
            |                                        |
   2. IDENTITY RESOLUTION                    4. AS-OF SELECTION
      raw id -> node id, versioned,             keep facts with
      reversible, evidence attached             t_avail <= t_decision
            |                                        |
            v                                        v
   +---------------------+                 +--------------------------+
   | IDENTITY MAP        |                 | SNAPSHOT at t_decision   |
   | version | evidence  |                 | immutable, addressable,  |
   | merge log           |                 | reproducible byte for byte|
   +---------------------+                 +--------------------------+
            |                                        |
            +------------------+---------------------+
                               |
        3. LAYOUT CHOICE       v
        |            edge table | CSR/CSC array | triple table |
        |            property graph | hypergraph incidence
        v
   +-------------------------------------------------------------------+
   | SERVING STRUCTURE   one position per (node, occurrence) or per row |
   +-------------------------------------------------------------------+
                               |
        5. PROJECTION           v
        |            node features, pair table, motif counts,
        |            bipartite 2-section, hypergraph star expansion
        v
   decision at t_decision  (score, authorise, escalate, report)
```

Read the diagram as five transformations, not one pipeline. Each arrow is a place where information can be
dropped, and each drop is acceptable only if the decision downstream is invariant to it. Step 1 decides
what a fact is; step 2 decides what an entity is; step 3 decides how a fact is laid out; step 4 decides
what is visible at the decision instant; step 5 decides what is aggregated. A defect at any step surfaces
as a symptom at the last step, which is why debugging graph features backwards from the metric usually
misattributes the cause to the model.

The diagram's most commonly skipped edge is the one from the identity map into the snapshot. A snapshot
that names a date but not an identity version is not a snapshot; it is a query that will resolve to
different bytes next week. The test is mechanical: two people running the same training job with the same
snapshot identifier must produce identical features. If they do not, the identifier names a schedule
rather than a state, and no backtest built on it is reproducible.

The second invariant in the picture is reconstructibility. Every derived structure must be reproducible
from the base fact table plus one identity version, with no other input. This is what makes lossy steps
safe: any loss can be re-derived with a different setting when the decision changes. It is also what
distinguishes a serving replica, which is disposable and always rebuildable, from a store of record, which
must be append-only and complete. A system that has one structure trying to be both will eventually lose
history, because the cheapest way to make a serving structure fast is to update it in place.

What varies across layout families is only which query shape is cheap. A row-per-fact table makes
arbitrary filtered scans cheap and multi-hop traversal expensive. A CSR array makes one-hop expansion
cache-friendly and per-edge filtering impossible without extra parallel arrays. A triple table makes
predicate-first pattern matching cheap and property-joining expensive. A property-graph store makes
pointer-chasing natural and bulk analytics awkward. Nothing about the facts differs between those four;
the choice is a bet on the dominant query, and the bet should be written down because it will be
re-litigated by whoever inherits the system.

The boundary that matters most for machine learning sits between the base graph and the projected feature
table. The base graph is high-cardinality, multivalued and time-indexed: nine edge occurrences over six
nodes in Section 5. The feature table is one row per entity per decision time with scalar columns. The
projection is where multiplicity becomes a statistic, and therefore where the largest class of silent bias
enters, because an aggregate computed over all edges ever seen is not the aggregate the serving path could
compute at any particular instant.

```text
Invariant ladder — each rung is a question a reviewer must be able to answer before sign-off
  rung 1  grain        what is one row? one fact per (src, rel, dst, t_event)?
  rung 2  direction    is the relation symmetric, asymmetric, or directed with a type per direction?
  rung 3  multiplicity is a repeated fact one row, a count column, or a collapsed edge?
  rung 4  time         which column decides visibility at a decision instant?
  rung 5  identity     which map version produced these node ids, and can a merge be undone?
  rung 6  layout       which query is cheap, and which is now a scan?
  rung 7  loss         what can this structure no longer answer, stated as a sentence?
  rung 8  rebuild      what command recreates this structure, and what does it read?
```

The ladder is short on purpose. A design review that answers all eight rungs in writing takes under an
hour and prevents most of Section 8. A review that answers rung 6 by naming a storage engine has skipped
rungs 1 through 5, which are the ones that determine whether the engine's cheap query is the query that
matters.

**Takeaway:** Five lossy transformations, two invariants — reconstructibility and pinned identity — and an
eight-rung checklist that turns a design argument into a set of written answers.

## 3. Foundational Definitions

Every definition below carries its type or unit, because most disputes in this area are disputes about
which definition was in force rather than about the algorithm.

### 3.1 The fact layer

**Fact.** One typed relationship assertion between two entities, with a value, an event time and an
availability time. A fact is immutable once written; a correction is a new fact that supersedes an earlier
one. The word is chosen deliberately over "edge", because an edge sounds like a graph object while a fact
sounds like a record — and the record view is the one that survives audits.

**Grain.** The answer to "what is one row?" expressed with units. "One row per transfer per account pair"
is ambiguous, because a transfer has two endpoints and appears once, while a pair can have many transfers.
"One row per transfer instruction, with the debiting account as source and the crediting account as
target" is a grain. The test is that a second reader can predict the row count from the business event
count without asking a question.

**Typed multigraph.** A graph whose edges carry a relation label and may repeat between the same ordered
pair. The type matters because it partitions aggregates: counting all edges between two entities mixes a
transfer with a device sighting, and the resulting number has no unit. The multigraph matters because
multiplicity is signal — three transfers is not the same fact as one transfer.

**Direction.** Whether a fact is symmetric (a shared-device sighting, a joint-account membership) or
directed (a transfer, an ownership declaration with an effective date). Direction is a property of the
relation type, not of the storage, and a symmetric relation stored once must be traversed in both
directions while a directed one stored once must be traversed in one, which are different code paths over
the same bytes.

**Value column and its unit.** The numeric payload of a fact, with an explicit unit and currency. A
nullable value is normal and must be distinguished from zero by policy, not by default. The distinction is
material: a null sighting and a zero-amount transfer are different facts, and coercing the first to a zero
turns a count into a sum that mixes populations.

**Provenance.** The reference from a fact back to the source record that produced it, precise enough to
re-read that record. Provenance is what makes a projection auditable, because a pair in a derived table
can be traced to the device that caused it only if the edge carries the reference. Without provenance, a
projection is a claim with no witness.

### 3.2 The identity layer

**Raw identifier.** The key a source system assigns, meaningful only inside that system. Two sources may
use the same integer for different entities, so a raw identifier is never globally unique and must never be
used as a node key without a namespace.

**Namespace.** The source qualifier that makes a raw identifier globally unique, typically a short system
code. Namespace collisions are the cheapest catastrophic defect in this area: two sources both numbering
accounts from 1 produces a merged graph in which every account has a spurious twin, and the resulting
degrees are too high by a factor near two.

**Resolved node.** The entity that a set of raw identifiers denotes, carrying a canonical identifier and,
critically, a member count. The member count is what converts a single identity error into a predictable
number of false facts, and it is cheap enough to materialise alongside the node.

**Identity version.** An immutable handle naming the mapping, the match rule and the algorithm version. It
is an input to every structure built from the graph, which is why it belongs in the snapshot identifier
and in the feature definition rather than in a configuration file. A feature store that joins on date
instead of version holds values computed under two different partitions sharing one label.

**Reversibility.** The property that a merge can be undone without recomputing the whole partition. It
requires a merge log, because a partition materialised as a single identifier column has discarded both
the members and the order of the decisions. Reversibility is a correctness property, not a convenience:
splits are requested for reasons unrelated to accuracy, including customer dispute, regulatory
per-source reporting, and downstream contract.

### 3.3 The time layer

**Event time (t_event).** When the fact happened in the world. It is a property of the source event and is
usually the later of two timestamps in a well-instrumented system, because events are reported after they
occur.

**Availability time (t_avail).** When the fact first became readable by the serving path. It includes
ingestion, validation and any batch delay. It is the only column that can answer what a model could have
known, and its absence is the most common cause of offline metrics that do not reproduce online.

**System time (also called processing time or validity time).** When the database believed a fact. It is
distinct from availability because a fact can be available to a pipeline but not yet committed to the
store that the model reads, and because corrections retroactively change belief without changing what was
available.

**Bitemporal.** Storing both the event axis and the belief axis, which requires four timestamps per fact:
event start, availability, system start, system end. The fourth is what permits reconstructing a past
belief state without destroying the current one. Four columns are the schema; the semantics additionally
require a write rule that permits at most one visible version per fact and belief instant.

**Decision time (t_decision).** The instant a decision is made, and therefore the cut-off for visibility:
a fact is eligible iff t_avail <= t_decision. Substituting t_event for the filter is the defect that makes
a backtest optimistic, and it is invisible in aggregate because the two timestamps agree on the majority of
well-instrumented facts and diverge heavily in the tail that matters.

**Event-time window versus processing-time window.** A window defined on event time is stable under
reprocessing and answers a question about the world. A window defined on processing time changes whenever
a batch is delayed and answers a question about the pipeline. Features must state which, because the two
produce different values for the same nominal period and both are defensible for different questions.

### 3.4 The layout layer

**Edge table (row store).** One physical row per fact, with all columns present. Every query is expressible,
filtering is applied before any join, and the cost is the byte width of a row times the number of rows
scanned. It is the natural store of record.

**Adjacency list.** One list per node holding its neighbours. Insertion is cheap, traversal is pointer-
chasing, and memory locality is poor because a list's elements are scattered by the allocator.

**CSR (compressed sparse row).** Two parallel arrays: an offset array of length n+1, where offset[i] to
offset[i+1] bounds node i's positions, and a position array holding the neighbour identifiers (plus any
parallel attribute arrays). One-hop expansion is a contiguous scan, which is why it is the layout of
choice for batch analytics and for batching graph computations. Insertion in the middle renumbers
everything after it, so a CSR is a snapshot, not a mutable table.

**CSC (compressed sparse column).** The transpose of CSR, grouping by destination. It makes reverse
traversal cheap and is the prerequisite for any bidirectional search or any feature that needs in-edges.

**Triple table.** One row per statement of the form (subject, predicate, object). Every fact is a row,
schema changes need no migration, and every aggregate becomes a join. Reification, the expression of an
edge as a resource with its own properties, is natural here and costs extra rows.

**Property graph.** Nodes and edges as first-class objects carrying key–value maps, traversed by pointer or
index. Natural for interactive exploration and per-edge annotation; awkward for bulk aggregation because
the engine's unit of work is a traversal rather than a scan.

**Hypergraph.** A generalisation in which an edge may join any number of nodes, which is what a group event
actually is: a joint account with four holders, a shared device used by three accounts, a meeting with
seven attendees. Its two binary encodings are the star expansion (a node per hyperedge) and the 2-section
(a clique per hyperedge), and they lose different things.

**Incidence structure.** The bipartite graph of nodes and facts. It is the representation that loses
nothing, because every fact remains an object; the node-only graph is a projection of it.

**Projection.** Any derived structure whose rows are not facts. Node-level features, pair tables and motif
counts are projections. A projection is the only place multiplicity becomes a statistic, and therefore the
place where the definition of the statistic must be written down.

### 3.5 The consumption layer

**Feature contract.** A written statement of numerator, denominator, population filter, time window, null
policy, unit and refresh cadence for one column. It is a contract because a downstream rule may be written
against it; changing the denominator without a version bump is a breaking change that no type system
catches. The toy graph supplies an instance: numerator is the sum of transfer values over eligible facts,
denominator is the count of those facts, null policy excludes entities with none rather than emitting
zero.

**Snapshot identifier.** An opaque handle naming a frozen fact set, an identity version and a relation
filter. Its testable consequence is byte-identical features for two runs with the same handle. A date
string fails that test whenever a late batch lands, because the date now resolves to different bytes.

**Scope.** The declared subset of facts a computation may read, expressed as tenant, purpose or
jurisdiction. Scope is a legal control before it is a modelling one: a shared graph across tenants can
disclose that one tenant's customer transacts with another's. Its testable consequence is that every
structure carries a scope key and a query without a scope predicate is rejected at the interface rather
than filtered later.

**Retention and deletion.** The rule that decides when a fact stops being visible and how that propagates
into every derived structure. Deletion is the hard case because a degree or a centrality encodes a fact
without storing it, so deleting the row leaves a residue that no row-level audit can find. The remedy is
lineage from published aggregates back to source fact identifiers.

**Cardinality constraint.** A declared bound on how many edges a node may hold, per relation type, used to
catch instrumentation collapse. A device token with 50,000 accounts violates a sensible bound, which turns
an invisible modelling problem into a rejected load. The constraint must be per relation type, because a
merchant legitimately has many counterparties while a person does not legitimately hold thousands of
accounts.

**Serving replica versus store of record.** Two obligations on two copies of the same facts. The store of
record is append-only, complete, bitemporal and slow. The replica is derived, filtered, fast and
disposable. Confusing them produces the two classic failures: a replica promoted to truth, which loses
history, and a store of record used for scoring, which loses latency.

### 3.6 Layout vocabulary that is routinely conflated

**Index versus materialised view.** An index is a redundant structure maintained by the engine on writes; a
materialised view is a redundant structure refreshed on a schedule. Both accelerate reads and both must be
invalidated on deletion. The distinction matters because an index is usually transactional and a
materialised view usually is not, so the latter can be stale in a way a latency budget must tolerate
explicitly rather than discover.

**Vertex-centric versus edge-centric storage.** Vertex-centric stores group facts by source so one-hop
expansion is local; edge-centric stores group them by relation or by time so filtered scans are local. A
design needing both must duplicate or accept one slow path. There is no layout in which a per-node
expansion and a per-relation time range are simultaneously contiguous, and claiming otherwise is a claim
about caching rather than about structure.

**Sparse versus dense.** Sparse means storage proportional to the number of facts; dense means storage
proportional to the square of the node count. Real relationship graphs have facts linear in nodes, so dense
storage wastes a factor of n and is justified only for spectral work at small n or for dense subproblems
such as a co-occurrence matrix over categories.

**Static versus dynamic.** A static graph is built once and read many times, which is what CSR assumes. A
dynamic graph accepts insertions, and an insertion into the middle of an offset array is O(n). Dynamic
graphs therefore use adjacency lists with per-node capacity, or a small mutable delta merged periodically.
Choosing CSR for a graph with intraday inserts is the most common layout-level performance defect in this
area, and it manifests as a rebuild that grows without bound.

**Normalised versus denormalised attribute storage.** Normalised means relation attributes live in their
own table and are joined; denormalised means each position carries its own copies. Denormalisation makes
one-hop expansion cheap and makes a correction a rewrite of every copy, which is a correctness risk that
must be accepted explicitly rather than inherited.

### 3.7 Cardinality, units and what a count means

Every aggregate over a graph must state whether it counts occurrences, entities or pairs, because those
three differ by the multiplicity surplus. On the toy graph of Section 5, occurrence degree sums to 18,
distinct-neighbour degree sums to 12, and distinct pairs number 6. The three numbers answer three different
questions: how much activity happened, how many entities were reached, and how many relationships exist.
Reporting one while a reader assumes another is the most common unit error in graph features.

The second unit question is whether a value is per entity or per entity per period. A sum of transfers
over a 90-day window and a mean of daily sums over the same window differ by a factor of 90 in magnitude
and are both called "average transfer volume" in practice. The fix is to write the denominator into the
column name, which is cheap and prevents the error permanently.

The third is whether a rate's denominator is the eligible population or the observed population. An
entity with no eligible facts has no rate, and emitting zero for it asserts a rate of zero rather than an
absence. The two populations differ by exactly the entities that are newest or most dormant, which is
usually the subpopulation the analysis cares about most.

### 3.8 Loss statements as first-class design artifacts

A loss statement is one sentence of the form "this structure can no longer answer whether X", written
next to the structure's definition. It is the single highest-value artifact in this sheet, because every
layout in Section 7 loses something and the only question is whether the loss matters.

Three loss statements for the common structures read as follows. A CSR of neighbour identifiers without
parallel arrays cannot answer which relation type a position carries. A 2-section projection cannot answer
which group two co-members shared. A snapshot filtered to a decision time cannot answer what a later
revision changed, because revisions after the cut-off are excluded by construction.

The test for whether a loss statement is adequate is that it names a question the structure cannot answer
rather than a column it does not have. "It has no relation column" is a schema observation, and schema
observations invite the response that the column can be added. "It cannot distinguish a transfer from a
device sighting, so any aggregate over it mixes units" is a loss statement, and it forces the design
question that the schema observation defers.

**Takeaway:** The vocabulary that prevents most graph defects is small: grain, direction, two timestamps,
two degree definitions, and the loss statement attached to each layout.

## 4. Core Formulas with Full Derivation

Each result uses the same anatomy: the formula, its symbols with units, the constraint that makes it
valid, a numerical example with intermediate values shown, and an interpretation that names the decision
the number supports. Every block ends with an explicit validity condition.

### 4.1 Base case: the incidence identity and per-edge storage

```text
FORMULA
  sum over nodes of occurrence degree  =  2 * m
  where m is the number of undirected edge occurrences counted with multiplicity.

WHERE
  d_v = occurrence degree of node v, dimensionless count, integer >= 0
  m   = number of edge occurrences in the snapshot, integer >= 0
  the identity holds per snapshot and per relation filter, not across them

CONSTRAINT
  Each undirected occurrence contributes exactly one to the degree of each endpoint,
  so summing degrees counts every occurrence twice. Self-loops are the exception:
  a loop contributes 2 to its node's degree in the standard convention, so the
  identity still holds, but a loader that counts a loop once produces a sum of
  2m - L where L is the loop count.

NUMERICAL EXAMPLE (toy graph, 9 occurrences over 6 nodes)
  E1 A1->M1   E2 A2->M1   E3 A1->M1   E4 A2->M1   E5 A1->M1
  E6 A1--D1   E7 A2--D1   E8 A1--D2   E9 A2--D3
  degrees: A1 = 5 (E1,E3,E5,E6,E8)
           A2 = 4 (E2,E4,E7,E9)
           M1 = 5 (E1..E5)
           D1 = 2 (E6,E7)
           D2 = 1 (E8)
           D3 = 1 (E9)
  sum = 5 + 4 + 5 + 2 + 1 + 1 = 18 = 2 * 9                      identity holds
  after excluding E9 (as-of snapshot): m = 8, sum = 16 = 2 * 8    identity holds

INTERPRETATION
  The identity is an exact assertion, so any mismatch is mechanical rather than
  statistical: a filter applied to one side, a deduplication applied to the other,
  a loop counted once, or a null degree from an outer join. It costs one pass over
  the edge column and should run on every snapshot build.

VALIDITY
  Holds for any finite multigraph, with or without loops, provided loops are counted
  with degree contribution 2 and the same filter is applied to both sides.
```

The identity's value is that it localises defects without knowing anything about the data. A team that
finds a sum of 17 against m = 9 knows three things immediately: the fact is mechanical, the defect is
either in the degree column or in the edge count, and the difference is exactly one. What it does not say
is which — but recomputing degrees from the fact table grouped by node and diffing the two columns answers
that in one query, and the differing nodes are the localisation.

The storage formula below is the second-most-used piece of arithmetic in this area, and its terms are worth
reading one at a time because each is a separate decision.

```text
FORMULA
  B_csr(n, m) = 4 * (n + 1) + m * w_pos
  w_pos       = 4 (int32 neighbour) + 8 (float64 weight) + 1 (uint8 relation)
                + optional parallel arrays

WHERE
  n     = number of nodes in the snapshot, count
  m     = number of positions = edge occurrences (2m for an undirected fact
          materialised in both directions), count
  w_pos = bytes per position, B
  B_csr = total bytes, B

CONSTRAINT
  int32 neighbour identifiers limit n to 2^31 - 1 = 2,147,483,647 nodes, which is
  a real ceiling and not a theoretical one: at n = 1e8 the identifier space is used
  to 4.7%, and an expected collision count of n^2 / (2 * 2^32) = 1.16e6 makes int32
  unusable without a checked allocator. Adding a per-edge timestamp adds 8 B to
  w_pos, which is a 38% increase on a 21 B position and is the most commonly
  underestimated term.

NUMERICAL EXAMPLE (three layouts over the same toy fact set)
  layout L1  neighbour + relation          5 B/position
             offsets 4*7 = 28 B; positions 9*5 = 45 B;  total 73 B  (8.11 B/edge)
  layout L2  + weight + event time        21 B/position
             offsets 28 B; positions 9*21 = 189 B;      total 217 B (24.11 B/edge)
  layout L3  + availability time          29 B/position
             offsets 28 B; positions 9*29 = 261 B;      total 289 B (32.11 B/edge)
  layout L4  + two system-time stamps     45 B/position
             offsets 28 B; positions 9*45 = 405 B;      total 433 B (48.11 B/edge)
  row store  176 B per row (derived in 4.3): 9 * 176 = 1,584 B (176 B/edge)
  ratios to the row store: 1584/73 = 21.7x, 1584/217 = 7.30x,
                           1584/289 = 5.48x, 1584/433 = 3.66x

INTERPRETATION
  The ratio between layout families is not a constant; it is set by how many columns
  the compressed layout must carry. A team that quotes "8x smaller than a row store"
  is quoting L2 with three attribute columns, and the same team at L4 is 3.7x. The
  honest statement is the formula plus the column set, never the ratio alone.

VALIDITY
  Assumes fixed-width columns and no compression, no page overhead and no index on
  the array. A columnar engine with delta encoding on sorted timestamps can beat these
  figures substantially, which is why the formula is for comparing layouts to each
  other rather than for predicting a deployment's footprint.
```

The 21 B position deserves comment because it is the figure most often quoted and the one most often
misapplied. It contains no time. Adding the availability timestamp takes it to 29 B, a 38% increase, and
adding the two system-time stamps takes it to 45 B, which is 114% above the base. A design review that
approves "CSR for the serving layer" without stating which of the four column sets is meant has approved
four designs whose storage differs by a factor of 2.1 and whose query capabilities differ by the ability
to answer what was believed at a past instant.

### 4.2 General case: the degree pair, multiplicity surplus and projection size

The second identity is the one that separates activity from reach, and it is where most aggregate
ambiguity originates.

```text
FORMULA
  d_occ(v)   = number of occurrences incident to v (with multiplicity)
  d_dist(v)  = number of distinct neighbours of v
  d_occ(v)   >= d_dist(v), with equality iff no pair repeats at v
  surplus(v) = d_occ(v) - d_dist(v) = sum over neighbours u of (c(v,u) - 1)
  where c(v,u) is the number of occurrences between v and u

WHERE
  all three quantities are dimensionless counts per node, per snapshot
  c(v,u) is an integer >= 1 for adjacent pairs

CONSTRAINT
  Requires that the snapshot be simple in the sense of direction, i.e. that a
  symmetric relation be stored once and counted once per endpoint. Storing both
  directions doubles d_occ without changing d_dist, which doubles the surplus and
  makes the two definitions incomparable across layouts.

NUMERICAL EXAMPLE (toy graph)
  A1: d_occ = 5, neighbours {M1, D1, D2}, d_dist = 3, surplus = 2
      c(A1,M1) = 3 so 3-1 = 2; c(A1,D1) = 1 and c(A1,D2) = 1 contribute 0
  A2: d_occ = 4, neighbours {M1, D1, D3}, d_dist = 3, surplus = 1
  M1: d_occ = 5, neighbours {A1, A2},       d_dist = 2, surplus = 3
  D1: d_occ = 2, neighbours {A1, A2},       d_dist = 2, surplus = 0
  totals: sum d_occ = 18, sum d_dist = 12, total surplus = 6
  distinct pairs = 6, and sum d_dist = 12 = 2 * 6                identity holds

INTERPRETATION
  M1 has the largest surplus (3) with the smallest reach (2), which is the signature
  of a collection point: many facts, few counterparties. Normalising a propagation
  feature by d_occ lets one busy counterparty absorb mass repeatedly; normalising by
  d_dist divides mass among neighbours. The choice is a modelling decision, and the
  surplus is the size of the error introduced by getting it wrong.

VALIDITY
  d_occ >= d_dist always holds for a snapshot without double-counted directions.
  Equality holds exactly on snapshots where no pair repeats, which is why a test
  fixture built from distinct pairs never exercises the distinction.
```

The projection formula is the third identity and the one with the widest range of practical consequences,
because its output can exceed its input without bound.

```text
FORMULA
  pairs(S) = C(k_S, 2) = k_S * (k_S - 1) / 2      per shared node S of degree k_S
  P        = sum over S of C(k_S, 2)              total projected pairs, before dedup
  marginal = C(k+1, 2) - C(k, 2) = k              pairs added by one new member

WHERE
  S  = a node of the bipartite side being projected away (a device, a merchant,
       a group), k_S = its degree in the incidence structure
  P  = number of unordered pairs emitted, count, before deduplication
  the marginal identity is exact for adding one member to a clique of size k

CONSTRAINT
  The full 2-section emits every pair inside every group, so P is quadratic in the
  largest group and can exceed the incidence edge count by an unbounded factor. Any
  pruning rule (minimum group size, hub exclusion, sampling) destroys the closed
  form: the pruned count depends on the data, so it must be measured rather than
  derived, and a report that quotes the closed form for a pruned projection is wrong
  in the direction of optimism.

NUMERICAL EXAMPLE (m = 1e9 incidence edges; one hub of k = 50,000 and 1e5 groups of k = 100)
  hub          C(50,000, 2) = 50,000 * 49,999 / 2      = 1,249,975,000
  other groups 1e5 * C(100, 2) = 1e5 * 4,950           =   495,000,000
  total P                                              = 1,744,975,000
  P / m                                                = 1.745
  marginal for the hub: one more member adds k = 50,000 pairs, since
  (C(50,001,2) - C(50,000,2) = 1,250,025,000 - 1,249,975,000 = 50,000)
  removing one member removes k - 1 = 49,999 pairs
  (C(50,000,2) - C(49,999,2) = 1,249,975,000 - 1,249,925,001 = 49,999)

INTERPRETATION
  The projection is 1.745x the edge count here, and 71.6% of it comes from one hub.
  That concentration means the projection's cost is a property of a single node, so
  capacity planning must use the maximum degree, not the mean. The marginal identity
  is the useful diagnostic: one new member of the hub costs 50,000 pairs, so a
  projection whose pair count jumps by an exact multiple of a degree is reporting a
  hub membership change rather than a general growth.

VALIDITY
  Exact for a full 2-section over an undirected incidence structure without dedup.
  Deduplication across groups reduces P by the number of pairs that co-occur in more
  than one group, which is data-dependent and typically small for sparse graphs.
```

The marginal identity corrects a mistake that appears in otherwise careful analyses: the difference
C(k,2) − C(k−1,2) is k − 1, not 2k − 1, and the difference C(k+1,2) − C(k,2) is k. So a projection that
grows by 49 pairs from an unknown cause indicates a member added to a group of degree 49, and a
projection that shrinks by 24 indicates a member removed from a group of degree 25. Getting this
arithmetic backwards by a factor of two sends an investigation to the wrong node size.

### 4.3 Variants

**Variant A — reification of a fact as a resource.** Expressing each fact as an object with property
statements multiplies the row count by the number of properties plus two structural statements.

```text
FORMULA
  rows(fact) = 2 + p
  bytes(fact) = (2 + p) * w_stmt
  multiplier  = (2 + p) * w_stmt / w_row

WHERE
  p       = number of properties stored on the fact (value, currency, event time,
            availability time, provenance), count
  w_stmt  = bytes per statement row, B
  w_row   = bytes per flattened fact row, B

CONSTRAINT
  Compares logical row counts. A triple store may pack several statements into one
  physical row, which lowers the physical multiplier while leaving the logical count
  unchanged; comparing a packed store's bytes to a flattened table's bytes therefore
  answers a different question from comparing their row counts.

NUMERICAL EXAMPLE (p = 3: value, event time, availability time)
  rows   = 5 per fact
  bytes  = 5 * 56 B = 280 B per fact, against 176 B flattened
  ratio  = 280 / 176 = 1.59x
  toy    = 9 * 280 = 2,520 B against 9 * 176 = 1,584 B

INTERPRETATION
  Reification is priced at roughly 1.6x for a three-property fact. The price buys the
  ability to attach a property to an existing fact without migrating a schema, and it
  buys per-fact provenance queries. Skip it when facts are only ever aggregated: the
  aggregate cannot see the difference and the traversal pays for the extra hops.

VALIDITY
  Assumes one statement per row and a 56 B statement width including subject,
  predicate, object and overhead. A store with dictionary encoding can reduce w_stmt
  by more than half on repeated predicates, which moves the multiplier below 1.
```

**Variant B — bitemporal widening.** Adding the belief axis to the event axis.

```text
FORMULA
  w_pos(bitemporal) = w_pos(base) + 4 * 8 = w_pos(base) + 32 B
  B_bitemporal(m)   = 4 * (n + 1) + m * (w_pos(base) + 32)

WHERE
  four int64 stamps are event start, availability, system start and system end;
  system end is nullable but occupies its fixed-width slot in a row store
  w_pos(base) = 13 B for neighbour, weight and relation in the toy layout

CONSTRAINT
  The storage cost is arithmetic; the semantic cost is not. Four columns do not
  provide bitemporal semantics unless the write path enforces at most one visible
  version per (fact, system time), and a bulk load that bypasses the write path
  breaks the invariant while the schema still looks correct.

NUMERICAL EXAMPLE
  toy: 4*7 + 9*45 = 28 + 405               = 433 B, 48.11 B per edge
       against 4*7 + 9*29  = 289 B, 32.11 B per edge   -> 1.50x on the toy
  scale: n = 1e8, m = 1e9, base 21 B
       4*(1e8+1) + 1e9*21  = 0.4 GB + 21.0 GB  = 21.4 GB
       4*(1e8+1) + 1e9*45  = 0.4 GB + 45.0 GB  = 45.4 GB
       ratio 2.12x; a 30-snapshot retention at 3 replicas:
       21.4 * 30 = 642 GB, * 3 = 1.93 TB versus 4.09 TB

INTERPRETATION
  The 2.12x figure is the price of answering "what did we believe on Tuesday about
  what happened on Monday". That price is usually correct for the base fact table and
  wrong for the serving structure, which is why the recommended shape is bitemporal
  base, monotemporal snapshot, and no timestamps at all in the array used for scoring.

VALIDITY
  Assumes fixed-width columns. A sorted columnar store with delta encoding compresses
  monotone timestamps heavily, so 32 B per fact is an upper bound for a row-oriented
  layout and an overestimate for a well-ordered columnar one.
```

**Variant C — partitioning a snapshot across hosts.** The cost of distribution is measured in copied
positions and boundary fetches, and both are computable before deployment.

```text
FORMULA
  vertex partition, p parts, hash on one endpoint:
    E[positions stored] = m * (2 - 1/p)          both directions materialised
    E[positions stored] = m * (1 + 1/p)          one direction, boundary copies added
    boundary fetches    = m * (1 - 1/p)
  edge-cut variant, node of degree d replicated across parts:
    E[copies of a node] = p * (1 - (1 - 1/p)^d)

WHERE
  p = number of partitions, integer >= 2
  m = edge occurrences, count
  d = node degree, count
  the expectations assume a uniform hash on the partition key, independent of degree

CONSTRAINT
  The vertex formula is exact in expectation for a random hash and a graph whose
  degrees are independent of the partition key. Real partition keys are correlated
  with degree — the largest tenants are the ones with the most edges — so the
  maximum partition is larger than the mean by a factor that grows with skew, and a
  p99 latency budget must be sized against the maximum rather than the expectation.

NUMERICAL EXAMPLE (p = 16, m = 1e9, 21 B per position)
  one direction with boundary copies: 1e9 * (1 + 1/16) = 1.0625e9 positions
                                      = 22.31 GB stored, 1.0625x the single copy
  both directions materialised:       1e9 * (2 - 1/16) = 1.9375e9 positions
                                      = 40.69 GB stored, 1.9375x
  per-partition share (both dirs):    40.69 / 16     = 2.54 GB
  boundary fetches:                   1e9 * (15/16)  = 9.375e8 per full scan
  edge-cut copies at d = 100:         16 * (1 - (15/16)^100) = 16 * 0.9984 = 15.97
                                      i.e. a degree-100 node is on essentially every host

INTERPRETATION
  A vertex partition costs 1.94x storage and 0.94 boundary fetches per edge, which is
  affordable; an edge-cut partition replicates a high-degree node onto nearly every
  host, which is not. The asymmetry explains the standard practice: partition by one
  endpoint and accept the boundary traffic, because the alternative is a hot node
  present everywhere and updated everywhere.

VALIDITY
  Expectation under hash independence. With a tenant key and a skewed tenant-size
  distribution, the largest partition can exceed the mean by an order of magnitude,
  so the formula sizes the average and the latency budget must be sized separately.
```

**Variant D — dense storage at small node counts.** The dense matrix is correct for spectral work and
wrong for everything else, and the boundary is arithmetic.

```text
FORMULA
  B_dense(n) = n^2 * b_cell         b_cell = 8 B for float64
  B_csr(n, m) = 4 * (n + 1) + m * w_pos
  ratio        = B_dense / B_csr

WHERE
  n = node count, count; m = edge count, count; w_pos = bytes per position, B
  b_cell = bytes per matrix cell, B (4 for float32)

CONSTRAINT
  The dense form is required by algorithms that need the full matrix (eigendecomposition,
  matrix products, some spectral clustering variants). It is not required by traversal,
  and using it for traversal converts a sparse problem into a memory problem whose
  failure appears only when n grows.

NUMERICAL EXAMPLE
  n = 1e4, m = 2e4, w_pos = 21 B
    dense = 1e8 cells * 8 B                  = 800,000,000 B = 800 MB
    csr   = 4 * 10,001 + 2e4 * 21 = 40,004 + 420,000 = 460,004 B
    ratio = 800,000,000 / 460,004            = 1,739x
  n = 1e6, m = 2e6
    dense = 1e12 cells * 8 B                 = 8 TB
    csr   = 4 * 1,000,001 + 2e6 * 21         = 46 MB
    ratio = 8e12 / 4.6e7                     = 173,913x

INTERPRETATION
  The ratio grows as n^2 / (n + m), so it is unbounded in n and bounded below by about
  n/2 for a sparse graph. At n = 10,000 anything under a gigabyte is acceptable; at
  n = 1,000,000 the dense form is not a choice at all. The numeric boundary is
  therefore n ~ 1e4 for a workstation and n ~ 1e3 for an edge device, and it moves with
  b_cell rather than with the algorithm.

VALIDITY
  Assumes a graph stored as an adjacency matrix of weights. A binary dense matrix at
  1 bit per cell is 64x smaller and moves the boundary by a factor of 8 in n, which is
  why bitset representations are a separate row in Section 7.
```

**Variant E — symmetric relations stored once versus twice.** The cheapest variant to get wrong, because
the two forms differ by a factor of two in storage and by an assertion in correctness.

```text
FORMULA
  storage factor = 1 row per fact, both directions traversed by the query layer
                 = 2 rows per fact, one traversal direction per row
  symmetry assertion: A[i][j] == A[j][i] for every symmetric relation
  partial-write exposure = 2 rows that must be written atomically

WHERE
  A is the adjacency structure for a symmetric relation type
  the assertion is exact and costs O(m) to check on a snapshot

CONSTRAINT
  Materialising both directions is safe only if the write is atomic across the pair.
  A partial write leaves one direction present and the other absent, producing an
  asymmetric structure that fails the assertion for exactly the interrupted writes,
  which is a pattern an operator can recognise rather than a random corruption.

NUMERICAL EXAMPLE (toy graph, symmetric SIGHTING relation only)
  stored once:  4 occurrences (E6..E9) -> 4 rows, 4 positions
  stored twice: 8 rows, 8 positions; sum of degrees 16 against 2m = 8 for the
                undirected reading, so the identity must be restated for the stored
                form or the assertion fires on a correct graph
  storage delta: 4 * 21 B = 84 B on the toy, 5e8 * 21 B = 10.5 GB at 5e8 sightings

INTERPRETATION
  Store symmetric relations once and traverse both directions in the query layer,
  because the alternative doubles storage and adds an atomicity requirement that a
  batch loader will eventually violate. The exception is a serving structure whose
  latency budget forbids a reverse index; then store twice, and make the symmetry
  assertion part of the load gate rather than part of a report.

VALIDITY
  The storage factor is exactly 2 for a fully symmetric relation and 1 for a fully
  directed one. Mixed schemas produce a factor between them, and the factor should be
  computed per relation type rather than quoted for the graph as a whole.
```

**Variant F — the hypergraph and its two binary encodings.** Group facts do not reduce to pairs without a
loss, and the two standard encodings lose different things.

```text
FORMULA
  star expansion:   nodes + hyperedges, incidence edges = sum of group sizes
  2-section:        nodes only, clique edges = sum of C(s_g, 2)
  inflation factor  = sum C(s_g,2) / sum s_g
  a group of size s contributes s incidence edges or C(s,2) clique edges

WHERE
  s_g = size of group g, count >= 2
  inflation is the ratio of clique edges to incidence edges, dimensionless

CONSTRAINT
  The 2-section preserves pairwise adjacency and destroys group identity: after the
  expansion, the statement "these two were in a group together" survives, and the
  statement "these four were in the same group" does not. The star expansion
  preserves both, at the cost of a node whose degree is the group size and which is
  not an entity of the domain.

NUMERICAL EXAMPLE
  s = 4: incidence 4 edges, clique 6 edges, inflation 1.5x
  s = 100: incidence 100 edges, clique 4,950 edges, inflation 49.5x
  s = 2: incidence 2, clique 1, inflation 0.5x, and the two encodings answer the same
         question, which is why the defect is invisible in tests built from pairs

INTERPRETATION
  Inflation is quadratic in group size, so the choice of encoding is a cost decision
  as well as a semantic one. Choose the 2-section for pairwise similarity features and
  the star for anything that must answer which group; a system that needs both should
  keep the incidence structure as the base and derive each expansion as a projection.

VALIDITY
  Exact for groups disjoint within a hyperedge set. Overlapping groups (a person in two
  households) make the clique edges deduplicate, so the realised inflation is below
  the formula and the group-identity loss is worse, because a pair can now share more
  than one group and the clique records only that they share at least one.
```

### 4.4 Limits and counterexamples

**The handshake identity fails when the two sides are filtered differently.** A snapshot built with
`t_avail <= t_decision` on the edge table and a degree column materialised at a different cut-off will
violate the identity by exactly the number of facts that arrived between the two cuts. The failure looks
like a data-quality problem and is a scheduling problem, so the assertion should carry the snapshot
identifier so that the two sides can be shown to share it.

**The degree-pair inequality fails when directions are double-counted.** If a symmetric relation is
materialised in both directions and the degree aggregate counts positions rather than neighbours, d_occ
doubles while d_dist does not, and the surplus doubles. The inequality still holds, which is the problem:
a defect that preserves the assertion is invisible to it. The discriminating check is to compare the
sum of positions against 2m and the sum of distinct neighbours against 2x the distinct pair count.

**The projection formula fails under deduplication and pruning.** A pair that co-occurs in three groups is
emitted three times before dedup and once after, so P overstates the deduplicated table by the number of
multi-group pairs. A pruned projection has no closed form at all, because the pruning rule depends on the
data. The boundary is measurable: compute P by formula, compute the realised row count, and report both;
a gap larger than a few percent means the formula is no longer describing the artifact.

**The storage formula fails on compressed storage, in both directions.** A columnar store with delta
encoding on a sorted timestamp column can use far less than 8 B per value, making the formula an
overestimate. A row store with page fill factor 70% and a 24 B page header uses more than the formula,
making it an underestimate. The formula is for comparing layouts to each other under one assumption set,
and quoting its absolute figure as a deployment prediction is the misuse.

**The partition formula fails under skew, and the failure is not small.** A tenant-keyed partition with a
heavy-tailed tenant-size distribution produces a maximum partition several times the mean, so a p99
latency computed from the expectation is optimistic by that factor. The check is cheap: compute the
partition sizes and report max/mean. A ratio above about 3 means the partition key is correlated with
size and the deployment needs either a salt or a split of the largest tenant.

**The dense-matrix ratio fails when the graph is genuinely dense.** A co-occurrence matrix over 2,000
categories is dense by construction, so the 1,739x ratio at n = 1e4 does not apply and the dense form is
correct. The counterexample is worth stating because it marks the boundary: the formula's ratio is a
statement about sparse graphs, and a graph with m near n^2/2 has no sparse advantage to lose.

**The reification multiplier fails when statements are packed.** A store that writes several statements per
physical row, or that dictionary-encodes repeated predicates, can land below the 1.6x figure and even below
the flattened table. The formula compares logical counts, and it should be labelled that way, because the
physical comparison depends on the engine's encoding rather than on the model.

**Takeaway:** Every formula in this section is an inequality or an expectation under stated assumptions;
the counterexamples are those assumptions failing, so the review question is which assumption is closest
to breaking rather than whether the formula is right.

## 5. Worked Numerical Example

The example carries one nine-fact graph through all four formulas so that each stage's arithmetic can be
checked against the previous stage's output. Every input is ILLUSTRATIVE.

### 5.1 The fact table

```text
fact_id  src  dst  rel       value    t_event              t_avail
E1       A1   M1   TRANSFER  380.00   2026-03-02 09:14:00  2026-03-02 09:14:41
E2       A2   M1   TRANSFER  615.00   2026-03-02 11:02:00  2026-03-02 11:02:38
E3       A1   M1   TRANSFER  120.00   2026-03-03 08:41:00  2026-03-03 08:41:52
E4       A2   M1   TRANSFER   90.00   2026-03-03 16:20:00  2026-03-03 16:21:14
E5       A1   M1   TRANSFER   45.00   2026-03-04 07:03:00  2026-03-04 07:03:29
E6       A1   D1   SIGHTING  (null)   2026-03-04 07:05:00  2026-03-04 07:05:31
E7       A2   D1   SIGHTING  (null)   2026-03-04 07:41:00  2026-03-04 07:41:44
E8       A1   D2   SIGHTING  (null)   2026-03-04 19:12:00  2026-03-04 19:12:36
E9       A2   D3   SIGHTING  (null)   2026-03-05 06:30:00  2026-03-05 06:31:02

nodes: A1, A2 (accounts), M1 (merchant), D1, D2, D3 (devices)
m = 9 occurrences; 6 distinct unordered pairs; 2 relation types; 5 non-null values

availability lag per fact (seconds): 41, 38, 52, 74, 29, 31, 44, 36, 62
mean lag = (41+38+52+74+29+31+44+36+62) / 9 = 407 / 9 = 45.2 s
max lag = 74 s (E4), min lag = 29 s (E5)

decision time for the snapshot: 2026-03-04 23:59:59
eligible facts: t_avail <= decision time -> E1..E8 (8 facts); E9 is excluded
```

The lag column is the reason the example carries both timestamps. The mean lag of 45.2 seconds looks
negligible against a daily feature, and it is, until a decision time falls inside the lag window: a
decision at 2026-03-05 06:30:30 would see E9's event time but not its availability, and a snapshot filtered
on event time would include a fact the serving path could not have had. That is the entire mechanism
behind F-02, and the toy graph contains one instance of it.

The nullable value column is the second deliberate feature. Four of nine facts have no value, and the null
is structural rather than missing: a sighting has no amount. Any aggregate that computes a mean over the
value column must decide whether the denominator counts nine facts or five transfer facts, and the two
choices produce 138.89 and 250.00 respectively.

### 5.2 Step 1 — degrees under both definitions

```text
occurrence degree (all 9 facts)
  A1: E1, E3, E5, E6, E8  = 5
  A2: E2, E4, E7, E9      = 4
  M1: E1, E2, E3, E4, E5  = 5
  D1: E6, E7              = 2
  D2: E8                  = 1
  D3: E9                  = 1
  sum = 18 = 2 * 9                                            identity check 1

distinct-neighbour degree
  A1: {M1, D1, D2} = 3      A2: {M1, D1, D3} = 3
  M1: {A1, A2}     = 2      D1: {A1, A2}     = 2
  D2: {A1}         = 1      D3: {A2}         = 1
  sum = 12 = 2 * 6 distinct pairs                            identity check 2

surplus per node: A1 2, A2 1, M1 3, D1 0, D2 0, D3 0; total 6
verification: 18 - 12 = 6 and sum over nodes = 2 + 1 + 3 = 6
```

The surplus concentrated on M1 is the quantitative version of the observation that a merchant is a
collection point. A1 also carries a surplus of 2 because it made three transfers to the same merchant,
which is ordinary behaviour and not an artifact. The check that separates ordinary multiplicity from an
instrumentation problem is whether the repeated pair has distinct event times: three transfers on three
dates is behaviour, three identical facts on one timestamp is a duplicate load.

### 5.3 Step 2 — the as-of snapshot

```text
decision time 2026-03-04 23:59:59 -> eligible E1..E8, m_snap = 8

occurrence degrees in the snapshot
  A1: E1, E3, E5, E6, E8 = 5
  A2: E2, E4, E7         = 3
  M1: E1..E5             = 5
  D1: E6, E7             = 2
  D2: E8                 = 1
  D3: (none)             = 0
  sum = 16 = 2 * 8                                            identity check 3

distinct neighbours in the snapshot
  A1: {M1, D1, D2} = 3   A2: {M1, D1} = 2   D3: {} = 0
  M1: {A1, A2}     = 2   D1: {A1, A2} = 2   D2: {A1} = 1
  sum = 10 = 2 * 5 distinct pairs in the snapshot              identity check 4

difference from the full graph: one fact (E9) removed, A2 4->3, D3 1->0,
distinct pairs 6->5, occurrence sum 18->16, distinct sum 12->10
```

The snapshot is where the arithmetic becomes operational, because it is the object the features are
actually computed from. Note that an entity can exist in the snapshot with degree zero: D3 is a known
device with no eligible facts, and whether it appears as a row with zero or is absent entirely is a
denormalisation decision that changes every mean computed over the node table. Both are defensible and
only one can be tested by a fixture, which is why the choice belongs in the feature contract.

### 5.4 Step 3 — CSR construction and cost for the snapshot

```text
node ordering by first appearance: A1=0, A2=1, M1=2, D1=3, D2=4, D3=5

positions in the layout L3 (29 B each)
  A1: M1, M1, M1, D1, D2   (5 positions; three separate TRANSFER positions to M1)
  A2: M1, M1, D1           (3)
  M1: A1, A2, A1, A2, A1   (5)
  D1: A1, A2               (2)
  D2: A1                   (1)
  D3: -                    (0)

offset array (n+1 = 7 entries, 4 B each)
  offset[0]=0  offset[1]=5  offset[2]=8  offset[3]=13  offset[4]=15  offset[5]=16
  offset[6]=16
  monotonicity: 0 <= 5 <= 8 <= 13 <= 15 <= 16 <= 16        holds
  last entry equals the position count 16                  holds

bytes
  offsets   4 * 7             =  28 B
  positions 16 * 29 B         = 464 B
  total                       = 492 B        (30.75 B per occurrence)
  same snapshot as L2 (21 B): 28 + 16*21 = 364 B
  same snapshot as L1 (5 B):  28 + 16*5  = 108 B
  row store: 8 * 176          = 1,408 B      ratio to L3 = 2.86x

per-node locality check
  A1's positions occupy bytes 28 .. 28+5*29-1, i.e. 145 contiguous bytes, one cache line
  of 64 B does not cover them, so a hub's expansion touches several lines but no page
  D1's positions are the two entries at index 13..14, i.e. 58 bytes
```

Two structural checks appear in the block and both are worth running on every build. Offset monotonicity
catches a merge that appends out of order. The last-offset equality catches a loader that allocated
positions and failed to write them, which is the defect that produces a graph with silent zero-degree
nodes. Neither check needs the fact table, so both can run in the build that produces the array.

The per-node locality reading is where the layout's advantage actually lives. A1's five positions are 145
contiguous bytes; the same five facts in a row store are five rows scattered across 880 bytes at whatever
locations the insert order produced. The difference is not the byte count, which is a factor of 2.9, but
the number of cache lines and the predictability of the access pattern, which is a factor that shows up in
latency but not in `du` output.

### 5.5 Step 4 — the projection and one feature

```text
projection: accounts over shared devices (SIGHTING relation only), snapshot 5 facts
  D1 has accounts {A1, A2}     -> C(2,2) = 1 pair (A1, A2)
  D2 has accounts {A1}         -> C(1,2) = 0
  D3 has accounts {}           -> 0
  projected pairs = 1
  channels supporting that pair = 1 (D1)
  marginal check: the pair exists once; adding a third account to D1 would add
                  2 pairs (C(3,2) - C(2,2) = 3 - 1 = 2 = k for k = 2)

feature: mean transfer value per account, snapshot 5 transfer facts, null policy exclude
  transfers: E1 380.00, E2 615.00, E3 120.00, E4 90.00, E5 45.00
  denominator = 5 transfer facts (not 8 snapshot facts, not 9 total facts)
  sum = 380 + 615 = 995; 995 + 120 = 1,115; 1,115 + 90 = 1,205; 1,205 + 45 = 1,250
  mean = 1,250 / 5 = 250.00 USD per transfer
  wrong version A: sum / snapshot facts = 1,250 / 8 = 156.25
  wrong version B: sum / all facts      = 1,250 / 9 = 138.89
  spread between correct and wrong B    = 250.00 - 138.89 = 111.11, i.e. 44.4% low

feature: distinct-device count per account
  A1: {D1, D2} = 2      A2: {D1} = 1 (D3 excluded by the snapshot)      A2 full graph = 2
  the snapshot changes A2's feature from 2 to 1, which is correct at the decision time
```

The two wrong versions of the mean are included because both are produced by real code. Version A divides
by the snapshot's fact count, which mixes transfers with sightings and therefore drifts with logging
volume: a device-sighting outage lowers the feature. Version B divides by all facts including those after
the cut-off, which mixes two time windows and drifts with the freshness of the load. The correct version
divides by the count of facts in its own relation and window, which is the only denominator that is stable
under changes to unrelated pipelines.

### 5.6 Independent re-check of every number

```text
check 1 — incidence by a second route
  sum of degrees computed from the edge list: for each fact, +1 to each endpoint.
  9 facts * 2 endpoints = 18 increments; the degree vector sums to 18.   consistent

check 2 — distinct pairs by enumeration
  pairs: (A1,M1) (A2,M1) (A1,D1) (A2,D1) (A1,D2) (A2,D3) = 6
  sum of distinct degrees = 2 * 6 = 12.                                 consistent

check 3 — snapshot identity by subtraction
  removing E9 removes one incidence from A2 and one from D3:
  occurrence sum 18 - 2 = 16 = 2 * 8.   distinct sum 12 - 2 = 10 = 2 * 5 pairs. consistent

check 4 — CSR bytes by the closed form
  4*(n+1) + positions*w_pos = 4*7 + 16*29 = 28 + 464 = 492 B.        consistent
  positions = sum of degrees = 16, which is the array length the builder wrote. consistent

check 5 — row-store width by column sum
  edge_id 8 + src 16 + dst 16 + rel 1 + value 8 + ccy 3 + t_event 8 + t_avail 8
  + ingest_batch 8 + provenance 4 = 80 B of columns
  + 24 B tuple header + 3 index pointers of 24 B = 72 B
  total = 80 + 24 + 72 = 176 B per row; 8 rows = 1,408 B.               consistent

check 6 — projection marginal
  C(3,2) - C(2,2) = 3 - 1 = 2 = k for k = 2, matching the "adds k pairs" rule.
  C(4,2) - C(3,2) = 6 - 3 = 3 = k for k = 3.                            consistent
```

The six checks are deliberately of different kinds: two count by enumeration, two recompute by closed
form, one derives a width from a column list, and one tests an identity on small cases. A model whose
numbers survive checks of different kinds is far more likely to be right than one that survives the same
check repeated, which is the failure mode of validating a formula against its own implementation.

### 5.7 Sensitivity

| Parameter moved | Baseline | Variant | Effect on the number | Reading |
|---|---|---|---|---|
| Snapshot cut-off | 2026-03-04 23:59:59 | 2026-03-05 12:00:00 | m 8 -> 9, occurrence sum 16 -> 18 | one late fact raises every degree it touches by 1 |
| Availability lag filter | none | exclude lag > 60 s | E4 excluded from a backtest at that instant | the longest-lag fact is the one a strict filter removes |
| Value null policy | exclude nulls | coerce to zero | mean 250.00 -> 138.89 | the denominator change, not the numerator, drives it |
| Position width | 29 B | 21 B (drop t_avail) | snapshot 492 B -> 364 B, -26.0% | one timestamp is a quarter of the position |
| Position width | 29 B | 45 B (add system times) | snapshot 492 B -> 748 B, +52.0% | the belief axis is the largest single addition |
| Node count at fixed m | n = 1e8 | n = 1e6 | offset array 0.4 GB -> 4 MB | the offset term is negligible until n ~ 1e7 |
| Projection hub degree | 50,000 | 25,000 | C(k,2) 1.25e9 -> 3.12e8, -75.0% | halving one degree quarters one term |
| Partition count | p = 16 | p = 64 | stored 1.0625x -> 1.0156x, boundary 0.9375 -> 0.9844 | more parts reduce duplication and raise boundary traffic |
| Reification properties | p = 3 | p = 8 | 280 B -> 560 B per fact, 3.18x the row | the multiplier is linear in the property count |

Two readings deserve comment because they invert an intuition. First, raising the partition count reduces
duplication while increasing boundary traffic per edge, so the two costs move in opposite directions and
the optimum depends on whether the workload is compute-bound or network-bound; the table's p = 64 row has
1.6% duplication and 98.4% boundary fetches, which is a network problem dressed as a storage win. Second,
the offset array is negligible until the node count approaches the edge count: at n = 1e6 and m = 1e9 the
offset term is 4 MB against 21 GB, which is 0.02%, so a design review focused on reducing node count is
optimising the wrong term.

### 5.8 Reading the same arithmetic at production scale

```text
scale: n = 1e8 nodes, m = 1e9 occurrences, layout L2 (21 B per position)

  offsets     4 * (1e8 + 1)               = 0.400 GB
  positions   1e9 * 21                    = 21.000 GB
  total                                   = 21.400 GB
  row store   1e9 * 176                   = 176.0 GB          ratio 8.22x
  layout L3   0.4 + 1e9*29                = 29.400 GB          ratio to L2 1.374x
  layout L4   0.4 + 1e9*45                = 45.400 GB          ratio to L2 2.121x
  information ceiling log2 C(m+n-1, n-1)  = 4.834e8 bits
                                          = 0.0604 GB          L2 is 354x above it

  projection with one hub k = 50,000 and 1e5 groups k = 100
    P = 1.249975e9 + 4.95e8 = 1.745e9 pairs = 1.745x m
    at 8 B per pair (two int32)           = 13.96 GB materialised
    emission at 1e7 pairs/s               = 174.5 s

  nightly build comparison
    CSR   allocate + write 21.4 GB        = 21.4 GB of writes
          scan 1e9 predicates at 1e8/s    = 10 s
          write 1e9 positions at 1e8/s    = 10 s
          total ~ 20 s plus I/O for 21.4 GB = 10.7 s at 2 GB/s  -> ~31 s
    row   read 176 GB at 2 GB/s           = 88 s
          index build 1e9 * log2(1e9) = 1e9 * 29.9 comparisons = 2.99e10
          at 1e8 comparisons/s            = 299 s
          total dominated by the index build -> ~387 s
```

The information ceiling is the line worth keeping. The number of distinct configurations of 1e9 positions
across 1e8 rows is C(m+n−1, n−1), so no encoding can use fewer than log of that many bits, which is 60 MB.
The CSR uses 21.4 GB, 354x above the ceiling, because it stores explicit identifiers rather than structure.
That is the correct trade for an access pattern that needs O(d_v) contiguous reads, and it is also the
reason compressed variants exist — and the reason a designer should not assume 21 B per position is a floor.

The build comparison shows a factor of 12.5 between the two layouts, and almost all of it is index
maintenance rather than scanning. A team that measures the scan and concludes the row store is affordable
has measured the smaller term; the index build is 299 of the 387 seconds, and it is the term that scales
with log(m) while the scan scales with m. At 1e10 facts the log term grows by another 10%, while the scan
grows by 10x, so the ratio narrows with scale — which is a reason to re-derive the comparison rather than
inherit it.

**Takeaway:** Derive the three numbers together — bytes, build time and projection peak — because a layout
that wins on one of them frequently loses on the other two.

## 6. Visual / Geometric Intuition

### 6.1 The adjacency matrix as a picture of the model

Draw the node set on both axes and mark a cell for each adjacent pair. A sparse relationship graph is a
nearly empty square with a thin diagonal pattern; a projection clique is a filled block whose side is the
group size. The area of the square is n², and the marked area is at most 2m, so the ratio of ink to paper
is the sparsity that justifies a sparse layout. At n = 1e6 the square has 1e12 cells and a typical graph
marks 2e6 of them, one in 500,000.

The picture makes multiplicity invisible in a way worth noticing. A cell is marked or not, so a
multigraph's three transfers and one transfer draw identically. Everything the multigraph knows lives in
a weight attached to the cell, and a weight map drawn as colour is the only way to see it. This is
precisely the collapse described in F-01: the matrix is the natural representation for presence, and
presence is exactly the question a frequency feature is not asking.

The picture also shows why symmetry assertions are visual. A symmetric relation draws a symmetric pattern,
so a defect that breaks symmetry is visible as a mark without its mirror. On 6 nodes that is a glance; on
1e6 nodes it is a sampled test, but the same check applies to a sampled submatrix and it localises to the
rows sampled.

### 6.2 The incidence view

Instead of a square, draw the bipartite graph of nodes and facts: two columns of dots with a line per
fact crossing between them. This is the representation that loses nothing, because each fact is an object
with its own line and its own attributes. The node-only graph is what you get by squashing the fact column
and joining lines that share a node.

Squashing is where the loss happens, and the geometric version of the loss is that lines crossing near a
shared node merge into one thick line whose width is multiplicity. A reader of the node-only picture sees
a thick line and cannot tell whether it means three facts or one fact with a weight of three. The
incidence picture keeps that distinction, which is why the base store should be the incidence structure and
every node-level graph should be derived.

The incidence view also explains the hypergraph encodings geometrically. A group of four is one dot in the
fact column with four lines in the incidence view. The star expansion keeps that dot and calls it a node.
The 2-section deletes the dot and draws lines between the four, which is six lines where there were four.
The picture of the deletion is the picture of the loss: the dot is gone, so the question "which group" can
no longer be pointed at.

### 6.3 The projection as a quotient

A projection is a map that identifies the facts of one type and joins the nodes they connect. Draw it as
a folding of the incidence graph along the fact column: pairs of lines sharing a fact are glued together.
The gluing is what creates C(k,2) edges from k lines, and it is also what makes the operation
non-invertible — an unglued pair cannot be un-glued into the specific facts that joined it.

The geometric consequence is that a projection can only shorten distances between nodes of the retained
type, never lengthen them, because every projected pair is adjacent by construction. A rule expressed as
"within two hops" therefore means something strictly larger on the projection than on the base graph, and
the difference is the folding. This is why a hop-distance feature computed on a projection and a hop-
distance feature computed on the base graph are different features that share a name.

### 6.4 The temporal axis

Draw time as a third dimension with the fact set as a stack of layers, one per availability instant. A
snapshot is a horizontal slice through the stack, and a bitemporal store is the same stack with each
fact carrying its own belief interval along a fourth axis. The picture makes two facts obvious that prose
obscures. A snapshot at time t is an intersection, not a filter, so it is cheap if the layer boundaries are
materialised and expensive if they must be computed by scanning. And a correction that arrives late
retroactively changes the layer it belongs to on the belief axis while leaving the event axis fixed, which
is why one timestamp cannot express both.

The picture also shows why retention multiplies storage: 30 daily snapshots of a 21.4 GB structure is
642 GB before replication, and three replicas take it to 1.93 TB. A design that stores snapshots to make
backtests reproducible pays that bill, and the alternative — recomputing a snapshot from the base facts —
costs a build per backtest run rather than a byte per day.

### 6.5 The hub as a geometric distortion

A hub is a node whose row in the matrix is dense and whose column is dense, so it appears as a cross
through the sparse square. Every pair in its neighbourhood is two hops apart through it, which means one
node sets the distance between k² pairs. On the toy graph, M1 has distinct degree 2 and already makes A1
and A2 adjacent at distance 2; a real merchant with 50,000 counterparties makes 1.25 billion pairs
distance-2, which is why a "two hops" rule on a payment graph is a rule about merchants.

The distortion is geometric in the literal sense: the hub collapses the graph's diameter and concentrates
its triangles. Any null model for a motif count that does not preserve the degree sequence will therefore
attribute the hub's triangles to coordination, which is the most common invalid inference in this area.

### 6.6 The sparse pattern in the incidence view

A real incidence structure is not uniformly sparse: it has a small number of very dense rows (the hubs),
a long tail of degree-1 and degree-2 nodes (devices seen once, accounts with one transfer), and a middle
that carries most of the facts. Plotted as a histogram of row degrees on a log axis, the shape is
characteristically heavy-tailed, and the two tails have different meanings. The upper tail is
instrumentation and aggregation; the lower tail is sparsity and thin evidence. Features that treat both
tails symmetrically — a mean degree, for instance — are dominated by the upper tail and unrepresentative
of the middle.

### 6.7 The projection lattice

Draw the layers as a lattice: raw facts at the bottom, the incidence structure above them, the node-level
graph above that, the projection above that, and the feature table at the top. Each upward step loses
something and each has a rebuild path downward. The lattice is the right mental picture because it makes
the rebuild dependency explicit: a feature cannot be rebuilt without its snapshot, a snapshot without its
identity version, and an identity version without the base facts. A system that stores only the top layer
has no rebuild path and therefore no way to answer a question about the middle.

**Takeaway:** Four pictures carry the section — the sparse square, the incidence bipartite graph, the
folding that creates a projection, and the time stack — and each answers a question that the formulas
alone leave ambiguous.

## 7. Comparison / Selection Table

### 7.1 Layout families

| Layout | Grain | Cheap query | Expensive query | Information lost | Choose when |
|---|---|---|---|---|---|
| Edge table, row store | one fact per row, all columns | filtered scan by time, type, node | multi-hop traversal | nothing if all columns are kept | base store of record, audit, corrections |
| Edge table with composite index | one fact per row, indexed pairs | one-hop by either endpoint | three-hop paths at scale | nothing | serving small snapshots per request |
| CSR array | one position per (node, occurrence) | one-hop expansion, batch analytics | per-edge filtering, mid-array insert | relation and time filters without parallel arrays | nightly feature builds, batched computation |
| CSC array | one position per (destination, occurrence) | reverse traversal, in-edge features | forward expansion | as CSR | bidirectional search, predecessor features |
| Adjacency list of lists | one list per node | iterative traversal, sampling | vectorised scans | nothing, but allocator-bound | small graphs, prototypes, dynamic graphs |
| Dense adjacency matrix | n² cells | matrix products, spectral work | anything sparse | nothing, wastes memory | n below about 1e4, spectral algorithms |
| Bitset adjacency | n² bits | set intersection, reachability closure | weighted aggregates | weights | n below about 1e5, reachability only |
| Triple table | one statement per row | predicate-first matching | multi-hop joins, aggregates | values need extra statements | heterogeneous sources, schema evolution |
| Reified property graph | fact as a resource | per-fact provenance queries | bulk aggregation | nothing | audited facts, per-fact annotations |
| Hypergraph, star expansion | node per group | incidence queries | pairwise adjacency | nothing; adds non-domain nodes | group events, joint accounts, shared devices |
| Hypergraph, 2-section | clique per group | pairwise adjacency | group identity, group count | group identity, hyperedge count | pairwise similarity features |
| Materialised projection table | one row per projected pair | downstream pair features | provenance of the pair | the fact layer from the lineage | fixed pair-level model with no audit need |

Read the table as a set of bets rather than a ranking, and note that only two rows lose information that a
regulator would ask for: the materialised projection loses provenance, and the 2-section loses group
identity. Every other loss is a column that can be added back with a parallel array, which is why those two
rows deserve the most scrutiny in a design review.

Three rows are commonly chosen for the wrong reason. The **dense matrix** row is chosen for spectral work,
where the dense form is genuinely required; choosing it to simplify code converts a sparse problem into a
memory problem whose failure appears only when n grows, and the arithmetic in 4.3(D) shows the ratio rising
as n²/(n+m). The **materialised projection** row is chosen when a downstream model wants one row per pair,
and it is legitimate there; the price is that the fact layer leaves the lineage, so an audit can no longer
ask which device caused the pair. The **triple table** row is chosen for schema flexibility, and the price
is that every aggregate is a join over a table whose join key is not constrained to be dense.

Two selection rules follow from the cost formulas and are worth stating as heuristics with their conditions
attached. If the dominant query is one-hop expansion over a frozen snapshot with more than roughly a
million facts, CSR wins on both memory and scan locality, and the advantage grows with m. If the dominant
query is "which facts involve this node, filtered by type and time", the row store wins despite its byte
overhead, because the filter is applied before the join rather than after the expansion — and the 8.22x
storage penalty at 1e9 facts buys an index that prunes most of the scan.

### 7.2 Selection procedure for a new dataset

1. Write the decision the graph serves, with its latency budget and refresh cadence. A nightly queue and
   an authorisation decision differ by five orders of magnitude in latency and cannot share a serving
   layout, no matter how similar their facts look.
2. Write the fact grain in one sentence containing exactly one "per". If the sentence needs two "per"s,
   the grain is under-specified: "one row per transfer per account" is ambiguous about whether a transfer
   appears once or twice.
3. Enumerate the queries that will run and the filter each applies. The dominant query determines the
   layout; the second and third determine whether a second copy is warranted.
4. Test each query against the loss column. If a query needs information a layout deletes, the layout is
   wrong regardless of its speed, and the speed is a distraction from the answer.
5. Compute storage from the formula, then multiply by retained snapshots and replication. Both multipliers
   are routinely forgotten and each is typically between 2 and 30.
6. Name the rebuild command and its trigger. A structure without a rebuild path is a store of record and
   must be append-only, which is a much stronger requirement than it sounds.
7. Write the assertion set: degree identity, offset monotonicity, relation registry coverage, distinct
   degree sum, and symmetry for symmetric relations. These are the checks that fail when a loader changes.

Steps 1 and 3 require talking to a consumer, and they are the steps most often replaced by a benchmark. A
benchmark answers which layout is faster for the query that was measured, which is a narrower question
than which is faster for the query that will dominate, and the difference between those two questions is
where most of the cost of a redesign comes from.

### 7.3 Two-layout architectures

The dominant production pattern is a row store as the store of record plus a CSR as the analytic replica.
It is justified by three conditions, all of which must hold. First, the analytic workload tolerates
staleness bounded by the rebuild interval. Second, the rebuild is deterministic, so a rerun produces
identical bytes. Third, deletion propagates, so a deleted fact invalidates the replica within the same
bound rather than surviving in it indefinitely.

When the third condition fails the architecture becomes a correctness liability: the store of record and
the replica disagree about whether a fact exists, and the replica is the one used for scoring. The failure
is not detectable by row-count reconciliation unless deletions are counted separately, which is why
deletion propagation needs an explicit test rather than a paragraph in a design document.

The interface between the two should be narrow and explicit: fact identifier, snapshot identifier,
identity version, and a build timestamp. Anything wider couples the two builds and makes a regression
ambiguous, because a change in the replica can no longer be attributed to a data change or a code change
without a bisect over both.

### 7.4 Choosing between the four timestamp sets

| Timestamp set | Answers | Cost per position | Choose when |
|---|---|---|---|
| none | current state only | 13 B (L1 5 B without weight) | the structure is derived and disposable |
| event only | when facts happened | 21 B | analytics on the event axis, no backtests |
| event + availability | what a model could have known | 29 B | any backtest, any decision-time feature |
| all four (bitemporal) | what was believed and when | 45 B | base store of record, corrections, audits |

The middle two rows are the ones confused in practice. A structure with event time only can compute
everything about the world and nothing about the pipeline, which means it can support a feature that is
recomputed identically but not a backtest that reproduces a decision. The 8 B that separates them is the
cheapest 38% in this sheet.

**Takeaway:** Choose the layout by naming the dominant query and the acceptable loss, and choose the
timestamp set by naming the question the consumer will ask about time.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, the causes that compete with it, and an observation that separates them.
The check is the deliverable; the mechanism is only the vocabulary for asking the question.

### F-01 Multiplicity collapsed to presence

**Mechanism.** A pair occurring three times is stored once, usually because a uniqueness constraint was
added to make a join safe. Every downstream count then reports presence, and a frequency feature becomes
constant at one. On the toy graph the collapse would take the occurrence sum from 18 to 12 and the mean
occurrences per pair from 1.50 to 1.00.

**Competing causes.** (a) collapse at write time via a constraint; (b) an as-of filter that is too
aggressive and drops repeat occurrences whose availability differs; (c) a `SELECT DISTINCT` added upstream
of the feature to fix a fan-out bug in a join.

**Discriminating check.** Count rows and distinct pairs on the same snapshot for one relation. If rows
exceed pairs and the feature maximum is one, the collapse is downstream of the fact table. If rows equal
pairs, it is at write time. On the toy graph the healthy ratio is 9/6 = 1.5, so the collapse appears as a
drop to 6 with the row count unchanged in the base table.

### F-02 Availability time backdated to event time

**Mechanism.** A batch loader has no load timestamp, so it fills availability from event time. Every
historical snapshot then appears to know facts it could not have known, offline metrics rise, and
production does not follow. The toy graph shows the signature: all nine lags become zero, whereas the
actual lags run from 29 s to 74 s.

**Competing causes.** (a) a missing column filled from event time; (b) a correction process that rewrites
the availability of old facts; (c) a timezone shift that moves availability earlier than the event.

**Discriminating check.** Compute the minimum and the distribution of t_avail − t_event. A negative minimum
proves a timezone or clock defect. A distribution with no mass above a few seconds proves the column is
synthetic, because real ingestion has a heavy tail. Then recompute one feature at a decision time inside
the tail window: a value that changes when the availability rule is restored shows the metric depended on
the backdating.

### F-03 Supernode fan-out in traversal

**Mechanism.** A shared token acquires 50,000 facts, and any two-hop expansion touching it visits the
entire neighbourhood. The traversal does not merely slow down; it changes the result's composition,
because the hub contributes 50,000 of the frontier's members and every aggregate over the frontier is
therefore an aggregate over the hub.

**Competing causes.** (a) a genuine hub such as a merchant or an institution; (b) a sentinel value such as
`device = 'unknown'`; (c) identifier recycling, where one token was reused across years for different
entities.

**Discriminating check.** Sample twenty facts at the hub and inspect the source payload. A constant string
or a null proves a sentinel. Real facts with disjoint account sets across years prove recycling. Real,
contemporaneous facts with varied payloads prove a genuine hub, which needs a policy (cap, downweight,
exclude) rather than a fix.

### F-04 Projection threshold applied to a closed form

**Mechanism.** A pipeline computes its expected output with C(k,2) and then prunes hubs below a degree
threshold. The pruned count is data-dependent, so the capacity estimate is wrong in the optimistic
direction and the job runs out of memory on the first hub-heavy day.

**Competing causes.** (a) the formula applied to a pruned projection; (b) a threshold applied to the input
degrees but not to the projected pair list, so the hub's pairs still emit; (c) deduplication that reduces
the realised count below the formula, which masks (a) until it does not.

**Discriminating check.** Emit the pair count per group before and after pruning, sorted by group. A ratio
close to one with a large formula prediction means the prune and the formula are measuring different
things; a ratio well below one means the prune works and the formula is simply the wrong estimate. The
useful report is the realised count and the formula side by side, with the gap attributed.

### F-05 Entity overmerge and undersplit

**Mechanism.** A resolution rule joins two records that are different people with the same name and
postcode, unioning their neighbourhoods; both nodes then have a degree equal to the sum of two, and every
two-hop feature doubles. The mirror defect splits one person into two nodes, deleting facts from each and
making both look sparse.

**Competing causes.** (a) a match threshold set too low; (b) a shared field such as a household address or
a generic name; (c) identifier recycling across time, where an old record's identifier was reused.

**Discriminating check.** Inspect the member count of the merged cluster and the field values of its
members. An overmerge has a member count above the population's 99.9th percentile with field values that
agree on the blocking key and disagree elsewhere. An undersplit has two clusters whose blocking keys
agree and whose distinguishing fields agree too, which means the match rule never compared them.

### F-06 Relation type used as a filter after projection

**Mechanism.** A feature wants transfer-based connectivity but is computed from a projection that mixed
sightings and transfers. The result is a number in mixed units, and it changes when the sighting
instrumentation changes even though no transfer did.

**Competing causes.** (a) the projection lost the relation column, as in the L1 layout without a parallel
array; (b) the feature filtered on relation but the projection had already merged two relations into one;
(c) a relation registry that maps two source codes to one canonical type.

**Discriminating check.** Compute the feature twice, once from the relation-filtered incidence structure
and once from the projection. A difference proves the projection lost or merged the type. Then check the
relation registry's distinct count against the source's code list, which localises (c) in one query.

### F-07 Null value coerced to zero

**Mechanism.** A nullable value column is summed with nulls treated as zero, so a mean is divided by a
denominator that includes facts with no value. On the toy graph the transfer mean falls from 250.00 to
138.89, a 44.4% error, while the numerator is unchanged.

**Competing causes.** (a) the coercion as described; (b) a join that produced a null for an unmatched fact
and the aggregate then treated it as a real null; (c) a unit change that stored a scaled value and
appeared as a large negative residual.

**Discriminating check.** Compare the aggregate's denominator against the count of facts with a non-null
value in the same relation and window. Equality means the denominator is correct; a larger denominator
means nulls were counted. Then sum the nulls explicitly: a non-zero count with a mean equal to the
non-null mean times the ratio identifies the coercion with no further work.

### F-08 Correction overwritten in place

**Mechanism.** A fact is corrected by updating the row, so the previous value and its availability are
gone. A backtest run before the correction cannot be reproduced, and a dispute about what the model saw
cannot be resolved. The defect is invisible in row counts, because the row count is unchanged.

**Competing causes.** (a) an in-place update as described; (b) a merge that collapsed two versions into one
by taking the latest per identifier; (c) an ingestion deduplication that kept the last write per natural
key.

**Discriminating check.** Count rows per fact identifier. A count of exactly one for every identifier in a
system that has ever issued a correction proves no version history. The complement is to ask for the value
of one fact at a past availability instant: a system with history answers, and a system without either
errors or returns today's value, and the two are distinguishable by comparing against a known correction.

### F-09 Bitemporal schema without a write rule

**Mechanism.** Four timestamp columns are added and the system is declared bitemporal. Nothing prevents two
visible versions of the same fact at the same belief instant, so a historical query double-counts and every
aggregate over it is inflated by the number of superseded versions.

**Competing causes.** (a) no enforcement at the write path; (b) a bulk load that bypassed the write path and
inserted overlaps; (c) a valid-time representation whose system-end convention is open-interval in one
loader and closed in another.

**Discriminating check.** Assert at most one visible row per (fact identifier, system instant) with a
group-by and having-count. A violation localises to the identifiers involved. The check is exact and cheap,
and it is the only assertion that distinguishes a bitemporal schema from a bitemporal system.

### F-10 Direction dropped for symmetric convenience

**Mechanism.** A directed relation is stored as undirected, so a transfer from A to M becomes a symmetric
edge. In-degree and out-degree are then the same number, and every feature that depends on direction —
net flow, fan-in from new counterparties — loses its signal.

**Competing causes.** (a) the direction dropped at ingest; (b) a symmetric flag in the relation registry set
wrongly; (c) a deduplication that collapsed A→M and M→A into one row because the pair matched.

**Discriminating check.** Compare the sum of out-degrees to the sum of in-degrees for the relation. For a
directed relation they are equal (both equal m), so equality proves nothing; instead check whether any node
has a non-zero out-degree and a zero in-degree. A merchant with many in-edges and no out-edges proves the
direction exists; a graph where every node has in-degree equal to out-degree for a supposedly directed
relation proves it was collapsed.

### F-11 Timestamp granularity mismatch

**Mechanism.** Two sources report at different granularities, one in seconds and one in days. A join on
timestamp equality then fails for all facts from one source, or succeeds for none, and the resulting graph
is partitioned by source with no error raised.

**Competing causes.** (a) granularity mismatch as described; (b) a timezone offset applied in one loader
only; (c) a string format difference that parses successfully into a wrong value, such as day-month
swapped.

**Discriminating check.** Compute the distribution of seconds-since-midnight for each source. A source whose
values are all zero has day granularity. Then compute the hour-of-day histogram: a source shifted by a
whole number of hours has a timezone defect, while one that is uniformly spread is granularity-independent.

### F-12 Self-loop counted once

**Mechanism.** A fact whose endpoints are the same entity, such as a transfer to oneself or a device
sharing event with itself, is counted once in the degree column. The handshake identity then fails by the
loop count, and the failure is attributed to the loader rather than to the loop.

**Competing causes.** (a) the convention as described; (b) a resolution merge that made two distinct
endpoints identical, creating loops retroactively; (c) a projection that emits a self-pair when a group of
size one is not filtered.

**Discriminating check.** Count loops and compare the degree sum to 2m + L. Equality with the loop term
included proves the convention is loop-aware and correct; equality with 2m proves loops are counted once.
A merge-created loop is distinguishable because the loop count changes between identity versions while the
base facts do not.

### F-13 Composite key collision across relation types

**Mechanism.** A key of (src, dst) without the relation type collides across types, so a transfer and a
sighting between the same pair of nodes overwrite each other. The graph loses one of the two facts silently,
and the surviving type is whichever loaded last.

**Competing causes.** (a) a key missing the relation as described; (b) a key that includes relation but
whose enum maps two source codes to the same value; (c) a staging table that partitioned by pair and kept
one row per partition.

**Discriminating check.** Count distinct (src, dst, rel) triples against distinct (src, dst) pairs per type.
A per-type count below the triple count proves collision. Then count per relation: the survivor's count
exceeds the others by exactly the collided number, which identifies the overwrite direction.

### F-14 Snapshot rebuilt from a different identity version

**Mechanism.** A snapshot identifier names a date and a relation filter but not an identity version, so a
rebuild after a resolution change produces a different graph under the same name. Feature values change,
the change is attributed to the model, and the investigation starts in the wrong place.

**Competing causes.** (a) the version omitted from the identifier; (b) a version recorded but not used in
the build; (c) a canonical-identifier rule that allocates fresh identifiers on each run even for an
identical partition.

**Discriminating check.** Rebuild the snapshot twice from the same identifier and diff the byte counts and
the member counts. Identical bytes prove determinism; identical member counts with different identifiers
proves the allocation rule is at fault; different member counts prove the partition changed, which means
the version was not pinned.

### F-15 Hub pruning applied at query time

**Mechanism.** A hub exclusion list is applied inside each query rather than at build time. The exclusion
costs a predicate evaluation per frontier node, and worse, the pruned structure is not what the aggregates
were computed against, so reports and features disagree about the graph's degree distribution.

**Competing causes.** (a) query-time pruning as described; (b) a build-time prune that was not propagated
to one consumer; (c) two prune lists maintained by different teams.

**Discriminating check.** Compute the total degree from the serving structure and from the stored
aggregate. A gap equal to the excluded hubs' degrees proves a prune is in effect on one side only. Then
diff the two hub lists by identifier, which is the cheapest way to find (c).

### F-16 Storage estimate that ignores indexes and replicas

**Mechanism.** A capacity plan uses the CSR formula and omits the secondary indexes on the row store, the
retained snapshots and the replication factor. The plan therefore understates by a factor between 4 and 90,
and the shortfall appears as a disk-full incident rather than as a design error.

**Competing causes.** (a) omitted multipliers as described; (b) index storage estimated at one pointer per
row when it is typically two to three; (c) a snapshot retention that counts days rather than decision
instants, which is larger when the decision is made more than once per day.

**Discriminating check.** Take the plan's figure and multiply explicitly by the number of retained
snapshots and the replication factor in front of the planner. Then compare against the measured bytes of
one snapshot: a factor above 1.2 between estimate and measurement identifies which term was omitted,
because the four terms are of very different sizes.

**Takeaway:** Each entry separates a symptom that several mechanisms share, so the check must change the
observation rather than re-examine the same number.

## 9. Complexity & Cost Analysis

Costs are stated in operation counts with ILLUSTRATIVE constants: 1e8 simple operations per second, 100 ns
per random memory access, 10 ns per contiguous access, 2 GB/s sequential read. They compare designs; they
do not predict a deployment.

| Operation | Layout | Time | Space | Notes |
|---|---|---|---|---|
| Ingest one fact | row store | O(1) amortised | O(1) | plus index maintenance at O(log m) |
| Ingest one fact | CSR | O(1) if the node is last, else O(n) | O(1) | mid-array insert shifts every later offset |
| One-hop expansion | CSR | O(d_v) contiguous | O(1) extra | 50,000 positions is 1.05 MB at 21 B |
| One-hop expansion | row store with index | O(log m + d_v) | O(1) | log2(1e9) = 29.9 seeks per starting point |
| Two-hop expansion | CSR | O(sum of d over frontier) | O(frontier) | dominated by the largest degree, not the mean |
| k-hop traversal | CSR plus visited set | O(nodes and positions touched) | O(visited) | the visited set is the memory bound |
| As-of filter, indexed | any | O(log m) per range | O(1) | requires an index the CSR does not have |
| As-of filter, unindexed | any | O(m) | O(1) | 1e9 predicates at 1e8/s = 10 s per request |
| Sort by degree | any | O(n log n) | O(n) | a heap gives O(n log k) for the top k |
| Projection emission | any | O(sum C(k_g,2)) | O(pairs) | 1.745e9 pairs here, 174.5 s at 1e7/s |
| Rebuild snapshot | CSR | O(n + m) | O(n + m) | 21.4 GB at n = 1e8, m = 1e9 |
| Triangle count | CSR, node iterator | O(m^1.5) bound | O(m) | 3.16e13 operations at m = 1e9 as a ceiling |
| Transpose to CSC | CSR | O(n + m) | O(n + m) | one extra copy of the position array |
| Degree-sum assertion | any | O(m) | O(1) | 10 s at m = 1e9, 1e8/s |
| Offset monotonicity | CSR | O(n) | O(1) | 1 s at n = 1e8, 1e8/s |

Three readings of the table. First, **as-of filtering is a full-graph operation when it is unindexed**, so
a design that filters at query time on a 1e9-fact table pays 1e9 predicate evaluations per request; the
correct pattern is a materialised snapshot with the filter applied once. Second, **CSR ingest is
append-only in the favourable case and a rebuild otherwise**, because inserting a position for node 3
shifts every offset after it; a mutable graph needs adjacency lists or a delta array merged periodically.
Third, **projection cost is set by the largest group**, so a capacity estimate must use the maximum degree
rather than the mean, and the mean is the statistic that makes pipelines look affordable before they time
out.

```text
worked build comparison at n = 1e8, m = 1e9, L2 layout (21 B per position)

CSR build
  allocate      4*(1e8+1) + 1e9*21        = 0.4 GB + 21.0 GB = 21.4 GB
  scan + write  1e9 predicate evaluations + 1e9 stores
                at 1e8 ops/s             = 10 s + 10 s     = 20 s
  write out     21.4 GB at 2 GB/s         = 10.7 s
  total                                    ~ 31 s
  second copy for immutability            + 21.4 GB

row-store build
  scan          1e9 rows * 176 B          = 176 GB
                at 2 GB/s                 = 88 s
  index build   1e9 * log2(1e9) = 2.99e10 comparisons
                at 1e8 comparisons/s      = 299 s
  total                                    ~ 387 s
  ratio to CSR build                       387 / 31 = 12.5x

projection at the same scale
  hub contribution      C(50,000,2)       = 1.249975e9
  other contributions   1e5 * C(100,2)    = 0.495e9
  total P                                  = 1.745e9 pairs = 1.745x m
  emission at 1e7 pairs/s                  = 174.5 s
  materialised at 8 B per pair             = 13.96 GB
```

The comparison's useful part is the decomposition: the index build is 299 of the row store's 387 seconds,
and the scan is 88. A design review that budgets only storage will approve a pipeline whose cost is 77%
index maintenance, and a review that budgets only the scan will predict a factor of 3.5 where the truth is
12.5. The projection's 174.5 seconds is larger than the entire CSR build, which is the arithmetic behind
the claim that a projection, not a traversal, is usually the dominant cost in a nightly graph job.

Memory pressure has a second-order effect that belongs here because it changes asymptotics in practice. A
CSR that does not fit in RAM is not merely slower: each expansion becomes a random read per position, so an
in-memory O(d_v) contiguous scan at 10 ns per access becomes O(d_v) random accesses at 100 ns, a 10x
constant. The design consequence is that the snapshot should be sized to the serving host's memory, and if
it is not, the correct move is to partition the graph rather than to accept thrash.

### 9.1 Where the cost lands at three scales

| Scale | n | m | CSR bytes | Dominant cost | Design consequence |
|---|---|---|---|---|---|
| Prototype | 5e3 | 1e4 | 230 KB | developer time | use whatever is expressible |
| Single host analytic | 2e6 | 1e7 | 218 MB | build time, intermediates | one process, in memory |
| Single host at limit | 1e7 | 1e8 | 2.14 GB | random access on hub expansion | hub policy becomes mandatory |
| Multi-host | 1e8 | 1e9 | 21.4 GB | shuffle and partition boundary | partition by tenant or component |
| Beyond memory | 1e9 | 1e10 | 214 GB | paging or distributed latency | aggregate first, traverse second |

The table's usefulness is in the fourth row, where the cost stops being arithmetic and becomes a
distribution question. Up to about 1e8 facts a single host works because the working set fits, and past
that the same code degrades by roughly a factor of ten per expansion because every offset read misses. The
remedy is structural — partition, or reduce the working set by aggregation — rather than procedural.

The first two rows deserve a warning of their own. At prototype scale the difference between the best and
the worst layout is a few hundred milliseconds, which is why benchmark-driven layout choices are so often
made at the wrong scale and why they survive review: the measurement was correct and the population was
not.

### 9.2 Amortisation of a rebuild

A nightly rebuild of a 21.4 GB structure costs the same every night regardless of what changed. If facts
accumulate at 1e7 per day into a 1e9 structure, the rebuild processes 1e9 facts to absorb 1e7, a ratio of
100:1 wasted work.

```text
full rebuild     1e9 predicate evaluations + 1e9 stores per night   = 20 s of CPU
incremental      1e7 evaluations + a merge of the delta arrays     = 0.2 s of CPU
ratio            100x less work, at the price of:
                 (a) a merge that doubles peak memory while it runs;
                 (b) a correctness dependency on append-only facts;
                 (c) a snapshot identity that records which delta was applied.
```

Condition (b) decides the design in practice. If corrections arrive as new versions, the delta is
append-only and the incremental path is sound; if corrections overwrite, the delta cannot express the
change and the rebuild must be full. That is a second, independent argument for versioned facts beyond
reproducibility, and it is the argument that usually wins a design review because it is a correctness
argument rather than a performance one.

### 9.3 Cost of the assertions

The integrity checks are cheap relative to the build and should never be skipped for cost. The degree
identity costs one pass over the position array, 10 s at m = 1e9. Offset monotonicity costs 1 s at
n = 1e8. The relation registry coverage costs a distinct count, which is O(m) without an index and O(1)
with one. Together they add under 1% to a build and localise defects that would otherwise surface as a
metric anomaly weeks later, at which point the fact table has moved on and the defect is unreproducible.

The expensive check is the pair-level audit of resolved identity, which costs a lookup per sampled pair and
is the one that deserves a budget line. Sampling 1,000 resolved pairs at 10 ms per lookup is 10 s, which is
negligible; auditing all pairs is a manual review and is infeasible. The sampling rate is therefore the
design variable, and a rate derived from the acceptable miss probability is more defensible than a round
number, because it can be defended when the sample misses.

**Takeaway:** Budget the index build, the projection emission and the snapshot retention; those three
dominate the real cost of a nightly graph job, and none of them is visible in a storage-only estimate.

## 10. Key Theorems / Results Reference

**Handshake lemma.** For a finite multigraph, the sum of occurrence degrees equals twice the number of
occurrences, with loops contributing 2 each. *Consequence:* degree sum is a checkable per-snapshot
invariant, and a mismatch localises immediately to a filter divergence or a deduplication. *Limit:* the
identity holds for any consistent filter, so it cannot detect a filter that was applied to both sides
wrongly — a snapshot and a degree column computed at the same wrong cut-off pass the assertion.

**Degree-pair inequality.** Occurrence degree is at least distinct-neighbour degree, with equality exactly
when no pair repeats at the node, and the surplus equals the sum over neighbours of (c − 1). *Consequence:*
the two definitions differ by a computable amount, so a feature can state which it uses and a reviewer can
bound the difference. *Limit:* the inequality is preserved by double-counted directions, so a symmetry
defect passes it; the check must compare the position sum against 2m and the distinct sum against 2 × pairs.

**Quadratic projection bound.** A 2-section over a group of size k contains C(k,2) pairs, and adding one
member adds exactly k. *Consequence:* projection cost is bounded by the square of the largest group, so hub
policy is a cost decision as well as a modelling one, and a pair-count jump identifies the group size.
*Limit:* the bound is on emitted pairs before deduplication; a pruned projection has no closed form because
the prune depends on the data.

**CSR storage identity.** Bytes equal 4(n+1) + m·w_pos with w_pos set by the attribute set. *Consequence:*
the ratio between layouts is a function of the column set, not a constant, and the offset term is
negligible until n approaches m. *Limit:* the identity excludes page overhead, fill factor, indexes and
compression, so it compares layouts and does not predict an engine's footprint.

**Information-theoretic floor for sparse structures.** The configuration of m positions among n rows has
log₂ C(m+n−1, n−1) bits of content, which is 0.0604 GB at n = 1e8, m = 1e9, against 21.4 GB for the CSR.
*Consequence:* explicit identifiers cost 354x the information content, which is why compressed variants
exist and why 21 B per position is not a floor. *Limit:* the floor ignores access cost, and an encoding
that reaches it cannot answer a one-hop query in O(d_v).

**Partition duplication identity.** A vertex partition into p parts stores 1 + 1/p copies per edge when one
direction is materialised and 2 − 1/p when both are, and requires m(1 − 1/p) boundary fetches per scan.
*Consequence:* the duplication and the boundary traffic move in opposite directions as p grows, so there is
an optimum set by whether the workload is compute-bound or network-bound. *Limit:* both figures are
expectations under a degree-independent hash; a partition key correlated with size makes the maximum part
several times the mean.

**Edge-cut replication.** A node of degree d appears on p(1 − (1 − 1/p)^d) hosts under a uniform edge cut,
which reaches 15.97 of 16 at d = 100. *Consequence:* edge-cut partitioning cannot be used for a graph with
hubs, because the hubs are replicated onto every host and updated on every host. *Limit:* the formula
assumes uniform assignment; a degree-aware assignment can hold a hub on one host at the price of a
lookup, which is a different architecture and a different cost.

**Bitemporal closure requirement.** Reproducible as-of queries require at most one visible version per
(fact identifier, belief instant), enforced by the write path. *Consequence:* the schema can be checked
exactly and cheaply, so a system can prove it is bitemporal rather than asserting it. *Limit:* enforcement
is a write-path property, so a bulk load that bypasses the write path breaks the invariant while the
schema remains correct.

**Reification cost identity.** A fact as a resource costs 2 + p statement rows for p properties, which is
1.59x a flattened row at p = 3 with a 56 B statement. *Consequence:* the migration cost is computable
before it is attempted, and the multiplier is linear in the property count. *Limit:* statement packing and
dictionary encoding reduce the physical multiplier below the logical one, sometimes below 1.

**Group encoding loss.** A star expansion preserves incidence and adds a non-domain node of degree s; a
2-section preserves pairwise adjacency and destroys group identity, inflating edges by C(s,2)/s. *Consequence:*
neither binary encoding answers both "are these two co-members" and "which group were they in", so the
choice follows the query. *Limit:* for s = 2 the two encodings coincide, which is why a test suite built
from pairwise fixtures cannot detect the loss.

**Diameter compression under projection.** A projection can only decrease hop distance between nodes of the
retained type, since every projected pair is adjacent by construction. *Consequence:* a "within two hops"
rule means different things on the base graph and the projection, so a hop feature must name its graph.
*Limit:* a pruned projection can remove a pair entirely, making the distance infinite rather than shorter,
which reverses the direction of the error for the discarded pairs.

**Monotonicity requirement for incremental maintenance.** Incremental snapshot maintenance is sound iff
every change is an append and every derived structure is monotone in the fact set. *Consequence:* counts
and sums can be maintained incrementally; percentiles and top-k cannot, because an appended fact can push a
value out of the top k. *Limit:* the condition is about the data lifecycle, so a system that mostly appends
and occasionally overwrites is unsound in exactly the cases the team remembers.

**Takeaway:** The usable results are those with a stated limit; each converts a design argument into an
assertion with a known failure boundary.

## 11. Interview / Exam Questions

**Q1. A graph has 9 occurrences over 6 nodes and the degree column sums to 17. What is the expected value and what is your first check?**
Answer: the expected sum is 18, so one incidence is missing or one fact was counted once. The first check is
whether any self-loop exists, because a loop-aware loader expects 2m + L and the convention changes the
target; then check whether the degree aggregate used a different as-of filter from the edge count, which is
the common cause. Reasoning: the identity is exact, so a mismatch is mechanical — a filter divergence, a
deduplication, or an outer join emitting a null degree. Follow-up: how do you localise it to one node?
Answer: recompute degrees from the fact table grouped by node and diff the two columns; the differing nodes
are the localisation. Red flag: "the graph is approximate". Scoring: 2 for the identity, 2 for the filter
hypothesis, 1 for the localisation.

**Q2. Compute the distinct-neighbour degree of M1 in the toy graph and explain why it is 2 rather than 5.**
Answer: M1 has five occurrences (E1–E5) but two distinct neighbours (A1, A2), so d_dist = 2 and the surplus
is 3, equal to (3−1) + (2−1) from the two pairs. The surplus is real behaviour: three transfers from A1 and
two from A2. Reasoning: occurrence degree measures activity, distinct degree measures reach. Follow-up:
which belongs in the denominator of a propagation feature? Answer: distinct degree, because mass divides
among neighbours rather than among events, and using occurrence degree lets one busy counterparty absorb
the mass repeatedly. Red flag: treating the surplus as a data-quality problem. Scoring: 2 for the numbers,
2 for the mechanism, 1 for the normalisation choice.

**Q3. Estimate the CSR footprint at n = 1e8, m = 1e9 with int32 neighbours, float64 weights, uint8 relation codes and an int64 event time, and compare with a 176 B row store.**
Answer: offsets 4 × (1e8+1) = 0.4 GB; positions 1e9 × 21 B = 21.0 GB; total 21.4 GB. The row store is
176 GB, so the ratio is 8.22x. Reasoning: the ratio is a property of the column set, not of the engine;
adding the availability stamp takes the position to 29 B, the total to 29.4 GB and the ratio to 5.99x.
Follow-up: when does the row store win? Answer: when the dominant query filters by type and time before
traversal, because the index prunes the scan and the byte overhead is paid only on survivors. Red flag:
quoting a ratio without the column list. Scoring: 2 for the arithmetic, 2 for the ratio, 1 for the
condition.

**Q4. A projection over devices yields 1.25e9 pairs from one hub of degree 50,000. Name two repairs and the loss each introduces.**
Answer: (a) exclude hubs above a degree threshold, which loses genuine co-occurrence for every account on
that device and biases against populations served by shared hardware; (b) sample pairs within the hub, which
makes the projection non-deterministic unless the sample is seeded and stored, breaking rebuild equality.
Reasoning: the pair count is C(k,2), so any repair either removes the hub or shrinks its clique, and the
marginal cost of one more member is exactly k = 50,000. Follow-up: what threshold and how defended?
Answer: one justified by the account-count distribution per device in the population — kiosk, corporate
pool, household — checked against an adjudicated pair sample, not a round number. Red flag: "raise the
memory". Scoring: 2 for two repairs, 2 for the losses, 1 for the threshold defence.

**Q5. Why does a fact table need availability time when it already stores event time?**
Answer: because the decision rule is "what did we know at t", and only availability answers it. A snapshot
filtered on event time includes facts that had not landed, which inflates offline metrics and cannot be
reproduced in production. Reasoning: the two stamps coincide on no real pipeline and their difference is
heavy-tailed, so any cut-off inside the tail is affected; the toy graph's lags run 29 s to 74 s, which is
enough to change a snapshot at the right instant. Follow-up: give a discriminating test. Answer: restrict
to facts with lag above 60 s and compare feature values at a cut-off inside that window under both filters;
a difference proves the filter matters and quantifies the leak. Red flag: "we can reconstruct it from the
load log", which is a rotated source rather than a stored column. Scoring: 2 for the reason, 2 for the
test, 1 for the follow-up.

**Q6. You find four timestamp columns and are told the table is bitemporal. What single assertion do you run first?**
Answer: for every fact identifier and belief instant, assert at most one visible version — group by
(identifier, system start) and require a maximum count of one. Reasoning: four columns are a schema;
bitemporal semantics are a constraint on writes, so the assertion tests the property rather than the
declaration. Follow-up: what does a violation imply about historical snapshots? Answer: they double-count,
because superseded and correcting rows are both visible at instants when only one should be, which inflates
every aggregate over facts that were ever corrected. Red flag: accepting the column list as proof. Scoring:
2 for the assertion, 2 for the implication, 1 for the precision about visible versions.

**Q7. An edge-count feature drops 12% overnight with no behaviour change. Give two causes and the counts that separate them.**
Answer: (a) a late batch moved the snapshot boundary, so m fell from 9 to 8; (b) a deduplication was added
upstream, collapsing multiplicity so the row count fell while the pair count did not. Check: compare m and
distinct pairs for the same snapshot identifier. If m changed, the boundary moved; if m is unchanged but
pairs fell, deduplication was introduced. Reasoning: the two causes move different counts, so one query
distinguishes them. Follow-up: what makes the check inconclusive? Answer: both happening at once, in which
case the effects can cancel; the fallback is to diff the fact identifier sets directly. Red flag:
attributing the drop to the model. Scoring: 2 for the causes, 2 for the counts, 1 for the cancellation
caveat.

**Q8. When is a dense adjacency matrix correct, and where is its boundary?**
Answer: correct for spectral work at small n, where the algorithm needs the full matrix. The boundary is
where n² × 8 B exceeds the affordable working set: n = 1e4 gives 800 MB, n = 1e6 gives 8 TB. Reasoning:
memory grows quadratically while facts grow linearly, so the ratio rises as n²/(n+m) — 1,739x at n = 1e4
against a CSR of 460 KB. Follow-up: at n = 1e4 with 2e4 facts, what does a bitset version change? Answer:
8 B per cell becomes 1 bit, an 8x reduction in n-boundary terms, because the boundary scales as sqrt(bytes
per cell). Red flag: "dense is faster" without n. Scoring: 2 for the use case, 2 for the boundary numbers,
1 for the ratio.

**Q9. How can overmerge and supernode collapse produce the same symptom, and how do you separate them?**
Answer: overmerge unions two nodes' neighbourhoods, raising degree for both by roughly the sum of two
moderate degrees; a sentinel hub gives one node thousands of facts, which then leaks into every neighbour's
second hop. Both raise apparent connectivity in aggregates. Separation: inspect the degree distribution —
a sentinel produces one node orders of magnitude above the 99.9th percentile, while overmerge shifts a
handful of nodes up by a moderate amount and leaves the maximum plausible. Reasoning: the shapes differ
even though the means move together. Follow-up: confirm the hub by sampling source payloads, where a
constant string proves the sentinel. Red flag: lowering the resolution threshold to "clean up". Scoring: 2
for the mechanisms, 2 for the shape distinction, 1 for confirmation.

**Q10. Adding one account to a shared device raises the projection by exactly 50,000 pairs. What do you conclude, and what is the wrong arithmetic?**
Answer: the device had degree 50,000 before the addition, because C(k+1,2) − C(k,2) = k, so the increment
equals the previous group size. The wrong arithmetic is 2k − 1, which would give k = 25,000.5 — a
non-integer, which is the tell. Reasoning: the marginal of a clique is its current size, so the increment
reads the group size directly and is a cheap attribution check. Follow-up: what if the projection fell by
24 pairs? Answer: a member was removed from a group of degree 25, since the decrement is k − 1. Red flag:
assuming the projection is linear in facts. Scoring: 3 for the identity, 2 for the wrong arithmetic, 1 for
the decrement case.

**Q11. What is lost when a joint account with four holders is encoded as a clique, and what is gained?**
Answer: group identity is lost — the clique asserts every pair co-holds, and cannot say the four formed one
group, nor distinguish a pair that also co-holds elsewhere. Gained: pairwise adjacency, so pair-level
models run unchanged. Reasoning: a 4-clique has 6 edges against 4 incidence edges, an inflation of 1.5x,
and at s = 100 the inflation is 49.5x, so the choice is also a cost decision. Follow-up: when choose the
star instead? Answer: whenever "which group" is a query, at the cost of a node of degree 4 that is not a
person and that must be excluded from entity-level aggregates. Red flag: claiming the clique preserves the
group. Scoring: 2 for the loss, 2 for the inflation, 1 for the star condition.

**Q12. Your build takes 387 s of which 299 s is index maintenance. What change addresses it and what new cost appears?**
Answer: drop the secondary indexes on the analytic replica and build a CSR for the batch; the new cost is
that any per-fact filtered query becomes a full scan, so the replica serves the batch and not the point
queries. Reasoning: an index is paid per row at build time and redeemed per query, and a nightly analytic
workload does not redeem it — 299 of 387 seconds is 77% of the build spent on a structure the job never
uses. Follow-up: how serve the point queries? Answer: keep a small row-store serving table for recent
facts, bounded by the retention window rather than by total history. Red flag: removing indexes from the
transactional store. Scoring: 2 for the change, 2 for the new cost, 1 for the two-store pattern.

**Q13. A motif count of 400 three-cycles is offered as evidence of a coordinated ring. What is missing?**
Answer: a null model preserving the degree sequence. A graph with the same degrees produces many 3-cycles
without coordination, because a hub of degree k contributes on the order of k² potential triples, so 400 is
a count rather than a finding. Reasoning: cycle counts are driven by degree distribution and local
clustering, so without the null the number measures the degree sequence. Follow-up: what would a defensible
comparison be? Answer: configuration-model samples or a degree-preserving rewiring ensemble, reporting the
observed count against the ensemble's quantiles plus the share of cycles touching the top-degree node. Red
flag: "400 is a lot". Scoring: 2 for the null model, 2 for the mechanism, 1 for the ensemble.

**Q14. A nullable value column has four nulls of nine facts and five transfers summing to 1,250.00. Give the correct mean, the wrong one, and the detection.**
Answer: correct is 1,250 / 5 = 250.00 per transfer; the wrong version divides by all nine and gives 138.89,
a 44.4% understatement with an unchanged numerator. Detection: the denominator equals the total fact count
rather than the transfer count, so recomputing both exposes it. Reasoning: the wrong version mixes two
populations, so its value drifts with logging volume — a sighting outage lowers a transfer feature, which
is the signature. Follow-up: how would you catch it in a test? Answer: assert the denominator equals the
count of non-null facts in the relation, and add a fixture with a null-valued fact of another type. Red
flag: "zero is a reasonable default". Scoring: 2 for the correct mean, 2 for the arithmetic, 1 for the
detection.

**Q15. You must serve one-hop expansion at p99 under 50 ms over 1e8 facts. Which layout, and what does the requirement forbid?**
Answer: a materialised CSR in memory, partitioned to fit the host, with the as-of filter applied at build
time. It forbids query-time as-of filtering over the base table, which is a 1e8-predicate scan per request
— about 1 s at 1e8 predicates per second — and it forbids per-request rebuilds for the same reason at a
larger constant. Reasoning: the budget permits an O(d_v) contiguous scan and nothing that touches m.
Follow-up: what if the graph changes intraday? Answer: a delta array merged on a schedule, with the
staleness bounded and stated in the feature contract. Red flag: "we will cache it". Scoring: 2 for the
layout, 2 for the forbiddance with its cost, 1 for the staleness plan.

**Takeaway:** The questions that matter ask for a number or a discriminating check; an answer that
describes a concept without either is not an answer.

## 12. Quick Reference Card

| Element | Statement | Condition |
|---|---|---|
| Handshake check | Σ occurrence degree = 2m (+L if loops counted once) | same filter on both sides |
| Degree pair | d_occ ≥ d_dist; surplus = Σ(c−1) | per snapshot; direction stored once |
| CSR bytes | 4(n+1) + m·w_pos, w_pos = 13/21/29/45 B | fixed width, no compression |
| Layout ratio | 8.22x row vs 21 B position at 1e9 facts | same column set; falls to 3.66x at 45 B |
| Projection | C(k,2) pairs; marginal = k per added member | full clique; pruning breaks the form |
| Dense bound | n² × 8 B; 800 MB at 1e4, 8 TB at 1e6 | small n or spectral work only |
| Time rule | eligible iff t_avail ≤ t_decision | never substitute t_event |
| Bitemporal | ≤ 1 visible version per (fact, belief instant) | enforced by the write path |
| Partition | 1 + 1/p copies one-way, 2 − 1/p two-way | degree-independent hash |
| Reification | 2 + p rows per fact, 1.59x at p = 3 | logical count; packing reduces it |
| Rebuild rule | derived = f(base facts, identity version) | any hand-edited edge breaks it |
