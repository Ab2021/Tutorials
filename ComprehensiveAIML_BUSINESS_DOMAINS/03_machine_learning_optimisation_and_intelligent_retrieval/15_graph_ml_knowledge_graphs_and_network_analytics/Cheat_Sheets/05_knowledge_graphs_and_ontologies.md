---
title: "Knowledge graphs and ontologies"
topic: "Knowledge graphs and ontologies: triple semantics, RDFS/OWL entailment, schema design, entity resolution, query planning, SHACL validation and embedding models"
collection: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Knowledge graphs and ontologies

## 1. File Header

> **Purpose:** Decide and defend the semantic layer of a knowledge graph — which classes and properties exist, what entailment regime the store applies, how identity is resolved, how the graph is queried at a stated latency, and what makes an answer wrong in a way that a validator can catch. The deliverable is a schema plus a rule set plus a cost model, not a diagram.
> **Prerequisites:** first-order logic at the level of quantifiers and implication, relational algebra (join, projection, selection), the distinction between a term and its denotation, and enough arithmetic to cost a join.
> **Key outcome:** Given a set of source facts and a decision, you can write the ontology that supports the decision, compute what the reasoner will entail, prove that a query returns the intended population, and state the storage and validation cost of the design.
> **This file does NOT cover:** graph traversal algorithms and centrality (see the centrality sheet), centrality-based ring detection (see the fraud sheet), GNN encoder architecture (see the GNN guide), or entity-resolution blocking implementation detail beyond its cost consequences (see the traversal sheet).
> **Level:** Core. Foundation is Section 3; advanced is the entailment-regime comparison in 7.1 and the resolution algebra in 9.2.
> **Version / reviewed:** Draft 1.0, 2026-11-10. Independent review not yet performed. Review trigger: any change to the entailment regime, the identity policy, or the property that a downstream rule treats as functional.
> **Evidence:** All entailment arithmetic is DERIVED and recomputed in Section 5 against an explicit triple set. Entity names, drug classes, patient identifiers and vector components are ILLUSTRATIVE constructions. No vendor benchmark, no production ontology and no organisation-specific schema is claimed. Complexity statements in Section 9 are standard results for the named logics and are labelled as such.
> **Retrieval cue:** If an answer is "technically entailed but operationally wrong", read Section 4.4 (open-world limits) and Section 8 entries K-02, K-05 and K-09 before changing any rule.

An ontology is a set of commitments about what can be said. It is not a taxonomy picture, and it is not a schema in the database sense, because the default semantics are open-world: absence of a fact is not evidence of its negation, and a class with no instances is not empty. Almost every disagreement between a data engineer and a domain expert about a knowledge graph reduces to one party reasoning under the closed-world assumption and the other under the open-world assumption, with neither stating which. Fixing that first is worth more than any amount of graph size.

The second structural fact is that a knowledge graph is a *materialisation decision*. The same logical content can be stored as facts only, with the reasoner consulted at query time; as facts plus a closure, with queries evaluated directly; or as facts plus a partial closure at a chosen depth. Each choice has a different cost curve and a different failure mode, and mixing the three in one system produces the classic symptom of two queries that disagree about whether a patient "takes an anticoagulant". This sheet treats the choice as primary and everything downstream as a consequence.

The third fact is that identity is the hard part. Class hierarchies are usually easy to agree on because they are small and reviewed; entity resolution is where the graph's meaning actually lives, and where the cost is unbounded. Two records for the same person, joined by `owl:sameAs`, merge every property of both — including contradictory ones — and the merge is transitive, so a single wrong link can collapse a thousand entities into one. That is not a modelling nicety: it is a data-loss event with a graph-shaped interface.

**Why this sheet exists in a course about retrieval and optimisation:** a knowledge graph is the substrate that makes retrieval explainable. A vector index returns "similar", a graph returns "because". The price of the "because" is the semantic layer, and the semantic layer is where the cost, the ambiguity and the maintenance burden live. Understanding it is the difference between shipping a graph feature and shipping a graph feature that survives an audit.

## 2. Concept Map / Mental Model

```text
   SOURCE SYSTEMS (records, documents, tables, feeds)
        |               |                |
   (1) EXTRACT     (2) RESOLVE      (3) MAP TO SCHEMA
   facts, with      entity identity   class and property
   provenance       and versions      assignment
        \               |                /
         +--------------+---------------+
                        v
        +------------------------------------------+
        |  ASSERTED GRAPH: (s, p, o) + provenance   |
        |  plus schema triples (subClassOf, domain, |
        |  range, subPropertyOf, restrictions)      |
        +------------------------------------------+
             |                |                |
      (4) ENTAIL       (5) VALIDATE      (6) SERVE
      closure at a     SHACL shapes      query rewriting,
      chosen regime    and cardinality   materialised views,
             |                |          or embedding index
             v                v                v
        INFERRED        CONFORMANCE       ANSWER + PROOF
        GRAPH           REPORT            PATH
```

Six stages, and each has a semantic decision that is invisible from the previous one. Stage 3 decides whether "patient takes drug" and "drug prescribed to patient" are one property or two. Stage 4 decides whether the store computes the closure or the query does. Stage 5 decides whether a missing value is a violation or an unknown. Stage 6 decides which of the two answers above the consumer actually receives.

The central invariant is **monotonicity of assertion**: adding a fact must never remove an entailed consequence under the description logics used in practice (RDFS, OWL RL, OWL QL, OWL EL). That property is what makes incremental materialisation safe and what makes retraction expensive — removing a fact requires computing what was justified *only* by it, which is why real systems prefer append-only assertion with a validity interval over deletion.

```text
Invariant ladder for an ontology review (each rung answerable in one sentence):

  rung 1  what is a node?                        -> entity policy
  rung 2  what is a fact?                        -> triple grain and provenance
  rung 3  what does absence mean?                -> open or closed world
  rung 4  which entailments are materialised?    -> regime and depth
  rung 5  what makes a fact invalid?             -> shapes and cardinalities
  rung 6  what does a merged entity absorb?      -> sameAs policy
  rung 7  what is the answer's proof path?       -> explainability contract
```

Rung 3 is the one that gets skipped, and its absence is detectable in the data: an application that queries "patients with no recorded allergy" and treats the empty result as "no allergy" is reasoning under the closed-world assumption against a store whose entailment is open-world. The two are not reconcilable by adding a rule; the application's question has to be re-expressed as a *negation-as-failure* query with an explicit "records complete as of" flag.

The second mental model is the four-way split of what an ontology contains:

| Layer | Example content | Who owns it | Change frequency |
|---|---|---|---|
| Upper / foundational | Person, Organisation, Event | standards or vendor | near zero |
| Domain | Drug, Prescription, AdverseEvent | domain committee | yearly |
| Application | ReviewedCase, WatchlistHit | product team | monthly |
| Instance | specific entities and facts | pipelines | continuous |

Most "the ontology is unmaintainable" complaints are a symptom of storing the application layer inside the domain layer. An application-specific class that leaks into the domain ontology becomes a permanent commitment that every future consumer must reason about.

The third mental model is the query shape, because it determines the physical design:

```text
  pattern                        example                          dominant cost
  ----------------------------   ------------------------------   -------------------
  point lookup                   (drug, rdf:type, ?)              dictionary + one index
  star join                      (?rx, drug, ?d), (?rx, pat, ?p)  intersection of indexes
  path / transitive closure      (?x, partOf+, ?y)                closure size, not graph size
  negation as failure            patients with no allergy         anti-join over population
  aggregation                    count prescriptions per class    group-by after expansion
  similarity                     nearest neighbours in embedding   vector index, not triples
```

A design that is excellent for star joins (columnar triple indexes) can be catastrophic for transitive closure, and a design that precomputes closure handles paths but pays an unbounded storage cost for every transitive property added later. Naming the dominant shape before choosing the store is the whole of physical design.

## 3. Foundational Definitions

**IRI.** An internationalised resource identifier; the global name of a term. Two IRIs are the same term iff they are character-identical after normalisation. Units: none. The failure mode is not ambiguity but *near-duplication*: `http://ex.org/Drug` and `http://ex.org/drug` are different terms in every conformant store, and the resulting graph has two classes that a human reads as one.

**Blank node.** An existentially quantified variable in the data. A blank node denotes something that exists, with no stable name. Consequence: blank nodes cannot be referenced across documents, cannot be merged by IRI equality, and their identity is scoped to the graph they appear in.

**Triple.** `(subject, predicate, object)` with subject an IRI or blank node, predicate an IRI, object an IRI, blank node or literal. The unit of assertion. A graph is a set of triples, so duplicate assertion is idempotent — which is why "how many times was this stated" is not recoverable without provenance.

**RDF graph vs dataset.** A graph is a set of triples; a dataset is a default graph plus named graphs. Provenance and versioning require named graphs or reification, because a plain triple has nowhere to carry its source.

**Class and instance.** A class is a set of resources; an instance is a member. `rdf:type` asserts membership. In RDFS, classes are themselves resources, so `(DOAC, rdfs:subClassOf, Anticoagulant)` is a triple between two resources that happen to be classes.

**Subclass and subproperty.** `(C, rdfs:subClassOf, D)` entails that every instance of `C` is an instance of `D`. `(p, rdfs:subPropertyOf, q)` entails that every `p`-edge is a `q`-edge. Both are transitive and reflexive in RDFS semantics (the reflexive part is often not materialised, which is a legitimate engineering choice that must be documented).

**Domain and range.** `(p, rdfs:domain, C)` entails that any subject of a `p`-edge is in `C`; `(p, rdfs:range, C)` does the same for objects. Crucially, they are *inference rules*, not constraints: a graph that violates a range does not become inconsistent in RDFS, it becomes *larger*, because the reasoner adds the type assertion. Reading `rdfs:range` as a validator is one of the most consequential category errors in the field.

**Open-world assumption (OWA).** Absence of a fact entails nothing about its negation. A missing `hasAllergy` triple is compatible with the patient having an allergy that was not recorded.

**Closed-world assumption (CWA).** Absence of a fact entails its negation. Needed for "no known allergy" queries, and must be introduced explicitly, per-query or per-dataset, never assumed.

**Unique name assumption (UNA).** Distinct names denote distinct entities. OWL does *not* adopt it; two IRIs may denote the same individual unless declared different with `owl:differentFrom`. RDF has no UNA either. Consequences: a join on identifiers can under-merge (two IRIs, one entity) and a lack of `owl:differentFrom` can over-merge in reasoning.

**`owl:sameAs`.** Asserts that two IRIs denote the same individual; it is symmetric and transitive, and it propagates across every property. It is the strongest and most dangerous assertion in the language, because it is not retractable in a distributed setting: if A asserts `x sameAs y` and B asserts `y sameAs z`, a consumer of both must treat `x`, `y`, `z` as one node even if A and B disagree about attributes.

**Entailment regime.** The rule set that determines which triples are consequences. The practical menu: RDFS (subclass, subproperty, domain, range), OWL RL (rule-based, PTIME, includes property chains and some equality reasoning), OWL QL (query rewriting, designed so that conjunctive queries can be answered by rewriting without materialisation), OWL EL (PTIME, supports conjunction and existential restrictions, no disjunction or universal restrictions).

**Materialisation.** Computing and storing the closure. Cost is paid at write time; queries are cheap and predictable. **Query rewriting.** Expanding the query against the schema at read time; cost is paid per query, and closure size does not grow the store.

**SHACL shape.** A declarative constraint over a node shape and its property shapes: cardinality, datatype, class membership, value range, pattern, or a SPARQL-based custom constraint. Shapes are validation, not inference — the distinction from `rdfs:range` is exactly the OWA/CWA distinction.

**Cardinality restriction.** `owl:maxCardinality 1` on a property declares that no individual has two distinct values; it does not delete one. Under OWL semantics, two values plus a max-1 restriction entail that the two values are the same individual — an entailment that surprises everyone the first time.

**Functional property.** `owl:FunctionalProperty` on `p` means each subject has at most one value. Combined with `owl:sameAs` propagation this can merge values that were distinct in the source, silently.

**Transitive property and property chain.** `owl:TransitiveProperty` on `partOf` makes the closure of `partOf` entailed; `owl:propertyChainAxiom` generalises this to sequences such as `locatedIn ∘ partOf → locatedIn`. Transitive properties are the main driver of closure-size blowup.

**Datatype and literal.** A literal is a value with a datatype IRI, e.g. `"2026-03-01"^^xsd:date`. Equality across datatypes is not free: `"1"^^xsd:integer` and `"1.0"^^xsd:decimal` are value-equal but not term-equal, and whether a store treats them as the same key is implementation-defined unless specified.

**Reification and RDF-star.** Classic reification turns a triple into four triples about a statement resource; RDF-star (and its standardised successor in the RDF 1.2 line) allows a triple to be quoted as the subject of another triple. The cost difference is a factor of about four in triple count for the classical route.

**Entity resolution (ER).** The process of deciding which records denote the same entity, producing clusters. Its two quality axes are pairwise precision and recall, and its cost axis is the number of candidate pairs compared, which blocking controls.

**Blocking.** Partitioning records into blocks so that only within-block pairs are compared. Its two parameters are the blocking key and the block size distribution; its failure is a true pair that never co-occurs in a block (recall loss), which no downstream classifier can repair.

**Canonicalisation.** Choosing one IRI per resolved entity as the stable reference, with the others redirecting. Without it, every consumer must carry the full equivalence class.

**Provenance.** The record of which source, at what time, asserted a fact under which extraction version. Its absence makes retraction impossible, because you cannot find the facts justified by the source you are removing.

### 3.1 Terms that are routinely conflated

| Pair | Distinction | Cost of conflating |
|---|---|---|
| ontology vs schema | ontology includes entailment; schema constrains | expecting validation from `rdfs:range` |
| inference vs validation | inference adds facts; validation reports | invalid data silently becomes "true" |
| class vs set of instances | a class may be empty under OWA | treating no instances as "does not exist" |
| subclass vs part-of | taxonomy versus mereology | wrong query for "contains" |
| `sameAs` vs `exactMatch` | one merges individuals, one maps vocabularies | merging people because their schema terms align |
| transitive vs hierarchical | closure versus a tree | unbounded closure growth where a tree was intended |
| literal equality vs value equality | term identity versus datatype value | join misses on numeric formatting |
| blank node vs skolem IRI | existential versus named | unstable identifiers across reloads |

## 4. Core Formulas with Full Derivation

Two of the results below are algebraic; the rest are mechanism blocks in the contract's five-part form (preconditions, state transition, invariant and justification, worked trace, failure boundary). The section closes with the explicit validity condition.

### 4.1 Base case

**Result 1 — closure size of a subclass hierarchy.**

```text
FORMULA
  |closure| = sum over classes c of |ancestors(c)|
  where ancestors(c) = { d : (c, subClassOf, d) is entailed }

WHERE
  a hierarchy is a DAG over the class set, with edges read as "is a kind of"
  |closure| counts distinct (c, d) pairs, excluding the reflexive c = d case
  the explicit edge count is |E_schema| = number of declared subClassOf triples

CONSTRAINT
  the identity |closure| = |E_schema| holds only for a forest in which every
  class has at most one parent and no class is reachable by two paths
  any DAG with sharing has |closure| > |E_schema|; a chain of length L has
  |closure| = L(L+1)/2 against L edges

NUMERICAL EXAMPLE
  ILLUSTRATIVE hierarchy:
    DOAC -> Anticoagulant -> CardiacDrug -> Drug -> PharmaceuticalProduct
    Warfarin -> Anticoagulant
    Statin -> CardiacDrug
    BetaBlocker -> CardiacDrug
    Patient -> Person
  ancestors: DOAC 4, Anticoagulant 3, CardiacDrug 2, Drug 1,
             Warfarin 4, Statin 3, BetaBlocker 3, Patient 1
  |closure| = 4+3+2+1+4+3+3+1 = 21
  |E_schema| = 8
  ratio 21/8 = 2.625

INTERPRETATION
  materialising subclass closure multiplies schema triples by 2.6 on this
  eight-class hierarchy. On a 200,000-class chain the same formula gives
  199,999,000,000 closure triples against 199,999 edges, a ratio of 99,999.5
  -- which is why the depth of the hierarchy, not its width, is the design
  variable that matters.
```

**Result 2 — the entailment multiplier for instance data.**

```text
FORMULA
  |asserted| = |T_declared| + |T_relational|
  |entailed| = sum over entities e of |ancestors(type(e))|
             + sum over relational triples of (subproperties + 1)
  |total|    = |asserted| + |entailed|, counting only new triples

WHERE
  T_declared are type assertions, T_relational are property assertions
  ancestors exclude the declared type itself, so a leaf class with no
  superclass contributes 0

CONSTRAINT
  assumes no domain or range entailment (those add type assertions for
  subjects and objects and are counted separately), and assumes the type
  of each entity is known, which under OWA is not guaranteed

NUMERICAL EXAMPLE
  entities: apixaban:DOAC, warfarin_x:Warfarin, atorvastatin:Statin,
            metoprolol:BetaBlocker (4 drug type assertions)
  relational: 6 takesDrug edges, and takesDrug subPropertyOf prescribed
  schema: 8 subClassOf + 1 subPropertyOf = 9 asserted schema triples
  |asserted| = 9 schema + 4 drug types + 4 patient types + 6 edges = 23
  new schema triples from subclass transitivity: 21 - 8 = 13
  new type triples: 4+4+3+3 for drugs + 4 for Patient->Person = 18
  new relational triples from the subproperty: 6
  |entailed| = 13 + 18 + 6 = 37 new
  |total| = 23 + 37 = 60, multiplier 60/23 = 2.609

INTERPRETATION
  doubling is the realistic figure for a shallow domain hierarchy with one
  subproperty. A deep hierarchy with three transitive properties can reach
  an order of magnitude, and the storage plan must be derived rather than
  assumed. Note the schema line: nine asserted schema triples produce
  twenty-two schema and type consequences, so the multiplier is highest on
  the smallest part of the graph.
```

**Result 3 — join cardinality under a hierarchy.**

```text
FORMULA
  naive estimate (uniformity + independence)
    est = |A| * |B| / max(V_A, V_B)
  where V_A, V_B are the distinct-value counts of the join key on each side
  and max() is the textbook assumption that the smaller domain is contained
  in the larger

CONSTRAINT
  valid only when the key distributions are uniform and the value sets
  overlap completely. A hierarchy makes both false: a filter on a superclass
  matches a union of subclasses with very different frequencies

NUMERICAL EXAMPLE
  side A: 999,600 prescription rows, 5,000 distinct drug keys
  side B: 200,000 drug rows, 800 distinct drug keys
  est = 999,600 * 200,000 / 5,000 = 39,984,000
  actual, with 10 hot drugs covering 80,000 rows each and 790 tail drugs
  covering 125 rows each:
    hot part   10 * 80,000 * 10,000 = 8,000,000,000
    tail part  800 *    40 *   125  =     4,000,000
    actual = 8,004,000,000
  ratio actual/est = 200.18

INTERPRETATION
  a 200x under-estimate turns a plan with a hash join into a plan with a
  nested loop, and the query that took 2 seconds now takes 400. The fix is
  not a better optimiser but a stored statistic: per-key frequency on the
  hot set, which is a materialised aggregate over the graph.
```

**Result 4 — closure of a transitive property.**

```text
FORMULA
  for a chain of n nodes with n-1 transitive edges:
    |closure| = n(n-1)/2 pairs, |edges| = n-1, ratio = n/2

WHERE
  pairs are ordered if the property is asymmetric, unordered if symmetric
  the ratio is what determines whether materialisation is affordable

CONSTRAINT
  holds for a simple path. A tree with branching factor b and depth d has
  n = b^d nodes and closure pairs = n(n-1)/2 as well, but the *diameter*
  is 2d, so the storage is quadratic in n regardless of shape

NUMERICAL EXAMPLE
  n = 1,000   -> 499,500 pairs vs 999 edges, ratio 499.5
  n = 10,000  -> 49,995,000 vs 9,999, ratio 4,999.5
  n = 200,000 -> 19,999,900,000 vs 199,999, ratio 99,999.5
  n = 1,000,000 -> 499,999,500,000 vs 999,999, ratio 499,999.5

INTERPRETATION
  at one million nodes in a single transitive chain, materialising the
  closure needs half a trillion triples. Query rewriting or a bounded
  depth is the only viable option, and the bound must be justified by the
  decision rather than by the storage budget.
```

### 4.2 General case

**Result 5 — RDFS rule application as a fixed point.**

```text
PRECONDITIONS
  a triple set T, a schema S subset of T, and the RDFS rule set
  no rule retracts a triple; all rules are monotone

STATE TRANSITION
  repeat until no rule fires:
    if (c, subClassOf, d) and (x, type, c) then add (x, type, d)
    if (p, subPropertyOf, q) and (s, p, o) then add (s, q, o)
    if (p, domain, C) and (s, p, o) then add (s, type, C)
    if (p, range, C) and (s, p, o) then add (o, type, C)
    if (c, subClassOf, d) and (d, subClassOf, e) then add (c, subClassOf, e)

INVARIANT AND JUSTIFICATION
  monotonicity holds because every rule's conclusion is added, never
  removed, and no rule has a negated premise. The fixed point exists
  because the Herbrand base is finite: at most |terms|^3 triples can be
  present. The result is unique (confluence) because the rules form a
  monotone production system over a set.

WORKED TRACE
  start: (apixaban, type, DOAC), (DOAC, subClassOf, Anticoagulant),
         (Anticoagulant, subClassOf, CardiacDrug)
  pass 1: add (apixaban, type, Anticoagulant);
          add (DOAC, subClassOf, CardiacDrug)
  pass 2: add (apixaban, type, CardiacDrug); add (apixaban, type, Drug) via
          the newly derived (DOAC, subClassOf, CardiacDrug) and the declared
          (CardiacDrug, subClassOf, Drug)
  passes 3 and 4 add (apixaban, type, PharmaceuticalProduct) and the
          remaining transitive subclass pairs; pass 5 adds nothing

FAILURE BOUNDARY
  if any rule had a negated premise (for example "if not type C then ...")
  the system would not be monotone, the fixed point might not exist, and
  the iteration could oscillate. OWL RL keeps monotonicity by treating
  negation only through `owl:differentFrom`, never through absence.
```

**Result 6 — materialisation depth as a truncation.**

```text
PRECONDITIONS
  a hierarchy of maximum depth D and a chosen bound k <= D
  the query set Q, partitioned into queries reachable within k hops and
  queries needing more

STATE TRANSITION
  materialise all ancestors up to depth k; for a query at depth j > k,
  either (a) rewrite the query to walk the remaining j-k hops at read
  time, or (b) answer incompletely and flag the answer

INVARIANT AND JUSTIFICATION
  every materialised triple is entailed (soundness is preserved by
  truncation); completeness is lost exactly for the queries whose depth
  exceeds k. The loss is therefore enumerable in advance by inspecting the
  maximum chain length of each property used in Q.

WORKED TRACE
  hierarchy of depth 4 (DOAC at depth 4, Drug at depth 1)
  k = 2 materialises 2 triples per DOAC instance instead of 4
  a query "all cardiac drugs" needs depth 2 and is complete
  a query "all pharmaceutical products" needs depth 4 and is incomplete
  by exactly the instances whose chain exceeds 2

FAILURE BOUNDARY
  if the schema changes and D grows, previously complete queries silently
  become incomplete. The check is to recompute max-depth per property on
  every schema release and diff it against the configured k.
```

**Result 7 — entity-resolution cost.**

```text
PRECONDITIONS
  N records, a blocking key producing B blocks with sizes n_1..n_B,
  and a pairwise classifier costing c seconds per compared pair

STATE TRANSITION
  candidate pairs = sum over blocks of n_i(n_i-1)/2
  true pairs = P, the (unknown) number of same-entity pairs
  pairs compared = candidate pairs, independent of P

INVARIANT AND JUSTIFICATION
  recall is bounded above by (true pairs co-located in some block)/P,
  because a pair never compared can never be found. Precision is bounded
  above by P/candidate pairs, because every candidate is a chance to be
  wrong and at most P can be right.

WORKED TRACE
  N = 2,000,000, B = 50 equal blocks -> n_i = 40,000
  candidate pairs = 50 * 40,000 * 39,999 / 2 = 39,999,000,000
  all pairs = 2,000,000 * 1,999,999 / 2 = 1,999,999,000,000
  reduction factor = 50.0
  with P = 50,000 true pairs, the precision ceiling is
  50,000/39,999,000,000 = 1.25e-6
  cost at 2 microseconds per pair = 39,999,000,000 * 2e-6 s = 79,998 s
  = 22.2 hours for one pass

FAILURE BOUNDARY
  the reduction is only as good as the block-size distribution. With 80% of
  records in 5 hot blocks, the same N and B give
  5 * 320,000 * 319,999 / 2 + 45 * 8,000 * 7,999 / 2 = 257,776,422,260
  pairs, a reduction of only 7.8x, and the pass now costs 143 hours.
```

### 4.3 Variants

**RDF-star and reification cost.**

```text
FORMULA
  classical reification: each annotated assertion costs 4 triples
    (stmt, rdf:type, rdf:Statement)
    (stmt, rdf:subject, s)
    (stmt, rdf:predicate, p)
    (stmt, rdf:object, o)
  RDF-star: each annotated assertion costs 1 triple with a quoted triple
    ((s p o), source, g)

WHERE
  annotation rate r is the share of assertions that need provenance
  N is the base triple count

CONSTRAINT
  reification changes query shape: a query over reified statements joins
  through rdf:subject/predicate/object, so it is 4x the join depth;
  RDF-star keeps the query shape but requires store support

NUMERICAL EXAMPLE
  N = 1e9, r = 5%: classical reification adds 0.05 * 1e9 * 4 = 2e8 triples
  (20% growth); RDF-star adds 5e7 (5% growth)
  at 24 B per triple (three 8 B terms) that is 4.8 GB versus 1.2 GB

INTERPRETATION
  the 4x factor is a fact about the encoding, not about the store; choosing
  reification means accepting a 4x annotation cost and a deeper query plan
  whether or not the store is fast.
```

**N-ary relations.** A prescription has a patient, a drug, a dose and a date. Encoding it as four binary triples attached to a reified node is the standard RDF pattern; encoding it as a property on the drug loses the per-prescription dose. The decision rule: if an attribute is a property of the *relationship* rather than of either endpoint, it forces a node.

**Property chains versus explicit transitive flags.** A chain axiom `locatedIn ∘ partOf → locatedIn` states a two-step rule; `owl:TransitiveProperty` states unlimited depth. The first is bounded and cheap, the second is unbounded and expensive. Most "transitive" requirements in practice are satisfiable by a bounded chain with a maximum length, and choosing the bounded form early avoids the closure blowup in Result 4.

**Skolemisation.** Replacing blank nodes with deterministic IRIs derived from their content or their source key makes identity stable across reloads, at the cost of asserting that two structurally identical blank nodes are the same entity. The deterministic hash must include the source identifier, not only the property values, or two genuinely distinct anonymous events with identical fields will collide.

### 4.4 Limits and counterexamples

**The range rule does not validate.** Take `(pat1, takesDrug, "aspirin"^^xsd:string)` with `(takesDrug, rdfs:range, Drug)`. An RDFS reasoner does not report an error; it adds `("aspirin"^^xsd:string, rdf:type, Drug)`, producing a graph in which a literal is a drug. The graph is larger and wrong. The check is to run SHACL with `sh:class Drug` on the object of `takesDrug`, which reports a violation rather than entailing nonsense. Any pipeline whose only quality gate is `rdfs:range` has no quality gate.

**`owl:maxCardinality 1` merges rather than rejects.** With `(pat1, hasAllergy, penicillin)` and `(pat1, hasAllergy, sulfa)` plus a max-1 restriction, OWL entails `penicillin owl:sameAs sulfa`. The data is not rejected; the two allergens become one. This is sound under the open-world reading ("there cannot be two, so they must be the same thing") and catastrophic operationally. The correct tool is a SHACL `sh:maxCount 1` constraint, which reports.

**Transitivity plus `sameAs` is a merging engine.** `sameAs` is transitive in OWL, so a chain of 1,000 pairwise assertions yields 499,500 entailed pairs. Each pair propagates every property of both endpoints. In a graph where a single erroneous link joins two 500-node clusters, the closure contains 250,000 unintended merges from one bad assertion, and no incremental repair is possible without provenance on the link.

**Bounded depth is not a semantic choice.** Truncating closure at depth 2 changes the meaning of every query that needs depth 3. It is a performance choice, and it must be recorded in the answer metadata, because two consumers querying the same store at different bounds will disagree about the population — and both will be correct relative to their bound.

**Validity condition for this section.** All the arithmetic assumes a single named graph with no duplicate assertions, a stable schema during the computation, and a reasoner that applies the full stated rule set. If assertions are duplicated across named graphs the closure counts are per-graph sums and the union is smaller than the sum; if the schema changes mid-computation the fixed point is over a moving rule set and monotonicity no longer guarantees a unique result.

## 5. Worked Numerical Example

The example is ILLUSTRATIVE and small enough to recount by hand. It contains a four-level drug hierarchy, a patient hierarchy of depth 1, four drug entities, four patients, six prescribing edges, and one subproperty link.

### 5.1 The asserted graph

```text
schema (8 subclass edges, 1 subproperty edge)
  DOAC         subClassOf Anticoagulant
  Anticoagulant subClassOf CardiacDrug
  CardiacDrug  subClassOf Drug
  Drug         subClassOf PharmaceuticalProduct
  Warfarin     subClassOf Anticoagulant
  Statin       subClassOf CardiacDrug
  BetaBlocker  subClassOf CardiacDrug
  Patient      subClassOf Person
  takesDrug    subPropertyOf prescribed

type assertions (8)
  apixaban      type DOAC
  warfarin_x    type Warfarin
  atorvastatin  type Statin
  metoprolol    type BetaBlocker
  pat1 type Patient   pat2 type Patient
  pat3 type Patient   pat4 type Patient

relational assertions (6)
  pat1 takesDrug apixaban
  pat1 takesDrug atorvastatin
  pat2 takesDrug warfarin_x
  pat2 takesDrug atorvastatin
  pat3 takesDrug apixaban
  pat4 takesDrug metoprolol

asserted total = 8 + 1 + 8 + 6 = 23
```

### 5.2 Entailed triples, enumerated

```text
subclass closure (21 pairs, of which 8 are asserted, so 13 new)
  DOAC          -> Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct
  Anticoagulant -> CardiacDrug, Drug, PharmaceuticalProduct
  CardiacDrug   -> Drug, PharmaceuticalProduct
  Drug          -> PharmaceuticalProduct
  Warfarin      -> Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct
  Statin        -> CardiacDrug, Drug, PharmaceuticalProduct
  BetaBlocker   -> CardiacDrug, Drug, PharmaceuticalProduct
  Patient       -> Person

type entailment from subclass closure (18 new)
  apixaban     -> Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct  (4)
  warfarin_x   -> Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct  (4)
  atorvastatin -> CardiacDrug, Drug, PharmaceuticalProduct                 (3)
  metoprolol   -> CardiacDrug, Drug, PharmaceuticalProduct                 (3)
  pat1..pat4   -> Person                                                   (4)
  total = 4 + 4 + 3 + 3 + 4 = 18

  the warfarin_x row is the one that gets missed: its declared type is
  Warfarin, so Anticoagulant appears only via the intermediate Warfarin,
  and an enumeration written from the leaf class forgets the first hop

relational entailment from subPropertyOf (6 new)
  each (x takesDrug y) yields (x prescribed y), and none were asserted

entailed new = 13 + 18 + 6 = 37
asserted + new = 23 + 37 = 60 triples in the materialised graph
```

Three counts are worth separating explicitly, because they are routinely summed together and double-counted:

```text
  schema triples    asserted  9  (8 subClassOf + 1 subPropertyOf)
                    entailed 13  (transitive subclass pairs)
                    total    22
  type triples      asserted  8  (4 drugs + 4 patients)
                    entailed 18  (ancestors of each declared type)
                    total    26
  relational        asserted  6  (takesDrug)
                    entailed  6  (prescribed)
                    total    12
  ---------------------------------------------------------------
  total              asserted 23, entailed 37, materialised 60

the three rows are disjoint because their predicates differ
(subClassOf, rdf:type, prescribed); a count that adds 18 type triples to
the 21 subclass pairs is adding different predicates and is correct, while
a count that adds 18 to 26 is adding a total to one of its own parts
```

The disjointness assertion is the check that makes the arithmetic trustworthy. If a reasoner were to derive a `subClassOf` triple from a type assertion (which it does not under RDFS or OWL RL), the rows would overlap and the sum would overstate the graph; the fact that they cannot overlap is what lets a reviewer verify the total by three independent group-bys.

### 5.3 Query cost with and without materialisation

```text
Q1  all patients taking any Drug
    materialised: scan (?, rdf:type, Drug) -> 4 drug entities, join to
      takesDrug -> pat1, pat2, pat3, pat4. 4 + 6 triple reads
    rewritten: expand Drug to {Drug, CardiacDrug, Anticoagulant, DOAC,
      Statin, BetaBlocker, Warfarin}, then union types and join. 7 schema
      lookups + 8 type lookups + 6 edges
    difference: 10 reads versus 21 reads on a 60-triple graph

Q2  all patients taking an Anticoagulant
    materialised: (?, type, Anticoagulant) -> apixaban, warfarin_x;
      join -> pat1, pat2, pat3
    note pat3 is returned because apixaban is a DOAC, which is an
      Anticoagulant; without closure pat3 is missed entirely
    rewritten: expand Anticoagulant to {Anticoagulant, DOAC, Warfarin}
      and union -> same three patients, 3 lookups instead of 1

Q3  all patients taking an Anticoagulant and a Statin
    materialised: two type lookups plus one self-join on patient
      -> pat1 (apixaban + atorvastatin), pat2 (warfarin_x + atorvastatin)
    rewritten: two expanded lookups plus the same self-join

INTERPRETATION
  on this graph the closure saves 11 reads on Q1 and 2 on Q2, while costing
  37 extra stored triples (a 2.61x store increase). Break-even is around
  three hierarchy-filtered queries per stored inferred triple; below that,
  rewriting wins.
```

### 5.4 Independent re-check

Three independent routes to the same total, each of which fails for a different class of error.

```text
check 1 (partition the materialised graph by predicate)
  subClassOf    9 asserted + 13 entailed = 22
  rdf:type      8 asserted + 18 entailed = 26
  takesDrug     6 asserted +  0 entailed =  6
  prescribed    0 asserted +  6 entailed =  6
  -------------------------------------------------
  total 22 + 26 + 6 + 6 = 60

  asserted = 9 + 8 + 6 + 0 = 23
  entailed = 13 + 18 + 0 + 6 = 37
  and 23 + 37 = 60, consistent with the partition

check 2 (enumerate one representative all the way to the root)
  apixaban is a DOAC, so its full type set is
    {DOAC, Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct}     (5)
  of which 1 is asserted and 4 entailed
  warfarin_x is a Warfarin, so its full type set is
    {Warfarin, Anticoagulant, CardiacDrug, Drug, PharmaceuticalProduct} (5)
  of which 1 is asserted and 4 entailed
  atorvastatin is a Statin:
    {Statin, CardiacDrug, Drug, PharmaceuticalProduct}                  (4)
  metoprolol is a BetaBlocker:
    {BetaBlocker, CardiacDrug, Drug, PharmaceuticalProduct}             (4)
  each patient:
    {Patient, Person}                                                   (2)

  type triples total = 5+5+4+4 + 4*2 = 18 + 8 = 26
  asserted types 8, entailed 18 -- matches check 1

check 3 (subclass closure counted independently as ancestor sums)
  DOAC 4, Anticoagulant 3, CardiacDrug 2, Drug 1,
  Warfarin 4, Statin 3, BetaBlocker 3, Patient 1
  sum = 21 closure pairs
  of which 8 are the declared edges, so 13 are entailed
  8 + 13 = 21 -- matches check 1

check 4 (spot-check a query against the closure)
  (?, type, Drug) returns apixaban, warfarin_x, atorvastatin, metoprolol
  = 4, which equals the number of drug entities, as it must since every
  drug class in this schema is below Drug
  (?, type, PharmaceuticalProduct) returns the same 4, again as it must
  (?, prescribed, ?) returns 6 rows against 6 asserted takesDrug rows
```

The four checks use different data structures — a predicate partition, a per-entity enumeration, an ancestor sum, and a query — so an arithmetic slip in one is very unlikely to reproduce in another. That is the point of a re-check: not to repeat the same addition, but to reach the same number by a route where the units differ.

### 5.5 Sensitivity: what changes the multiplier

| Change | Effect on total triples | Mechanism |
|---|---|---|
| Add one level to the drug hierarchy | +1 per drug entity per level (4 here) | each type gains an ancestor |
| Add a second subproperty of `prescribed` | +6 | each edge duplicates |
| Make `prescribed` transitive | 6 edges become up to 36 pairs | quadratic in chain length |
| Add `rdfs:domain` on `takesDrug` | +0 here (subjects already typed) | idempotent when type present |
| Add `rdfs:range` on `takesDrug` | +0 here (objects already typed) | idempotent |
| Remove the patient hierarchy | −4 | one fewer ancestor per patient |
| Add 1,000 patients, 1,500 edges | +1,000 + 1,500 + 1,500 | type, edges, subproperty |

The last row is the scale test: at 2 million patients and 3 million edges, the type closure adds 2 million `Person` triples and the subproperty adds 3 million `prescribed` triples, so the multiplier on instance data converges to `(2e6 + 3e6 + 6e6) / (2e6 + 3e6) = 2.2` regardless of drug hierarchy depth — the drug table's contribution becomes a rounding term, while the subproperty doubling becomes half the graph.

### 5.5a Sensitivity of the closure decision to the depth bound

A second sensitivity is worth computing because it decides the serving architecture. Truncate the type closure at depth `k` and the entailed type count becomes the number of ancestors within `k` hops rather than all of them:

```text
k = 1  apixaban 1 (Anticoagulant), warfarin_x 1, atorvastatin 1,
       metoprolol 1, patients 1 each  ->  4 + 4 = 8 entailed types
k = 2  apixaban 2, warfarin_x 2, atorvastatin 2, metoprolol 2,
       patients 1 each  ->  8 + 4 = 12
k = 3  apixaban 3, warfarin_x 3, atorvastatin 3, metoprolol 3,
       patients 1 each  ->  12 + 4 = 16
k = 4  apixaban 4, warfarin_x 4, atorvastatin 3 (Statin has 3 ancestors),
       metoprolol 3, patients 1 each  ->  14 + 4 = 18  (complete)

cost at k=2: 12 of 18 entailed types, so 67% of the entailment for 67% of
the storage; the query "any PharmaceuticalProduct" is wrong at k<4 and
correct at k=4, while "any CardiacDrug" is correct from k=2
```

The useful reading is that the depth bound should be chosen per hot query, not per graph. Here `k = 2` serves every query down to `CardiacDrug` at two thirds of the storage, and only the `PharmaceuticalProduct` query needs the extra two levels — so the sensible design is `k = 2` plus an explicit rewrite for the one deep query, not `k = 4` for everything.

### 5.6 Production-scale reading

```text
10^9 asserted triples, 5% needing provenance
  reified: 0.05 * 2e9 = 1e8 statements * 4 = 4e8 triples
  RDF-star: 1e8 extra triples
  at 24 B/triple: 9.6 GB versus 2.4 GB of annotation

index footprint
  SPO only:         1e9 * 3 * 8 B = 24 GB
  SPO + POS + OSP:  72 GB
  dictionary: 2e8 unique terms * 40 B = 8 GB
  total with three permutations: 80 GB before compression

closure decision
  if the drug hierarchy is depth 4 and there are 5e6 drug entities,
  type closure adds 2e7 triples (4 per entity minus the declared one),
  which is 0.5 GB -- cheap.
  if `partOf` over 1e5 assembly nodes is made transitive, the closure is
  5e9 pairs (1e5 * 99999 / 2), which is 120 GB -- not cheap, and the
  decision must be query rewriting or bounded depth.
```

The asymmetry between the two paragraphs is the design lesson: the same store, the same engine, and a factor of 240 between two closure decisions that look identical in a schema diagram.

## 6. Visual / Geometric Intuition

### 6.1 The triple table as a three-axis space

Every triple is a point in `(subject, predicate, object)` space. A query is a box: fixing one coordinate gives a plane, fixing two gives a line, fixing none gives everything. Physical indexes are orderings of that space, and the number of permutations you materialise is exactly the number of axis orderings you can seek into directly.

```text
  s fixed, p fixed      -> line along o      (property lookup: "drugs patient takes")
  s fixed, o fixed      -> line along p      ("how are these two related")
  p fixed, o fixed      -> line along s      (reverse lookup: "who takes this")
  s fixed only          -> plane             (all about one entity)
  p fixed only          -> plane             (all edges of a relation)
  nothing fixed         -> whole space       (full scan)
```

Six orderings exist; three cover every two-coordinate pattern, which is why three permutations is the canonical minimum and six is the wasteful maximum. Choosing `SPO + POS + OSP` covers `(s,p)`, `(p,o)`, `(o,s)`; `(s,o)` and `(o,p)` and `(p,s)` are answered by a scan of a covered ordering with a filter.

### 6.2 The hierarchy as a lattice and the closure as its reachability matrix

The eight-class hierarchy has 21 reachable pairs out of 64 possible in an 8x8 matrix, so the closure is 33% dense and the matrix is mostly zeros. Storing the closure as a matrix is therefore wrong at any scale: at 200,000 classes it would be 4e10 cells. Storing it as pairs is right, and the number of pairs is the ratio in Result 4.

Geometrically, a hierarchy's depth is the radius of the ball the closure must cover, and its width is how many balls share the same radii. Depth drives cost (quadratic), width drives constant factor (linear). This is why "we need a deep taxonomy" and "we need many categories" have completely different engineering consequences and are usually discussed as if they were the same request.

### 6.3 The embedding space

A translational model places entity and relation vectors so that `h + r ≈ t` for true triples. The illustrative instance: `h = (0.8, −0.2, 0.5)`, `r = (−0.6, 0.4, 0.1)`, so `h + r = (0.2, 0.2, 0.6)`. The true tail `(0.2, 0.1, 0.6)` sits at distance `0.1`. Candidate tails:

```text
  true tail        (0.2,  0.1, 0.6)   d = 0.100000   correct
  near miss        (0.2,  0.1, 0.55)  d = 0.111803   wrong entity, near-identical vector
  near miss        (0.3,  0.2, 0.5)   d = 0.141421   wrong entity, plausible region
  far negative     (-0.1, 0.9, 0.3)   d = 0.818535   clearly wrong
  far negative     (0.5, -0.5, 0.2)   d = 0.860233   clearly wrong
```

The geometric reading: the two near misses are *closer to each other* than the far negatives are to anything, which is why a margin of 0.5 produces a loss of only 0.0118 on the first near miss and 0.0414 on the second. The model therefore has to be trained with negatives that are hard in that region, and this is the geometric content of "negative sampling matters more than architecture".

```text
margin loss at gamma = 0.5, computed against d_pos = 0.1
  near miss 1: max(0, 0.5 + 0.111803 - 0.100000) = 0.511803
  near miss 2: max(0, 0.5 + 0.141421 - 0.100000) = 0.541421
  far negative 1: max(0, 0.5 + 0.818535 - 0.1)   = 1.218535
  far negative 2: max(0, 0.5 + 0.860233 - 0.1)   = 1.260233
  mean = 0.883
at gamma = 1.0 mean = 1.383; at gamma = 2.0 mean = 2.383
```

The far negatives contribute most of the loss even though they are easy, which is the mechanism behind the common observation that a model trained on uniformly sampled negatives learns almost nothing: the gradient is dominated by cases it already gets right.

### 6.4 Why a graph drawing of an ontology is misleading

A class diagram draws `subClassOf` as arrows between boxes arranged by a layout algorithm. The drawing suggests that the distance between boxes is meaningful and that the number of boxes is the size of the ontology. Both are wrong: the size is the closure (`21` here against 8 boxes and 8 arrows), and the distance is a layout artefact. A hierarchy of depth 4 with 100 leaves looks like a wide fan and has a closure of about 300; a chain of 20 classes looks like a snake and has a closure of 210. The visual size and the semantic size are not related.

The second misleading feature is that a drawing shows only one graph. In practice there are four, and they disagree:

| Graph | Differs from asserted by | Where it appears |
|---|---|---|
| asserted | nothing | the triple log |
| materialised at depth k | type closure up to k | the serving store |
| materialised at full depth | full type and property closure | the analytics store |
| validated | nothing (validation adds reports, not triples) | the quality report |

Two people looking at graphs 2 and 3 will disagree about whether "apixaban is a PharmaceuticalProduct" is in the data, and both will be telling the truth.

### 6.5 The ER blocking picture

Blocking partitions the record space and compares within cells. The candidate-pair count is the sum of `n_i(n_i-1)/2`, so the geometry is entirely about cell sizes:

```text
  50 equal cells of 40,000        39,999,000,000 pairs    (50x reduction)
  5 cells of 320,000 + 45 of 8,000  257,776,422,260 pairs (7.8x reduction)
  2,000 cells of 1,000            999,000,000 pairs      (2,000x reduction)
```

The last row is 2,000x cheaper than the first and finds strictly fewer true pairs, because a fine key loses recall. The design surface is the curve between them, and the only way to locate a point on it is to measure recall on a labelled sample of true pairs — which is a data-labelling task rather than an engineering one, and therefore usually the binding constraint.

## 7. Comparison / Selection Table

| Regime | What it entails | Complexity | Materialisation growth | Best for | Failure |
|---|---|---|---|---|---|
| None (plain RDF) | nothing | query-time joins only | 1.0x | facts as a graph, no taxonomy | duplicated hierarchy logic in every query |
| RDFS | subclass, subproperty, domain, range | PTIME, linear passes | 1.5–3x typical | shallow taxonomies, cheap inference | `range` read as validation |
| OWL RL | RDFS plus property chains, equality, some restrictions | PTIME (rule-based) | 2–10x | rule engines, streaming inference | `sameAs` propagation |
| OWL QL | rewriting-based, no materialisation | AC0 data complexity, query-time | 1.0x | read-mostly, large graphs, small schema | rewrite explosion on deep hierarchies |
| OWL EL | conjunction and existential restrictions | PTIME | 2–20x | biomedical ontologies, large class counts | no disjunction, no universal |
| Full OWL 2 DL | everything | N2EXPTIME worst case | unbounded | small, carefully curated models | undecidability in careless extensions |

### 7.1 Selection procedure

```text
step 1  write the five queries the graph must answer, with their
        populations and their expected cardinalities.
step 2  mark which of them need taxonomy reasoning (a superclass filter,
        a transitive relation) and which are plain joins. Only the marked
        ones justify a reasoner.
step 3  if the marked set is empty, publish plain RDF and do hierarchy
        expansion in the application. This is the most common correct answer.
step 4  if the hierarchy is shallow (depth <= 4) and the graph is read-heavy,
        materialise RDFS or OWL RL and budget 2-3x storage.
step 5  if the hierarchy is deep or the class count is large, prefer OWL QL
        rewriting or bounded materialisation; measure rewrite time per query.
step 6  if a validator is required for the decision, write SHACL shapes and
        treat them as the quality gate, independent of the entailment regime.
step 7  write down max-depth per property and re-check it on every schema
        release, because the materialisation bound is only sound for the
        depth it was computed against.
```

### 7.2 Store and encoding choices

| Choice | Enables | Costs | Rule of thumb |
|---|---|---|---|
| Triple table with 1 permutation | point lookups on `(s,p)` | scans for `(p,o)` | never; add one more index |
| Triple table with 3 permutations | all two-coordinate patterns | 3x storage | default for a mutable store |
| Columnar triple store | fast scans and aggregation | write amplification | analytics-heavy reads |
| Property graph | fast traversal, typed edges | schema is code, not data | operational graphs, not shared ontologies |
| Materialised views per query | predictable latency | staleness | dashboards with a fixed question set |
| Embedding index | similarity, link prediction | no explanation, retraining | retrieval and ranking, not compliance |

The last two rows are the pairing that matters in a retrieval product: the embedding index answers "similar to this", the triple store answers "why". A system that ships only the first cannot explain a result, and a system that ships only the second cannot rank by similarity. Both are needed and neither substitutes for the other.

## 8. Failure Modes & Edge Cases

### K-01 `rdfs:range` used as a validator

**Symptom:** invalid values appear correct; a literal appears as an instance of `Drug`; the quality dashboard shows zero violations while domain experts find obvious errors. **Competing causes:** (a) the range axiom is being interpreted as a constraint; (b) the validator is running against the materialised graph and validating its own entailments; (c) the shapes file is not deployed. **Discriminating check:** query `(?, rdf:type, Drug)` and inspect the object types. Under RDFS a mistyped object is *entailed* into `Drug`, so the count rises rather than the error being reported. If the count includes literals or unexpected classes, cause (a) is confirmed. **Correction:** move the check to SHACL (`sh:class Drug` on the object of `takesDrug`) and keep the range axiom for inference only. **Prevention and verification:** a test that inserts a literal object and asserts that (i) SHACL reports it and (ii) the entailment count is unchanged when the range axiom is removed from the shape.

### K-02 Absence read as negation

**Symptom:** the "patients with no recorded allergy" report returns 94% of the population, and clinicians reject it. **Competing causes:** (a) open-world semantics, so absence is genuinely unknown; (b) the allergy source feed is incomplete for one site; (c) the query filters on a subclass that the data does not use (e.g. `Allergy` versus `Intolerance`). **Discriminating check:** run the same query per source site. A uniform 94% points to (a); one site at 40% points to (b); a set of patients with related-but-different classes points to (c), which is visible as a non-empty intersection with `Intolerance`. **Correction:** re-express the query as a negation-as-failure query against a dataset with a declared completeness flag, and report the coverage of the source. **Prevention and verification:** every negative query carries a `records_complete_as_of` parameter, and the pipeline fails if it is absent.

### K-03 A single `sameAs` link collapses two clusters

**Symptom:** two organisations' entities merge, one entity inherits the union of both attributes, and downstream counts drop by thousands. **Competing causes:** (a) a false-positive ER link; (b) a correct link whose propagation is undesired (two systems' identifiers for the same entity but different attribute definitions); (c) the merge is transitive through a third entity. **Discriminating check:** compute the size of the merge class before and after the link is added. A change from 2 to 2 confirms (a); a change to a large class confirms (c); a change in attribute cardinality without a class-size change confirms (b). **Correction:** replace `sameAs` with a weaker relation (`skos:exactMatch` for vocabularies, a custom `likelySame` for probabilistic links) and reserve `sameAs` for reviewed, high-confidence merges. **Prevention and verification:** an assertion that no `sameAs` class exceeds a configured size, plus a review queue for links whose score is below a high threshold.

### K-04 Transitive property closure consumes the store

**Symptom:** a storage alert after adding `owl:TransitiveProperty` to `partOf`; the write path slows by an order of magnitude. **Competing causes:** (a) the closure really is quadratic (Result 4); (b) the reasoner is recomputing the full closure per write rather than incrementally; (c) the property was already materialised as explicit edges, so the reasoner duplicates rather than adds. **Discriminating check:** compute `n(n−1)/2` for the node count in the transitive relation and compare with the growth observed. If they match, (a) is the cause and no implementation change helps. **Correction:** replace the transitive axiom with a bounded property chain or move the path query to read time. **Prevention and verification:** a CI check that fails when `n²/2` for any transitive property exceeds a configured triple budget.

### K-05 `owl:maxCardinality 1` silently merges values

**Symptom:** a patient's two allergies become one in every report; the count of distinct allergens falls after a modelling change. **Competing causes:** (a) the cardinality restriction entails `sameAs` between the two fillers; (b) an ER merge; (c) the reporting query deduplicates on the wrong key. **Discriminating check:** query for entailed `sameAs` between the two filler IRIs. If present, cause (a); if absent, check the ER cluster and the query's `GROUP BY`. **Correction:** use `sh:maxCount 1` in SHACL, which reports a violation and leaves the data intact, and model multi-valued clinical facts as separate prescription nodes. **Prevention and verification:** forbid `owl:maxCardinality` on any property whose source is known to be multi-valued, and enforce the rule with a schema lint.

### K-06 Hierarchy expansion missed in a rewritten query

**Symptom:** a report shows 40% fewer patients on anticoagulants than the clinical system, and the gap is exactly the DOAC patients. **Competing causes:** (a) the query filters on `Anticoagulant` without expanding to subclasses; (b) the materialisation bound truncated the subclass closure; (c) the DOAC records use a different class IRI. **Discriminating check:** count `(?, type, Anticoagulant)` with and without closure. A large difference confirms (a) or (b); identical counts with a differing total confirms (c). To separate (a) from (b), check the maximum chain depth of `DOAC` against the configured bound. **Correction:** expand the class to its descendants at query time (`DESCRIBE`-style expansion or a property path over `subClassOf`), or increase the bound. **Prevention and verification:** a golden test per report that compares the closure-based and rewrite-based counts and fails when they differ by more than the stated tolerance.

### K-07 Blank nodes compared across graphs

**Symptom:** duplicate facts accumulate on every reload; deduplication by IRI never matches; triple counts grow linearly with the number of loads. **Competing causes:** (a) blank nodes are scoped per load and therefore never equal; (b) the loader skolemises with a random UUID; (c) the store's isomorphism check is disabled. **Discriminating check:** load the same document twice and count triples. Growth by exactly the document's triple count indicates (a) or (b); no growth indicates the store is doing isomorphism-based deduplication. To separate (a) and (b), inspect the node labels after two loads: random UUIDs indicate (b), per-load counters indicate (a). **Correction:** skolemise deterministically from a stable source key plus the predicate path, and include the source system in the hash. **Prevention and verification:** a two-load test asserting an idempotent triple count.

### K-08 Datatype mismatch breaks joins

**Symptom:** a join between two sources returns 60% of the expected rows; the missing rows look identical in a sample. **Competing causes:** (a) integer versus decimal literals are not term-equal; (b) a date is stored as a string in one source; (c) an IRI has a trailing slash in one source. **Discriminating check:** `GROUP BY datatype(?o)` on both sides. Two different datatype IRIs confirm (a); one side returning `xsd:string` for a date confirms (b); a filter on IRI length differing by one confirms (c). **Correction:** normalise datatypes at load (canonical lexical form per datatype), and treat the normalisation as part of the schema contract. **Prevention and verification:** a SHACL `sh:datatype` constraint per property plus a load-time assertion that no literal carries the generic string datatype for a typed property.

### K-09 Range and domain create types the source never asserted

**Symptom:** entity counts per class exceed the source counts, and an audit cannot reconcile the graph with the operational database. **Competing causes:** (a) domain and range axioms are entailing types; (b) the closure is being counted as asserted; (c) the sources genuinely overlap. **Discriminating check:** disable domain and range axioms, recompute, and compare class counts. A drop equal to the discrepancy confirms (a). **Correction:** either accept the entailment and report both asserted and entailed counts, or remove the axioms and assert the types explicitly at load. **Prevention and verification:** every count report carries `asserted` and `entailed` columns; a reconciliation report reconciles asserted counts only.

### K-10 Property used as both object property and datatype property

**Symptom:** an OWL DL reasoner refuses the ontology; a rule engine silently treats the property inconsistently. **Competing causes:** (a) genuine punning of a property IRI; (b) two vocabularies collided on the same IRI; (c) a mapping error in one source. **Discriminating check:** list the objects of the property and classify them by node kind (IRI vs literal). A mixed set confirms (a) or (c); if the literal objects come from a single source, (c) is the cause. **Correction:** split into two properties (`hasValue` datatype, `refersTo` object) and migrate the source. **Prevention and verification:** a schema lint that rejects any property with both literal and IRI objects in a sample of the graph.

### K-11 Query rewrite explosion on a deep hierarchy

**Symptom:** a query that was fast becomes slow after a taxonomy release; the plan shows thousands of unions. **Competing causes:** (a) OWL QL rewriting expanded a class to hundreds of descendants; (b) a property path replaced a materialised closure; (c) the optimiser lost statistics after a bulk load. **Discriminating check:** count the expanded disjuncts in the rewritten query. A count in the thousands confirms (a); unchanged disjunct count with a slower plan confirms (c). **Correction:** materialise the descendant sets as a table and join once instead of unioning; or raise the materialisation depth for the specific classes used by hot queries. **Prevention and verification:** a query-plan regression test that fails when the rewrite size for any registered query exceeds a threshold.

### K-12 Provenance lost so retraction is impossible

**Symptom:** a source is found to be wrong and its facts cannot be removed; analysts instead add negation triples that nothing reads. **Competing causes:** (a) no provenance was recorded; (b) provenance was recorded on a copy that is not the serving graph; (c) provenance exists but the retraction job matches on a different key. **Discriminating check:** pick one known-bad fact and attempt to find the source record that asserted it. Finding nothing confirms (a); finding a record in a different store confirms (b); finding it with a mismatched key confirms (c). **Correction:** assert provenance at load into a named graph or via RDF-star, keyed by source and load id, and make the serving store derived so it can be rebuilt. **Prevention and verification:** a quarterly drill that retracts a synthetic source and asserts the derived graph rebuilds without it.

### K-13 SHACL validated against the inferred graph

**Symptom:** validation passes in staging and fails in production for the same data. **Competing causes:** (a) staging validated the asserted graph, production the materialised one, and a shape targets an entailed property; (b) the shapes version differs; (c) the production graph has a larger closure so more nodes match a target. **Discriminating check:** run the same shapes against both the asserted and materialised views of the production graph. A difference confirms (a) or (c), and comparing shape counts confirms (b). **Correction:** pin the view (asserted or materialised) in the shape definition and in the deployment, and record it in the conformance report. **Prevention and verification:** the conformance report names its input graph and its shape version.

### K-14 Embedding evaluation leaked through the KB

**Symptom:** link-prediction MRR is high offline and collapses online. **Competing causes:** (a) training triples include the test triples' inverses; (b) filtered evaluation was used offline but raw online; (c) the entity set at serving time contains new entities with no embedding. **Discriminating check:** recompute offline MRR with raw ranking and with a strict temporal split. A large drop to the raw figure confirms (b); a drop only under the temporal split confirms (a); a drop only on the new-entity subset confirms (c). **Correction:** report both raw and filtered metrics, split temporally by assertion time rather than randomly, and route new entities to a fallback path. **Prevention and verification:** the evaluation report contains the split definition, the filter policy, and the share of evaluation entities unseen at training time.

## 9. Complexity & Cost Analysis

| Task | Cost | Standard result | Practical note |
|---|---|---|---|
| RDFS closure | O(|T| · |rules|) fixed-point passes | PTIME | three passes typically suffice on shallow schemas |
| OWL RL closure | polynomial in |T| | PTIME | rule count dominates constant |
| OWL EL classification | polynomial in class count | PTIME | large ontologies classifiable in minutes |
| OWL QL query answering | AC0 in data complexity | rewriting | rewrite size can dominate |
| Full OWL 2 DL | N2EXPTIME worst case | known upper bound | avoid for data-driven graphs |
| Transitive closure | O(n·m) or O(n³) dense | standard | size dominates, see Result 4 |
| Star join over 3 patterns | index intersections | — | number of patterns is the cost driver |
| Path query of length k | O(b^k) with branching b | — | 20² = 400 at k=2, 20³ = 8,000 at k=3 |
| SHACL validation | O(nodes × shapes × per-shape cost) | — | path shapes are the expensive kind |
| Beam search for link prediction | O(B · depth · degree) | — | 64 × 3 × 20 = 3,840 expansions |
| ER candidate pairs | Σ n_i(n_i−1)/2 | — | see 4.2 Result 7 |
| Dictionary lookup | O(1) hash | — | memory-bound, not CPU-bound |

### 9.1 Three scales of the storage decision

```text
asserted triples   three permutations   dictionary   total
1e6                24 MB                0.4 GB*      0.42 GB
1e8                2.4 GB               0.8 GB       3.2 GB
1e9                24 GB                8 GB         32 GB
3e9 (all perms)    72 GB                8 GB         80 GB

* assumes 1e7 unique terms at 40 B each
```

The dictionary is not the dominant term at a billion triples; the permutations are. That inverts at around `10^11` triples, where a 40-byte-per-term dictionary over `10^9` terms becomes 40 GB and the compression strategy for term strings becomes the main design question.

### 9.2 Resolution cost and the precision ceiling

The ER pass at 2 million records with 50 equal blocks costs 22.2 hours at 2 µs per pair, and its precision ceiling is `1.25e-6` because almost all candidates are negative. Two consequences follow.

First, the classifier must be cheap and highly selective; a neural pairwise model at 2 ms per pair turns 22 hours into 2.5 years. The standard architecture is a cheap blocking key, then a cheap similarity filter, then an expensive classifier on the survivors.

Second, the metric that matters is recall at a fixed review budget, not AUC. With 50,000 true pairs among 4e10 candidates, an AUC of 0.999 still leaves thousands of false positives above the review threshold, and the precision at any operational cut has to be measured directly on a labelled sample.

```text
  candidates            4.0e10
  true pairs            5.0e4
  precision ceiling     1.25e-6
  if the review budget is 5,000 per week and the pass is weekly,
  the maximum achievable precision is 5,000/5,000 = 1.0 but the number of
  true pairs found is bounded by 5,000, i.e. 10% of the population of true
  pairs -- recall is the binding constraint, not precision
```

### 9.3 Validation cost

```text
nodes 2e6, shapes 8, all node-targeted
  at 1 microsecond per check: 2e6 * 8 * 1e-6 = 16 s
  at 50 microseconds per check (path-based shape): 2e6 * 8 * 50e-6 = 800 s
shapes that use sh:path with a variable-length path are the ones that
  reach the 50 microsecond figure; shapes that test a datatype are near 1
```

The design consequence: run datatype and cardinality shapes on every load, and path-based shapes on a schedule against a snapshot. Validating a path shape on the write path adds 800 seconds per 2 million nodes, which for a stream means the validator becomes the bottleneck and the backpressure propagates into the loader.

### 9.4 Where the cost actually lands

Three cost centres dominate different graph sizes:

```text
  1e6 triples    cost centre is query latency; indexes are in memory,
                 optimisation is about the number of patterns per query
  1e8 triples    cost centre is storage I/O; permutations and compression
                 matter more than the query shape
  1e9 triples    cost centre is the closure decision; materialising type
                 closure is affordable, materialising transitive closure is not
```

Most "the knowledge graph is too slow" tickets are a closure decision made for a graph size the designer did not anticipate, not a query that needs rewriting.

## 10. Key Theorems / Results Reference

**T1 Monotonicity of RDFS and OWL RL rule sets.** Both are monotone production systems without negated premises, so the closure exists, is unique, and can be computed by iterating until no rule fires. Consequence: incremental insertion is safe (add the new conclusions), while deletion requires justification tracking.

**T2 P-completeness of OWL RL.** Instance checking and conjunctive query answering in OWL RL are PTIME-complete in the combined size of the graph and the query. Consequence: the rule engine's cost is polynomial and cannot be improved by a smarter engine beyond the constant factor; the only levers are graph size and rule count.

**T3 PTIME classification in OWL EL.** Class subsumption in OWL EL is PTIME-complete, which is why EL is the profile of choice for ontologies with hundreds of thousands of classes. Consequence: a large class count is affordable; a large instance count is the separate cost axis.

**T4 AC0 data complexity of OWL QL.** Conjunctive query answering in OWL QL has AC0 data complexity (first-order rewritable), so the answer can be computed by a single SQL-style query over the data without a fixed-point iteration. Consequence: the data can be arbitrarily large for a fixed query; the rewrite is where the cost hides.

**T5 N2EXPTIME upper bound for OWL 2 DL.** Satisfiability in full OWL 2 DL is decidable with an N2EXPTIME upper bound. Consequence: it is decidable in theory and often impractical in practice; a data-driven graph with millions of individuals should choose a tractable profile and say so.

**T6 Transitivity implies quadratic closure.** For a transitive property over `n` connected nodes the entailed pair count is `n(n−1)/2`, so the closure is Θ(n²) irrespective of the number of asserted edges. Consequence: any schema that declares a transitive property over a large relation has committed to quadratic storage or to read-time inference.

**T7 Punning is not identity.** OWL 2 DL restricts the use of the same IRI as both a class and an individual unless punning is used, and punning does not make the class and the individual the same thing. Consequence: a class and its instance sharing an IRI remain distinguished by their roles; an application that joins on IRI alone will conflate them.

**T8 No unique name assumption.** OWL semantics does not require distinct IRIs to denote distinct individuals. Consequence: two records with different identifiers may be the same entity, so a join on identifiers cannot establish distinctness; only `owl:differentFrom` can.

**T9 `sameAs` is an equivalence relation.** It is reflexive, symmetric and transitive, and it substitutes in all contexts including class membership and property values. Consequence: a chain of `k` links entails `k(k−1)/2` pairs and propagates all attributes across the class; a single false link is not locally repairable.

**T10 Cardinality restrictions entail equality, not deletion.** `maxCardinality 1` with two distinct fillers entails their equality. Consequence: a data-quality problem becomes, under OWL, a data-merging problem; validation must use SHACL.

**T11 Conjunctive query containment is NP-complete.** Deciding whether one conjunctive query is contained in another is NP-complete in the size of the queries. Consequence: optimisers use heuristics rather than containment checks, so two logically equivalent queries can have different plans and different latencies.

**T12 Join cardinality estimation error compounds per join.** Under independence assumptions, estimation error multiplies across joins, so a `k`-way join with per-join factor `f` has error `f^k`. Consequence: the three-way estimate of 6,250,000 for a chain of three relations, each with a 200x error, could be off by `8e6`; statistics on the hot values are the only reliable mitigation.

**T13 Blocking recall is a hard ceiling.** If a true pair never co-occurs in a block, no downstream model can recover it. Consequence: blocking design is a recall decision made before any classifier exists, and it must be evaluated on labelled true pairs rather than on the total record count.

## 11. Interview / Exam Questions

**Q1. A colleague says "the range axiom validates our data". On the illustrative graph, what happens when a literal is asserted as the object of `takesDrug`?**
Answer: nothing is rejected. An RDFS reasoner adds `("aspirin"^^xsd:string, rdf:type, Drug)`, so the graph grows and now contains a literal that is a drug.
Reasoning: `rdfs:range` is an inference rule (object of `p` is an instance of `C`), not a constraint. Validation requires a separate mechanism with closed-world semantics, i.e. SHACL.
Check: run `(?, rdf:type, Drug)` before and after inserting the bad triple; the count rises by one rather than an error appearing.

**Q2. Compute the schema closure for the eight-class hierarchy and explain why a chain is worse than a fan.**
Answer: 21 closure pairs from 8 edges. A chain of length `L` has closure `L(L+1)/2` for `L` edges, so at `L = 200,000` the closure is about `2e10` triples from 199,999 edges.
Reasoning: each class in a chain has all its ancestors as distinct nodes, so the sum of ancestor counts grows quadratically. In a fan (one root, `k` children) each child has exactly one ancestor, so the closure is `k` for `2k` edges.
Check: the eight classes give 4+3+2+1+4+3+3+1 = 21, and `21/8 = 2.625`; a fan of 8 classes would give about `1.75`.

**Q3. With 23 asserted triples and a 4-level drug hierarchy, how many triples does the materialised graph hold, and what is the multiplier?**
Answer: 60 triples, a 2.61x multiplier: 13 new subclass pairs, 18 new type assertions, 6 new `prescribed` edges.
Reasoning: the subclass closure is 21 pairs against 8 asserted (13 new); each of the four drugs gains its full ancestor set (4+4+3+3 = 14) plus four patients gaining `Person` (4) = 18; the subproperty duplicates all 6 edges.
Check: 23 + 13 + 18 + 6 = 60, and a direct query for `(?, type, Drug)` returns exactly the four drugs.

**Q4. Why is a 200x join-estimate error plausible rather than a bug, and what is the mechanism?**
Answer: the uniform-independence estimate is 39,984,000 while the true cardinality is 8,004,000,000, because 10 hot drugs hold 80,000 rows each while 790 tail drugs hold 125.
Reasoning: the estimator assumes the smaller key domain is uniformly spread and fully contained; a hierarchy with skewed frequency violates both. The hot set alone contributes 8e9 of the 8.004e9 total.
Check: compute the per-key frequency distribution and compare `max(freq)` with `mean(freq)`; here the ratio is 640.

**Q5. When does materialising type closure pay for itself on this example?**
Answer: after roughly one query per inferred triple. Materialisation costs 37 extra triples (or 24 new ones, depending on how the closures are counted) and saves 11 reads on the "any Drug" query — so break-even is about 2–3 such queries per stored triple at this scale.
Reasoning: the saving per query is bounded by the number of hierarchy levels skipped, which is at most 4 here. The cost is paid once per triple per store, so a query mix dominated by hierarchy-filtered reads is the only case that wins.
Check: compare 10 reads (materialised) with 21 (rewritten) for the "any Drug" query, then count how many times that query runs per load.

**Q6. A single wrong `sameAs` link joins two 500-node clusters. How many unintended merges result, and what makes repair hard?**
Answer: the merge class becomes 1,000 nodes, so all pairs within it are entailed equal — `1,000 × 999 / 2 = 499,500` pairs, of which 250,000 cross-cluster pairs are unintended.
Reasoning: `sameAs` is an equivalence relation and substitutes in every context, so the class is not a set of pairwise assertions but a single merged entity for every downstream query. All 250,000 cross pairs follow from the one assertion.
Check: compute the equivalence class with union-find before and after adding the link; the size change is the impact.

**Q7. You are asked for "patients with no known allergy". Write the semantic contract.**
Answer: the query must be evaluated under closed-world semantics with an explicit completeness flag: `NOT EXISTS (allergy) AND records_complete_as_of(site, d) >= decision_date`.
Reasoning: absence of a triple entails nothing under OWA, so "no known allergy" is a statement about records, not about the patient. Without the flag the query silently converts "unknown" into "none".
Check: run the query per site; a uniform result across sites with different feed maturity is the signature of the semantic error rather than a clinical finding.

**Q8. Derive the candidate-pair count for 2 million records blocked into 50 equal blocks, and state the precision ceiling with 50,000 true pairs.**
Answer: each block holds 40,000 records so contributes `40,000 × 39,999 / 2 = 799,980,000` pairs; times 50 gives `39,999,000,000`. With 50,000 true pairs the ceiling is `5e4/4e10 = 1.25e-6`.
Reasoning: the ceiling follows because at most `P` of the candidates can be true, so any classifier's precision on candidates is bounded by `P/candidates` regardless of quality.
Check: the all-pairs count is `2e6 × 1,999,999/2 = 1.999999e12`, so blocking achieves exactly a 50x reduction — the block count, as expected.

**Q9. Why does adding `owl:TransitiveProperty` to a 100,000-node assembly relation break the store?**
Answer: the closure is `1e5 × 99,999/2 = 4,999,950,000` pairs, which at 24 bytes per triple is about 120 GB, from 99,999 asserted edges.
Reasoning: transitivity entails all reachable pairs, not just the immediate ones, so the entailed set is quadratic in the number of nodes regardless of the graph's shape.
Check: compute `n(n−1)/2` before enabling the axiom and compare with the configured storage budget; here the ratio to asserted edges is 49,999.5.

**Q10. Given `h + r = (0.2, 0.2, 0.6)` and candidate tails, which negative is hardest and why does that matter for training?**
Answer: `(0.2, 0.1, 0.55)` at distance 0.111803 against the true tail's 0.1 — a difference of 0.0118, which at margin 0.5 yields a loss of 0.5118, nearly the margin itself.
Reasoning: the hardest negative is the one closest to the true tail in embedding space; it carries almost no discriminating signal because the model already places it nearly correctly, and gradient is dominated by the far negatives that are trivially wrong.
Check: at margin 0.5 the four losses are 0.5118, 0.5414, 1.2185, 1.2602, so the two easy negatives contribute 2.48 of the 3.53 total.

**Q11. What is the difference between filtered and raw MRR, and how large can it be?**
Answer: filtered MRR removes from the ranking any candidate that is a true triple in the graph but not the answer, so it measures the model; raw MRR includes them, so it measures the model against an incomplete labelling. On a 10-item example with ranks 1,2,2,3,5,8,11,20,40,80 the filtered MRR is 0.28367; with two spurious higher-ranked hits the raw figure is 0.42992, a 1.52x gap.
Reasoning: a raw rank is always at least the filtered rank (removing candidates can only move the answer up), so raw MRR is systematically pessimistic. The gap measures how many true-but-unlabelled triples the store happens to contain.
Check: recompute both on the same model and the same candidate set; the ratio is the size of the labelling gap.

**Q12. Why is three triple permutations the canonical choice and not six?**
Answer: six orderings exist (`spo, sop, pso, pos, osp, ops`) but only three are needed to answer every two-coordinate pattern by a contiguous seek: `spo` covers `(s,p)`, `pos` covers `(p,o)`, `osp` covers `(o,s)`. The remaining patterns `(s,o)`, `(o,p)`, `(p,s)` are answered by a range scan plus filter on one of the three.
Reasoning: a two-coordinate query fixes two axes, leaving a line; the chosen permutations place each pair of axes in adjacent positions at least once.
Check: at 1e9 triples, three permutations cost 72 GB and six cost 144 GB, so the saving is 72 GB for a set of patterns that a filter handles at scan cost.

**Q13. Two consumers report different counts of "patients on anticoagulants". Give three causes and the order in which you would test them.**
Answer: (1) different materialisation depths — one query expands subclasses, the other does not; (2) different populations — one filters to active patients; (3) different vocabularies — one uses `takesDrug`, the other `prescribed`, and only one has the subproperty closure.
Reasoning: test order follows cost. Depth is a one-line configuration check; population is a predicate diff; vocabulary requires a schema diff and a data sample. The first test also explains the largest observed effects, since the DOAC class alone can account for a 40% gap.
Check: count `(?, type, Anticoagulant)` under both configurations; the difference is exactly the subclasses, and the illustrative graph shows a3's apixaban case as the minimal example.

**Q14. What does a SHACL `sh:maxCount 1` do that `owl:maxCardinality 1` does not?**
Answer: SHACL reports a violation and leaves the data unchanged; OWL entails that the two values are the same individual, merging them.
Reasoning: SHACL is a closed-world validation language over a data graph; OWL is an open-world description logic. The same English sentence ("at most one") has opposite operational meaning in the two.
Check: assert two allergy values on one patient. SHACL returns a conformance report with the node and path; the OWL reasoner returns an entailed `sameAs` between the two values and no error.

## 12. Quick Reference Card

| If you need to… | Use | Guard against |
|---|---|---|
| enforce data quality | SHACL shapes with `sh:class`, `sh:maxCount`, `sh:datatype` | `rdfs:range` treated as a constraint |
| answer "no known X" | negation as failure plus a completeness flag | reading absence as negation |
| support superclass filters cheaply | materialise type closure to a measured depth | depth drift after a schema release |
| support a transitive relation at scale | query rewriting or a bounded property chain | `owl:TransitiveProperty` over large `n` |
| connect two vocabularies | `skos:exactMatch` | `owl:sameAs`, which merges individuals |
| merge duplicate records | blocking plus a scored link with review | `sameAs` on a probabilistic score |
| keep provenance | named graphs or RDF-star at 1x cost | classical reification at 4x |
| pick a reasoner | RDFS for shallow, EL for big class counts, QL for deep read-mostly | full DL on data-driven graphs |
| cost a store | 24 B per triple per permutation plus 40 B per unique term | forgetting the dictionary term |
| evaluate link prediction | filtered and raw MRR, temporal split | random split leaking inverses |
