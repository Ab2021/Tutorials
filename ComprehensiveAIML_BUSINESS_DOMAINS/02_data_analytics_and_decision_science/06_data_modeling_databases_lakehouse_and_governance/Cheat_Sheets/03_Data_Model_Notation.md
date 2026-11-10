---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "03"
topic: "Data model notation"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A relationship, key, temporal rule, business definition, or notation convention changes."
notation_style: "plain-text"
prerequisites: ["01_Data_Contract_Checklist.md", "Module 02 relational keys and SQL joins"]
pattern_tags: ["entity-relationship", "cardinality", "temporal-modeling", "notation-selection"]
source_status: "OPEN"
owner_role: "Data architect"
source_index: "INDEX.md"
---

# Data Model Notation

> **Purpose:** Read and write a compact model that preserves entity identity, relationship cardinality, optionality, and temporal meaning well enough that an implementer can build the table and an auditor can prove the build matches the business sentence.
> **Prerequisites:** [01 Data contracts](01_Data_Contract_Checklist.md) for grain, identity, denominator, and change-control vocabulary; SQL `GROUP BY`, joins, and null handling; the distinction between an event time and a load time.
> **Key outcome:** Given a business sentence and a sample of rows, you can (a) decide which nouns are entities, (b) state natural identity and optionality for each, (c) choose one notation whose marks are enforceable, (d) write the temporal predicate in closed form, and (e) compute the join multiplicity the model implies before running the join.
> **This file does NOT cover:** vendor DDL dialects and physical index design (handoff to warehouse query patterns), grain and fan-out arithmetic at query level ([02 Grain and join safety](02_Data_Model_Grain_and_Join_Safety.md)), or stakeholder discovery interviews. Notation is the compression layer; it is not discovery.
> **Level:** 1 — Foundations, with a production-grade temporal and complexity section.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** Notation semantics and the ER/relational results in Sections 4 and 10 are DERIVED from standard definitions (Chen's entity-relationship model, Codd's relational model, the SQL:2011 temporal extension, and the Armstrong axioms). All counts, byte figures, and money amounts in Sections 5 and 9 are ILLUSTRATIVE teaching inputs whose arithmetic is VERIFIED by hand and by an independent recomputation. No vendor engine, product schema, company model, or production deployment is described or measured. `source_status: OPEN` records that no primary specification was reviewed in full for this revision; each such dependency is named where it is used.
> **Retrieval cue:** If two teams use the same noun (`customer`, `order`, `active`) for different identity boundaries, start at Section 3. If a diagram is agreed but cannot be built, start at Section 4.1 and Section 8. If a joined total is wrong by a stable percentage, start at Section 4.3 and then Section 5.

## 1. File Header

### 1.1 What this sheet is for

A data model notation is a lossy compression format for a business agreement. The business agreement is long, contested, and written in a natural language that overloads its own nouns. The notation is short, typed, and mechanically checkable. The whole value of the notation is that a specific mark on the page corresponds to a specific test that a specific table must pass.

That correspondence is what separates a model from a drawing. A drawing records what someone believed at a meeting. A model records what the system will enforce and what the reader may conclude without asking again. This sheet teaches the second thing, and it teaches the notation families in the order a practitioner meets them: entity-relationship shapes, crow's-foot and UML class marks, IDEF1X and dimensional conventions, and finally the temporal and property-graph marks that modern lakehouse schemas actually need.

### 1.2 The decision this sheet supports

The recurring decision is not "which diagram tool." It is: **given this sentence and this sample of rows, what is the smallest set of claims that a reader must be able to check, and which notation displays each claim in a form that an implementer cannot silently drop?** Everything else in this file is machinery for answering that question.

Three consequences follow, and they organize the rest of the sheet.

First, notation choice is subordinate to claim choice. If the team has not decided whether `customer` means a login or a bill-to account, no arrow style resolves it. Section 3 forces the identity boundary into words before any box is drawn.

Second, every mark must map to a constraint with a name. "Optional parent" is not a mark; it is a nullable foreign key or a deferred check. "One-to-many" is not a mark; it is a uniqueness constraint on the child's parent reference plus, where required, a minimum-participation rule that SQL cannot express declaratively. Section 4 derives each mapping.

Third, a model that cannot state its join multiplicity is unfinished. The most expensive modeling errors in analytical work are not missing columns; they are silent relationship promotions — a one-to-many modelled as one-to-one in the reader's head, or a many-to-many modelled as a straight join. Section 4.3 makes multiplicity a computed number rather than a feeling.

### 1.3 Reader task ladder

The ladder below is the order in which a reader should acquire the skill, and it matches the section order.

| Rung | Reader task | Where | Minimum acceptable output |
|---|---|---|---|
| 1 | Name the entity boundary and its natural key | §3 | A sentence of the form "one row per ___ identified by ___" |
| 2 | Assign cardinality and optionality to every edge | §3, §4.2 | A `(min,max)` pair per direction, or an explicit "unknown" |
| 3 | Write the temporal rule in closed form | §4.4 | A half-open predicate with a stated time basis |
| 4 | Predict the join multiplicity | §4.3, §5 | A ratio `α` computed before executing the query |
| 5 | Choose a notation family for the audience | §7 | A choice with a stated disqualifier |
| 6 | Diagnose a wrong diagram from a wrong total | §8 | A competing-cause table with a discriminating check |
| 7 | Price the model | §9 | Rows, bytes, and comparison counts with units |
| 8 | Defend the choice under changed constraints | §11 | An answer that names the condition that would reverse it |

### 1.4 Scope boundary and handoffs

Notation sits between discovery and implementation. Upstream of it is the business definition work: who owns the term, which process generates the record, which decision consumes it. Downstream is physical design: partitioning, clustering, indexes, materialization. This sheet deliberately stops at the logical marks and the constraints they imply, because those are the parts that survive a change of engine.

Two handoffs are explicit. Grain arithmetic at query level — loss detection, fan-out measurement, allocation conservation — lives in [02 Grain and join safety](02_Data_Model_Grain_and_Join_Safety.md); this sheet produces the multiplicity that the other sheet measures. Delivery, freshness, and change classification live in [01 Data contracts](01_Data_Contract_Checklist.md); this sheet produces the identity and version marks that a contract's uniqueness test then enforces.

### 1.5 Evidence posture

Notation is a specification practice, so most of the content here is DERIVED: given the definitions in Section 3, the claims in Section 4 follow by substitution and by the standard results named in Section 10. Where a result is a named theorem (lossless join, Armstrong completeness, the SQL temporal predicate form), the theorem's hypotheses are stated, because a mnemonic is not a theorem.

Every number in this file is constructed for teaching. The customer/order/line/product scenario in Section 5 does not describe any real retailer, and no figure is a benchmark. The arithmetic is independently checked, which makes the calculations VERIFIED without making the workload measured. Where a fact about a specific product or standard version would be needed and was not verified in this revision, the text says `OPEN` and names what would close it.

### 1.6 Version and review record

Version 1.0 of this sheet was 150 lines: a plausible outline with correct instincts and no derivations, no worked arithmetic, no discriminating checks, and eleven interview questions whose answers were one clause long. It failed the depth gate on every axis that matters — a reader could not compute anything from it, could not distinguish two causes of the same symptom, and could not defend a notation choice against a colleague who preferred another.

Version 2.0 rebuilds around a single spine: identity, cardinality, time, multiplicity, cost. It keeps the correct topic-specific material from 1.0 (the half-open interval rule, the bridge warning, the "noun is not an entity" caution, the four failure themes) and expands each into a derived, testable form. `reviewed` remains `null` because no independent reviewer has checked this revision. That field is not a formality: an unreviewed temporal rule is exactly the kind of claim that reads correctly and fails at a month boundary.

**Takeaway:** Notation earns its place only by mapping each mark to a constraint someone can test, so this sheet is organized by testable claim rather than by diagram style.

## 2. Concept Map / Mental Model

### 2.1 One-screen orientation

```text
[Business sentence in natural language]
        |
        |  (1) identify nouns that have independent existence
        v
[Entity candidate] --fails identity test--> [attribute of another entity]
        |
        |  (2) state natural key + domain + version rule
        v
[Entity with declared identity]  <---- Section 3 table row
        |
        |  (3) place edges: which entity references which
        v
[Relationship with (min,max) in both directions]
        |                                   \
        | (4a) cardinality 1:1, 1:N, M:N     \ (4b) optionality: 0.. or 1..
        v                                     v
[Uniqueness constraint on child]      [nullable FK / deferred check /
        |                              minimum-participation rule]
        |
        |  (5) time: is the row a state, version, or event?
        v
[Temporal mark: half-open interval / bitemporal pair / event instant]
        |
        |  (6) predict join behaviour from the marks
        v
[Multiplicity alpha = |F join B| / |F|] --> [Section 5 arithmetic]
        |
        |  (7) choose notation family for the audience
        v
[Crow's foot | IDEF1X | UML class | Chen | dimensional | property graph]
        |
        |  (8) implement, then test each mark
        v
[Constraint tests: uniqueness, FK, exclusion, as-of exactly-one]
        |
        v
[Feedback: a failed test reopens step (2) or (3), not the drawing tool]
```

### 2.2 Why the order is not negotiable

The flow above has a property worth naming: each step consumes the output of the previous one in a way that cannot be reordered without losing information. Cardinality without identity is meaningless, because "one-to-many" is a statement about instances of two identified sets. Optionality without cardinality is under-specified, because "may be missing" does not say how many may exist when present. Temporal marks without a grain cannot be written, because an interval attaches to a version of a row and a row requires a grain.

The common failure — and it is the single most common failure in practice — is to start at step 7. A team picks crow's foot because the tool defaults to it, draws boxes for the nouns in the requirements document, and then discovers at implementation that two of the boxes are the same entity, one box is an attribute of a third, and the cardinality on the most important edge is a guess. The diagram is then not wrong in any single mark; it is wrong in its ordering, and no amount of arrow correction repairs it.

### 2.3 Reading a model backwards

Strong reviewers read a diagram backwards, from the marks to the sentences. For each edge they ask: what test would fail if this cardinality were false? If no test comes to mind, the mark is decorative. This backward reading is what Section 8 formalizes: every failure entry begins with an observable symptom, lists the competing causes that produce that same symptom, and names the check that separates them.

The backward reading also explains why the notation wars are mostly unproductive. Crow's foot, IDEF1X, and UML class diagrams can all express one-to-many with mandatory participation; they differ in where the mark sits and whether the mandatory side is drawn inside or outside the foot. A team that argues about mark placement while leaving identity and time undefined has optimized the least consequential layer.

### 2.4 The four-question gate

Before any diagram is accepted, four questions must have written answers. This is the operational form of the map.

| Gate | Question | Rejection signal |
|---|---|---|
| G1 identity | What does one row mean, and what identifies it? | The answer uses a verb or a process |
| G2 cardinality | For each edge, what is the maximum count in each direction? | "It depends" with no `(min,max)` |
| G3 time | Is the row a current state, a version, or an event? | "Current" used without a cutoff |
| G4 consequence | Which decision breaks if a mark is dropped? | No consumer can be named |

A model that passes G1–G3 but fails G4 is complete and unused, which is a legitimate outcome for an inventory exercise and not for a data product. Section 7 treats G4 as the tie-breaker when two notations are equally expressive.

**Takeaway:** Identity, then edges, then time, then cost — a diagram drawn in that order has marks that can be tested, and a diagram drawn in the reverse order cannot be repaired by changing arrow styles.

## 3. Foundational Definitions

### 3.1 The marks, stated precisely

The table below is the controlled vocabulary for this sheet. Each row names a notation element, its formal meaning, the constraint that implements it, and a nonexample — a case that superficially resembles the element but must not be modelled with it. The nonexample column is the part readers skip and the part that prevents the most errors.

| Element | Formal meaning | Constraint that enforces it | Nonexample |
|---|---|---|---|
| Entity type | A set of distinguishable business objects with independent existence | A relation with a declared key | A noun that only ever appears inside another noun's description, such as "order status text" |
| Attribute | A function from entity instances to a declared value domain | Column with type and nullability | A derived value with no stored definition, such as "margin" with no cost basis |
| Natural key | A minimal attribute set that identifies the instance in the business domain | `UNIQUE` constraint on business columns | A warehouse sequence number |
| Surrogate key | A system-assigned identifier with no business meaning | Surrogate `PRIMARY KEY` plus a separate natural `UNIQUE` | A natural key renamed as a surrogate |
| Candidate key | Any minimal set that determines all attributes of the relation | Uniqueness test per candidate | A superset key, which is unique but not minimal |
| Relationship type | An association among entity types with declared participation | FK on the child plus uniqueness where the direction requires it | A free-text field listing related identifiers |
| Cardinality | The maximum number of associated instances on one side, read from the other | Uniqueness of the FK tuple for 1:1 and 1:N-from-parent | Using "many" for anything countable but not bounded by a rule |
| Optionality (modality) | The minimum number of associated instances: 0 or 1 | Nullable FK for 0..1 parent; minimum-1 needs a deferred or trigger check | Assuming null means optional without checking the load order |
| Identifying relationship | Parent key is part of the child key | Composite PK containing the parent key | A composite PK used only to save a surrogate |
| Non-identifying relationship | Parent key appears as a plain FK | FK not in the PK | Calling a nullable FK "identifying" because it is important |
| Associative entity | An entity that carries the relationship and may have its own attributes | Table with the two FKs in its key plus attribute columns | A pure cross-reference table with no attributes, modelled as an entity anyway |
| Recursive relationship | Both roles of an edge point at the same entity type | Self-referencing FK with a role name per direction | A hierarchy flattened into repeated columns |
| Role name | Disambiguates an entity's function in one edge | Distinct FK column names | One FK reused for two different roles |
| Generalization | A subtype is a specialization of a supertype | Shared PK across subtype and supertype tables | Subtyping when the difference is a status value |
| Exclusive subtype | An instance belongs to at most one subtype | Check or exclusion across subtype tables | Overlapping subtypes with no rule |
| Temporal validity | The interval during which a stated attribute value is true in the world | Half-open interval plus non-overlap exclusion per key | A `current_flag` with no rule for the previous row |
| Transaction time | The interval during which the database asserted a version | Second interval pair on the same row | Using load timestamp as validity |
| Fact grain | What one fact row measures, at one combination of dimension keys | Unique constraint on the dimension key tuple | A fact table whose rows are "a sale" when some are returns |
| Bridge entity | Resolves many-to-many and holds relationship attributes | Composite key of both FKs plus optional attributes | Joining M:N directly and accepting duplicated measures |
| Property graph | Nodes and typed edges with attributes on both | Edge type plus endpoint constraints | A relational star rendered as if it were a graph |

### 3.2 Identity before attributes

The ordering claim of Section 2 has a precise justification, and it is the same argument that makes normalization meaningful. An attribute is a function; a function needs a domain. Until the domain of instances is fixed, "the customer's tier" has no truth value, because there is no answer to which customer instance is being described. In relational terms, an attribute is a column of a relation whose tuples are instances, and the key determines which tuple a given sentence is about.

This is why the phrase "we will add attributes later" is safe in a way that "we will decide identity later" is not. Attributes can be added to a stable relation without disturbing its consumers, subject to the compatibility rules in the contract sheet. Changing identity changes the cardinality of every relation that references it, and therefore changes the arithmetic of every aggregate built on those relations.

### 3.3 The identity boundary test

Three questions decide whether a candidate noun is an entity or an attribute of another entity. They are cheap to apply and they resolve most disputes before a diagram exists.

| Test | Passes as entity when | Fails and is likely an attribute when |
|---|---|---|
| Independent lifecycle | It can be created, changed, and retired on its own schedule | It exists only while its parent exists and changes only with the parent |
| Independent reference | Other entities refer to it directly, or it has its own owner | Nothing else needs to point at it |
| Independent attributes | It has at least one attribute not determined by its parent | All its attributes are parent attributes restated |

A worked application: in a retail order model, `order_line` passes all three. It can be created after the order, it is referenced by fulfilment and returns, and it carries `quantity` and `unit_price` that the order header does not determine. `shipping_address` fails the first test if the business never edits an address independently of the order; it is then a set of attributes on the order, or a value object rather than an entity. `customer` passes all three even when the marketing team wishes it did not, which is why customer identity disputes are expensive.

### 3.4 Cardinality vocabulary and its ambiguity

The word "many" hides three different situations, and the notation must distinguish them. A bounded many has a business maximum (an order has at most, say, 200 lines by policy). An unbounded many has none. A many with a stable average is what actually drives cost, and the average is a modelling input in Section 4.3 rather than a fact about the world.

| Situation | Declared as | Enforced by | Modelling obligation |
|---|---|---|---|
| Parent has 0 or 1 child | `0..1` from parent, `1..1` from child | Unique nullable FK on child | State what the child's creation event is |
| Parent has 0..* children | `0..*` from parent | FK on child, no uniqueness beyond key | Declare typical and worst-case counts |
| Parent has 1..* children | `1..*` from parent | Deferred check or transaction-level rule | Name the transaction that creates both |
| Symmetric many-to-many | `*..*` both ways | Bridge table with composite key | Declare whether the bridge carries attributes |
| Exactly one parent from a set of alternatives | Exclusive arc | Check constraint over nullable FKs, or a role column | State the exclusivity rule and its open cases |

The minimum side deserves emphasis because it is where SQL and diagrams diverge most. A relational foreign key enforces that a non-null child reference resolves to an existing parent. It does not enforce that a parent has at least one child, and it does not enforce that the parent of a child is the parent the business intended — only that some row with that key exists. A diagram showing a mandatory foot therefore asserts something the schema cannot guarantee, and the gap must be closed by a documented rule, a deferred constraint, or an explicit accepted risk.

### 3.5 Time marks are not optional decoration

Three temporal questions arise for every entity, and conflating them is the second most expensive notation error after identity ambiguity.

The first is whether the row is an event or a state. An event is a thing that happened at an instant and is never edited: a shipment, a payment, a click. A state row describes what is true over an interval and may be revised in the world or in the database. An event row with a validity interval is usually a modelling mistake, because an event's time is a point and the interval is being used to encode processing delay.

The second is which interval the reader means. Valid time is when the fact was true in the business. Transaction time is when the database believed it. A price that was 42 currency from February to April, but was recorded in the database only in March, has valid time from February and transaction time from March. Reporting "the price in February as of a February-close report" and "the price in February as known today" are different queries, and the model must support whichever the decision needs.

The third is closure. Intervals in a model must state whether they are closed, half-open, or open, and which end contains the boundary instant. Section 4.4 derives why half-open is the only convention that gives exactly one match at a boundary with a single-column test, and Section 5 exercises it at a month boundary where a closed-closed convention double counts.

### 3.6 Notation families and what they actually encode

Practitioners sometimes treat notation families as taste. They are not: each family has a different expressive centre, and using a family outside its centre produces marks that readers misinterpret.

| Family | Expressive centre | Weakness in practice | Typical audience |
|---|---|---|---|
| Chen ER | Entities, relationships, attributes as separate shapes; explicit cardinality labels | Visual weight grows fast; attributes crowd the diagram | Teaching, formal specification |
| Crow's foot (IE) | Cardinality and optionality at line ends; compact for many entities | Recursive and subtype detail is awkward | Analytics engineering, warehouse design |
| IDEF1X | Identifying versus non-identifying relationships; strong key discipline | Dense; the key rules are not intuitive to newcomers | Enterprise data architecture |
| UML class | Attributes and operations in compartments; rich multiplicities; generalization | Operations imply behaviour a warehouse does not have | Software and integration design |
| Dimensional (star/snowflake) | Facts, dimensions, grain, conformed dimensions | Suppresses transactional relationships behind dimensions | BI and reporting |
| Data Vault | Hubs, links, satellites; insert-only history | Deliberately non-semantic; needs a presentation layer | Raw historical integration |
| Property graph | Nodes, typed edges, attributes on edges | No native constraint language for many cardinalities | Traversal and path queries |
| Physical DDL | Exact types, constraints, defaults | Engine-specific; hides intent behind syntax | Implementation and review |
| Schema document (JSON, Avro, Protobuf) | Wire format, compatibility rules | Cannot express referential or cross-record rules | Interface contracts |

The practical instruction is to pick the family whose expressive centre matches the claim that the audience most needs to check, then to write the claims the family cannot show as named constraints beside the diagram. A crow's-foot diagram plus a written list of uniqueness and exclusion tests is stronger than a perfect UML model with no test list, because the tests are what actually run.

**Takeaway:** A noun becomes an entity only when it has independent lifecycle, reference, and attributes; and every edge needs a `(min,max)` pair in both directions, because the minimum is precisely the part SQL cannot enforce.

## 4. Core Formulas with Full Derivation

### 4.0 How to read this section

Every formula below is presented with validity stated before use, a derivation that starts from a definition and applies named algebra, then the five-part block `FORMULA / WHERE / CONSTRAINT / NUMERICAL EXAMPLE / INTERPRETATION`. The numerical examples are small enough to check by hand, and each is labelled ILLUSTRATIVE even where the arithmetic has been verified, because a checked calculation about a constructed schema is not a measurement of any system.

The section is ordered as a ladder. Section 4.1 derives the base claim — what a key does — because every later formula assumes identity is settled. Section 4.2 derives cardinality as arithmetic, because that is the step most often performed by intuition instead of computation. Section 4.3 derives the variants that appear in real schemas: versioned dimensions, bitemporal rows, bridges, subtypes, and hierarchies. Section 4.4 is the counterexample section, and it is the most important one, because it shows the exact configurations in which the earlier formulas produce a confident wrong answer.

### 4.1 Base: identity, functional determination, and the as-of predicate

#### 4.1.1 Derivation of functional determination from the key definition

**Validity before use:** the relation is in first normal form (all attribute values atomic), the key attributes are drawn from the same relation, and no attribute is multi-valued. These are the hypotheses of the relational model, not optional preferences.

Start from the definition. A relation `R(A1..An)` is a set of tuples; a candidate key `K ⊆ {A1..An}` is a minimal subset such that no two distinct tuples agree on all of `K`. Set semantics gives one direction immediately: if two tuples agree on `K` they are the same tuple, therefore they agree on every attribute. Writing `X → Y` for "`X` functionally determines `Y`", the key definition is exactly `K → {A1..An}`.

Minimality is the second half, and it is what makes the key useful rather than merely true. If `K` determines all attributes and some proper subset `K' ⊂ K` also determines all attributes, then `K` contains a redundant component. The redundancy is not harmless: a composite key with a redundant component makes every foreign key that references it wider than necessary, and it hides the real identity from the reader. A candidate key is a key that is both determining and minimal; a primary key is the candidate key the designer elects to use as the row's identity.

Two consequences follow, and both are used later. First, functional determination is not causality. `customer_id → tier` says that in this relation, one tier value is associated with each customer; it says nothing about why. Second, determination is relative to the relation's current extension and declared rules. A rule that holds today may be violated by a new business process, which is why the determination claim is a versioned commitment rather than a permanent truth.

```text
FORMULA [MN-01]: K is a candidate key of R  iff  (K -> attrs(R))  and
                 forall A in K: NOT ((K \ {A}) -> attrs(R))
WHERE: R is a relation over attribute set attrs(R);
       K is a subset of attrs(R);
       X -> Y means every pair of tuples agreeing on X agrees on Y.
CONSTRAINT: first normal form; set semantics (no duplicate tuples);
            determination is evaluated against declared business rules,
            not only against a convenient sample of rows.
NUMERICAL EXAMPLE: R(order_id, line_no, product_id, qty, unit_price).
       K1 = {order_id, line_no} determines all five attributes.
       Is K1 minimal? order_id alone does not determine line_no, so no
       proper subset determines all attributes. K1 is a candidate key.
       K2 = {order_id, line_no, product_id} is unique (it contains K1)
       but not minimal, so it is a superkey and not a candidate key.
INTERPRETATION: the child's identity is the pair, not the triple; adding
       product_id to the key would make every reference carry a payload
       that identity does not require.
```

The distinction between `K1` and `K2` is the difference between a model that states identity and one that states uniqueness. Both reject duplicate rows; only `K1` tells a reader that `(order_id, line_no)` is the business identity of a line. When a downstream team needs to reference a line, `K1` gives them a two-column reference and `K2` gives them a three-column reference whose third column can change — which turns a stable reference into a mutable one and forces a cascade update. This is the concrete cost of a non-minimal key, and it recurs at scale in Section 9.

#### 4.1.2 Derivation of the Armstrong closure bound

**Validity before use:** the rule set is a set of functional dependencies over a fixed attribute set, and only the three Armstrong axioms (reflexivity, augmentation, transitivity) are used. If multivalued dependencies or inclusion dependencies are in play, this closure is a lower bound on the implied set, not the full set.

The closure `X⁺` is the set of attributes determined by `X`. It is computed by repeatedly applying transitivity: if `Y ⊆ X⁺` and `Y → Z`, then `Z ⊆ X⁺`. The process terminates because `X⁺` only grows and is bounded by `attrs(R)`.

The practical value is that closure lets a designer decide whether a proposed key is redundant without inspecting data. If `X⁺ = attrs(R)` and `(X \ {A})⁺ ≠ attrs(R)` for every `A ∈ X`, then `X` is a candidate key by definition, and this is a proof rather than a sample. It also explains why key disputes are resolvable by argument: two teams that disagree about identity are disagreeing about which dependencies hold, and that is a factual question about the business process.

```text
FORMULA [MN-02]: X+ = the smallest set containing X, closed under:
                 if Y subset of X+ and (Y -> Z) in F, then Z subset of X+.
WHERE: F is the declared dependency set; X is an attribute set;
       X+ is the attribute closure [set of attribute names].
CONSTRAINT: dependencies are over a single relation schema; the axiom set
            is {reflexivity, augmentation, transitivity}; F is finite.
NUMERICAL EXAMPLE: attrs = {o, l, p, q, u} abbreviating order_id, line_no,
       product_id, qty, unit_price.
       F = { o,l -> p ; o,l -> q ; o,l -> u ; p -> u }.
       Start X+ = {o,l}. Apply o,l -> p : X+ = {o,l,p}.
       Apply p -> u : X+ = {o,l,p,u}. Apply o,l -> q : X+ = {o,l,p,q,u}.
       X+ = all five attributes.
       Remove l: {o}+ = {o}, not all attributes. Remove o: {l}+ = {l}.
       Therefore {o,l} is a candidate key.
INTERPRETATION: the closure proved minimality without scanning data, and it
       showed that unit_price is transitively dependent on (o,l) via p, which
       is a normalization signal rather than a modelling error here.
```

The last observation is important and easy to over-read. `p → u` means that if `unit_price` is the catalogue price, it belongs on the product, not on the line; if it is the price actually charged, the dependency is `o,l → u` and the `p → u` rule is false. The letter of the closure is mechanical; deciding which dependencies the business intends is the modelling judgement, and no axiom set makes it for you. This is the honest limit of formal methods in notation work, and Section 4.4 returns to it.

#### 4.1.3 Derivation of the half-open as-of predicate

**Validity before use:** each business key has at most one version whose interval contains the as-of instant; intervals are stored as instants in a single time basis (UTC, fixed precision); and the fact timestamp is comparable to the interval endpoints. If the fact timestamp is in local time and the interval in UTC, the predicate is invalid before it is evaluated.

Read the predicate in two parts. The key equality selects the version family; the interval test selects the version within the family. Writing `d.from ≤ t < d.to` uses left-closed right-open closure, and the choice is forced by a counting argument rather than by convention.

Consider a sequence of versions for one key with change instants `t0 < t1 < t2`. Under closed-closed intervals, version 1 covers `[t0, t1]` and version 2 covers `[t1, t2]`, so the instant `t1` is covered twice. Under open-open, `(t0, t1)` and `(t1, t2)` leave `t1` covered zero times. Under half-open, `[t0, t1)` and `[t1, t2)` cover `t1` exactly once, and no other instant twice. The half-open convention is therefore the unique choice among these three that partitions the timeline, which is why it is the default in temporal modelling.

The second reason is computational. Half-open membership is a single range test with two comparisons and no special case for the boundary. Closed-closed membership needs an additional rule to decide which of two adjacent versions owns the boundary, and that rule is a tie-break that must be stored or inferred — extra state for no additional information.

```text
FORMULA [MN-03]: version d matches fact f  iff
                 d.key = f.key  AND  d.from <= f.time  AND  f.time < d.to
WHERE: d is one version row of a versioned dimension;
       f.time is the fact's business-event instant [UTC instant];
       d.from, d.to are validity start and end [UTC instants];
       d.to may be a sentinel such as 9999-12-31 for the open version.
CONSTRAINT: for each key, versions have disjoint interiors and cover the
            declared domain; the fact instant lies inside that domain;
            both columns use the same time basis and precision.
NUMERICAL EXAMPLE: tiers Silver [2026-01-01, 2026-02-01) and
       Gold [2026-02-01, infinity). Fact A at 2026-01-31 23:59 matches
       Silver (from <= t, t < to). Fact B at 2026-02-01 00:00 does not
       match Silver (t < to is false) and does match Gold.
INTERPRETATION: each instant resolves to exactly one version, so an as-of
       join is a lookup rather than a decision about which row to trust.
```

The derivation also tells you what to test. The predicate itself never fails; the model around it fails when the version set has a gap or an overlap. That is a property of the data, checked by a separate query, and Section 4.4 gives the exact overlap count that such a check should produce on a clean versus broken model.

### 4.2 General: cardinality as arithmetic, not adjective

#### 4.2.1 Derivation of join multiplicity

**Validity before use:** the fact relation's key is unique; the bridge's degree is defined per fact key; the join is an equijoin on the fact key; and no filtering is applied after the join. The last condition matters because a `WHERE` clause on a bridge attribute changes the effective degree, and that is a different quantity derived below.

Start from the definition of a join. For each fact row `f` with key `k`, the join produces one output row per bridge row whose foreign key equals `k`. The number of such bridge rows is the degree `deg(k)`. Summing over all facts gives the output cardinality, and dividing by the input cardinality gives the average degree.

The average degree is exactly the multiplicity factor `α`. It is the ratio of output rows to fact rows under an equijoin with no other predicate. When `α > 1`, any additive measure in the fact table is multiplied by `α` in the output, because each fact contributes its measure once per matching bridge row. That statement is exact for sum and for count, and it is the whole explanation of the classic "revenue went up 25% after adding tags" incident.

The definitional identity `|F ⋈ B| = |B|` is worth stating explicitly, because it is a strong check. When you inner-join a fact table to a bridge on the bridge's foreign key, the output row count equals the bridge row count, not something between the two. If your query returns a different number, either a fact key appears more than once in the fact table (a key violation) or the join predicate is not a pure equijoin on that key.

```text
FORMULA [MN-04]: alpha = |F join B| / |F| = (1/|F|) * sum over f in F of deg(f)
                 and, for a pure equijoin on the fact key, |F join B| = |B|
WHERE: F is the fact relation [rows]; B is the bridge relation [rows];
       deg(f) = number of B rows whose FK equals f's key [bridge rows/fact];
       alpha is dimensionless [output rows per input row].
CONSTRAINT: F key is unique; the join is an equijoin on that key with no
            additional predicate; B's FK is non-null; inner join semantics.
NUMERICAL EXAMPLE: |F| = 168,000 order lines; |B| = 210,000 tag assignments;
       alpha = 210,000 / 168,000 = 1.25 exactly.
       A line at 42.00 currency contributes 42.00 per tag. Total measure
       before join = 168,000 * 42.00 = 7,056,000.00;
       after join = 210,000 * 42.00 = 8,820,000.00;
       inflation = 1,764,000.00, i.e. 1,764,000 / 7,056,000 = 0.25.
INTERPRETATION: a 25% overstatement with no data error at all; the query is
       arithmetically correct and semantically wrong for a line-level total.
```

Note what the example does not claim. It does not claim that tags are bad, that 210,000 is a realistic tag count, or that any system behaved this way. The tag count was chosen as `1.25 × 168,000` precisely so that `α` would be a round number and the inflation factor would be easy to verify by hand. The teaching point survives the artificiality: the error is a multiplicative constant that looks exactly like a business improvement.

#### 4.2.2 Derivation of composed multiplicity across chained bridges

**Validity before use:** two bridges both key on the same fact key; degrees are measured on the full, unfiltered relations; the covariance term is computed from the same population, not from a sample with a different denominator.

When a fact is joined to two independent bridges in sequence, the output degree is the product of the two degrees per fact. Averaging the product is not the same as multiplying the averages, and the gap is exactly the population covariance of the two degree variables. That identity is algebraic, not approximate.

Write `α₁ = (1/n)Σd₁`, `α₂ = (1/n)Σd₂`, and `α₁₂ = (1/n)Σd₁d₂`. Expanding the definition of covariance gives `α₁₂ = α₁α₂ + Cov(d₁,d₂)`. Therefore the product rule is exact only when the covariance is zero. In practice degrees are often positively correlated, because both bridges are driven by the same underlying richness — a large order has more lines, more promotions, more shipments. Then `α₁₂ > α₁α₂` and the naive product understates the inflation.

The practical instruction is to measure `α` on the fully joined query rather than to multiply per-bridge estimates. If you must estimate, measure the covariance and state it, or bound the result with the worst-case degree.

```text
FORMULA [MN-05]: alpha_12 = alpha_1 * alpha_2 + Cov(d1, d2)
                 = (1/n) * sum over f of d1(f)*d2(f)
WHERE: d1(f), d2(f) are the two bridge degrees for fact f [rows/fact];
       Cov is the population covariance over the same n facts;
       alpha_12 is the combined output-rows-per-fact factor.
CONSTRAINT: both bridges key on the same unique fact key; degrees measured
            on unfiltered relations; the covariance uses the same fact set.
NUMERICAL EXAMPLE: three facts with degrees (1,2), (2,2), (3,5).
       alpha_1 = (1+2+3)/3 = 2; alpha_2 = (2+2+5)/3 = 3;
       product of means = 6.
       alpha_12 = (1*2 + 2*2 + 3*5)/3 = (2+4+15)/3 = 21/3 = 7.
       Cov = 7 - 6 = 1. Naive multiplication understates by 1/6 = 16.7%.
INTERPRETATION: with positively correlated fan-out the true inflation is 7x,
       not 6x, so an estimate built from two separate averages is optimistic
       in exactly the direction that hurts a revenue number.
```

The example is deliberately tiny so the arithmetic is transparent; three facts with those degrees give a combined factor of 7 against a product-of-means of 6. The 16.7% gap is the size of the mistake a reviewer would make by multiplying two dashboard numbers. At warehouse scale the gap can be much larger, and it is invisible in any single bridge's metrics because each bridge is individually correct.

#### 4.2.3 Derivation of loss fraction and the concealment identity

**Validity before use:** the fact relation is complete and keyed; the bridge covers a subset of facts; the join is inner. Under a left join the loss is zero by construction and the diagnostic is different — you must count null-extended rows instead.

An inner join silently drops facts with no matching bridge row. The dropped fraction is the share of facts whose degree is zero, and the surviving fraction is its complement. This is the loss control. Separately, the surviving facts are inflated by their average degree conditional on survival. The two effects act in opposite directions on row count, which is exactly why a row-count check fails to detect them.

Deriving the concealment identity: let `S` be the surviving fact set, `s = |S|/|F|` the survival share, and `α_S = |B|/|S|` the conditional multiplicity. Total output rows are `|B| = s·α_S·|F|`. If the designer expected `α = 1` and `s = 1`, the observed row count is `s·α_S` times the expectation.

Because each surviving fact carries its measure once per matching bridge row, the joined measure is `s·α_S·M_F`, where `M_F` is the true total over all facts. Row count and measure therefore scale by the *same* factor `s·α_S`. The consequence is the important one: the two errors cancel together, not separately. When `α_S = 1/s`, both the row count and the joined total return exactly the naive expected values, while the population is wrong — some facts are missing and others are counted twice. A 1.43% loss paired with a 1.014493x fan-out produces a row count of `0.985714 × 1.014493 × 168,000 = 168,000` and a total of exactly `7,056,000.00`, with 2,400 lines absent from the answer and 2,400 line-measures duplicated elsewhere.

This is why a total-level reconciliation is not a control. It is sensitive only to the product `s·α_S`, so it is blind to the composition of that product. Only checks that are sensitive to one factor are diagnostic: the anti-join measures `1 − s` and the degree ratio measures `α_S`. Reporting the product alone is reporting the one number the two errors are designed to leave unchanged.

```text
FORMULA [MN-06]: s = |S| / |F|;  alpha_S = |B| / |S|;
                 output rows = s * alpha_S * |F|;
                 measure error factor = s * alpha_S
WHERE: S = facts with at least one bridge row; |B| = bridge rows;
       s is survival share [dimensionless]; alpha_S conditional multiplicity.
CONSTRAINT: inner join; bridge FK non-null; each fact's measure is repeated
            once per matching bridge row; no DISTINCT applied.
NUMERICAL EXAMPLE: |F| = 168,000; 2,400 lines have no tag, so |S| = 165,600;
       s = 165,600/168,000 = 0.985714...;
       |B| = 207,000 so alpha_S = 207,000/165,600 = 1.25;
       output rows = 0.985714 * 1.25 * 168,000 = 207,000 (matches |B|);
       measure = 207,000 * 42.00 = 8,694,000.00 versus true 7,056,000.00;
       error factor = 8,694,000 / 7,056,000 = 1.232143, i.e. +23.2%.
INTERPRETATION: 1.43% of lines vanished and the survivors were counted 1.25
       times; the net total is 23.2% high, and no single control sees both.
```

The identity is worth restating in words because it is the diagnostic insight of the whole sheet: **loss and fan-out are conjugate errors.** A model that under-joins loses measure and a model that over-joins duplicates measure, and the two can be tuned by accident to produce a plausible total. This is why "the number looks about right" is not evidence, and why the verification in Section 8 always includes a key-level check rather than a total-level check.

### 4.3 Variants: versioning, bitemporality, bridges, subtypes, hierarchies

#### 4.3.1 Derivation of version-row growth for a type-2 dimension

**Validity before use:** each business key's history is a sequence of non-overlapping half-open intervals; the current row is open-ended; and each change closes one interval and opens another. Under these conditions the number of rows is the sum of per-key change counts, not a function of elapsed time alone.

The derivation is a counting argument over change events. A key that never changes contributes one row. A key that changes `c` times contributes `c + 1` rows, because the initial state is a row and each change opens a new one. Summing over keys gives total rows `= n + Σc`, where `n` is the number of keys. There is no dependence on whether the changes were value changes or corrections; a correction to a historical row's payload is a change event in transaction time even if the valid time is unchanged, which is the distinction Section 4.3.2 formalizes.

Two derived quantities follow. The average versions per key is `1 + Σc/n`, which is the factor by which the dimension grows relative to a current-state table. The number of adjacent interval boundaries is `Σc`, which is also the number of pairs that must be checked for overlap under a closed-closed convention. That coincidence is useful: the same number that tells you the storage multiple tells you how many boundary instants a wrong closure convention would double-count.

```text
FORMULA [MN-07]: N = n + sum over keys of c(key);
                 v_bar = N / n = 1 + (sum c)/n;
                 adjacent boundaries = sum c = N - n
WHERE: N is total version rows [rows]; n is distinct business keys [keys];
       c(key) is the number of changes for that key [changes];
       v_bar is mean versions per key [rows/key].
CONSTRAINT: half-open, contiguous, non-overlapping intervals per key;
            one row per key at any instant; no deletions of history.
NUMERICAL EXAMPLE: 3,000 customer keys: 1,200 with 0 changes, 1,200 with 2
       changes, 600 with 5 changes.
       sum c = 0 + 1,200*2 + 600*5 = 2,400 + 3,000 = 5,400.
       N = 3,000 + 5,400 = 8,400 rows; v_bar = 8,400/3,000 = 2.8.
       Boundaries = 5,400, each of which would be double-matched under a
       closed-closed convention, producing 5,400 duplicated as-of matches.
INTERPRETATION: a 3,000-row current-state table becomes 8,400 rows; the
       change history is 2.8x the key count, and the boundary count tells
       you the exact size of a closure-convention error.
```

The example also shows why "we will just keep the current value" is a decision, not a default. Dropping history restores 3,000 rows and saves 5,400 rows of storage, at the cost of making every historical report unreproducible. That trade is legitimate when the decision has no lookback requirement; it is illegitimate when a regulatory or contractual lookback exists, and the model should record which case applies rather than leaving it implicit.

#### 4.3.2 Derivation of the bitemporal state bound

**Validity before use:** valid time and transaction time are both tracked per attribute value or per row; a correction that changes only transaction time is stored as a new row rather than an update; and the query specifies both an as-of-valid instant and an as-of-transaction instant.

A bitemporal row is indexed by two intervals, and the number of rows is driven by two independent revision processes. The worst case is multiplicative: if a key has `v` valid-time versions and `r` transaction-time revisions of the whole history, the model can require up to `v·r` rows. That is the bound a capacity estimate should use when the business can restate history, because restatement is the operation that multiplies the two dimensions.

The common case is much smaller, and the model should say which case it is in. If corrections only ever apply to the current version, the row count is `v + r` rather than `v·r`, because revisions and versions do not cross. The difference between these two regimes is a factor of `min(v,r)`, which is large enough that guessing wrong on a capacity estimate is a real problem.

The query consequence is sharper than the storage consequence. A bitemporal row has four possible query types: as-of-valid with current transaction, as-of-transaction with current valid, as-of-both, and full audit. Each returns a different answer for a corrected value, and a model that stores only one interval can answer only some of them. The notation must therefore mark which intervals exist; a single `valid_from/valid_to` pair on a table that receives retroactive corrections is a model that has silently chosen to overwrite belief.

```text
FORMULA [MN-08]: N_bounded <= sum over keys of v(key) * r(key);
                 for independent uniform versions, N ~ n * v_bar * r_bar;
                 if revisions only touch current versions, N = n(v_bar + r_bar - 1)
WHERE: v(key) = valid-time versions [versions]; r(key) = transaction-time
       revisions [revisions]; n = keys; v_bar, r_bar are means.
CONSTRAINT: every (valid, transaction) pair that the audit needs is retained;
            both intervals are half-open; the query pins both coordinates.
NUMERICAL EXAMPLE: 8,400 valid-time rows for 3,000 keys (v_bar = 2.8).
       If each history is revised up to 4 times, the bound is
       8,400 * 4 = 33,600 rows. If revisions touch only current versions,
       the count is 3,000*(2.8 + 4 - 1) = 3,000*5.8 = 17,400 rows.
INTERPRETATION: the capacity spread between the two regimes is 33,600 vs
       17,400 rows, so the model must state which revision rule applies
       before anyone sizes storage or promises an audit capability.
```

#### 4.3.3 Derivation of bridge normalization and its measure discipline

**Validity before use:** the many-to-many set is genuinely many-to-many in the business (the same tag can apply to many lines and a line can carry many tags); the bridge's own attributes are properties of the relationship, not of either endpoint; and measures are aggregated before or after the bridge according to a stated rule.

A many-to-many relationship cannot be represented by a foreign key in either direction, because a foreign key is single-valued per row. The bridge resolves this by making each relationship instance a row whose key is the pair. The derivation is a set-cardinality argument: if each side can be associated with many instances of the other, then the association itself is an entity with cardinality equal to the number of pairs, and it must be stored as such.

Bridge attributes are the reason the bridge is an entity rather than a pure cross-reference. `assigned_at`, `assigned_by`, and `weight` are properties of the pairing, not of the order line or the tag. When such attributes exist, the bridge has a lifecycle and needs its own key, which is usually `(left_key, right_key, version)` rather than the bare pair.

Measure discipline is the part readers get wrong. Summing a fact measure after joining to a bridge multiplies it; summing a bridge-owned measure such as `weight` across facts would be meaningless. The rule is that each measure is summed at its own grain, and the model must label which grain each numeric column belongs to. A column with no declared grain is the modelling equivalent of a unitless number.

```text
FORMULA [MN-09]: |M:N| = number of distinct pairs;  with attribute a on the
                 bridge, the bridge key is (left, right) if a is fixed per
                 pair, else (left, right, version).
                 Fact-measure rule: sum(measure) must be computed at the
                 fact grain, before any bridge join, or with DISTINCT on the
                 fact key if the bridge join is unavoidable.
WHERE: pairs are distinct by business definition; a is a bridge attribute;
       measure is additive at the fact grain.
CONSTRAINT: pair duplication requires a version or a validity interval;
            the fact key must be recoverable in the joined result.
NUMERICAL EXAMPLE: 168,000 lines, 210,000 tag assignments -> 1.25 tags per
       line on average. Correct line revenue: aggregate lines first,
       168,000 * 42.00 = 7,056,000.00. Incorrect: aggregate after the
       bridge join, 207,000 * 42.00 = 8,694,000.00 at the observed coverage
       or 8,820,000.00 at full coverage.
INTERPRETATION: the bridge is a legitimate entity; the error is aggregating a
       line-grain measure at the pair grain, which the model can prevent by
       declaring the grain of every numeric column.
```

#### 4.3.4 Derivation of subtype exclusivity and its enforcement gap

**Validity before use:** the supertype holds the shared identity; each subtype holds only its own attributes; and the exclusivity rule is stated as a business rule before it is enforced by a constraint.

A generalization partitions a supertype's instances into subtypes. Exclusivity says an instance belongs to at most one subtype; completeness says every instance belongs to at least one. These are independent properties, giving four combinations, and the model must state which one holds. The common mistake is to assume exclusivity because the subtypes look different, when the business actually allows an entity to be both a supplier and a customer.

Enforcement is where notation and SQL diverge again. Exclusivity across several subtype tables can be enforced by a check over a discriminator column when the subtypes share one table, or by a trigger or assertion when they are separate tables. Completeness across separate tables cannot be enforced by a foreign key at all, because the constraint is on the parent, not on the child. A crow's-foot diagram with a mandatory subtype symbol therefore asserts a rule that most engines will not check.

The design consequence is that exclusive subtypes sharing most attributes are usually better modelled as one table with a type column plus a check constraint, and subtypes with genuinely different attribute sets are better modelled as separate tables with a shared key and an accepted, documented completeness risk. Both choices are defensible; leaving the choice unstated is not.

```text
FORMULA [MN-10]: subtype partition states = {exclusive, overlapping} x
                 {total, partial} = 4 named states;
                 exclusivity across k subtype tables requires a count
                 constraint: sum over i of I(instance in subtype i) <= 1
WHERE: I(.) is the indicator function; k is the number of subtype tables;
       the sum is over the instance's memberships.
CONSTRAINT: instance identity is shared across supertype and subtype;
            the count constraint is declarative only in single-table designs.
NUMERICAL EXAMPLE: 4,800 products, of which 3,000 are simple, 1,200 are
       kits, and 600 are bundles. Sum of memberships = 4,800 = supertype
       count, consistent with exclusive and total. If 200 kits were also
       bundles, the sum would be 5,000 > 4,800, which is the exact signal
       that the exclusivity assumption is violated.
INTERPRETATION: the sum-of-memberships test is a one-query check that
       distinguishes exclusive from overlapping subtypes; it is cheaper and
       more reliable than reading the diagram.
```

#### 4.3.5 Derivation of hierarchy depth and path cost

**Validity before use:** the hierarchy is a directed acyclic graph with one parent per node (a tree) or an explicitly acyclic DAG; a cycle would make a recursive traversal non-terminating; and the traversal depth is bounded by the height, not by the node count.

A recursive relationship stores a hierarchy as repeated self-references. Its cost has two components: the storage, which is linear in nodes plus edges, and the traversal, which is proportional to the number of nodes visited. For a tree the visited set for a subtree query is the subtree size, which can be as large as the node count; for a path-to-root query it is the depth.

The distinction that matters for notation is between a hierarchy that is a tree and one that is a DAG. Trees answer "all descendants" with a simple recursive query and a natural cycle-free guarantee. DAGs allow a component to belong to several assemblies, which is often the real business rule, and they make both the traversal and the correctness argument harder because a node can be reached by multiple paths. A model that draws a tree for a DAG will undercount roll-up costs, because a shared component is charged once instead of once per path.

```text
FORMULA [MN-11]: path depth h = number of edges on the longest root-to-leaf
                 path;  subtree query visits |subtree(v)| nodes;
                 DAG multi-path reachability can yield up to
                 (number of distinct paths from v to u) charges for one pair.
WHERE: h is the tree height [edges]; |subtree(v)| is the node count below v;
       a path is a distinct edge sequence, not a distinct node sequence.
CONSTRAINT: the edge set is acyclic; a cycle check is required before a
            recursive query is declared terminating.
NUMERICAL EXAMPLE: 48,000 products with an average of 3.5 lines per order
       implies 168,000 line rows, but says nothing about hierarchy depth.
       For a 5-level bill of materials, a root query visits every descendant
       once per path; if one component is shared by 3 subassemblies, its cost
       is charged 3 times, so a roll-up that assumed a tree understates by
       2 charges for that component.
INTERPRETATION: state tree or DAG explicitly; a tree mark on a DAG model is a
       silent undercount in every cost roll-up that traverses it.
```

### 4.4 Limits and counterexamples

The formulas above are all true under their stated constraints. This subsection states the configurations where they produce confident wrong answers, because that is what a practitioner is actually paid to avoid.

#### 4.4.1 The lossless-join counterexample: why M:N cannot be flattened

**Validity before use:** a lossless join decomposition requires that the common attributes of the two relations form a superkey of at least one of them. This is the standard test, and it is the reason a many-to-many relationship cannot be decomposed into two relations without a bridge.

Consider facts `F(line_id, amount)` and tags `T(line_id, tag)`. Their common attribute is `line_id`, which is a key of `F` but not of `T` (a line has many tags). The join is therefore lossless in one direction only: joining `F` to `T` on `line_id` reproduces `T` exactly and does not invent pairs, but it duplicates `F`'s measure once per tag. If instead we tried to store the relationship as a repeated column in `F`, we would violate first normal form. If we tried to store it as two relations whose common attribute is a non-key on both sides, the join would create spurious pairs — the classic lossy decomposition.

The counterexample to remember is the one where the join looks harmless. Joining `F` to `T` does not lose rows and does not create invalid pairs; every output row is a true statement ("this line has this tag"). The error is not in the join's correctness but in the grain of the result: the output grain is `(line_id, tag)`, and a measure from `F` is not additive at that grain. Notation must therefore mark the grain of the output, not only the connectivity of the inputs.

```text
Counterexample [MN-04 limit]: |F| = 168,000; |T| = 210,000; join on line_id.
  Every output row is a true (line, tag) statement. Row count = 210,000.
  Sum of amount = 8,820,000.00 against a true line total of 7,056,000.00.
  No row is wrong; every total is wrong. The defect is grain, not data.
```

#### 4.4.2 The closure-convention counterexample

**Validity before use:** the counterexample applies to any version table whose intervals are stored as inclusive endpoint pairs without a tie-break rule.

Take the 8,400 version rows from Section 4.3.1 with 5,400 adjacent boundaries. If the model stores `valid_from` and `valid_to` as inclusive dates and the query uses `BETWEEN`, each boundary date matches two versions. The as-of query then returns duplicate attribute rows and the downstream join doubles its measure. The symptom is identical to a bridge fan-out, and the discriminating check is whether the duplication is at the fact grain or at the attribute grain.

The fix is not to add a tie-break to the query; it is to change the stored convention to half-open and re-derive `valid_to` as the next version's `valid_from`. A tie-break in the query is a per-query patch that every future query must remember, which is precisely the kind of knowledge that does not survive a team change.

```text
Counterexample [MN-03 limit]: 5,400 adjacent boundaries, inclusive endpoints,
  BETWEEN predicate. Each boundary date yields 2 matching versions.
  As-of row count = 8,400 + 5,400 = 13,800 (one duplicate per boundary).
  A fact join on such a dimension returns 13,800 rows for 8,400 facts.
```

#### 4.4.3 The independence counterexample for composed multiplicity

**Validity before use:** the product rule `α₁α₂` requires zero covariance, which is not a property you can assume from the fact that the two bridges are "different."

Section 4.2.2 already gave the identity. The counterexample here is the case where the covariance is structurally positive and large: two bridges that both count members of a shared parent. For instance, promotions per line and shipments per line are both larger for large orders, so the two degrees are positively correlated by construction. Multiplying the two averages then understates the joined inflation systematically, and the understatement grows with order-size dispersion. The bounded check is to compute `α` on the joined result and compare it to the product; if the gap exceeds the tolerance for the decision, the estimate is not usable.

#### 4.4.4 The natural-key-reuse counterexample

**Validity before use:** surrogate keys are stable; natural keys are stable only within their declared domain and lifetime. A model that treats a natural key as permanent assumes the business never reuses it.

Business identifiers get reused: account numbers are reassigned, SKUs are recycled across seasons, store codes are reissued after closure, and employee numbers are reused by some organisations after a statutory gap. When a natural key is reused, historical facts that reference it are silently re-attributed to the new entity. This is worse than a missing value because it produces a plausible, wrong answer that no null check can detect.

The modelling response is to carry both keys: a surrogate that never changes, and the natural key with a validity interval so that reuse is representable. The check is a query for natural keys whose version intervals are non-contiguous, which is exactly the signature of reuse.

```text
Counterexample [MN-01 limit]: store code S-1042 closed 2026-03-31 and
  reopened 2026-09-01 for a different location. A fact from 2026-06 (no
  store) and a fact from 2026-10 (new store) both reference S-1042.
  Without a validity interval the 2026-06 attribution is impossible to
  distinguish from a data error, and the new store inherits old sales.
```

#### 4.4.5 What no notation can express

The final limit is the honest one. Notation can express existence, uniqueness, cardinality, optionality, typing, and temporal containment. It cannot express that a field means what its name suggests, that a business process is followed, or that a consumer's interpretation matches the producer's intent. Those are semantic agreements, and they are enforced by review, contracts, and tests rather than by marks.

This matters because it bounds the ambition of the sheet. A perfect diagram with no semantic review is a false assurance, and the failure mode it produces — everyone agrees the model is correct, and everyone is wrong in the same way — is harder to detect than a diagram with visible gaps. The practical instruction is to keep the marks modest and the written definitions explicit, and to test the definitions against data rather than against the drawing.

**Takeaway:** Half-open intervals, covariance-aware multiplicity, and a grain label on every measure are the three marks that prevent the majority of arithmetic errors; everything else in notation is expressiveness, and expressiveness without a test changes nothing.

## 5. Worked Numerical Example

### 5.0 Scenario and evidence label

**Evidence:** ILLUSTRATIVE. The schema below is a constructed teaching model for a retail order domain. It is not a real company's schema, no vendor engine was used, and no production measurement is claimed. **Arithmetic:** VERIFIED — every figure in this section was computed twice, once by hand from the stated inputs and once by an independent recomputation, and the two agree. **Validity:** the example assumes unique fact keys, non-null bridge foreign keys, half-open dimension intervals, and a single time basis.

The example is built so that the reader must exercise all four base formulas from Section 4 and must encounter one instance of each counterexample. The inputs are deliberately round so that the arithmetic is checkable in a review meeting without a calculator.

### 5.1 The business sentence

> "Customers place orders; an order contains order lines; each line refers to one product; lines may be tagged with promotions; a customer's tier changes over time; and managers want revenue by product for any past month."

This sentence is typical: short, plausible, and containing at least five unresolved modelling decisions. The walkthrough resolves them in the order Section 2 prescribes.

### 5.2 Step 1 — Entity identification and identity statements

Apply the three identity tests from Section 3.3 to each noun.

| Noun | Independent lifecycle | Independent reference | Independent attributes | Verdict |
|---|---|---|---|---|
| Customer | yes | yes (orders, invoices, tier history) | yes (tier, region) | entity |
| Order | yes | yes (shipments, payments) | yes (order date, channel) | entity |
| Order line | yes (editable after order) | yes (returns, tags) | yes (qty, unit price) | entity |
| Product | yes | yes (lines, catalogue, hierarchy) | yes (name, category, list price) | entity |
| Promotion tag | yes | yes (applied to many lines) | yes (code, campaign, weight) | entity |
| Tier | no (only exists per customer-version) | no | no (it is the value) | attribute |
| Order status text | no | no | no | attribute |
| Shipping address | depends | no (only via order) | no | attribute or value object |

The output of this step is a set of identity statements, each of which must be a single sentence with a countable noun and a key.

| Entity | Identity statement | Natural key | Candidate-key check |
|---|---|---|---|
| Customer | One row per customer account | `(customer_id)` | minimal: no proper subset exists |
| Order | One row per placed order | `(order_id)` | minimal |
| Order line | One row per line within an order | `(order_id, line_no)` | `order_id` alone repeats, so the pair is minimal |
| Product | One row per sellable product | `(product_id)` | minimal |
| Promotion tag | One row per promotion code | `(promo_code)` | minimal |
| Tag assignment | One row per tag applied to a line | `(order_id, line_no, promo_code)` | pair is not unique when a line has two tags |

The last row is where the model would have failed in version 1.0 of this sheet. If the tag assignment were modelled as a column list on `order_line`, the second tag would either overwrite the first or be stored in a delimited string, and both choices destroy the ability to aggregate by tag. Applying the identity test to the relationship — not just to the nouns — is what produces the bridge.

### 5.3 Step 2 — Cardinality derivation and count arithmetic

The business supplies totals rather than per-row counts, which is the normal situation. Derive the per-row counts from the totals, then check that the derived counts are consistent with the stated cardinalities.

Given: 3,000 customers, 24,000 orders, 168,000 lines, 48,000 products, 210,000 tag assignments.

| Quantity | Arithmetic | Result | Unit |
|---|---|---|---|
| Lines per order | 168,000 / 24,000 | 7.0 | lines/order |
| Orders per customer | 24,000 / 3,000 | 8.0 | orders/customer |
| Lines per customer | 168,000 / 3,000 | 56.0 | lines/customer |
| Lines per product | 168,000 / 48,000 | 3.5 | lines/product |
| Tags per line | 210,000 / 168,000 | 1.25 | tags/line |

Each count maps to a cardinality in both directions. `lines per order = 7` is the average of the `1..*` side, and it does not bound the maximum; a model that declares `0..7` would be wrong unless the business actually caps lines at seven. This is the single most common cardinality error: converting an average into a maximum.

The declarations that follow from the sentence, with the enforcement column showing what is achievable:

| Edge | Read as | Declared `(min,max)` | Enforcement |
|---|---|---|---|
| Customer → Order | one customer places many orders | `(0..*, 1..1)` | FK on order; 0..* is unenforced |
| Order → Line | one order contains many lines | `(1..*, 1..1)` | FK on line; minimum 1 needs a transaction rule |
| Line → Product | one product appears on many lines | `(0..*, 1..1)` | FK on line; product minimum is unenforced |
| Line ↔ Tag | many lines carry many tags | `(*..*, *..*)` | bridge with composite key |
| Customer → Tier version | one customer has many versions | `(1..*, 1..1)` | FK on version; contiguity needs an exclusion rule |

### 5.4 Step 3 — Bridge multiplicity, computed before the join

The tag bridge holds 210,000 rows over 168,000 lines. Applying formula MN-04:

`α = 210,000 / 168,000 = 1.25`.

The line-level money measure is 42.00 currency per line. The true line total is the sum at the line grain:

`168,000 × 42.00 = 7,056,000.00`.

The total obtained by joining lines to tags and summing is at the pair grain:

`210,000 × 42.00 = 8,820,000.00`.

The inflation is `8,820,000.00 − 7,056,000.00 = 1,764,000.00`, which is `1,764,000 / 7,056,000 = 0.25`, exactly 25%.

Three independent checks confirm this is arithmetic and not coincidence. First, the ratio of the two totals must equal `α`: `8,820,000 / 7,056,000 = 1.25`. Second, the inflation as a fraction must equal `α − 1`: `1.25 − 1 = 0.25`. Third, the per-unit view: a tagged line with two tags contributes `2 × 42.00 = 84.00` instead of `42.00`, and the average contribution is `1.25 × 42.00 = 52.50`, so `168,000 × 52.50 = 8,820,000.00`. All three agree.

### 5.5 Step 4 — Add the loss, and watch the errors interact

Now suppose 2,400 of the 168,000 lines carry no tag at all. An inner join drops them. Applying formula MN-06:

`|S| = 168,000 − 2,400 = 165,600`.

`s = 165,600 / 168,000 = 0.985714285...`

With 207,000 tag rows over the surviving lines, `α_S = 207,000 / 165,600 = 1.25`.

Output rows `= s · α_S · |F| = 0.985714... × 1.25 × 168,000 = 207,000`, which matches `|B|` as the identity requires.

The measure after the join is `207,000 × 42.00 = 8,694,000.00`. Against the true `7,056,000.00`, the error factor is `8,694,000 / 7,056,000 = 1.2321428...`, so the reported total is 23.21% high.

Now examine the row-count control. Expected rows under a naive assumption of one row per line is 168,000. Observed is 207,000. The observed-to-expected ratio is `207,000 / 168,000 = 1.2321`, which is *the same as the measure error factor*. This is not a coincidence: both equal `s · α_S`. The consequence is that a row-count reconciliation against "expected 168,000" detects that something is wrong, but it cannot tell you whether the cause is fan-out, loss, or both, because both scale the same quantity.

To separate them you need a check that is sensitive to one effect only. The key-level anti-join is sensitive to loss: `count(distinct line_id in F) − count(distinct line_id in F⋈B) = 168,000 − 165,600 = 2,400`. The degree histogram is sensitive to fan-out: `count(distinct line_id) = 165,600` and `count(*) = 207,000` gives `α_S = 1.25`. Running both turns one ambiguous symptom into two definitive diagnoses.

| Check | Query shape | Expected clean | Observed | Diagnosis |
|---|---|---|---|---|
| Row count | `count(*)` | 168,000 | 207,000 | anomaly, cause unspecified |
| Key survival | anti-join on `line_id` | 0 missing | 2,400 missing | loss confirmed |
| Degree | `count(*)/count(distinct key)` | 1.00 | 1.25 | fan-out confirmed |
| Measure | independent control total | 7,056,000.00 | 8,694,000.00 | +23.21% |

### 5.6 Step 5 — Temporal versioning applied to the same model

The customer tier changes over time. Using the distribution from Section 4.3.1: 3,000 customers, of whom 1,200 never change tier, 1,200 change twice, and 600 change five times.

`Σc = 1,200×2 + 600×5 = 2,400 + 3,000 = 5,400`.

`N = 3,000 + 5,400 = 8,400` version rows, and `v̄ = 8,400 / 3,000 = 2.8` versions per customer.

At 320 bytes per version row, storage is `8,400 × 320 = 2,688,000` bytes `= 2,688,000 / 1,048,576 = 2.5635` MiB. If the model were extended to bitemporal with up to four revisions per history, the bound is `8,400 × 4 = 33,600` rows and `33,600 × 320 = 10,752,000` bytes `= 10.2539` MiB.

Now the as-of join. Each of the 168,000 lines carries an order timestamp, and the model wants tier at order time. The correct predicate is MN-03. To verify it, construct three test cases around a boundary.

| Case | Customer | Tier interval | Fact instant | Expected match | Reason |
|---|---|---|---|---|---|
| T1 | C1 | Silver `[2026-01-01, 2026-02-01)` | 2026-01-31 23:59 | Silver | `t < to` |
| T2 | C1 | Gold `[2026-02-01, ∞)` | 2026-02-01 00:00 | Gold | `from ≤ t` |
| T3 | C1 | (gap) `[2026-03-01, 2026-04-01)` | 2026-02-15 12:00 | none | gap between versions |

Case T3 is the one that catches a broken load. A model whose version loader failed to insert the February row has no interval covering 15 February, and the as-of join returns zero rows for those facts. If the join is a left join, the tier becomes null and the downstream aggregate silently excludes the facts; if it is an inner join, the facts vanish. Both outcomes are wrong in the same direction, and both are invisible unless the model asserts "exactly one match for every fact in the declared domain."

The verification query for that assertion is a three-part test: total facts must equal the sum of match counts, the maximum match count must be 1, and the minimum must be 1. On the 168,000 lines with a clean model, all three hold. With one duplicate boundary, the sum exceeds 168,000. With one gap, the sum is less.

### 5.7 Step 6 — Full reconciliation and the decision

Combine the checks into one reconciliation table, because a model is only trustworthy when its controls agree in the same run.

| Control | Formula | Computed | Pass condition | Status |
|---|---|---|---|---|
| Identity uniqueness | `count(*) − count(distinct key)` | 0 | must be 0 | pass |
| Cardinality sanity | lines/order | 7.0 | within declared max | pass (max unstated) |
| Bridge fan-out | `α` | 1.25 | declared and accepted | disclosed |
| Key survival | anti-join | 2,400 | must be 0 for line totals | fail for revenue |
| Measure | `α_S · s · M` | 8,694,000.00 | equals 7,056,000.00 | fail |
| Version rows | `n + Σc` | 8,400 | matches storage plan | pass |
| As-of exactly-one | min/max match count | 1 / 1 | both equal 1 | pass |

The decision that follows is not "fix the query." The query is correct for the question "how many tags were applied," and it is incorrect for "what was line revenue." The model-level fix is to declare the grain of `line_amount` as the line grain, so that any aggregate over a joined result must either pre-aggregate lines or carry `DISTINCT line_id`. The review consequence is that the revenue metric is blocked until the join is rewritten, and the tag metric is released because its grain matches.

This is what notation buys. A diagram with a bridge and a grain label on `line_amount` makes the error a specification violation that a reviewer can name. A diagram without the grain label makes it a query bug that is discovered when someone compares two dashboards.

### 5.8 Sensitivity: which input changes the conclusion

The conclusion "revenue is overstated by 23.21%" depends on `s`, `α_S`, and the measure. Vary each and observe which one can reverse the release decision while leaving the model unchanged.

| Variation | New value | New reported total | Change in conclusion |
|---|---|---|---|
| No untagged lines (`s = 1`) | `α_S = 1.25` | 8,820,000.00 | worse: +25.0% |
| Untagged lines also tagged (`α_S = 1`) | `s = 1` | 7,056,000.00 | error vanishes |
| Both (no bridge) | `α = 1`, `s = 1` | 7,056,000.00 | model correct by construction |
| 3,000 missing lines | `s = 0.982143` | 8,694,000 (row count) | direction unchanged |
| Tags per line 2.0 | `α_S = 2.0` | 14,112,000.00 | +100% |

The second row is the important one: a fan-out with compensating loss can produce an exactly correct total, which is the concealment case from Section 4.2.3. Here the coincidence requires `s · α_S = 1`, i.e. `α_S = 1/s`. At `s = 0.985714`, that means `α_S = 1.014493`. A bridge with that average degree would make the revenue total correct to within rounding while the population was 1.43% wrong. Any control that only compares totals would pass. This is why the reconciliation table above has five rows and not one.

**Takeaway:** The worked example's value is not the 23.21% figure; it is that four independent controls were needed to explain a single wrong total, and that a compensating pair of errors can make the total look correct.

## 6. Visual / Geometric Intuition

### 6.1 Cardinality is a shape you can see

The reason diagrams work at all is that cardinality has a spatial signature. A one-to-many edge is a tree: one parent above, several children below, no child shared. A many-to-many edge is a diamond: two sets with a bridge between them, and a path from left to right that goes through two vertices rather than one. Once the reader sees the shape, the arithmetic in Section 4.2 becomes obvious, because the shape *is* the arithmetic.

```text
ONE-TO-MANY (a tree, no shared children)

      Customer
         |
    +----+----+----+
    |    |    |    |
   Ord1 Ord2 Ord3 Ord4        each order belongs to exactly one customer
                              join adds no rows to the order side

MANY-TO-MANY (a diamond, one extra vertex on every path)

   OrderLine                         Tag
   (168,000)                       (n tags)
       \                            /
        \                          /
         +---- TagAssignment -----+
              (210,000 pairs)
        /                          \
       /                            \

   Every path from a line to a tag passes through the bridge.
   alpha = |bridge| / |lines| = 210,000 / 168,000 = 1.25
```

The two shapes explain two different failure modes without any algebra. In the tree, a join cannot duplicate the parent because each child has exactly one parent reference; duplication requires a broken key, not a wrong join. In the diamond, duplication is structural: the number of paths is the bridge size, and any measure carried along a path is repeated once per path. The reader who internalizes the diamond predicts the 25% inflation before running a query.

### 6.2 The interval line: where closure lives

Temporal notation is a one-dimensional geometry, and the whole subject is which side of a boundary point belongs to which segment.

```text
Half-open intervals — each instant covered exactly once

   Jan 1            Feb 1            Mar 1            Apr 1
     |                |                |                |
     [---- Silver ----)                |                |
                      [----- Gold -----)                |
                                       [--- Platinum ---)
     ^                ^                ^
     |                |                |
   covered by       covered by       covered by
   Silver only      Gold only        Platinum only
   (t >= Jan1,      (t >= Feb1,      (t >= Mar1,
    t <  Feb1)       t <  Mar1)       t <  Apr1)

 Closed-closed intervals — boundaries covered twice

     [---- Silver ----]
                      [----- Gold -----]
                                       [--- Platinum ---]
                      ^                ^
                   Feb 1 matched     Mar 1 matched
                   by both           by both
```

Draw the same line for a gap and the third failure mode becomes visible: two intervals that do not touch leave an uncovered segment, and every fact in that segment matches nothing. A reviewer who can sketch the line can classify the three defects — overlap, gap, wrong closure — without running anything.

```text
A gap is a segment with no interval

     [---- Silver ----)        (gap)        [----- Gold -----)
     Jan 1           Feb 1   Feb 1..Mar 1   Mar 1          ...
                                ^
                                |
                        facts here match zero versions
                        (left join -> null tier; inner join -> lost facts)
```

### 6.3 The version ladder and its row count

Versioning has a natural picture too: each change inserts a rung on a ladder for one key, and the ladder's height is the key's change count plus one.

```text
One customer, changing twice -> three version rows

  row 1: [Jan 1 --------)  tier = Silver
  row 2:        [Feb 1 ----------)  tier = Gold
  row 3:                  [Apr 1 ------------>  tier = Platinum

  heights: 1 + c, where c = 2 changes -> 3 rows
  boundaries between rungs: 2 (Feb 1, Apr 1)
  total boundaries across 3,000 keys: 5,400
```

The picture makes the storage formula MN-07 self-evident: total rows equal the number of keys plus the number of changes, because every key contributes one base rung and every change contributes one more. It also makes the bitemporal bound visible: give each rung a second dimension for revisions and the ladder becomes a grid, with rows at the intersections rather than only along the diagonal.

### 6.4 Subtype as a partition of a set

Generalization is set geometry. The supertype is the full set; each subtype is a region; exclusivity says the regions do not overlap; completeness says they cover the whole set.

```text
EXCLUSIVE + TOTAL (a partition)          OVERLAPPING (regions share area)

  +---------------------------+            +---------------------------+
  |  +-------+  +-------+     |            |  +-------+  +-------+     |
  |  | kit   |  |bundle |     |            |  | kit   |  |bundle |     |
  |  +-------+  +-------+     |            |  +-------+--+-------+-+   |
  |      +-----------+        |            |      +-----------+   | |   |
  |      |  simple   |        |            |      |  simple   |   | |   |
  |      +-----------+        |            |      +-----------+---+ |   |
  +---------------------------+            +---------------------------+
   sum of memberships = |super|             sum of memberships > |super|
   4,800 = 3,000+1,200+600                  5,000 > 4,800
```

This picture converts a modelling argument into a one-line query. Counting memberships across subtypes and comparing to the supertype count decides exclusivity empirically, which is the only way to settle it when two stakeholders disagree about whether a product can be both a kit and a bundle.

### 6.5 The graph view: trees versus DAGs

Hierarchies are the place where the picture and the cost diverge most. A tree has one path between any two nodes; a DAG has several; and the cost of a roll-up depends on which one you have.

```text
TREE: one path per pair                DAG: several paths per pair

        A                                      A
        |                                     / \
        B                                    B   C
        |                                     \ /
        C                                      D

  paths A->C: 1                          paths A->D: 2 (A-B-D, A-C-D)
  charge C once                          charge D twice in a naive roll-up
```

The DAG picture explains why a roll-up that assumes a tree understates cost, and why the notation must mark the edge set as acyclic but not necessarily a forest. It also explains why recursive traversal has a termination obligation: a cycle in the picture turns a finite traversal into an infinite one, so the model needs a cycle check before any recursive query is declared safe.

### 6.6 The notation-view lattice

Finally, it helps to see notation families as points in an expressiveness lattice rather than as rivals. Each family is strong on one axis and weak on another, and no single family dominates.

```text
                    high semantic precision
                              ^
                              |
              IDEF1X          |        Chen ER
          (key discipline)    |     (relationship clarity)
                              |
    low <---------------------+---------------------> high
    visual                     |                    visual
    compactness                |                    richness
                              |
              Data Vault      |        UML class
          (history, hubs)      |     (types, behaviour)
                              |
              Crow's foot     |
          (analytics breadth) |
                              v
                    low semantic precision
```

Two practical readings follow. If the audience is an analytics team that must maintain forty tables, crow's foot plus a grain table is the best cost-benefit point: compact, familiar, and the missing semantics are written as tests. If the audience is a standards body or a regulated schema review, Chen or IDEF1X earns its density because the reader needs to check participation and key structure explicitly. Choosing by preference rather than by what the audience must verify is the failure this lattice exists to prevent.

**Takeaway:** Cardinality is a shape (tree or diamond), time is a line with covered and uncovered segments, and versioning is a ladder whose height is the change count — three pictures that let a reviewer predict the arithmetic error before writing a query.

## 7. Comparison / Selection Table

### 7.1 Choosing a notation family for a real review

The selection table below is organized by the claim the audience must verify, not by the diagram's appearance. Read the first column as "the reviewer's binding question," and each row's choice follows from whether the family can show that claim without supplementary text.

| Binding question | Choose | Why it fits | Disqualifier | Fallback |
|---|---|---|---|---|
| Which tables exist and how do analytics join them? | Crow's foot | Cardinality at the line ends; wide schemas stay readable | Recursive and subtype detail needs annotation | Crow's foot plus a written constraint list |
| Which attribute determines which, and what is the real key? | IDEF1X or a dependency list | Key structure is first-class, not implied | Non-technical audience finds it opaque | Armstrong closure table from §4.1.2 |
| Who participates in this relationship, and is participation mandatory? | Chen ER | Participation is explicit on both sides | Crowds quickly beyond ~15 entities | Chen for the core, crow's foot for the rest |
| What are the types, interfaces, and allowed operations? | UML class | Types and multiplicity are native | Implies runtime behaviour a warehouse lacks | UML for the interface, relational for storage |
| What is the measurement grain and which dimensions conform? | Dimensional (star/snowflake) | Grain and conformed dimensions are the point | Hides transactional relationships | Star for the presentation layer, normalized for the source layer |
| What is the raw historical record of integrations? | Data Vault | Insert-only hubs, links, satellites; auditable | Deliberately non-semantic; needs a presentation layer | Data Vault raw plus a star presentation |
| Which paths exist and how deep do they go? | Property graph | Traversal is native; path semantics explicit | Weak declarative cardinality constraints | Graph for traversal, relational for the canonical store |
| What exactly will the engine enforce? | Physical DDL | Constraints are the ground truth | Engine-specific; intent is implicit | DDL plus a comment naming each business rule |

The first row is the default answer for most analytics organisations, and the reason is not that crow's foot is better but that the binding question for an analytics team is breadth of join path, and crow's foot is the most compact way to show breadth. The second row matters when the dispute is about identity, because that is the only row where a dependency notation actually settles an argument that a cardinailty notation merely records.

### 7.2 Choosing an identity strategy

Identity strategy is a separate decision from diagram family, and it is the one that most often reverses silently during a migration.

| Strategy | Use when | Benefit | Cost | Failure it invites |
|---|---|---|---|---|
| Natural key only | The business guarantees non-reuse and immutability | No join needed to resolve identity; readable | Every attribute change that touches the key cascades | Reuse re-attributes history |
| Surrogate only | The business key is unstable or composite | Stable references; compact joins | Identity must be resolved by lookup; duplicates enter easily | Two surrogates for one business entity |
| Surrogate plus natural unique | Production analytical models | Both stability and business-level uniqueness | Slightly wider tables; two indexes | Natural-key drift breaks the unique test |
| Composite natural key | Genuine dependent entity | Identity is the business statement | Wide foreign keys everywhere | Non-minimal key grows over time |
| Hash of natural key | Distributing or merging sources | Deterministic and join-free | Collision policy; rehash on key change | Hash computed over a mutated key |

The last row deserves a warning that is specific to lakehouse practice. A hash of the natural key is a surrogate with the useful property that it can be recomputed independently in two systems, which is why it is popular for merges. Its weakness is that any change to the key definition — adding a source-system prefix, normalizing case, trimming whitespace — silently produces a new identity for every existing entity, and the resulting duplicates are indistinguishable from genuine new entities. The bounded fix is to version the hash input and to keep the raw natural key alongside the hash so that duplicates can be re-resolved.

### 7.3 Choosing a temporal representation

Temporal representation is where the cost difference between options is largest, and where teams most often accept a representation that cannot answer the question they will be asked in six months.

| Representation | Answers | Cannot answer | Storage multiple | Reversal risk |
|---|---|---|---|---|
| Current state only | "What is true now?" | Any historical question | 1.0× | High — history is gone |
| Type-2 valid time | "What was true at instant t?" | "What did we believe at instant t?" | ~`1 + Σc/n` | Low for valid time |
| Type-2 plus `current_flag` | Current and historical, faster current reads | Corrections still overwrite belief | ~`1 + Σc/n` plus flag maintenance | Medium — flag drift |
| Bitemporal | Valid and belief, both as of any instant | Nothing temporal; it is the full record | up to `v̄ · r̄` | Low |
| Event-sourced | Full causal history; rebuild any state | Cheap direct reads without a projection | depends on event volume | Low, but projection cost |
| Snapshot per period | Point-in-time comparisons | Intra-period changes | linear in periods | Medium — period boundary choice |

The `current_flag` row is the most common compromise and the one that produces the most subtle bugs. The flag is derived from the intervals, so it is redundant state, and redundant state drifts. The failure signature is a key with two current rows or none, which is exactly the same signature as a broken interval set — and the discriminating check is to compare the flag against the interval predicate. If they disagree, the flag is wrong; if they agree and the count is still not one, the intervals are wrong.

### 7.4 Choosing how to represent a many-to-many

| Option | Use when | Disadvantage | Verify with |
|---|---|---|---|
| Bridge table with composite key | The relationship set is stable and pair-level attributes are minimal | One join per path | `count(*) = count(distinct (left,right))` |
| Bridge with version or validity | Membership changes over time | Wider key; as-of predicate on every query | exactly-one as-of match per fact |
| Bridge with surrogate key | Memberships are edited frequently | Business uniqueness needs a separate test | unique on `(left,right,version)` |
| Array or delimited column | The set is small and only ever read whole | Cannot join, index, or aggregate per member | none — this is the anti-pattern |
| Two mirrored columns | The relationship is genuinely one-to-one in practice | Breaks as soon as it is not | uniqueness in both directions |
| Graph edge | Traversal is the dominant access pattern | Constraint enforcement is weaker | edge endpoint existence |

The anti-pattern in row four is worth naming precisely because it appears in lakehouse schemas where a JSON array feels natural. An array of promotion codes on an order line can be read whole, but it cannot be joined to the promotion dimension without exploding it, and it cannot be constrained to valid codes at the member level. The moment any report asks "revenue by promotion," the array must be unpacked, and the unpacking is a fan-out that the model never declared.

### 7.5 Decision guide: which defect to fix first

When a model fails several checks at once, repair order matters because some defects mask others.

| Condition | Priority | Action | Why first |
|---|---|---|---|
| Identity ambiguous between two nouns | 1 | Settle the definition and the key | Every downstream count is undefined |
| Grain unlabeled on a measure | 2 | Label the grain; fix aggregates | Prevents the fan-out error at source |
| Interval overlap or gap | 3 | Rebuild intervals half-open; exclusion test | Duplicated or lost facts at boundaries |
| Cardinality overstated as a maximum | 4 | Replace average with declared bound | Avoids a false rejection at peak |
| Optionality unenforced | 5 | Add the deferred check or record the risk | Completeness claims are otherwise unsupported |
| Notation family mismatch | 6 | Switch family or add the written constraint list | Cosmetic if 1–5 are unresolved |

Priority 6 last is deliberate. Teams frequently respond to a modeling review by redrawing in a different notation, which changes nothing that the tests check. The tests in the table are ordered by how many other checks depend on them, and notation style depends on none of them.

**Takeaway:** Choose the family by the claim the reviewer must verify, choose identity by whether the business guarantees key stability, and repair identity and grain before style — in that order, because style has no downstream dependents.

## 8. Failure Modes & Edge Cases

### 8.0 How these entries are structured

Each entry below has the same five-part anatomy, because a failure catalogue is only useful if a reader can act on it under time pressure. The **symptom** is what is observable, stated in the terms a person would actually notice. The **competing causes** are at least two mechanisms that produce that same symptom, because a single-cause entry teaches pattern matching rather than diagnosis. The **discriminating check** is a query or inspection whose outcome differs across the causes. The **bounded fix** is the smallest action that restores correctness without a redesign. The **verification** is the evidence that the fix worked, and it must be a check that would have failed before.

The entries are ordered by the cost of the error, not by how often it occurs. An identity ambiguity is first because it invalidates every downstream control; a notation-style mismatch is last because it invalidates nothing that a test checks.

### 8.1 F-01 — The same noun has two identity boundaries

**Symptom.** Two dashboards report customer counts that differ by a factor of 3 to 10, and both teams defend their number. Cohort retention computed on one is not comparable to the other. A "customer" count from the order system exceeds the count from the CRM.

**Competing causes.** (a) The order system counts bill-to accounts while the CRM counts legal organisations, and one organisation has several accounts. (b) The order system counts guest checkout identities that the CRM has deduplicated to a person. (c) One system includes soft-deleted records. (d) A join to a bridge inflated one side, which is the Section 4.2 case rather than an identity case.

**Discriminating check.** Take twenty records from each system that purport to describe the same customer and align them by name, tax identifier, and address. If the alignment is many-to-one consistently, cause (a) or (b). If the counts reconcile at the distinct-key level, cause (d). If the difference is exactly the soft-deleted count, cause (c). The decisive query is `count(*)` versus `count(distinct <candidate identity>)` in each system, which separates a genuine identity difference from a duplication.

**Bounded fix.** Do not choose a winner by seniority. Declare both entities with distinct names — `billing_account` and `legal_organisation` — add the relationship between them with its cardinality, and publish a mapping with a stated coverage. Any metric that previously said "customer" must be renamed to name its entity.

**Verification.** Recompute the disputed metric on both definitions and confirm it now reproduces each team's number exactly. Then confirm the mapping table's cardinality: every billing account maps to exactly one organisation, and the count of organisations with zero accounts is reported rather than hidden. A mapping that leaves orphans is incomplete, not clean.

**Why this is first.** Every other control in this sheet presumes identity. A uniqueness test on the wrong identity passes perfectly and is worthless.

### 8.2 F-02 — A row-count check passes while the measure is wrong by 23 percent

**Symptom.** The joined revenue figure does not match the finance ledger, but the joined row count is close to the expected line count, so the pipeline is considered healthy. The discrepancy is stable across periods and proportional to revenue.

**Competing causes.** (a) A bridge join duplicated measures and a filter dropped an almost equal number of rows. (b) The ledger includes tax or freight that the warehouse does not. (c) A currency conversion difference. (d) A dimension join duplicated because the as-of predicate over-matched.

**Discriminating check.** Run three checks in one query batch: the anti-join count (sensitive to loss), the degree ratio `count(*)/count(distinct fact_key)` (sensitive to fan-out), and the ratio against an independently produced control total (sensitive to definitional difference). If degrees are greater than 1 and the anti-join is nonzero, cause (a). If degrees are exactly 1 and the anti-join is 0, the remaining candidates are definitional, and a line-item comparison against the ledger settles (b) or (c). If degrees are exactly 1 but the as-of match count exceeds 1, cause (d) and Section 8.4 applies.

**Bounded fix.** For cause (a), pre-aggregate the fact measure at its declared grain before joining to the bridge, or aggregate with `DISTINCT` on the fact key, and record the declared grain in the model. For (b), reconcile the measure definition by naming the included components rather than adjusting the query.

**Verification.** The reconciled total must equal the control total at the same cutoff, and the degree ratio must be exactly 1.0 on the repaired query. Both conditions are required: a repaired total with a degree of 1.25 means the repair worked by coincidence, which is the concealment case and will break in the next period.

### 8.3 F-03 — Optionality was assumed, and the orphans are legitimate lifecycle states

**Symptom.** A referential integrity check reports thousands of child rows whose parent key does not exist. The team's first instinct is corrupt data.

**Competing causes.** (a) Genuine orphan rows from a load ordering bug, where the parent arrives after the child. (b) Legitimate lifecycle states: the child was created for an entity whose parent row is intentionally absent, such as a prospect order with no customer record. (c) A parent key that was changed without cascading. (d) A comparison across different snapshots of the parent.

**Discriminating check.** Examine the orphan set's age distribution and its event-time ordering. Load-order orphans cluster at partition boundaries and vanish after a full reload. Lifecycle orphans persist and have a distinct status or channel value. Key-change orphans pair one-to-one with a deleted parent key. Snapshot orphans disappear when both sides are restricted to the same cutoff. The single most informative query is to group orphans by the child's status field: if they all share one status, the model lacked a lifecycle state, not a constraint.

**Bounded fix.** For (a), make the load idempotent and order-independent, or introduce a late-arriving-parent window with a documented bound. For (b), model the lifecycle explicitly — either a nullable parent with a stated meaning, or a distinct child subtype for the pre-parent state — and record the optionality rather than pretending it is mandatory.

**Verification.** The orphan check returns zero for genuine corruption and returns exactly the declared lifecycle population for optional cases. The second half matters: a check that always returns zero because it was weakened is not a control.

### 8.4 F-04 — As-of joins return two attribute rows for one fact

**Symptom.** A fact-to-dimension join increases the row count by a small, suspiciously round amount, and the duplicated rows have different attribute values. Revenue by region shows a segment appearing twice in a period.

**Competing causes.** (a) Intervals are stored closed-closed and the query uses `BETWEEN`. (b) A genuine overlap in the version data from a double load. (c) A missing version so the fact matches the wrong neighbour. (d) The dimension has both a current-state row and version rows, and the query does not filter to the version set. (e) Duplicate dimension rows from a join in the dimension's own build.

**Discriminating check.** Count matches per fact key and examine the boundary instants. If duplicates occur only at change instants, cause (a). If duplicates occur at arbitrary instants, cause (b) or (e). If matches are zero for a contiguous range, cause (c). If duplicates occur for every fact regardless of instant, cause (d). The exact test is `select fact_key, count(*) ... group by fact_key having count(*) > 1`, followed by inspecting whether the duplicate pairs share a boundary instant.

**Bounded fix.** For (a), rewrite `valid_to` as the next version's `valid_from` and change every predicate to `>= from and < to`. For (b), add an exclusion constraint over the key and interval so the overlap cannot be inserted again, then repair the overlapping rows by re-deriving them from the change log. For (c), backfill the missing interval from the change log rather than interpolating.

**Verification.** Run the as-of assertion: total facts equals the sum of matches, `min(matches) = 1`, and `max(matches) = 1`. Then re-run it on a synthetic fixture with a change exactly at midnight, because a boundary bug that is absent in daytime test data is present in production. Section 5.6 case T2 is that fixture.

### 8.5 F-05 — Cardinality was recorded as an average and became a hard limit

**Symptom.** A validation rule rejects a legitimate order with 250 lines because the model declared a maximum of 200, which was in fact the average line count times a safety factor.

**Competing causes.** (a) A genuine business cap exists and the order is a policy exception. (b) The declared maximum came from an observed average, not from a rule. (c) The order is the result of a merge or a bulk import that should be a different entity. (d) The line count is inflated because the lines were joined to something.

**Discriminating check.** Ask for the rule's source. If no policy document states a cap, cause (b). If a policy exists, check whether exceptions are allowed and who approves them. To separate (d), count distinct line keys rather than rows. The check that actually distinguishes (b) from (a) is a historical distribution: if the 99.9th percentile is above the declared maximum and the system has been running, the maximum is empirical and was never a rule.

**Bounded fix.** Replace the empirical maximum with either the stated policy cap plus an exception path, or an unbounded declaration with a monitoring threshold that alarms rather than rejects. Never silently raise the limit — a raised limit with no recorded source repeats the error.

**Verification.** Replay the rejected order and confirm acceptance; then confirm the monitoring threshold still fires on a synthetic order at the historical maximum plus one. The pair of checks proves the fix accepts what it should and still observes what it should.

### 8.6 F-06 — A many-to-many was flattened into a delimited column and history was lost

**Symptom.** A tag or category report cannot be filtered by member, and when someone unpacks the column with a string split, the resulting counts do not reconcile to any known total. Editing tags overwrites the previous set with no record of what changed.

**Competing causes.** (a) The column is genuinely a delimited list and the unpacking is correct but the denominator is wrong. (b) The delimited values contain the delimiter inside a value. (c) The column is a JSON array and the split function is being applied to the serialized text. (d) Different rows use different delimiters.

**Discriminating check.** Profile the value lengths and delimiter counts. If the delimiter count per row varies and rows with embedded delimiters exist, cause (b). If values look like `["a","b"]`, cause (c). If two delimiters appear in the population, cause (d). Comparing the split count to the stated business expectation distinguishes (a).

**Bounded fix.** Normalize to a bridge table with the pair as its key, backfilling from the existing column with a documented parse, and quarantine rows that do not parse cleanly rather than dropping the malformed members. Keep the original column until the bridge's reconciliation passes, then retire it — the contract sheet's dual-run rule applies.

**Verification.** The bridge row count equals the sum of parsed members across rows, with the quarantine count reported separately. Then confirm a member-level aggregate (revenue by tag) reconciles to a manually computed sample. After retirement, the bridge must be the only source, and a check should fail if the old column reappears.

### 8.7 F-07 — Subtype exclusivity was assumed and a product belongs to two subtypes

**Symptom.** A roll-up that sums subtype-level metrics exceeds the supertype total by a few percent.

**Competing causes.** (a) The subtypes genuinely overlap and the business allows it. (b) A product was migrated and left in both subtype tables. (c) A subtype table contains a duplicate row. (d) The roll-up double-counts because it joins the supertype to two subtype tables without a union.

**Discriminating check.** Compute the sum of memberships and compare to the supertype count. Equal means exclusive. Greater means overlapping — then decide whether it is (a) or (b) by checking whether the dual membership has a migration timestamp. Cause (d) is distinguished because it produces duplicates only in the roll-up, not in the membership tables.

**Bounded fix.** For (a), declare overlapping subtypes and change the roll-up to a union with a rule that prevents double-charging shared products. For (b), remove the stale membership in the subtype that the migration superseded, keeping an audit row.

**Verification.** Membership sum equals the supertype count for exclusive designs, or the overlap population is enumerated and priced for overlapping designs. In both cases the roll-up must reconcile to a control total computed from the supertype alone.

### 8.8 F-08 — A hash-derived key changed silently

**Symptom.** The entity count jumps after a pipeline release, and the new entities have attributes identical to existing ones.

**Competing causes.** (a) The hash input changed (case normalization, whitespace trim, added source prefix). (b) A genuine influx of new entities. (c) The hash function or salt changed. (d) The natural key itself changed upstream.

**Discriminating check.** Join the new entities to the old on the attributes other than the key. If they pair one-to-one, cause (a) or (c). Compare the count of distinct natural keys before and after: unchanged means the input or hash changed, not the population. Cause (d) shows up as a partial shift affecting one source system.

**Bounded fix.** Version the hash input explicitly (a column that records the recipe version), keep the raw natural key so duplicates can be re-resolved, and re-derive the affected keys with a mapping rather than deleting and reloading.

**Verification.** The natural-key distinct count is unchanged across the release, and the duplicate resolution query returns zero unresolved pairs. Both checks together separate a re-keying from a population change; either alone is ambiguous.

**Takeaway:** Eight entries, eight distinct mechanisms — but the same shape: an observable that at least two mechanisms produce, a check whose outcome differs across them, and a verification that would have failed before the repair.

## 9. Complexity & Cost Analysis

### 9.1 What is being counted, and in what units

Cost analysis for a model is about three quantities, and they are not interchangeable. **Rows** count stored or produced tuples. **Comparisons** count the elementary key or interval tests an algorithm performs, which is the quantity that predicts runtime rather than I/O. **Bytes** count storage, which is what a capacity plan buys. A model review that reports only one of the three is incomplete, because a design can be cheap in rows, expensive in comparisons, and moderate in bytes — a hash-joined version set is exactly that.

The units used throughout this section are explicit: rows are tuples of the stated relation; comparisons are key-equality or range tests; bytes are decimal bytes unless MiB is stated, with `1 MiB = 1,048,576 bytes`. All figures are ILLUSTRATIVE capacity arithmetic on the Section 5 scenario, computed from stated per-row sizes. They are not measurements of any engine, and they exclude page overhead, indexes, compression, replication, and concurrency.

### 9.2 Base costs of the model operations

| Operation | Time | Space | What drives it | Consequence |
|---|---|---|---|---|
| Uniqueness check on a key | `O(n log n)` sort or `O(n)` hash | `O(d)` for `d` distinct keys, or `O(1)` streaming if sorted | distinct key count, not row count | a small key space makes even a large table cheap to validate |
| Parent existence check | `O(n)` with indexed lookup per child | `O(1)` per probe | child rows times index probe cost | remote lookups dominate if the parent is not co-located |
| Interval overlap check | `O(n log n)` after sort per key | `O(n)` | adjacent pairs per key, `n − keys` | the check's cost equals the boundary count from §4.3.1 |
| As-of join | `O(n log n)` sort-merge or `O(n)` hash | `O(n)` | fact rows plus version rows | the interval test adds a constant per candidate |
| Recursive hierarchy traversal | `O(nodes + edges)` per connected region | `O(h)` stack, `O(n)` visited | subtree size, not depth | a DAG can revisit nodes per path |
| Nested-loop join without index | `O(n · m)` | `O(1)` streaming | product of the two relations | the exact failure that makes naive as-of joins unusable |

The last row is worth quantifying, because it is the difference between a model that works and one that does not. For 168,000 facts and 8,400 versions, a nested loop is `168,000 × 8,400 = 1,411,200,000` comparisons. With a sort-merge on key, the cost is the sort of both sides plus a linear merge: roughly `168,000·log₂168,000 + 8,400·log₂8,400 ≈ 168,000×17.3576 + 8,400×13.0362 ≈ 2,916,077 + 109,504 ≈ 3,025,581` comparisons, and `1,411,200,000 / 3,025,581 ≈ 466` times fewer. Both are algorithms, not engine benchmarks; the ratio is a property of the asymptotics at this size.

### 9.3 Version-set growth and its derived multiple

Applying MN-07 to the scenario: 8,400 version rows for 3,000 keys, `v̄ = 2.8`. The as-of join's candidate set per fact is the version family of its key, which averages 2.8 rows, so the interval test is applied about `168,000 × 2.8 = 470,400` times under a nested formulation, or once per merged candidate under a merge formulation. The number of *adjacent boundaries* that a closure check must inspect is `8,400 − 3,000 = 5,400`, which is the same figure Section 4.3.1 derived for the double-match population under a closed-closed convention.

| Quantity | Formula | Value | Unit |
|---|---|---|---|
| Version rows | `n + Σc` | 8,400 | rows |
| Mean versions per key | `N/n` | 2.8 | rows/key |
| Adjacent boundaries | `N − n` | 5,400 | pairs |
| Interval tests (nested) | `n_F · v̄` | 470,400 | comparisons |
| Interval tests (merged) | `n_F + N` | 176,400 | comparisons |
| Double matches if closed | `N − n` | 5,400 | extra rows |

The comparison between the last two interval rows is the practical lesson: a merge-based as-of join does work proportional to the sum of the two relations, while a nested formulation does work proportional to their product with the version multiple. The ratio here is `470,400 / 176,400 ≈ 2.67`, which is smaller than the nested-loop-vs-sort example because the version families are short; with a high-change dimension the ratio grows toward `v̄`.

### 9.4 Storage sizing with units shown

Per-row size is the input that makes storage arithmetic honest. The scenario assumes a fixed-width version row of 320 bytes, which is a stated assumption, not a measurement.

| Object | Rows | Bytes/row | Bytes | MiB |
|---|---|---|---|---|
| Customer versions | 8,400 | 320 | 2,688,000 | 2.5635 |
| Bitemporal bound (×4) | 33,600 | 320 | 10,752,000 | 10.2539 |
| Tag bridge | 210,000 | 24 | 5,040,000 | 4.8065 |
| Line facts at 40 B measure | 168,000 | 40 | 6,720,000 | 6.4087 |
| Line facts at 80 B payload | 168,000 | 80 | 13,440,000 | 12.8174 |

The conversion check is explicit: `2,688,000 / 1,048,576 = 2.5634765625`, and `33,600 / 8,400 = 4` confirms the bitemporal multiple. The line-fact row shows why column pruning matters more than model elegance at this scale: doubling the payload from 40 to 80 bytes doubles storage while the model is unchanged, so a storage argument that ignores payload width is not an argument about the model at all.

Add the bridge's own index footprint and the picture shifts. A 24-byte bridge row with two 8-byte key columns and a payload is small, but the pair index is a separate structure whose size is comparable to the table for short rows. This is why bridge tables are cheap in bytes and expensive in joins, and why the Section 4.3.3 measure discipline matters more than the storage line.

### 9.5 Comparison counts for the diagnostic suite

The five-row reconciliation from Section 5.7 has a cost, and it should be priced because a control that cannot finish before the decision deadline is not a control.

| Control | Comparisons | Dominant term | Deadline risk |
|---|---|---|---|
| Identity uniqueness | `O(n)` hash | 168,000 distinct keys | low |
| Key survival anti-join | `O(n)` hash build plus probe | 168,000 + 210,000 | low |
| Degree ratio | `O(n)` single pass | 210,000 rows | low |
| Measure reconciliation | `O(1)` against a control total | one scalar comparison | none |
| As-of exactly-one | `O((n_F + N) log)` merge | 176,400 candidates | moderate |
| Interval overlap | `O(N log N)` sort per key | 8,400 rows, 5,400 pairs | low |
| Bitemporal audit | `O(N · r)` candidate pairs | up to 33,600 rows | high if per-instant |

The bitemporal audit is the one that can breach a deadline. A per-instant audit of every valid version against every transaction revision is `8,400 × 4 = 33,600` candidate pairs for this scenario, which is small; but the same structure applied to every fact instant is `168,000 × 4 = 672,000` pairs, and if the revision count grows the product grows with it. The bounded fix is to audit on a sample of instants for the routine check and to run the full audit on a slower cadence, with the sample's blind spot stated rather than assumed away.

### 9.6 The cost of getting the model wrong

Model errors have a cost profile that differs from query errors, and it is worth stating in the same units. A fan-out at degree `α` multiplies every downstream aggregate by `α`, so the cost is proportional to the measure volume, not to the number of queries. A missing interval loses a share `1 − c` of the affected facts, and the loss is silent, so the cost includes the decisions made on the wrong number before discovery.

| Error class | Blast radius | Detection cost | Repair cost | Recurrence risk |
|---|---|---|---|---|
| Ambiguous identity | every metric naming the noun | high: requires cross-system alignment | high: rename and remap | high without a definition owner |
| Unlabelled grain | every aggregate over the joined result | low with a degree check | low: pre-aggregate | medium: recurs on the next bridge |
| Interval closure | every as-of join at a boundary | medium: needs a boundary fixture | low: re-derive endpoints | low once half-open is standard |
| Missing version | facts in the gap period | low with exactly-one assertion | medium: backfill from change log | medium if load order is not fixed |
| Overstated maximum | legitimate peak-volume records | immediate: rejection is visible | low: restate the rule | low once sourced from policy |
| Hash re-keying | entity continuity across sources | medium: needs attribute join | medium: mapping and re-derive | medium if the recipe is undocumented |

The table's point is that detection cost dominates repair cost in the two most expensive classes. An ambiguous identity is cheap to rename and expensive to notice, which inverts the usual engineering instinct to fix what is easy to see. Budgeting review effort in proportion to detection cost rather than repair cost is the correct allocation, and it is the reason Section 11 asks about discriminating checks rather than about fixes.

**Takeaway:** Price the model in rows, comparisons, and bytes with units attached; the as-of join's cost is the sum of the two relations when merged and their product when nested, and a control that cannot finish before the deadline is not a control.

## 10. Key Theorems / Results Reference

### 10.0 How to use this section

Each entry names a result, states the hypotheses that must hold before it applies, gives the modelling consequence, and names the boundary at which the claim stops being true. A result used without its hypotheses is a mnemonic, and mnemonics are how teams arrive at confident wrong models. Where this sheet relies on a result, the reliance is stated; where the primary specification was not reviewed for this revision, that is marked `OPEN`.

### 10.1 Relational and key results

**Candidate keys and functional determination.** A candidate key `K` of relation `R` satisfies `K → attrs(R)` and is minimal under that property (MN-01). The consequence is that identity is a set-level claim about determination, not a property of the data currently loaded. The boundary is first normal form: a multi-valued attribute voids the claim, because a tuple no longer has a single value per attribute and the key's determination becomes ambiguous.

**Armstrong's axioms are sound and complete for functional dependencies.** Soundness means every derivable dependency holds in every relation satisfying the given set; completeness means every dependency that holds in every such instance is derivable. The consequence is that a key dispute about a closed set of dependencies is decidable by closure computation (MN-02) rather than by sampling. The boundary is that the axiom system covers functional dependencies only. Multivalued dependencies need a different axiom set (Armstrong's complementation and augmentation rules), inclusion dependencies are not captured at all, and no axiom set can tell you which dependencies the business intends — that remains a design judgement. Evidence: DERIVED from the standard statement of the result; the primary treatment was not re-read for this revision, so the citation is `OPEN` pending a versioned reference.

**The lossless-join test.** A decomposition of `R` into `R₁` and `R₂` is lossless if and only if the common attributes form a superkey of at least one of the two. The consequence for notation is direct: a many-to-many relationship cannot be decomposed into two relations without a bridge, because the common attribute is not a superkey of the side with many values (Section 4.4.1). The boundary is that the test guarantees no spurious tuples, not that the result is at the intended grain. A lossless join can still be at the wrong grain, which is exactly the Section 4.4.1 counterexample. Evidence: DERIVED.

**Normal forms and anomaly elimination.** Third normal form removes transitive dependencies of non-key attributes on the key; Boyce-Codd normal form removes every non-trivial dependency whose determinant is not a superkey. The consequence is that a relation in BCNF has no update anomaly of the insertion/deletion/update family arising from functional dependencies. The boundary is important and commonly overstated: normal forms eliminate *those* anomalies and nothing more. They do not define a business metric, do not stop fan-out in a query, and do not prevent a semantically wrong key. A fully normalized schema with an ambiguous customer definition is still wrong.

### 10.2 Entity-relationship results

**Cardinality determines join multiplicity for equijoins.** For an equijoin of a fact relation `F` to a relation `B` on `F`'s key, the output cardinality is `|B|` when `F`'s key is unique and every `B` row has a non-null foreign key (MN-04). The consequence is that `α = |B|/|F|` is computable before the join. The boundary: a non-unique `F` key, a null foreign key, an additional predicate, or an outer join all change the identity, and the formula must be restated rather than stretched.

**The composed-multiplicity identity.** For two bridges on the same fact key, the combined degree average equals the product of the averages plus their population covariance (MN-05). The consequence is that multiplying two measured averages is exact only under independence, which is an empirical condition, not a structural one. The boundary is the population: a covariance computed on a filtered subset does not apply to the unfiltered join.

**The concealed-error identity.** With survival share `s` and conditional multiplicity `α_S`, both the row count and the measure scale by `s·α_S` (MN-06). The consequence is that row-count and total-level controls cannot separate loss from fan-out, because both errors map to the same scalar. The boundary is that the identity assumes measures are additive at the fact grain; for non-additive measures such as distinct counts, the effect is different and usually larger.

**Half-open intervals partition the timeline.** With change instants strictly increasing, `[t_i, t_{i+1})` covers each instant exactly once (MN-03). The consequence is deterministic as-of matching with a two-comparison predicate. The boundary is coverage of the declared domain: the partition is exact only if the intervals are contiguous and cover the domain, and it fails silently as a gap if they are not.

**Bitemporal state is bounded by the product of version and revision counts.** The worst case is `v·r` rows per key (MN-08), and the count collapses to `v + r − 1` when revisions touch only the current version. The consequence is that a capacity promise depends on which revision rule the business actually follows. The boundary is that the bound is a maximum over independent processes; if revisions are strongly correlated with versions the actual count is smaller, and the bound remains valid but pessimistic.

### 10.3 Temporal and SQL-standard results

**SQL:2011 application-time and system-time periods.** The standard defines period columns with `PERIOD FOR` declarations and temporal primary keys or uniqueness constraints that prevent overlapping periods for the same key. The modelling consequence is that overlap prevention is declarative in engines that implement the feature, rather than trigger-based. The boundary is implementation coverage: not every engine implements the temporal features, and syntax-level support does not guarantee the same overlap semantics. The exact supported feature set per engine is `OPEN` for this revision and must be verified against the target version's documentation before a model depends on it.

**Snapshot isolation and the meaning of transaction time.** Under snapshot isolation, a transaction reads a consistent snapshot, so the transaction-time coordinate of a bitemporal model corresponds to the snapshot at which the assertion became visible. The consequence is that a bitemporal model's transaction intervals must be derived from commit ordering, not from wall-clock insertion time, or the audit can show a value as believed before it was visible. The boundary is that different isolation levels and different engines expose different commit semantics; this sheet does not assert a specific engine's behaviour. Evidence: DERIVED for the general principle; engine specifics `OPEN`.

**Event time versus processing time.** A model must not use a load timestamp as a validity endpoint, because the two coordinates measure different things and diverge under late arrival and replay. The consequence is that the interval endpoints must be derived from the business change instant and the belief change instant separately. The boundary is that some facts genuinely have only a processing-time proxy, in which case the model must label it as such rather than presenting it as valid time.

### 10.4 Graph and hierarchy results

**Acyclicity is required for terminating recursive traversal.** A recursive query over a cyclic edge set does not terminate under naive recursion. The consequence is that a hierarchy model needs a cycle check before any recursive traversal is declared safe. The boundary: some engines offer cycle-detection clauses, in which case termination is restored but the result set must be interpreted as a path enumeration rather than a subtree.

**Tree reachability is unique-path; DAG reachability is not.** Under a tree edge set there is exactly one path between any connected pair, so a roll-up charges each descendant once. Under a DAG there can be several paths, so a naive roll-up charges a shared node once per path (MN-11). The consequence is that the tree-or-DAG decision is a cost decision, not a cosmetic one, and the model must mark which it is.

**Bounded-depth traversal has cost proportional to visited nodes, not to depth.** A subtree query visits every node in the subtree; a path-to-root query visits the depth. The consequence is that a "small" hierarchy can have an expensive descendant query, and the notation should record both the depth and the expected subtree size where a cost promise exists.

### 10.5 Results that do not exist, and what to use instead

Three things practitioners hope for are not available, and pretending otherwise produces the wrong model.

There is no general algorithm that infers the intended business model from data. Schema and dependency discovery can propose keys and functional dependencies from an instance, but the proposals are about the observed data and not about the rules the business intends; a discovered dependency that holds on today's rows may be violated by tomorrow's process. The correct substitute is a declared rule plus a test that alarms when observed data diverges.

There is no notation that enforces semantic equivalence between producer and consumer. Cardinality, typing, and temporal marks are structural. Whether `tier` means the same thing in two systems is a review question, and the substitute is a written definition with an owner and a change process, which is the contract sheet's subject.

There is no universal quality score for a model. Any single scalar collapses the layers that Section 5.7 keeps separate, and the collapse destroys the owner and the repair path. The substitute is a small set of controls with pass conditions, which is what the reconciliation table provides.

**Takeaway:** Use key, lossless-join, interval-partition, and multiplicity results as proofs under stated hypotheses; treat engine-specific temporal support and any claim about a business's intended dependencies as OPEN until verified.

## 11. Interview / Exam Questions

### 11.0 How these answers are scored

Fourteen questions follow, ordered from recall through application to judgement. Each has a checked answer, the reasoning that makes the answer more than a definition, and a **red flag** — a phrase that indicates the candidate has memorized the vocabulary without acquiring the diagnosis. Roughly two thirds require a calculation, a selection, or a diagnosis rather than recall, because a notation interview that only tests vocabulary selects for vocabulary.

Interviewers should prefer answers that name a condition that would reverse the claim. A candidate who says "half-open intervals" is correct; a candidate who says "half-open, because closed-closed double-matches the boundary instant, and the fix is to derive `valid_to` from the next version's `valid_from`" has the mechanism and the repair. The red flags below are diagnostic of the difference.

### 11.1 Question 1 — Why is a schema registry insufficient as a data model?

**Checked answer.** A registry records structure — field names, types, and compatibility rules. A model additionally records identity, cardinality, optionality, grain, and temporal meaning. A field can remain `decimal` while its semantics change from tax-inclusive to tax-exclusive, and every registry check passes while every metric breaks.

**Reasoning.** Registry validation is a statement about parseability across versions. Model validation is a statement about whether a specific consumer's arithmetic remains correct. The two are independent: a fully compatible change can be semantically breaking, and a breaking change can be semantically harmless if no consumer reads the field.

**Red flag.** "The registry catches breaking changes, so the model is covered." That answer equates compatibility with correctness and has no account of grain.

### 11.2 Question 2 — Define the grain of `order_line` and prove the key is minimal.

**Checked answer.** One row represents one line within one placed order. The candidate key is `(order_id, line_no)`. Minimality: `{order_id}⁺ = {order_id}`, which does not contain `line_no`, so the singleton is not determining; `{line_no}⁺ = {line_no}`, which does not contain `order_id`. Therefore no proper subset determines all attributes and the pair is minimal.

**Reasoning.** The proof is a closure computation, not a data sample. A sample can show that the pair is unique today; only the closure argument shows that no smaller attribute set is the identity. The distinction matters because a non-minimal key widens every foreign key that references the line.

**Red flag.** "It is unique in the table, so it is the key." Uniqueness without minimality describes a superkey, and the candidate says "key" for a superset.

### 11.3 Question 3 — A joined revenue figure is 25% high and the row count is close to expected. Diagnose.

**Checked answer.** Compute two separate signals. First, the key survival rate via an anti-join between the fact table and the joined result; this detects loss. Second, the degree ratio `count(*) / count(distinct fact_key)`; this detects fan-out. If the degree is 1.25, a bridge is duplicating the measure and the reported total is `α_S · s · M`. Then decide at which grain the measure is additive and pre-aggregate before the bridge join.

**Reasoning.** The 25% figure is exactly what `α = 1.25` produces, and the near-correct row count is what a compensating loss produces. Row count alone cannot separate the two because both errors scale the same scalar, `s·α_S`. The discriminator has to be key-sensitive and not total-sensitive.

**Red flag.** "Add `DISTINCT` to the query." `DISTINCT` over the whole row does not fix duplicated measures when other columns differ, and it does not address the loss at all. The candidate has a patch without a diagnosis.

### 11.4 Question 4 — Why half-open intervals, and what exactly breaks with closed-closed?

**Checked answer.** Half-open `[from, to)` assigns each instant to exactly one version. With closed-closed endpoints and a `BETWEEN` predicate, every change instant matches two versions, so an as-of join duplicates the attribute row and inflates any measure joined through it. With `n` keys and `Σc` changes, the double-match population is exactly `Σc`. The repair is to store `valid_to` as the next version's `valid_from` and to use `>= from and < to` everywhere.

**Reasoning.** The choice is forced by a partition argument, not by taste: among closed, open, and half-open, only half-open covers every instant once with a single two-comparison predicate. The count `Σc` is useful because it predicts the size of the defect, which lets a reviewer confirm the diagnosis numerically.

**Red flag.** "It is just a convention; either works if you are consistent." Consistency does not repair double counting, and the candidate has no mechanism.

### 11.5 Question 5 — Given 3,000 keys with change counts distributed as 1,200 with 0, 1,200 with 2, and 600 with 5, compute the version row count and the boundary count.

**Checked answer.** `Σc = 1,200×2 + 600×5 = 5,400`. Version rows `N = 3,000 + 5,400 = 8,400`. Mean versions per key `= 8,400/3,000 = 2.8`. Adjacent boundaries `= N − n = 5,400`, which is also the number of instants that a closed-closed convention would double-match.

**Reasoning.** Each key contributes one base row plus one row per change, so the total is `n + Σc`. The equality between the boundary count and `Σc` is not a coincidence: every change creates exactly one boundary, and every boundary is one potential double match.

**Red flag.** "8,400 rows, so about 2.8 times bigger." The arithmetic is right but the candidate has missed that the same number predicts the closure defect, which is the part that matters for the decision.

### 11.6 Question 6 — When may you multiply two measured fan-out factors?

**Checked answer.** Only when the two degree variables have zero population covariance. In general the combined factor is `α₁α₂ + Cov(d₁,d₂)`, computed over the same fact population. Since degrees are often positively correlated — larger orders have more lines, more promotions, and more shipments — the product of averages typically understates the true inflation.

**Reasoning.** The identity is algebraic: expanding the covariance definition gives `α₁₂ = α₁α₂ + Cov`. Reporting the product without the covariance term is an unstated independence assumption. The correct practice is to measure `α` on the fully joined result rather than composing estimates.

**Red flag.** "The bridges are independent because they come from different systems." Independence is a property of the variables, not of the provenance, and the candidate is asserting it rather than measuring it.

### 11.7 Question 7 — A many-to-many is stored as a delimited string column. What is the specific cost, and what is the migration order?

**Checked answer.** The cost is that member-level operations are impossible without unpacking: no per-member join, no per-member constraint, no per-member index, and no recorded history of membership changes. The migration order is: parse into a bridge with the pair as key, quarantine unparsable rows with their raw text, reconcile the bridge count against the sum of parsed members, run a member-level aggregate against a hand-computed sample, dual-run against the old column, then retire the column and add a check that fails if it reappears.

**Reasoning.** The order matters because the retirement is the irreversible step. Quarantining rather than dropping is what keeps the migration auditable; a parse that silently discards malformed members loses exactly the records whose handling was ambiguous.

**Red flag.** "Split it in the ETL and load a bridge." That is the first step only, and it says nothing about quarantine, reconciliation, or retirement.

### 11.8 Question 8 — Which notation would you choose for a review that must settle a key dispute, and why?

**Checked answer.** IDEF1X or an explicit functional-dependency list, because both make key structure and identifying-versus-non-identifying relationships first-class. Crow's foot shows cardinality well but leaves determination implicit, which is the claim in dispute. Pair either with an Armstrong closure computation for the contested relation.

**Reasoning.** The selection principle is to choose the family whose expressive centre matches the claim the reviewer must verify. A key dispute is a determination dispute; a join-path review is a cardinality review; a hierarchy review is a traversal review. Choosing by visual preference decouples the notation from the question.

**Red flag.** "Crow's foot, because it is the team standard." A standard is a constraint to acknowledge, not a reason that the disputed claim becomes checkable.

### 11.9 Question 9 — What does a foreign key prove, and what does it not prove?

**Checked answer.** It proves that a non-null child reference resolves to an existing parent row at the time the constraint is checked. It does not prove that the parent is the business-correct parent, that the parent was valid at the fact's instant, that the parent still exists in the as-of snapshot being queried, or that a parent has at least one child. The last gap means a mandatory-participation mark is not enforceable by a plain foreign key.

**Reasoning.** A foreign key is an existence claim over a key value. Every stronger claim — temporal validity, business correctness, minimum participation — needs a different mechanism: an interval predicate, a review, or a deferred check. Conflating them produces diagrams that assert more than the schema delivers.

**Red flag.** "The foreign key guarantees referential integrity, so the relationship is correct." That collapses existence into correctness.

### 11.10 Question 10 — Design the check that would have caught a missing version before anyone noticed the report was wrong.

**Checked answer.** Three assertions on the as-of join, run as one pass: total facts equals the sum of match counts; `max(matches per fact) = 1`; `min(matches per fact) = 1`. The third assertion is the one that catches a gap, because a missing interval produces zero matches for facts inside the gap. Run it against a fixture with a change exactly at midnight, since a closure bug that is invisible in daytime data is present in production.

**Reasoning.** The first assertion catches overlap on aggregate, the second catches duplicate matching locally, and the third catches loss. A single "no duplicates" check misses the gap entirely. The fixture requirement exists because the defect is a boundary phenomenon and boundary defects are invisible in non-boundary test data.

**Red flag.** "Check that the join returns rows." An inner join returning some rows is compatible with losing every fact in a gap.

### 11.11 Question 11 — Your bitemporal estimate assumed 33,600 rows but the business says corrections only touch the current version. What changes?

**Checked answer.** The bound drops from `v·r` to `v + r − 1` per key, so the estimate becomes `3,000 × (2.8 + 4 − 1) = 17,400` rows rather than `8,400 × 4 = 33,600`. The storage difference is a factor of about 1.93 here, and it grows with the version count. More importantly, the query set changes: if corrections never touch history, an as-of-transaction query on a historical valid interval returns the same value as the current belief, and the audit capability that justified the second interval may not be needed.

**Reasoning.** The two regimes differ in whether revisions and versions cross. A capacity promise built on the worst case is safe but expensive; a promise built on the current-only rule is cheaper and depends on a business invariant that must then be stated and monitored. Choosing between them is a decision, and the model should record which rule applies.

**Red flag.** "Just use the worst case to be safe." Safety at twice the storage without stating the invariant is not a design, and the invariant can break silently.

### 11.12 Question 12 — A hash-derived surrogate key changed its input. What is the diagnosis, and what is the bounded fix?

**Checked answer.** Join the new entities to the old on attributes other than the key. If they pair one-to-one, the input recipe or the hash changed rather than the population. Confirm by comparing the distinct natural-key count before and after: unchanged means re-keying. The bounded fix is to record the recipe version in a column, keep the raw natural key alongside the hash so duplicates can be re-resolved, and re-derive the affected keys through a mapping rather than deleting and reloading.

**Reasoning.** A hash surrogate is attractive because it can be recomputed independently in two systems, which makes merges join-free. That same property means any change to the input recipe re-keys every entity at once, and the resulting duplicates are indistinguishable from genuine new entities unless the raw key is retained. Retaining the raw key is the cheap insurance that makes the failure recoverable.

**Red flag.** "Recompute the hashes and delete the old rows." That destroys the ability to pair old and new, which is the only evidence that would distinguish re-keying from population growth.

### 11.13 Question 13 — What is a plausible wrong method that produces a correct-looking model?

**Checked answer.** Inferring the model from a data profile: taking the observed maximum as the cardinality bound, the observed distinct values as the domain, and the observed dependencies as the key. It looks rigorous because it is quantitative, and it produces a model of the sample rather than of the business. The specific failure appears at the first policy exception: an order above the observed maximum, a status code not yet used, or a dependency that the next process violates.

**Reasoning.** Profiling answers "what is true in these rows," while a model must state "what the business permits and requires." The two coincide only while the sample covers the operating range, and that coincidence is not a guarantee. The corrective is to source bounds from policy and to keep profiling as a monitor that alarms when observed data leaves the declared range.

**Red flag.** "The data shows a maximum of 200 lines, so the cardinality is `0..200`." The candidate has converted an observation into a rule without asking who owns the rule.

### 11.14 Question 14 — What changed constraint would reverse your choice of a type-2 dimension?

**Checked answer.** A hard latency requirement on the current-state read path, combined with an accepted loss of historical replayability. If the decision only ever asks "what is true now," the version table adds `v̄ − 1` extra rows per key and forces a `current_flag` or a range predicate on every query, and dropping history restores a simple unique key. The reversal is legitimate only if the lookback requirement is genuinely absent and that absence is recorded, because a regulatory or contractual lookback converts the same choice into a defect.

**Reasoning.** Type-2 history is an insurance policy with a storage and query cost. The cost is quantifiable from MN-07 (`N = n + Σc`); the benefit is reproducibility of every past report. Reversing requires evidence that no consumer needs the benefit, and the evidence is a recorded decision, not an absence of complaints.

**Red flag.** "History is always best practice." A policy with no cost accounting and no named consumer is a default, not a decision, and defaults are how unreproducible reports get built.

**Takeaway:** The distinguishing signal in every answer is a named condition that would reverse it; an answer that cannot be reversed has not been reasoned from the mechanism.

## 12. Quick Reference Card

| Element | Record this | Enforce with | Avoid |
|---|---|---|---|
| Entity | One-row statement: what it is and what identifies it | Unique constraint on the minimal key | A noun without an identity sentence |
| Identity | Natural key, stable surrogate, and any reuse interval | Surrogate PK, natural `UNIQUE`, reuse check | Treating a recycled business code as permanent |
| Cardinality and optionality | `(min,max)` both ways, plus why zero is allowed | FK for the maximum; deferred check for the minimum | Turning an observed average into a maximum |
| Many-to-many | A bridge keyed on the pair, carrying its own attributes | Composite key, or pair plus version | A delimited or array column |
| Grain | A label on every numeric column saying what it counts | Pre-aggregation or `DISTINCT` on the fact key | Summing a line measure after a bridge join |
| Time | Event instant, valid interval, belief interval, named apart | Half-open `from <= t < to` | `BETWEEN` on inclusive endpoints |
| Multiplicity | `α = bridge rows / fact rows`, computed before the join | Degree ratio equals the declared value | Multiplying two separately measured `α` values |
| Structure | Tree or DAG, subtype exclusivity, membership counts | Acyclicity and membership-sum assertions | Drawing a tree for a DAG |
| Notation family | The claim the reviewer must verify | A written constraint list for what the family cannot show | Choosing by visual preference |
| Review | Section 5.7 pass conditions, run in one pass | One reconciliation per load | A single composite quality score |

| Control | Formula | Pass condition |
|---|---|---|
| Identity | `count(*) − count(distinct key)` | exactly 0 |
| Fan-out | `count(*) / count(distinct fact_key)` | equals the declared `α` |
| Loss | anti-join on the fact key | 0 for additive totals; declared otherwise |
| As-of | `min` and `max` matches per fact | both exactly 1 |
| Versions | `N = n + Σc` | matches the storage plan |
| Overlap | adjacent pairs per key | 0 pairs overlapping |
| Membership | sum of subtype memberships | equals the supertype count when exclusive |
| Reconciled measure | joined total vs control total at one cutoff | equal, with degree equal to 1 |

**Card conditions.** The card is valid only for models whose identities are settled, whose intervals are stored half-open in a single time basis, and whose numeric columns carry a grain label. Under those conditions the eight controls above are sufficient to detect the failures in Section 8; outside them, the first repair is identity, not the control query.
