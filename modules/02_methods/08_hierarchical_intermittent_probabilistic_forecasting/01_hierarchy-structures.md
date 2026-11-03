# Hierarchy Structures — product, location, customer, and temporal aggregation and the paths between them

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 07 (hierarchy design); Module 07 topic 05 (hierarchy and aggregation)
> Version 2026.5

---

## 1. Why this topic exists

A monthly demand review opens with two numbers for the same month. The commercial director
quotes 13,000 units, rolled up through the product families. The supply planner's system sums to
12,800 units, rolled up through the channels. Both figures are correct outputs of correct
queries. Neither can be reconciled, because the two people are not disagreeing about demand.
They are using two different structures and neither knows it.

This is the normal condition of a supply chain that has more than one grouping variable, which is
every supply chain. The failure is not statistical. It is that the structure was never written
down as an object, so nobody could see that the same six bottom series feed two different trees
and that the two trees meet at a total which therefore has two values.

Every remaining topic in this module operates on one mathematical object:

- Aggregation consistency (Topic 02) asks which linear projection of a forecast vector onto the
  coherent subspace to use.
- Minimum-trace reconciliation (Topic 03) estimates a covariance for that projection.
- Temporal and cross-temporal reconciliation (Topic 04) tensor that projection with a second one
  over time buckets.
- Probabilistic reconciliation (Topic 07 onward) does the same thing to distributions rather than
  point forecasts.

All of them take a matrix as an input. In the literature, that matrix appears in the first line of
the method section, labelled `S`, with a sentence like "where `S` is the summing matrix". The
paper then spends thirty pages on the estimator, because the estimator is the interesting
mathematics. The matrix is treated as given.

In practice the matrix is the deliverable nobody produced. Projects fail at the structure, not at
the estimator:

- A retailer's "product hierarchy" turns out to have one item listed under two categories, so
  two nodes each claim it and the total is overstated by 4%.
- A distributor's "region" and "channel" are both groupings of the same orders, and the
  reconciliation silently picks one route and ignores the other.
- A manufacturer's case pack of twelve is modelled as one unit at the parent level, so a forecast
  of 100 cases reconciles against 1,200 eaches and the tool spreads a 1,100-unit residual across
  every SKU in the family.

Each of these produces a coherent-looking output from a structure that was wrong before the
estimator ran. No amount of covariance shrinkage repairs a mis-specified `S`.

**What this topic is, and what it is not.** Module 01A topic 07 taught hierarchy design for
dimensional modelling: grain, partitions and covers, fan and chasm traps, bridge tables,
additivity classes, and as-was versus as-is reporting. Module 07 topic 05 taught the choice of
forecasting level, the aggregation gain, and the four classical strategies, and previewed
reconciliation. Neither built the structure as an algebraic object, because neither needed to.
This file does. It assumes the vocabulary from both and spends its length on the matrix, the
subspace, the three structure types, and the nodes that will not fit.

> **Note:** If you have not read 01A.07, the sections on partitions, additivity, and ragged
> hierarchies will read as assertions. Read that file first; this one links to it rather than
> repeating it.

Why the structure deserves a whole topic rather than a paragraph:

| If the structure is wrong | What the estimator does | What the business sees |
|---|---|---|
| A node has two parents, both used unweighted | Spreads a residual that should not exist | A total that is 4% high, stable, and unexplained |
| A node is a mean or a ratio, not a sum | Forces it to be a weighted sum and moves the other nodes | A service metric that drifts whenever the mix changes |
| A case pack enters with weight 1 instead of 12 | Treats a 12× scale error as forecast noise | Chronic over-forecast on small items, under on large |
| The bottom level is the stored grain, not the decision grain | Optimises coherence where nobody decides | Coherent numbers that no planner uses |
| The structure changes mid-history unversioned | Reconciles January against March's map | History that will not reproduce |

Each row is a structure defect. None of them is visible in a reconciliation residual, because a
reconciliation residual is computed against the structure you gave it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a hierarchy as a triple of nodes, a parent relation, and summing weights, and
   **distinguish** this object from a diagram, a folder tree, or a dimension table.
2. **Build** the summing matrix `S` by hand for a structure of up to twenty nodes, and **verify**
   its four invariants without a library.
3. **Test** any pair of grouping variables for nesting versus crossing, and **classify** the
   resulting structure as a tree, a grouped structure, or a cross-temporal structure.
4. **State** what a coherent forecast vector is, **compute** the coherence residual for a given
   forecast set, and **explain** why the coherent set has dimension equal to the number of bottom
   series.
5. **Diagnose** a node that is not a sum of its children — a mixed case pack, a kit, a shared
   component, a ratio — and **choose** among the three legitimate repairs.
6. **Count** nodes, bottom series, and covariance cells for a proposed structure, and **predict**
   from the counts whether the estimator will be tractable.
7. **Decide**, with a stated criterion, which levels of a structure need coherent numbers and
   which are better left as derived metrics.
8. **Write** a structure specification that a second analyst could implement without asking you
   a question.

Outcomes 2, 3, 4, and 6 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures, in this order: a structure with two routes to the same total; a matrix that
encodes it; and a subspace that contains every coherent forecast.

### 3.1 Two routes to one total

```text
        PRODUCT ROUTE                          CHANNEL ROUTE
             TOTAL  ------------------------------ TOTAL
            /      \                             /   |   \
     DRESSINGS   SUTURES                  RETAIL  HOSP  EXPORT
      /  |  \     /  |  \                  / \     / \    / \
   D-R D-H D-E  S-R S-H S-E             D-R S-R D-H S-H D-E S-E

   The same six boxes appear at the bottom of both trees.
   Each box has one parent above it and two parents in total:
   D-R belongs to DRESSINGS and to RETAIL.
   TOTAL is reachable from D-R by two paths, and the two paths
   are required to give the same number.
```

This is the whole of the topic in one picture. A **tree** has one route from every leaf to the
root. A **grouped structure** — the name the forecasting literature uses for crossed
hierarchies — has more than one, because the grouping variables are crossed rather than nested.
The six bottom boxes are the same units counted once; the two upper trees are two different
partitions of them.

The consequence is blunt: **a grouped structure does not have a hierarchy at the top.** It has a
partial order. "Dressings in Retail" is below both Dressings and Retail and above nothing. The
only node that is above everything is the total, and the total is reachable by several routes.

> **Note:** The distinction is not academic. Every reconciliation method that assumes a tree
> either picks one route (and silently ignores the information in the other) or produces a total
> that does not equal the sum of the level below it. Module 07 topic 05 stated this as the
> `hts` versus `gts` distinction; here it becomes a property of a matrix.

### 3.2 The same structure as a matrix

Fix an order for the six bottom series. Then every node in the picture is described by which
bottom series it contains:

```text
              D-R  D-H  D-E  S-R  S-H  S-E
   TOTAL       1    1    1    1    1    1
   DRESS       1    1    1    0    0    0
   SUT         0    0    0    1    1    1
   RETAIL      1    0    0    1    0    0
   HOSP        0    1    0    0    1    0
   EXPORT      0    0    1    0    0    1
   D-R         1    0    0    0    0    0
   D-H         0    1    0    0    0    0
   D-E         0    0    1    0    0    0
   S-R         0    0    0    1    0    0
   S-H         0    0    0    0    1    0
   S-E         0    0    0    0    0    1
```

Twelve rows, six columns. The matrix is `S`, the **summing matrix** or **structural matrix**. Its
row `i`, column `j` entry is the weight of bottom series `j` in node `i`. For an aggregation
structure the weights are 0 or 1, but they need not be: a case pack of twelve gives a 12, and an
assortment case of six of one and six of another gives a row with two sixes in it.

Read three things off the matrix:

**Every column has a 1 in its own row.** The bottom six rows form an identity block. This is what
makes `S` full column rank, always, for any structure, which in turn is why `S'S` is invertible
and why the coherent subspace has exactly six dimensions.

**Row sums are subtree sizes.** The TOTAL row sums to 6, the DRESS row to 3, the D-R row to 1. If
row sums are not the number of bottom series you believe the node contains, the structure is
wrong, and you can find that out before any data arrives.

**Column sums are ancestry counts.** D-R's column sums to 4: TOTAL, DRESS, RETAIL, D-R. In a tree
this equals depth plus one. In a grouped structure it equals the number of nodes containing the
series, which is larger than depth plus one when two grouping variables are crossed at more than
one level.

### 3.3 Coherence is a subspace

With `b` the six-vector of bottom series and `y` the twelve-vector of all nodes:

```text
   y  =  S b                        the coherent set
   dim( col(S) ) = 6               the coherent subspace has six dimensions
   N - n_b = 12 - 6 = 6            the number of independent constraints
```

A forecast vector `ŷ` is **coherent** when it lies in the column space of `S`, which is the same
as saying `ŷ = S b` for some `b`. Equivalently, it satisfies six independent linear constraints,
which can be written `C ŷ = 0` with `C` a 6 × 12 matrix. The most natural `C` is
`[I | −A]`, where `A` is the top six rows of `S`: it says "each aggregate minus the sum of its
children is zero", once per aggregate.

Two consequences that matter for everything later:

1. **A summing matrix cannot be incoherent. Only forecasts can.** If you compute `S b` you get a
   coherent vector by construction, no matter how bad `b` is. The incoherence you observe in a
   company comes from the process that produced `ŷ`, never from the structure.
2. **Reconciliation is a projection.** Every linear reconciliation method has the form
   `ỹ = S G ŷ` for some matrix `G`. The composition `P = S G` maps any vector into the coherent
   subspace. Methods differ only in `G`, and the requirement that the projection be unbiased is
   exactly `S G S = S`.

That single identity, `S G S = S`, is the reason the rest of the module is short on surprises. It
is also the condition a vendor means when it says its reconciliation is "unbiased".

### 3.4 Three structure types, drawn small

```text
   (a) TREE                    (b) GROUPED                 (c) CROSS-TEMPORAL
       one route                   many routes                 structure x time

        T                           T                           T   T   T
       / \                        /   \                         |   |   |
      A   B                     AB     BA                     A   A   A
     / \   \                    / \    / \                     |   |   |
   a1  a2   b1                aX aY  bX bY                    a   a   a
                                                              wk  mo  qtr
   every leaf one parent      leaves shared by two         one matrix per
                              grouping variables            frequency, plus
                                                             the time map
```

| Type | Definition | Test | Where it comes from |
|---|---|---|---|
| Tree | Every node except the root has exactly one parent | Count nodes with more than one parent: zero | Product family trees, geographic rollups, calendars that nest |
| Grouped | Two or more trees over the same leaves, no unique path | Count leaves with more than one parent: greater than zero | Product × location, product × channel, customer × product |
| Cross-temporal | A structure replicated at several time frequencies, with an aggregation map between them | Does the business forecast or decide at more than one bucket? | Monthly plan, weekly schedule, daily execution |

The three compose. A real planning structure is usually grouped across product, location, and
customer, and cross-temporal across month, week, and day. That composition is where the node
counts in section 6.5 come from, and it is where most implementations quietly give up and
reconcile one axis at a time.

When somebody presents a rollup, ask: **"Which structure produced this number, and what are the
other routes to the same total?"** A blank look means the number is a leaf of an undeclared tree.
Two routes with two totals mean the structure is grouped and nobody has built the matrix.

## 4. Definitions and vocabulary

The fourth column gives the unit or coding case, because a definition of a structure without a
statement of what the entries mean numerically is the defect this file exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Hierarchy | A set of nodes with a containment relation and weights, used to aggregate a measure | Used for any grouping, including crossed ones | Node list plus weight function |
| Structure | The complete object: nodes, parent relation, weights, and bottom level | Used as a synonym for "the diagram" | Triple `(V, E, W)` |
| Node | One member of the structure: an aggregate series or a bottom series | Called a "level" or a "dimension" | Stable key plus display name |
| Aggregate node | A node whose value is defined as a weighted sum of bottom series | Assumed to be a reporting convenience | Row of the summing matrix |
| Bottom series | The finest series in the structure; the unit that carries the forecast | Confused with the stored grain | Column of the summing matrix |
| Leaf | A node with no children in a particular tree | Confused with bottom series in a grouped structure | Node key |
| Root | The single node containing all bottom series | Assumed present; often absent in source data | One node key per structure |
| Parent | A node that directly contains another | Assumed unique | One key in a tree; many in a grouped structure |
| Child | A node directly contained by another | Assumed to partition the parent | Node key |
| Depth | Number of edges from a node to the root | Used interchangeably with level | Integer, root = 0 |
| Level | A named depth, such as family or region | Assumed identical across structures | Ordinal integer |
| Tree | A structure in which every non-root node has exactly one parent | Assumed whenever a hierarchy is drawn | Parent count per node = 1 |
| Grouped structure | Two or more crossed trees over the same bottom series | Modelled as one tree with a chosen route | Several aggregation factors |
| Cross-temporal structure | A structure replicated at several time frequencies with a nesting map | Handled as several unrelated problems | Structure matrix per frequency |
| Summing matrix (`S`) | The `N × n_b` matrix mapping bottom series to every node | Treated as a lookup table | Rows = nodes, columns = bottom series |
| Aggregation matrix (`A`) | The top `N − n_b` rows of `S` | Used for the whole structure | Dimensionless weights |
| Identity block | The bottom `n_b` rows of `S`, equal to the identity | Omitted when the bottom level is chosen casually | Unit entries |
| Constraint matrix (`C`) | A matrix whose null space is the coherent subspace, often `[I \| −A]` | Assumed unique | `(N − n_b) × N` |
| Coherent vector | A forecast vector in the column space of `S` | Confused with an accurate vector | Units per node per bucket |
| Coherence residual | `r = ŷ − S b`, the part of a forecast vector outside the column space | Reported as forecast error | Units per node |
| Coherent subspace | The set of all vectors satisfying every aggregation identity | Imagined as a single number | Dimension `n_b` |
| Base forecast (`ŷ`) | A forecast produced at a node from that node's own history | Called "the forecast" | Units per node per bucket |
| Reconciled forecast (`ỹ`) | A coherent forecast derived from base forecasts by a linear map | Assumed more accurate by construction | Units per node per bucket |
| Projection matrix (`P`) | `P = S G`, mapping any vector into the coherent subspace | Ignored in favour of `G` | `N × N`, rank `n_b` |
| Mapping matrix (`G`) | The matrix in `ỹ = S G ŷ`; the only thing methods differ on | Called "the reconciliation" | `n_b × N` |
| Unbiasedness condition | `S G S = S`, the requirement that a correct base forecast stays correct | Checked by inspection | Matrix identity |
| Structural scaling | A weighting where each node's weight is proportional to its size | Confused with statistical weighting | Diagonal of `S 1` |
| Aggregation ratio | The ratio of a node's value to the sum of one child's value | Assumed to be one | Dimensionless factor |
| Mixed case pack | A case containing several different items in stated quantities | Treated as a unit conversion of one item | Bill of materials per case |
| Kit | An assembly sold as one item but built from several stocked items | Treated as a parent of its components | Bill of materials |
| Shared component | An item consumed by more than one product | Treated as a child of one of them | Bill of materials edge |
| Requirements matrix (`M`) | A matrix mapping end-item demand to component demand | Confused with the summing matrix | Bill-of-materials coefficients |
| Non-summing node | A node whose value is not a linear combination of its children | Forced into `S` with unit weights | Function type plus parameters |
| Derived metric | A quantity computed after aggregation, never reconciled | Reconciled as if it were a node | Formula plus inputs |
| Partition | A division into non-overlapping subsets | Assumed whenever a rollup is drawn | Disjoint leaf sets |
| Cover | A division permitting overlap | Mistaken for a partition | Possibly overlapping leaf sets |
| Node count (`N`) | Number of series in the structure, aggregates plus bottom | Quoted as the size of the problem | Count |
| Bottom count (`n_b`) | Number of bottom series; the dimension of the coherent subspace | Confused with node count | Count |
| Covariance cells | `n_b × n_b`, the size of the object a minimum-trace estimator must estimate | Assumed to be small | Count |
| Sparsity of `S` | Fraction of entries that are zero | Ignored when choosing an implementation | Ratio |
| Kronecker product | The tensor operation that builds a cross-temporal summing matrix from a structure matrix and a time matrix | Implemented by looping over frequencies instead | Matrix of dimension `N m × n_b m_b` |
| Temporal summing matrix (`R`) | The matrix mapping the finest time bucket to every coarser bucket | Assumed to be all ones | `m × m_b` |
| Cross-temporal matrix (`F`) | `S ⊗ R`, mapping bottom series at the finest frequency to every node at every frequency | Built by hand and mis-sized | `N m × n_b m_b` |
| Structure version | An identifier for a structure that changes over time | Overwritten in place | Version key plus effective dates |
| Immutable node | A node whose forecast may not be adjusted, such as a contractual commitment | Ignored by the estimator | Boolean per node |

Four of these deserve expansion, because they are the ones that get conflated in practice.

**Structure versus diagram.** A diagram shows boxes and lines. A structure additionally states
the weights and the bottom level. Two people can draw the same diagram and build different
matrices, because one includes a node the other treats as a derived metric. The diagram is not
the specification; the matrix is.

**Bottom series versus leaf.** In a tree these coincide. In a grouped structure they do not: the
bottom series are the boxes at the bottom of the picture, and each of them is a leaf of two
different trees. Saying "the leaf level" in a grouped structure is ambiguous, and the ambiguity
hides which tree you meant.

**Summing matrix versus requirements matrix.** `S` aggregates: it says "the total is the sum of
these parts". A bill of materials says "making one of these consumes two of those". Both are
linear maps with 0/1 or integer coefficients, and they look similar in a spreadsheet. They answer
different questions and obey different constraints: aggregation demands equality, requirements
demand sufficiency.

**Coherence versus accuracy.** A coherent set of forecasts can be worse than an incoherent set,
and often is when the base forecasts are biased. Coherence is a property of the arithmetic;
accuracy is a property of the prediction. Section 11 states this as a limit rather than a caveat,
because it is the single most overclaimed result in the reconciliation literature.

## 5. The mechanics, step by step

This section builds a structure from a business description to a verified matrix, then tests it.
Each step names its input, its output, and the branch where the analysis can stop.

### Phase A — Fix the object

**Step 1. Name the measure and its unit before naming any node.**

A structure is a structure *of something*. "Units shipped, in eaches, per calendar month" and
"revenue, in euro, per fiscal period" produce different structures over the same products,
because revenue cannot be summed across a case-pack level without a price. Write the measure and
the unit first; the nodes follow.

*Input:* the decision the forecast serves. *Output:* one sentence naming measure, unit, and
bucket. *Branch:* if two measures are needed, build two structures. One matrix cannot serve both.

**Step 2. Choose the bottom level and say why.**

The bottom level is the finest series the structure will forecast, not the finest series the
warehouse stores. The warehouse may hold transaction lines; the structure may bottom out at
SKU × DC × week. The choice is a modelling decision with three inputs:

| Criterion | Question | If the answer is no |
|---|---|---|
| Decision | Is a commitment made at this level? | Going finer adds cost without changing a decision |
| Data | Are there enough non-zero observations per series? | Going finer makes every series unforecastable |
| Estimability | Is `n_b` small enough to estimate a covariance? | Going finer makes minimum-trace reconciliation impossible |

*Output:* a named bottom level with its dimensions. *Branch:* if the decision level and the
estimable level differ, the structure will need a reconciliation between them, which is what the
rest of this module is for. Record the difference; do not paper over it.

> **Note:** Module 07 topic 05 gives the evidence on where to forecast. This step does not
> re-open that question. It fixes the bottom level of the *structure*, which may be coarser than
> the decision level if the decision level is handled by allocation downstream.

**Step 3. List every grouping variable that touches the measure.**

There is never one. A consumer goods manufacturer typically has product, location, customer, and
time; a distributor adds a supplier or contract axis; a hospital supply chain adds a clinical
category that cuts across the commercial one. List them with their levels, and note which are
authoritative for which decision. Module 01A topic 07 calls this the inventory of hierarchies and
it is the step most often skipped.

*Output:* a table of grouping variables, levels, member counts, and owning function.

**Step 4. Test every pair of grouping variables for nesting.**

For variables `X` and `Y`, ask whether every member of `Y` sits inside exactly one member of `X`.
If yes for one direction, they nest and can be one tree. If no in both directions, they cross and
the structure is grouped.

| Test | Result | Structure |
|---|---|---|
| Every DC belongs to exactly one region | Nested | One location tree |
| Every SKU belongs to exactly one family | Nested | One product tree |
| A SKU sells through several channels | Crossed | Grouped structure |
| A customer buys several families | Crossed | Grouped structure |
| Every week belongs to exactly one month | False in general | Cross-temporal structure with an explicit rule |

The test is a count, not a judgement: `count(distinct parent) per child`. One means nested, more
than one means crossed, zero means orphaned.

### Phase B — Build the summing matrix

**Step 5. Fix the row order and the column order, and never change them silently.**

Put aggregates first in a documented order and bottom series last, so the identity block is
visible. Any order works mathematically; a stable order is what lets two people compare matrices
and lets a reconciliation residual be attributed to a named node.

**Step 6. Fill column `j` by walking the ancestry of bottom series `j`.**

Set a 1 in every node that contains bottom series `j`, and a 0 elsewhere. In a tree this is the
path from the leaf to the root. In a grouped structure it is the union of the paths in each
crossed tree, which is why the column sum exceeds depth plus one.

**Step 7. Check the four invariants before looking at any data.**

| Invariant | Statement | What a failure means |
|---|---|---|
| I1 Bottom block | The last `n_b` rows form the identity | The bottom level was not included, or a column is duplicated |
| I2 Row sum | Row `i` sums to the number of bottom series node `i` contains | A node is missing a child, or a weight is wrong |
| I3 Column sum | Column `j` sums to the number of nodes containing series `j` | A node was omitted from the structure |
| I4 Rank | `rank(S) = n_b` | Impossible if I1 holds; a failure means the matrix was built by concatenating pieces |

Invariant I4 is worth stating formally: because the identity block is present, `S` always has
full column rank, so `S'S` is always invertible, so both the projection and the least-squares
reconciliation always exist. The interesting failures are never in the algebra; they are in the
correspondence between the algebra and the business.

**Step 8. Write the constraint form as well.**

`C = [I_{N−n_b} | −A]` satisfies `C S = 0`, so `C y = 0` is exactly the coherence condition. The
constraint form is useful because it separates the structure (which nodes exist) from the
identities (which sums must hold), and because optimisation-based reconciliation, including
non-negativity, is stated as a problem in `C`.

```text
   S  (12 x 6)      y = S b        coherent set
   A  (6 x 6)       top block of S
   C  (6 x 12)      [ I6 | -A ]    C y = 0  <=>  y coherent
   nullity(C) = 12 - 6 = 6 = n_b
```

### Phase C — Test a forecast set

**Step 9. Compute the residual `r = ŷ − S b̂`.**

Take the bottom entries of the forecast vector as `b̂` and compute what the structure says every
node should be. The residual is zero on the bottom block by construction and non-zero only where
the aggregate forecasts disagree with the bottom forecasts.

**Step 10. Attribute the residual to a route, not just to a node.**

A residual on one aggregate is an ordinary disagreement between two forecasts. A residual on the
total with zero children is a *route* disagreement: two paths to the total were computed from two
different partitions. The first is fixed by the estimator; the second is a defect in how the
structure was populated.

**Step 11. Decide whether the residual is a data error or a legitimate disagreement.**

If the bottom entries of `ŷ` produced the aggregate entries, a non-zero residual is arithmetically
impossible and points to a different source, filter, or period. If the aggregate was forecast
independently — the normal case — the residual is expected and is what the estimator reduces.

### Phase D — Classify and compose

**Step 12. Identify the structure type.**

Tree, grouped, or cross-temporal, using the tests in section 3.4. Write the answer down. Most
implementation failures begin with a grouped structure described in a slide as "our hierarchy".

**Step 13. Compose the grouping variables explicitly.**

Two crossed variables produce a node set that is the *product* of the two node sets, minus any
combination that does not occur. A product tree with 3,073 nodes crossed with a location tree
with 49 nodes does not produce 150,577 real series; it produces at most that many, and the ones
that exist are the ones with data. The structure must list the combinations that exist, because
the empty ones are not zeros — they are absences, and a forecast of zero for a combination that
cannot occur contaminates every aggregate above it.

**Step 14. Add time as a second structure, not as a column.**

If the business forecasts or decides at more than one bucket, the structure is cross-temporal.
The time axis is not a grouping variable like product; it has its own summing matrix `R` and its
own nesting rules, and it composes with `S` by Kronecker product. Module 08 topic 04 treats the
composition; here it is enough to record that the structure has a second dimension and that the
total series count multiplies by the number of frequencies.

### Phase E — Handle nodes that are not sums

**Step 15. Classify every node by the function that defines it.**

| Node function | Example | Fits `S`? | Treatment |
|---|---|---|---|
| Sum | Total units = sum of SKUs | Yes, weights 0/1 | A row of `S` |
| Weighted sum | Cases = 12 × eaches | Yes, weight 12 | A row of `S`, with the unit stated |
| Weighted sum over items | Assortment case = 6 of A + 6 of B | Yes, two non-zero weights | A row of `S`, if the unit is the case |
| Mean | Average price per unit | No | Derived metric |
| Ratio | Fill rate, margin percent | No | Derived metric |
| Maximum | Peak inventory across DCs | No | Derived metric, or a separate model |
| Bill of materials | Kit = 1A + 2B + 1C | Not as aggregation | Requirements matrix |
| Shared resource | One component in four products | Not as aggregation | Requirements matrix |

**Step 16. Repair by weights where the relation is genuinely linear.**

A mixed case pack is linear. A case of six A and six B is `case = 6a + 6b`, which is a perfectly
good row of `S` provided every entry is expressed in the same unit as the bottom series. The
failure is not that the relation is non-linear; it is that the unit changed and nobody converted.
State the conversion factor in the specification, and test it: the row's coefficient of the
bottom series should equal the pack quantity, not 1.

**Step 17. Reclassify or separate where it is not.**

Three legitimate responses to a node that will not fit:

- **Derived metric.** Remove it from `S` and compute it after reconciliation from coherent
  inputs. A fill rate is `units served / units demanded`, both of which are coherent nodes; the
  ratio is then coherent by construction and never needs reconciling.
- **Requirements matrix.** Keep the node out of `S`, define a separate matrix `M` mapping
  end-item demand to component demand, and state the constraint as an inequality
  (`supply ≥ M × demand`) rather than an equality. This is the correct treatment of kits and
  shared components.
- **Separate structure.** If the node belongs to a different measure with a different unit,
  build a second structure and reconcile the two only at the nodes they share.

What is not legitimate is putting the node in `S` with unit weights and hoping. The estimator
will find a residual it cannot remove, and it will remove it somewhere else.

### Phase F — Size the problem and choose levels

**Step 18. Count nodes, bottom series, and covariance cells.**

`N` is the number of series in the structure. `n_b` is the number of bottom series, which is the
dimension of the coherent subspace and the order of the covariance a minimum-trace estimator must
estimate. Report both, and report `n_b²`. The first number is what the tool displays; the second
is what determines whether the method is feasible.

**Step 19. Choose which levels need coherent numbers.**

Not every level does. A level needs coherence when two conditions hold: a decision consumes the
number at that level, and another decision consumes a number at a different level in the same
structure. If only one level is ever consumed, coherence is a property nobody observes and the
reconciliation cost is wasted. A total consumed by finance and a family plan consumed by sales
operations both need it, because they are reported side by side. A daily scheduling profile
inside the week usually does not, because nothing above it is reported at that bucket.

**Step 20. Write the structure specification.**

One page, and it is the artefact the rest of the module consumes.

| Field | Content | Example |
|---|---|---|
| Measure and unit | What is being aggregated | Units shipped, eaches |
| Bottom level | The finest series in the structure | SKU × DC × week |
| Grouping variables | Each with its levels and owner | Product (family, SKU); location (region, DC) |
| Structure type | Tree, grouped, or cross-temporal | Grouped, product × location |
| Node count and bottom count | `N` and `n_b` | 4,812 and 3,900 |
| Weights | Any entry not 0 or 1, with its unit | Case node: weight 12 |
| Excluded nodes | Nodes deliberately not in `S`, with the reason | Fill rate: derived metric |
| Version | Identifier and effective dates | 2026-04, effective 1 April 2026 |

The specification is short because the structure is simple once it is written down. Its value is
that a dispute about a reconciled number is settled by reading one page rather than by
reverse-engineering a matrix from four systems.

## 6. Worked example

Halvorsen Surgical distributes two product families through three channels. The structure is
grouped: product and channel are crossed, and the same six bottom series feed both trees. The
figures are constructed for teaching and are internally consistent; every sum in this section can
be checked by hand.

### 6.1 The structure

| Level | Node | Members |
|---|---|---|
| 0 | Total | 1 |
| 1 | Product family | Dressings, Sutures |
| 1 | Channel | Retail, Hospital, Export |
| 2 | Bottom series | Six combinations |

`N = 12`, `n_b = 6`. The coherent subspace has six dimensions and there are six independent
constraints.

### 6.2 Base forecasts for month M

The six bottom series are forecast at SKU-family × channel level: D-Retail 4,200 units,
D-Hospital 3,100, D-Export 700, S-Retail 1,800, S-Hospital 2,400, S-Export 600, summing to 12,800.
The six aggregates are forecast independently at their own level, which is what a reconciliation
input looks like in practice.

| Node | Base forecast (units) | Sum of children | Residual (units) |
|---|---:|---:|---:|
| Total | 13,000 | 12,800 | +200 |
| Dressings | 8,000 | 8,000 | 0 |
| Sutures | 5,000 | 4,800 | +200 |
| Retail | 6,100 | 6,000 | +100 |
| Hospital | 5,400 | 5,500 | −100 |
| Export | 1,300 | 1,300 | 0 |
| Six bottom series | as forecast | — | 0 |

Three readings, in order of importance.

**The product route and the channel route disagree.** Family forecasts sum to 8,000 + 5,000 =
13,000. Channel forecasts sum to 6,100 + 5,400 + 1,300 = 12,800. The difference is 200 units,
1.56% of the total, and it is not an error in either route. It is what happens when two
partitions of the same six numbers are forecast separately.

**The residual is not distributed evenly.** Dressings is exactly coherent while Sutures is 200
high, so the family forecaster believed something about Sutures that the SKU forecaster did not.
A reconciliation moves both, and the movement is a modelling choice, not a correction.

**The bottom block is coherent by construction.** Every zero in the last six rows is zero because
`b̂` was used to compute the sum, not because the bottom forecasts are good.

### 6.3 Two coherent sets from the same inputs

**Bottom-up.** Take `b̂` and compute every aggregate from it.

| Node | Bottom-up (units) | Change from base (units) |
|---|---:|---:|
| Total | 12,800 | −200 |
| Dressings | 8,000 | 0 |
| Sutures | 4,800 | −200 |
| Retail | 6,000 | −100 |
| Hospital | 5,500 | +100 |
| Export | 1,300 | 0 |

**Top-down proportional.** Scale every bottom series by 13,000 / 12,800 = 1.015625 and recompute.

| Node | Scaled bottom series | Rounded | Aggregate after rounding |
|---|---:|---:|---:|
| D-Retail | 4,265.6 | 4,266 | Dressings 8,125 |
| D-Hospital | 3,148.4 | 3,148 | Sutures 4,875 |
| D-Export | 710.9 | 711 | Retail 6,094 |
| S-Retail | 1,828.1 | 1,828 | Hospital 5,586 |
| S-Hospital | 2,437.5 | 2,438 | Export 1,320 |
| S-Export | 609.4 | 609 | Total 13,000 |

Both tables are coherent. They differ by 200 units at the total and by up to 1.6% at the bottom,
and the difference is entirely a statement about which level the analyst trusts. Topic 02 turns
that statement into an estimator; this topic's point is only that **the structure admits many
coherent vectors and chooses none of them.**

> **Note:** A third option, ordinary least squares, weights every node equally regardless of its
> size, so a total of 13,000 and a leaf of 700 contribute equally to the fit. On structures with
> several levels this produces implausible leaf adjustments. Structural scaling, which weights
> each node by its size, is the usual repair, and it is the default in most implementations.
> Module 08 topic 02 derives it.

### 6.4 The mixed case pack

Halvorsen sells D-Retail in eaches and in cases of twelve. Both are legitimate nodes. The
question is what the case node's row of `S` contains.

| Treatment | Case row weights | Forecast of 100 cases reconciles to | Error |
|---|---|---:|---:|
| Correct | 12 on D-Retail | 1,200 eaches | 0 |
| Weight 1 | 1 on D-Retail | 100 eaches | 1,100 eaches short |
| Case as separate bottom series | 1 on the case column | 100 cases, 1,200 eaches in a parallel column | 0, if the unit is carried |

The third row is the point. A pack level can be either a *different unit of the same series* or a
*separate series with its own column*. The first is a weight in `S`; the second is a second
column and a second bottom series, and the two must not both be summed into the same total or the
volume is counted twice.

The failing pattern in practice is neither. It is a data model that stores cases and eaches in
the same column with no unit flag, so the sum is meaningless before the structure is even
involved. Module 01A topic 01 treats the unit machinery; here the narrow lesson is that a
summing matrix is dimensionally homogeneous, and a node measured in a different unit needs a
conversion factor that is written down.

### 6.5 A kit, a shared component, and why they are not parents

A sterile procedure pack K contains one D-Export dressing, two S-Export sutures, and one item
from a third family. It is sold as one unit and has its own demand history.

```text
   AGGREGATION (fits S)                 REQUIREMENTS (needs M)
   Total                                K demand
    |                                    |
    +-- Dressings                        +--> 1 x D-Export
    +-- Sutures                         +--> 2 x S-Export
    +-- Procedure packs  <- a node      +--> 1 x T-Export
                                        (inequality: supply >= requirement)
```

If K is placed in the product tree as a parent of D-Export, S-Export, and T-Export, then D-Export
is counted twice: once as its own demand and once as a component of K. The total is overstated by
the component content of every kit sold. The correct treatment is:

- K is a bottom series in the aggregation structure, with its own column.
- The component requirement is a separate linear map `M` applied to the K *plan*, not to the K
  history.
- The constraint is `supply_c ≥ sum over products of (M[c, p] × plan_p)`. It is an inequality,
  because surplus components are allowed and shortages are not.

The same reasoning applies to a shared component used in four products. It is not a child of any
of them; it is a column of `M` with four non-zero entries.

This distinction is the most useful thing in the topic for a practitioner, because it separates
two questions that look identical in a spreadsheet: *how do the numbers add up* (S) and *how much
material does the plan require* (M). Confusing them produces a total that is wrong by exactly the
kit content, and a component plan that is short by the same amount.

### 6.6 Node counts and the combinatorial problem

Take a mid-sized distributor with a product tree of 12 families, 60 subfamilies, and 3,000 SKUs,
and a location tree of 8 regions and 40 DCs.

| Quantity | Count | Arithmetic |
|---|---:|---|
| Product nodes | 3,073 | 1 + 12 + 60 + 3,000 |
| Location nodes | 49 | 1 + 8 + 40 |
| Full cross, all nodes | 150,577 | 3,073 × 49 |
| Bottom series | 120,000 | 3,000 × 40 |
| Constraints | 30,577 | 150,577 − 120,000 |
| Covariance cells | 14.4 × 10⁹ | 120,000² |
| Covariance memory at 8 bytes | 115 GB | 14.4e9 × 8 / 1e9 |

Three things follow.

**Reducing `N` does not reduce `n_b`.** Drop the intermediate levels and keep only total, family
× region, and SKU × DC: `N` falls from 150,577 to 120,097, a 20% saving, while `n_b` stays at
120,000. The cost that matters is unchanged.

**The covariance is the binding constraint, not the structure.** A dense 120,000 × 120,000
matrix is 115 GB before any arithmetic. This is why minimum-trace reconciliation needs a
shrinkage estimator, a diagonal approximation, or a coarser bottom level. Module 08 topic 03
treats the estimators; the structural point is that the choice of bottom level determines whether
they are needed.

**The only lever that changes `n_b` is the bottom level.** Forecasting at SKU × region instead of
SKU × DC takes `n_b` from 120,000 to 24,000, and the covariance from 115 GB to 4.6 GB. That is a
decision-grain change, not a software change, and it is why step 2 comes before everything else.

The public benchmark for scale is the M5 competition: 42,840 series across 12 aggregation levels,
of which 30,490 are bottom-level product × store series. That is a small structure by the
standards of the calculation above, and it is still large enough that no competitor estimated a
dense covariance.

## 7. Code, query, or calculation

One block. It builds the matrix from section 6 by hand rather than from a library, because the
building is the skill and the library hides it. Plain Python; no imports.

```python
# Summing matrix for the 2-family x 3-channel structure. Rows: aggregates, then bottom series.
leaves = ["D-Retail", "D-Hospital", "D-Export", "S-Retail", "S-Hospital", "S-Export"]
rows = {
    "TOTAL":     leaves,
    "DRESSINGS": ["D-Retail", "D-Hospital", "D-Export"],
    "SUTURES":   ["S-Retail", "S-Hospital", "S-Export"],
    "RETAIL":    ["D-Retail", "S-Retail"],
    "HOSPITAL":  ["D-Hospital", "S-Hospital"],
    "EXPORT":    ["D-Export", "S-Export"],
}
S = [[1 if leaf in rows[name] else 0 for leaf in leaves] for name in rows]
S += [[1 if i == j else 0 for j in range(6)] for i in range(6)]       # identity block

n_b, N = len(leaves), len(S)
assert N == 12 and all(len(r) == n_b for r in S)
assert all(sum(S[i][j] for i in range(N)) >= 1 for j in range(n_b))  # no orphan column
print("shape", (N, n_b), "| row sums", [sum(r) for r in S])

b = [4200, 3100, 700, 1800, 2400, 600]
y_hat = [13000, 8000, 5000, 6100, 5400, 1300] + b
coherent = [sum(w * x for w, x in zip(row, b)) for row in S]
print("S b   ", coherent)
print("resid ", [a - c for a, c in zip(y_hat, coherent)])
```

Expected output: `shape (12, 6) | row sums [6, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1]`, then
`S b [12800, 8000, 4800, 6000, 5500, 1300, 4200, 3100, 700, 1800, 2400, 600]`, then
`resid [200, 0, 200, 100, -100, 0, 0, 0, 0, 0, 0, 0]`.

Three things to verify by hand first. The row sums are the subtree sizes, so a row sum of 2 on
RETAIL says Retail contains two bottom series. The identity block is rows 7 to 12, which is why
`S` has full column rank and the coherent subspace has dimension six. The residual is non-zero
only on aggregates, which is the signature of an ordinary incoherent forecast set.

Add one assertion when the structure includes a case node: the coefficient must equal the pack
quantity, not one. `assert S[case_row][leaf_col] == 12` is a one-line test that catches the most
expensive structure error in the topic.

## 8. Data, standards, and regulatory context

No international standard defines a planning hierarchy. That sentence is worth keeping, because
analysts regularly cite a product classification standard as though it governed one. What exists
is a set of identifier standards, one unit-of-measure standard, one calendar standard, and a
financial reporting requirement that fixes the top of any structure that must tie to an annual
report.

### 8.1 Identity standards that constrain the bottom level

| Standard | Governs | What it fixes for the structure | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity | That each pack level is a distinct item, not a conversion of one item | B |
| GS1 GLN | Party and place identity | That a ship-to, a DC, and a legal entity are different nodes | B |
| UN/LOCODE | Trade and transport locations | A place code that is not a sales territory | B |
| ISO 3166-1 and 3166-2 | Country and subdivision codes | The geographic levels above the site | B |
| GS1 GPC | Product classification | Four levels that are not the commercial tree | B |

The GTIN row is the structural one. A case of twelve and a single unit are different GTINs, which
is the standards body's way of saying they are different items and not a unit conversion. A
structure that treats them as one series with a conversion factor is defensible, but it is a
choice that must be recorded, because the standard does not make it for you.

### 8.2 Units, which the matrix assumes

| Standard | Governs | Consequence for `S` | Grade |
|---|---|---|---|
| UN/CEFACT Recommendation 20 | Unit of measure codes | Every row of `S` must be dimensionally homogeneous with the bottom block | B |
| ISO 80000 (quantities and units) | Physical quantity semantics | "Cases" and "eaches" are different quantities, not different scales | B |

A summing matrix is a linear map between vectors in one unit. Any node in another unit needs a
factor, and the factor is a modelling assumption rather than a measurement. That is why the
conversion belongs in the structure specification and not in a transformation layer upstream.

### 8.3 Time, which the cross-temporal structure assumes

| Standard or convention | Governs | What it fixes | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation | Week numbering and ordering | B |
| NRF 4-4-5 retail calendar | Retail period structure | Weeks nest into periods by construction | C |
| Gregorian month | Civil calendar | Months of 28 to 31 days | B |

Only the 4-4-5 convention makes weeks nest into months. Without it, the cross-temporal structure
needs an explicit rule for the straddling week, and that rule is a convention, not a standard.
Module 01A topic 04 treats the calendars; here the point is that the temporal summing matrix `R`
is only well defined once the nesting rule is.

### 8.4 Financial reporting, which fixes the top

| Requirement | Where | What it constrains | Grade |
|---|---|---|---|
| IFRS 8 Operating Segments | EU-adopted IFRS, many jurisdictions | Segment disclosures and the reconciliation of segment totals to the financial statements | B |
| ASC 280 Segment Reporting | United States | The same function under US GAAP | B |
| Local statutory consolidation | National company law | The legal-entity hierarchy, which is not a planning choice | B |

The top of a structure that must tie to published accounts is not negotiable. IFRS 8 requires
disclosure of revenue by product and service and by geographical area, and a reconciliation of
segment totals to the entity's statements. That gives at least two mandated aggregation axes at
the top, and they cross, which means the top of the finance structure is grouped whether or not
the planning structure is.

> **Note:** This constrains the *reported* structure, not the *planning* structure. The two can
> differ, and usually should; the bridge between them is a documented mapping, not an identity.

### 8.5 What the standards do not fix

Four things look standardised and are not: the commercial product tree (GPC bricks are
classification, not merchandising); the customer hierarchy (there is no global standard for
"account"); the planning location set (GLNs identify places, not planning nodes); and the bottom
level. None of these is fixed by a standard, and the last is the decision in section 5, step 2.

## 9. Industry practice and case evidence

### 9.1 The reconciliation literature did not start in forecasting

This is the piece of history most practitioners do not know, and it explains why the algebra is
older and better developed than the forecasting applications suggest.

| Period | Contribution | What it established | Grade |
|---|---|---|---|
| 1942 | Stone, Champernowne and Meade formulate national-accounts balancing as simultaneous equations | Aggregation identities as a linear system to be solved, not a table to be edited | A |
| 1961 | Stone, *Input-Output and National Accounts* | Balanced estimates as a weighted combination of initial estimates | A |
| 1978 | Byron, "The estimation of large social account matrices", JRSS A | A computationally efficient projection, `M = I − WC'(CWC')⁻¹C` | A |
| 1982 | van der Ploeg, JRSS A | The same problem over a sequence of matrices, with reliability weights | A |
| 2011 | Hyndman, Ahmed, Athanasopoulos and Shang | The same projection applied to forecasts at every level of a hierarchy | A |
| 2019 | Wickramasuriya, Athanasopoulos and Hyndman | Minimum-trace reconciliation, which turns out to have the same form as Byron's projection | A |

The 2023 review by Athanasopoulos and colleagues makes the lineage explicit: the minimum-trace
solution has the same algebraic form as a national-accounts balancing formula published forty
years earlier, derived from a different motivation and a different literature. That is a useful
thing to know when someone presents reconciliation as a recent invention.

The practical consequence is that the *structure* was always the hard part. National accountants
spent decades on which accounts must balance against which, and the estimator was the easy part
once the constraint set was written. Forecasting inherited the estimator and, in most
implementations, did not inherit the discipline about constraints.

### 9.2 The grouped archetype

The canonical grouped example in the forecasting literature is Australian domestic tourism,
where a geographic hierarchy (Australia, states, zones, regions) is crossed with purpose of
travel (holiday, visiting friends and relatives, business, other). In the version used in
Hyndman's teaching material, 76 to 82 regions crossed with four purposes give 304 bottom-level
series, and every region appears under its state and under its purpose. There is no unique path
from a region-purpose series to the national total.

That example is worth internalising because it is the same shape as product × location and
product × channel. The structure is not a tree with a wide bottom; it is two trees sharing a
bottom.

### 9.3 Scale in public benchmarks

| Benchmark | Structure | Size | Grade |
|---|---|---|---|
| Australian tourism (teaching version) | Region × purpose, grouped | 304 bottom series | A |
| M5 accuracy competition | 12 aggregation levels over product, store, state, category, department | 42,840 series, 30,490 bottom | A |
| M5 history and horizon | Daily sales | 1,941 days, 28-day horizon | A |

The M5 numbers are the most useful calibration available. Twelve levels, 30,490 bottom series,
and a value-weighted metric that weights all twelve levels equally. Two findings transfer to
structure design. First, the bottom level dominates the series count but not the metric, because
each level is weighted equally. Second, no competitive entry estimated a dense covariance over
30,490 series; the winners used simple coherent methods or alignment heuristics. The structure
was large, and the estimator was deliberately small.

### 9.4 What vendors and companies describe

| Source | What it describes | Grade |
|---|---|---|
| SAS conference paper on Electrolux | Forecasts built at product-line level, reconciled to SKU × customer × location; overrides propagate both directions | D |
| SAS conference paper on a US retailer | Department, category, SKU, DC hierarchy; store–SKU reconciled top-down from DC–SKU | D |
| ToolsGroup case study, Ackermans | Forecasting moved from style level to SKU and size, with top-down and bottom-up visibility | D |
| Vendor case study, Mexican consumer goods manufacturer | Region, state, city, route, brand, SKU, customer, with reconciliation to an aggregate plan | D |

Read these as descriptions of what is sold, not as evidence that it worked. The recurring pattern
is consistent and worth noting: every one of them forecasts at an aggregate, reconciles to a
finer level, and allows overrides in both directions. None of the public descriptions states
`n_b`, the covariance method, or how the structure was versioned. Those are the three numbers a
practitioner would need to judge the claim.

### 9.5 What practitioners argue about

**"Coherence is a constraint, not a model."** The purist position: the structure is data, the
forecast is a model, and reconciliation is arithmetic that should never be described as
forecasting. The opposing position: choosing `G` is a modelling decision with a loss function,
and describing it as arithmetic hides the choice. Both are right about different things, and the
argument matters because it determines who owns the reconciled number.

**"Reconcile history too."** If the actuals are incoherent, then scoring a reconciled forecast
against incoherent actuals measures the wrong thing. The awkward consequence is that reconciling
history changes reported actuals, which is politically unacceptable in most organisations. The
usual compromise is to reconcile history only inside the evaluation sandbox.

**"The bottom level should be the decision level."** It sounds obviously true and is often
impossible, because the decision level may have too few observations for any method and too many
series for any covariance.

> **Note:** The volume of published work on reconciliation estimators is much larger than the
> volume on structure specification. That imbalance is a property of the research incentive, not
> of the difficulty. The estimator is a solved problem under assumptions; the structure is a
> negotiation.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Grouped structure described as a tree | Two reports give two totals for the same month and both are "correct" | Product and channel both partition the same leaves | Reconciliation picks a route silently | Build `S`; test column sums above depth plus one |
| A node with two parents summed unweighted | One category total exceeds the company total by a stable amount | Overlapping leaves counted in both branches | Volume overstated, often by 2–8% | Primary-assignment flag, or weights that sum to one per leaf |
| Non-summing node forced into `S` | The residual never reaches zero on one specific node | A mean or ratio cannot be a weighted sum | The estimator moves other nodes to compensate | Remove and compute as a derived metric |
| Case pack entered with weight 1 | Small items chronically over-forecast, large items under | A 12× scale error read as noise | Inventory imbalance across the pack ladder | Assert the coefficient equals the pack quantity |
| Kit treated as a parent of its components | Component demand is roughly double the kit's content | Aggregation and requirements conflated | Over-buy of every kit component | Separate requirements matrix; inequality constraint |
| Structure changes mid-history unversioned | Last quarter's reconciliation no longer reproduces | Node list overwritten in place | Audit trail lost; comparisons invalid | Version key plus effective dates |
| Bottom level is the stored grain | `n_b` in the millions; the covariance is "estimated" by a diagonal | Nobody chose the level | Method runs and means nothing | Choose the bottom level explicitly |
| Node count quoted as the problem size | "We have 150,000 series" with no `n_b` stated | `N` and `n_b` conflated | The real cost is invisible | Report both, and `n_b²` |
| Level with no decision owner reconciled | A level exists only because a system produced it | Coherence enforced where nobody looks | Adjustment cost with no benefit | List the levels a decision consumes |
| Weeks assumed to nest into months | The quarterly plan is not the sum of the monthly plans | Calendar, not arithmetic | Cross-temporal structure ill-defined | State the nesting rule; use 4-4-5 or a straddle rule |
| Empty combinations treated as zeros | A product–region cell with no history forecasts zero and drags its parents down | Absence confused with zero demand | Systematic under-forecast in sparse cells | Mark absent cells and exclude them from `S` |
| A ratio reported as a node | Fill rate moves when the mix changes and nobody can say why | Non-additive quantity in an additive structure | Metric drifts without operational cause | Store numerator and denominator as nodes |
| Structure inferred from data | The "parent" is whichever grouping has more volume | Volume used as a proxy for containment | Structure changes when demand changes | Declare the structure; test it against the data |
| Two structure versions in one report | Totals reconcile in one column and not the next | Two extracts, two node lists | Silent inconsistency | One structure per report, version stated |

Two of these deserve a note. **The two-parent node is the quiet one.** It reconciles internally:
the leaves sum to the parents, and the parents sum to the reported total, because the total was
computed from the same overlapping query. Only an independent distinct count, or the structure's
own column-sum test, exposes it. **The forced non-summing node is the expensive one.** It produces
a residual the estimator cannot remove, so the estimator removes it elsewhere, and every
reconciled number is then slightly wrong in a way that gets attributed to forecast error.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The measure is additive over the structure's edges | `S` is a linear map | Ratios, means, and maxima are not sums | Classify every node by function |
| Every row of `S` is in the bottom series' unit | Matrix multiplication is dimensionally homogeneous | Case versus each, currency mix | Check the coefficients that are not 0 or 1 |
| The bottom level is fixed for the forecast horizon | The matrix has fixed dimensions | Network change, new channel, store closure mid-horizon | Compare structure versions at both ends of the horizon |
| Every node in `S` exists for every period | Padding a missing node with zero is a modelling claim | A product launched in month 4 | Test for structural zeros versus absent nodes |
| The node list is complete | A missing node is invisible to the estimator | A regional node omitted, so its volume disappears | Column-sum test against the source system |
| Weights are stable over the horizon | Mix changes are handled by reconciliation, not by `S` | A pack change from 12 to 10 | Version the weights with the structure |
| Coherence is the relevant constraint | Some structures need inequalities instead | Component supply, capacity, service floors | Ask whether the relation is equality or sufficiency |
| The grouped structure's crossed routes are all real | Some combinations are impossible, not zero | A channel that cannot sell a product | List the combinations that exist |
| `n_b` is small enough to estimate a covariance | Minimum-trace needs an estimator | 120,000 bottom series | Compute `n_b²` before choosing the method |
| The structure the analyst built is the one the business uses | Reconciliation output is consumed by people | Two structures in two systems | Ask the decision owner to name the levels |
| History and forecast use the same structure version | Evaluation compares like with like | Reorganisation mid-window | Reconcile history under the applicable version |
| Rounding does not break the identities | Integer forecasts are often required | Largest remainder applied inconsistently | Recheck `C y = 0` after rounding |

Three limits are worth stating plainly. The matrix guarantees coherence, not meaning: a correct
`S` says the arithmetic is consistent and nothing about whether the grouping suits a decision.
The structure is a snapshot of an organisation, so one without a version will silently disagree
with last year's report. And a grouped structure is not a defect to be repaired — the business
genuinely sells through several channels — the defect is treating it as a tree and summing
without weights, and the repair is a matrix, not a reorganisation.

## 12. Exercises

### Exercise 1 — Build the matrix by hand (application, 40 minutes)

Take the six bottom series and twelve nodes in section 6.1 and construct `S` without a computer.

**Deliverable:** a 12 × 6 grid with 0/1 entries, the four invariants verified in writing, and the
row and column sums.
**Acceptance criteria:** the identity block is in the last six rows; every row sum equals the
subtree size; every column sum is stated and explained; `rank(S) = 6` is argued from the identity
block rather than computed.

### Exercise 2 — Test nesting versus crossing (analysis, 30 minutes)

A distributor has three grouping variables: product family, DC, and customer channel. You are
given a table of 40,000 order lines with one column per variable.

**Deliverable:** for each pair, the count of distinct parents per child, and a classification of
the resulting structure.
**Acceptance criteria:** the query or procedure is stated; each pair is classified as nested or
crossed with the count as evidence; the resulting structure type is named; one impossible
combination is identified and its treatment stated.

### Exercise 3 — Compute the coherence residual (calculation, 35 minutes)

Base forecasts: total 20,000; family A 12,000; family B 9,000; channel X 11,000; channel Y 8,500.
Bottom series: A-X 7,000; A-Y 5,000; B-X 4,000; B-Y 4,500.

**Deliverable:** `S`, `S b`, the residual vector, and its Euclidean norm.
**Acceptance criteria:** the residual is zero on the bottom block; both routes to the total are
computed separately; the difference between them is stated in units and as a percentage.

### Exercise 4 — Two coherent sets (calculation, 40 minutes)

Using Exercise 3's inputs, produce a bottom-up coherent set and a top-down proportional coherent
set.

**Deliverable:** both twelve-vector sets, the scaling factor, and the rounding table.
**Acceptance criteria:** both sets satisfy `C y = 0`; the rounding uses largest remainder and the
residual is stated; the two sets differ at the total by a stated amount.

### Exercise 5 — Diagnose the node type (critique, 30 minutes)

Eight nodes are described: total units, family units, case units, assortment case units, average
selling price, fill rate, kit units, component units.

**Deliverable:** a table classifying each as sum, weighted sum, derived metric, or requirements
relation, with the treatment.
**Acceptance criteria:** every classification is justified by the defining function; the two
nodes that require a separate matrix are named; the unit of each weighted sum is stated.

### Exercise 6 — Repair the mixed case pack (design, 35 minutes)

A product is sold in eaches, in cases of 10, and in an assortment case of 4 of item A and 6 of
item B. A forecast of 50 assortment cases exists.

**Deliverable:** the rows of `S` for all three pack levels, and the resulting bottom-level
quantities.
**Acceptance criteria:** every coefficient equals a pack quantity; the unit of each row is stated;
the arithmetic from 50 assortment cases to A and B quantities is shown.

### Exercise 7 — Size a structure (calculation, 30 minutes)

A structure has 5,000 SKUs, 30 DCs, and three time frequencies.

**Deliverable:** `N`, `n_b`, the number of constraints, the covariance cell count, and the memory
at 8 bytes per cell, for the full cross and for a version that drops intermediate levels.
**Acceptance criteria:** both versions computed; the statement of which quantity changed and
which did not; a recommendation on whether a dense covariance is feasible.

### Exercise 8 — Choose the coherent levels (design, 45 minutes)

A retailer plans at total, department × region, and SKU × store; executes at SKU × store × day;
and reports fill rate by department.

**Deliverable:** a table of levels, consumers, and a coherence decision with a reason, plus the
treatment of fill rate.
**Acceptance criteria:** at least one level is excluded with a reason; fill rate is classified and
its inputs named; the structure type is stated.

### Exercise 9 — Write the structure specification (design, 50 minutes)

Using any structure you know from work, fill in the eight fields of section 5, step 20.

**Deliverable:** a one-page specification.
**Acceptance criteria:** bottom level justified on decision, data, and estimability; every
non-0/1 weight stated with its unit; excluded nodes listed with reasons; version and effective
dates present; a second analyst could build `S` from the page alone.

## 13. Self-check answers

**Exercise 1.** Rows in the order total, Dressings, Sutures, Retail, Hospital, Export, then the
six bottom series. Row sums are 6, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1. Column sums are 4, 4, 4, 4, 4,
4, because every bottom series belongs to the total, one family, one channel, and itself. Depth
plus one is 3 here, not 4, because D-Retail has two parents at level 1, the family and the
channel; the column sum counts ancestors, not levels, so the two coincide only in a strict tree.
Rank is six because the last six rows are the identity, so the six columns are independent.

**Exercise 2.** For each pair, run `count(distinct parent) per child`. Product family against DC
returns one parent per SKU for most SKUs and more than one for items stocked in several DCs, so
the pair is crossed at the stocking level even though the product tree itself is nested. DC
against channel returns several channels per DC, so crossed. Family against channel returns
several channels per family, so crossed. The structure is grouped on all three axes. An
impossible combination is a channel that cannot ship a product, for example an export channel for
an item with no export licence; it must be absent from the node list rather than forecast as
zero, because a zero would drag the family aggregate down.

**Exercise 3.** Bottom: A-X 7,000; A-Y 5,000; B-X 4,000; B-Y 4,500. Sum 20,500.

```text
   S b : total 20,500 | A 12,000 | B 8,500 | X 11,000 | Y 9,500 | bottom unchanged
   y   : total 20,000 | A 12,000 | B  9,000 | X 11,000 | Y 8,500
   r   : total   +500 | A      0 | B   -500 | X      0 | Y +1,000 | bottom 0
   norm: sqrt(500^2 + 500^2 + 1000^2) = sqrt(1,500,000) = 1,224.7 units
```

The product route gives 12,000 + 9,000 = 21,000; the channel route gives 11,000 + 8,500 = 19,500;
the bottom gives 20,500. Three routes, three totals, spread of 1,500 units or 7.3%. That is the
signature of a structure whose aggregates were forecast independently, not of a data error.

**Exercise 4.** Bottom-up: 20,500; A 12,000; B 8,500; X 11,000; Y 9,500. Top-down factor
20,000 / 20,500 = 0.97561: A-X 6,829.3; A-Y 4,878.0; B-X 3,902.4; B-Y 4,390.2. Rounded to
6,829, 4,878, 3,902, and 4,391 (largest remainder on B-Y at 0.2), sum 20,000. Aggregates then read
A 11,707; B 8,293; X 10,731; Y 9,269; total 20,000, and every identity holds. Both sets are
coherent and they differ by 500 units at the total, 2.4%.

**Exercise 5.** Total units and family units are sums with 0/1 weights. Case units are a
weighted sum with the pack quantity as coefficient. Assortment case units are a weighted sum over
two items with two coefficients. Average selling price is a derived metric, computed as revenue
over units after reconciliation. Fill rate is a derived metric, computed from two coherent nodes.
Kit units are a bottom series in the aggregation structure, with a separate requirements relation
to components. Component units are not an aggregation node at all; they belong to the
requirements matrix, with an inequality constraint against the plan.

**Exercise 6.** Three pack rows: each row has coefficient 1 on the each column; case row has
coefficient 10; assortment row has coefficient 4 on A and 6 on B. Fifty assortment, or 200 A and
300 B, are bottom quantities. Two traps. If the assortment row is written as 1 on a single
"assortment" column, the composition is lost and A and B cannot be planned. If the case and each
rows are both summed into the same total, the volume is counted twice; carry one unit through the
structure and express the others as weights.

**Exercise 7.** `n_b` = 5,000 × 30 = 150,000. With a total above the SKUs and a total above the
DCs, product nodes are 5,001 and location nodes 31, so `N` = 155,031 and the constraints,
`N − n_b`, number 5,031. Covariance cells are `n_b²` = 2.25 × 10¹⁰, or 180 GB at 8 bytes per cell.
The three frequencies multiply every count by three and change no conclusion. Dropping
intermediate levels lowers `N` and leaves `n_b` at 150,000, so the covariance is unchanged; the
only lever that moves it is the bottom level. A dense covariance is not feasible, so the method
must use a diagonal, a shrinkage estimator, or a coarser bottom level.

**Exercise 8.** Coherent: total, department × region, SKU × store. Excluded: SKU × store × day,
because it is a scheduling profile inside the week and no other level consumes it. Fill rate is
a derived metric, computed from units served and units demanded, both of which are coherent
nodes at department level. The structure is grouped, department × region crossed with SKU ×
store, and it is cross-temporal only if the day profile is reconciled, which by the decision
above it is not.

**Exercise 9.** A strong specification states the measure and unit, the bottom level with its
three justifications, every grouping variable with its levels and owner, the structure type, `N`
and `n_b`, every non-unit weight with its unit, the excluded nodes with reasons, and a version
with effective dates. A weak specification lists the levels and says "standard hierarchy", which
is not implementable.

## 14. Interview questions

**1. "What is a hierarchy, formally?"**

Strong: a set of nodes, a parent relation, and weights, with a designated bottom level; the whole
thing is a linear map from bottom series to nodes, written as a matrix. Weak: "a way of grouping
products", which describes a diagram rather than an object.

**2. "What is the difference between a tree and a grouped structure?"**

Strong: a tree has one route from every leaf to the root; a grouped structure has several, because
two grouping variables are crossed over the same leaves, and the matrix then has a column sum
greater than depth plus one. Weak: "a grouped structure is a hierarchy with more levels".

**3. "What does coherent mean, and what does it not mean?"**

Strong: every parent equals the sum of its children at every level and bucket, equivalently the
vector lies in the column space of `S`; it says nothing about accuracy, and a coherent set can be
worse than an incoherent one. Weak: "the numbers add up", with no distinction from being right.

**4. "A category total exceeds the company total. What is the first thing you check?"**

Strong: the structure, not the query — a node with two parents, counted in both branches; then an
independent distinct count. Weak: rerun the query and check the join condition.

**5. "Why does a summing matrix always have full column rank?"**

Strong: because the bottom rows form an identity block, so the columns are linearly independent,
so `S'S` is invertible and both the projection and the least-squares solution always exist; the
interesting errors are in the correspondence to the business, not the algebra. Weak: "because it
is a matrix of ones and zeros".

**6. "How would you handle a case pack of twelve?"**

Strong: as a weight of 12 in the row, with the unit stated, or as a separate bottom series with
its own column; the failure mode is a weight of 1, which turns a scale error into apparent
forecast noise. Weak: "convert everything to eaches upstream and forget it", without stating
where the conversion lives.

**7. "A kit is sold as one item and built from three components. Where does it go in the
hierarchy?"**

Strong: the kit is a bottom series with its own demand; the components belong to a separate
requirements matrix with an inequality constraint, not to the aggregation structure. Weak: "the
kit is the parent of its components", which double counts the components.

**8. "How many series are in a hierarchy of 3,000 SKUs and 40 DCs?"**

Strong: asks what the intermediate levels are, then gives `n_b` = 120,000 and notes that `N`
depends on the levels kept, and that `n_b²` is 14.4 billion covariance cells. Weak: "120,000",
without distinguishing bottom count from node count.

**9. "Which levels need coherent forecasts?"**

Strong: the levels a decision consumes where another decision consumes a different level in the
same structure; a level consumed by nobody is a candidate for exclusion. Weak: "all of them,
because coherence is a property of the whole".

**10. "Why does reconciliation need a matrix rather than a set of rules?"**

Strong: because every linear method is `ỹ = S G ŷ`, methods differ only in `G`, and unbiasedness
is the single identity `S G S = S`; rules cannot be compared or checked without the matrix. Weak:
"because the software needs a data structure".

## 15. Key terms

- **Structure** — nodes, parent relation, weights, and a bottom level; the object `S` encodes.
- **Summing matrix** — the `N × n_b` map from bottom series to every node.
- **Aggregation matrix** — the top `N − n_b` rows of `S`.
- **Identity block** — the bottom rows of `S`; the reason `S` has full column rank.
- **Constraint matrix** — `C = [I | −A]`, whose null space is the coherent subspace.
- **Coherent vector** — a vector in the column space of `S`.
- **Coherence residual** — `r = ŷ − S b`, the part of a forecast outside the subspace.
- **Bottom series** — the finest series in the structure; the columns of `S`.
- **Aggregate node** — a node defined as a weighted sum of bottom series.
- **Tree** — one route from each leaf to the root.
- **Grouped structure** — crossed trees over the same leaves; several routes.
- **Cross-temporal structure** — a structure replicated at several frequencies.
- **Temporal summing matrix** — `R`, mapping the finest bucket to coarser ones.
- **Kronecker product** — the operation composing `S` and `R` into `F = S ⊗ R`.
- **Projection matrix** — `P = S G`, mapping any vector into the coherent subspace.
- **Mapping matrix** — `G`, the only thing reconciliation methods differ on.
- **Unbiasedness condition** — `S G S = S`.
- **Structural scaling** — weights proportional to node size.
- **Mixed case pack** — a case of several items in stated quantities.
- **Kit** — an assembly with its own demand and a bill of materials.
- **Requirements matrix** — `M`, mapping end items to components; not an aggregation.
- **Non-summing node** — a mean, ratio, or maximum; not a row of `S`.
- **Derived metric** — a quantity computed after reconciliation, never reconciled.
- **Partition** — non-overlapping division; **cover** — division permitting overlap.
- **`N` and `n_b`** — node count and bottom count; the second is the problem size.
- **Covariance cells** — `n_b²`, the object a minimum-trace estimator must estimate.
- **Structure version** — identifier plus effective dates for a changing structure.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Stone, R., Champernowne, D. G., and Meade, J. E. (1942). National income estimation and the balancing of accounts as a simultaneous system. Cited in the 2023 review below.
- Stone, R. (1961). *Input-Output and National Accounts*. O.E.E.C., Paris.
- Byron, R. P. (1978). "The Estimation of Large Social Account Matrices." *Journal of the Royal Statistical Society, Series A*, 141(3), 359–367.
- van der Ploeg, F. (1982). "Reliability and the Adjustment of Sequences of Large Economic Accounting Matrices." *Journal of the Royal Statistical Society, Series A*, 145(2), 169–194.
- Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., and Shang, H. L. (2011). "Optimal combination forecasts for hierarchical time series." *Computational Statistics and Data Analysis*, 55(9), 2579–2589. DOI: 10.1016/j.csda.2011.03.006.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Petropoulos, F. (2017). "Forecasting with temporal hierarchies." *European Journal of Operational Research*, 262(1), 60–74.
- Wickramasuriya, S. L., Athanasopoulos, G., and Hyndman, R. J. (2019). "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *Journal of the American Statistical Association*.
- Wickramasuriya, S. L., Turlach, B. A., and Hyndman, R. J. (2020). "Optimal non-negative forecast reconciliation." *Statistics and Computing*, 30(5), 1167–1182.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "M5 accuracy competition: Results, findings, and conclusions." *International Journal of Forecasting*, 38(4), 1346–1364. DOI: 10.1016/j.ijforecast.2021.11.013.
- Di Fonzo, T., and Girolimetto, D. (2023). "Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives." *International Journal of Forecasting*, 39(1), 39–57.
- Girolimetto, D., Athanasopoulos, G., Di Fonzo, T., and Hyndman, R. J. (2024). "Cross-temporal probabilistic forecast reconciliation." *International Journal of Forecasting*, 40(3), 1134–1151.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Panagiotelis, A. (2024). "Forecast reconciliation: A review." *International Journal of Forecasting*, 40(2), 430–456.

**Books, textbooks, and practitioner canon — Grade B or C**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition, chapters on hierarchical and grouped time series. Grade B.
- Hyndman, R. J., and Athanasopoulos, G. (2014). "Optimally Reconciling Forecasts in a Hierarchy." *Foresight*, Fall 2014. Grade C.
- Hyndman, R. J. (2022). "Notation for forecast reconciliation." Personal technical note. Grade C.

**Standards and official — Grade B**

- GS1 General Specifications, current edition, for GTIN and GLN; GS1 Global Product Classification documentation, gs1.org.
- UNECE Recommendation No. 20, *Codes for Units of Measure Used in International Trade*.
- UNECE Recommendation No. 16, *UN/LOCODE*, Revision 4 (2020), ECE/TRADE/459.
- ISO 8601, *Date and time — Representations for information interchange*; ISO 80000, *Quantities and units*.
- IFRS 8 *Operating Segments*; ASC 280 *Segment Reporting* (FASB).

**Vendor, software, and case material — Grade D or E**

- FoReco R package documentation, cross-temporal reconciliation functions. Grade D.
- The `hts` and `gts` R packages, software documentation. Grade C.
- SAS conference paper, "Demand-Driven Planning and Optimization at Electrolux." Grade D.
- SAS conference paper, retail forecasting case study. Grade D.
- ToolsGroup case study, Ackermans. Grade D.
- Vendor case study, Mexican consumer goods manufacturer. Grade E.

**Note on citations.** The M5 figures are reported as the competition documentation states them,
and the Australian tourism series count as it appears in the teaching material, not as a measured
statistic. The Halvorsen figures in section 6 are constructed for teaching and every sum is
shown. No source measures how often structures are mis-specified in practice; the claims in
section 10 are inferences from the algebra, not prevalence estimates.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Part map for Methods: [`../_index.md`](../_index.md)
- Survey file for this module:
  [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- Predecessor, hierarchy and aggregation at the forecasting level:
  [`../07_time_series_forecasting/05_hierarchy-aggregation.md`](../07_time_series_forecasting/05_hierarchy-aggregation.md)
- Predecessor, hierarchy design for dimensional modelling:
  [`../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md`](../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md)
- Unit-of-measure foundations, needed for the pack-level nodes:
  [`../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file for the forecasting module:
  [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Next topic, aggregation consistency and the classical reconciliation methods:
  [`02_aggregation-consistency.md`](02_aggregation-consistency.md)
- Minimum-trace reconciliation in detail: `03_mint-reconciliation.md` (planned)
- Temporal and cross-temporal structures: `04_temporal-cross-temporal.md` (planned)
- Intermittent demand, where the bottom level usually lives: `05_intermittent-classification.md` (planned)
- Probabilistic forecasts and their decision use: `07_probabilistic-forecasts.md` (planned)
