> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 07 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01, 01A.04

---

# Hierarchy Design

---

## 1. Why this topic exists

Every number a supply chain analyst reports is a rollup. Revenue by category, inventory by
region, service level by customer group, cost per case by lane: each is a sum or an average
taken over a hierarchy, and in most reports the hierarchy is doing more work than the
arithmetic.

A hierarchy is not a filing convention. It is a claim that a set of numbers may be added
together. The claim is either true or false, and when it is false the error is invisible,
because every row is valid, every subtotal is a correctly computed sum, and the total looks
plausible. This is the same failure pattern as Topics 03 and 04, in a third dimension.

The module has now discretised three things:

| Topic | What is discretised | Unit introduced | Failure if it is wrong |
|---|---|---|---|
| 01A.01 | Quantity | each, case, layer, pallet | Plans and costs that cannot be executed |
| 01A.04 | Time | day, week, 4-5-4 period, fiscal year | Trends and variances measured on incomparable buckets |
| 01A.07 | Membership | node, level, path, grain | Totals that double count or silently lose volume |

Quantity, time, and membership are the three axes of every aggregate. A number without a unit is
not a measurement; a number without a period is not a rate; a number without a hierarchy is not
a total.

Four failures set the shape of the problem, all drawn from the worked example in section 6.

**Failure one: the category total exceeds the company total.** A distributor reports 40,000 in
category revenue against 37,000 in total revenue for the same month. Both figures come from
correct `SUM` statements. One post-operative dressing is carried in two commercial categories
because it is sold standalone and inside a procedure pack, so the category query returns it
twice. Nothing in either query is wrong; the mapping is a cover rather than a partition.

**Failure two: header-level values multiplied by line-level rows.** An order carries one freight
charge of 240 and three lines. Joining the header to its lines and summing freight returns 720,
so the freight ratio reads 28.8% instead of 9.6%. This is a fan trap, the most common defect in
dashboards built directly on transaction tables.

**Failure three: two fact tables joined through a shared dimension.** Sales and returns both key
on product. Joined directly they produce a row for every sales-return pair, so each side inflates
by the other's row count. A return rate of 13.3% is reported as 20.0%. This is the chasm trap, and
`COUNT(DISTINCT)` does not repair it because it fixes counts and leaves every sum wrong.

**Failure four: a balance summed over time.** Month-end inventory of 1,000, 1,100, and 1,050 units
is reported as 3,150 units. Inventory is additive across locations, not across time. The number
3,150 is a valid sum of three valid numbers and is not an inventory.

**Why this belongs in a foundations module.** Every later model consumes an aggregate. Forecast
error metrics in 01A.09 are computed at a level. Hierarchical forecasting in Module 08 reconciles
levels and depends on the same identities. Network design in Module 11 allocates shared capacity
across regions. Assortment analytics in Module 17 lives or dies on category definitions. If the
hierarchy is wrong, each of those models is fitted to a number that does not mean what its name
says, and the resulting error is attributed to the model.

**Where it leads.** 01A.08 separates measurement error from bias, which requires knowing what was
measured. 01A.09 and 01A.10 compute error metrics whose denominators are aggregates. Module 08
turns the reconciliation identity into a forecasting constraint.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a partition from a cover in a hierarchy, and **predict** from the shape alone
   whether a rollup will reconcile.
2. **Declare** the grain of a fact table in one sentence in business terms, and **test** whether
   a second analyst could disagree with the declaration.
3. **Identify** fan traps and chasm traps in a schema, **compute** the inflation factor for each,
   and **rewrite** the query as a drill-across.
4. **Classify** a measure as additive, semi-additive, or non-additive, and **state** the
   dimensions over which each classification holds.
5. **Choose** among flattened columns, parent-child recursion, a closure or bridge table, and
   path enumeration for a given hierarchy shape, with the query cost and failure mode stated.
6. **Handle** a ragged hierarchy without inventing a level, and **explain** what a placeholder
   member is and why unknown, not applicable, and not assigned must be distinct.
7. **Design** an allocation for a shared node, choose a driver on causal grounds, and **show**
   that the allocated shares sum to the pool.
8. **Reconcile** any level of a hierarchy to its parent using the four tests in section 5, and
   **diagnose** which test fails from the pattern of the residual.
9. **Model** a hierarchy that changes over time, and **state** whether a report shows the
   hierarchy as it was or as it is.
10. **Write** a hierarchy specification containing grain, levels, parent rule, placeholder policy,
    measure additivity, allocation rule, and effective dates.

Outcomes 2 to 8 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures. The first is a set diagram: a hierarchy is a claim about which sets contain
which. The second is a four-shape taxonomy that tells you what will break. The third is an
identity with three conditions, each of which is a test.

### 3.1 A hierarchy is a partition claim

A partition divides a set into non-overlapping parts. A cover allows overlap. Additive rollups
require a partition at every level. Almost every hierarchy defect is a cover pretending to be a
partition.

```text
   PARTITION — reconciles                    COVER — does not reconcile

          TOTAL 130                                TOTAL 190?
          /       \                                /       \
      CAT A      CAT B                        CAT A      CAT B
      /   \        |                         /   \       |   \
    s1    s2      s3                      s1    s2     s3   s2'
    40    60      30                      40    60     30   60
                                                             ^
   every leaf has exactly one parent      s2 is listed under both categories

   sum(leaves)  = 40+60+30        = 130    sum(leaves)  = 40+60+30+60 = 190
   sum(parents) = 70+30           = 130    sum(parents) = 100+90    = 190
   both equal the total                   both equal the total, but only
                                          130 units exist
```

Read the right-hand diagram twice. The cover *does* reconcile internally: the leaves sum to the
parents, and the parents sum to the reported total. That is why the defect survives review. The
total is internally consistent and externally wrong, because the total was recomputed from the
overlapping members rather than compared with an independent count.

The test that catches it is not internal consistency. It is an independent measure of the same
population, such as a distinct count of items, a count of order lines, or a physical stock count.

### 3.2 Four shapes, four failure modes

```text
   (a) BALANCED TREE        (b) RAGGED TREE         (c) DAG / SHARED NODE
       every leaf at           leaves at different      one node with two parents
       the same depth          depths

        root                    root                     root
       /    \                  /    \                   /    \
      A      B                A      B                 A      B
     / \    / \              / \      \               / \    / \
   a1  a2 b1  b2           a1  a2     b1            a1  a2 b1  b2
                              \                       \    /
                               a1x                     \  /
                                                        s   <- shared

   (d) LATTICE — many-to-many at the same level, usually with a weight
       product <---> category, with an allocation percentage per pair
```

| Shape | Typical example | Rollup behaviour | First thing to check |
|---|---|---|---|
| Balanced tree | Fiscal calendar: year, quarter, month, week | Reconciles at every level | Depth is genuinely equal, not padded |
| Ragged tree | Organisation: some spans of control are two levels, some five | Reconciles if each leaf has one root path | Level-skip policy and placeholder members |
| Directed acyclic graph (DAG) | Product used in several assemblies; DC serving several regions | Double counts unless weighted | Is the weight a share that sums to one? |
| Lattice | Item-to-category with dual listing; customer-to-sales-territory overlap | Does not reconcile by construction | Whether a primary assignment exists |

The DAG and lattice cases are not defects. Shared components genuinely belong to several
products, and a hospital genuinely buys through two contracts. The defect is treating them as
trees and summing without a weight.

### 3.3 The reconciliation identity

For a measure `m`, a set of leaves `L`, and a parent function `p`:

```text
   sum over l in L of m(l)  =  m(root)

   iff  (i)   every fact row is assigned to exactly one leaf
        (ii)  every leaf has exactly one path to the root
        (iii) m is additive over the levels traversed
```

Each condition is a separate test, and each fails in a recognisable way.

| Condition | Test | Residual pattern when it fails |
|---|---|---|
| (i) Single assignment | Count fact rows with no leaf key, or with more than one | Rollup smaller than the leaf total (orphans) or larger (duplicates) |
| (ii) Single path | Count leaves with more than one parent, or with none | Rollup larger than the total at the duplicated branch only |
| (iii) Additivity | Ask whether summing two rows is meaningful | Total is a sum of valid numbers with no unit that describes it |

The third condition is the one analysts skip, because it is not a data quality check. It is a
semantic check on the measure, and section 3.4 gives the classification.

### 3.4 Additivity belongs to the measure, not the column

A column type is not additive. A measure is additive over a stated set of dimensions.

| Measure | Across product | Across location | Across time | Correct aggregate |
|---|---|---|---|---|
| Units shipped | Yes | Yes | Yes, within a period | Sum |
| Revenue | Yes | Yes | Yes | Sum |
| Order count | Yes, if an order belongs to one category | Yes | Yes | Sum or distinct count, stated |
| Inventory on hand | Yes | Yes | **No** | Last value, average, min, or max |
| Headcount | Yes | Yes | **No** | Last value or average |
| Backlog | Yes, with care | Yes | **No** at a point; flow over a period | Last value, or sum of new minus cleared |
| Gross margin % | **No** | **No** | No | Sum of margin / sum of revenue |
| Price per unit | **No** | **No** | No | Sum of extended value / sum of units |
| Forecast accuracy | **No** | **No** | No | Recompute from summed errors |
| On-time in full (OTIF) | **No** | **No** | No | Recompute from summed counts |
| Distinct customers served | **No** across overlapping sets | No | No | Distinct count at the reporting grain |

Two rules follow, and they cover most of the ground:

1. **Store the components of a ratio, never only the ratio.** A margin percentage cannot be
   aggregated; a margin amount and a revenue amount can.
2. **A balance is a state, not a flow.** Anything measured "as of" a date is semi-additive at
   best, and summing it over dates produces an observation count wearing the name of a stock.

> **Note:** The common phrase "additive across all dimensions" is a statement about the
> dimensions in one fact table. A measure can be additive there and non-additive in a different
> model where the grain differs. Always attach the additivity claim to a grain.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system, because a definition without one is the
defect this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Hierarchy | A set of nodes related by containment, used to aggregate measures | Treated as a folder structure | Level list plus parent relation |
| Level | A named depth in a hierarchy, such as category or region | Assumed to be the same across systems | Ordinal integer, root = 0 |
| Node | One member of a hierarchy, with a key and a name | Called a "dimension" | Surrogate key plus natural key |
| Root | The single node that contains all others | Assumed to exist; often absent in source data | One node key per hierarchy |
| Leaf | A node with no children; the level at which facts are recorded | Confused with the lowest level that happens to be populated | Node key, depth = maximum |
| Parent node | The node that directly contains a given node | Assumed unique | One node key, or many in a DAG |
| Child node | A node directly contained by a given node | Assumed to be a fact owner | Node key |
| Path | The ordered sequence of nodes from a leaf to the root | Stored as a display string and parsed later | Node key sequence |
| Depth | The number of edges from a node to the root | Used interchangeably with level | Integer, root = 0 |
| Grain | The business meaning of one row in a fact table | Described as a list of keys | One sentence in business terms |
| Fact table | A table of measurements at a declared grain | Used for any numeric table | Grain declaration plus measures |
| Dimension table | A table of descriptive attributes joined to facts | Assumed to be normalised | Surrogate key plus attributes |
| Conformed dimension | A dimension reused across fact tables with identical attribute names and domains | Confused with a shared key | Attribute name and domain |
| Drill-down | Moving from a parent to its children | Confused with filtering | Level change, one step |
| Roll-up | Aggregating from children to a parent | Confused with any `GROUP BY` | Level change, one step |
| Drill-across | Running separate queries on separate fact tables and aligning them on conformed row headers | Replaced by a join | Sort-merge on shared headers |
| Bridge table | A table storing every ancestor-descendant pair, with depth | Assumed to hold only direct parents | `(ancestor, descendant, depth)` |
| Closure table | A bridge table that includes the zero-depth self row | Omitted self rows, losing leaf-level facts | `(ancestor, descendant, depth)` |
| Many-to-many relationship | A relationship where one member of each side relates to several of the other | Modelled as a tree | Associative table with optional weight |
| Fan trap | Inflation caused by joining a one-side table to two or more many-side tables in one query | Called a "join bug" | Row multiplication factor |
| Chasm trap | Inflation caused by joining two fact tables through a shared dimension | Called a "cartesian product" | Row multiplication factor per side |
| Additive measure | A measure that can be summed across every dimension in its fact table | Assumed for every numeric column | Measure name plus dimension list |
| Semi-additive measure | A measure additive across some dimensions but not time | Summed over dates in reports | Measure name plus excluded dimension |
| Non-additive measure | A measure that cannot be summed across any dimension without recomputation | Averaged across rows | Numerator and denominator |
| Allocation | Assigning a shared cost or quantity to receiving nodes by a rule | Presented as a measurement | Pool, driver, shares |
| Shared node | A node that serves more than one branch of another hierarchy | Counted once in each branch | Node key plus service list |
| Cost driver | The observable quantity used as the allocation base | Chosen for convenience | Quantity with unit |
| Double counting | Including the same underlying volume in a total more than once | Detected only by external comparison | Ratio of reported to distinct total |
| Orphan record | A fact row whose dimension key has no matching node | Silently dropped by inner joins | Row count |
| Surrogate key | A system-generated integer identifying a dimension row | Exposed to business users | Integer |
| Natural key | The business identifier of a member | Assumed unique across systems | Source code, such as a GTIN |
| Slowly changing dimension (SCD) | A dimension that records attribute change over time | Overwritten in place | Type 1 or type 2 with effective dates |
| Type 2 change | A change recorded by adding a row with validity dates | Applied without an as-of rule | `valid_from`, `valid_to` |
| As-was / as-is reporting | Reporting with the hierarchy in force then, versus now | Chosen by accident | Report parameter plus version |
| Reconciliation test | A check that a rollup equals an independent total | Replaced by checking that the query runs | Pass condition per test |
| Partition | A division into non-overlapping subsets | Assumed whenever a hierarchy is drawn | Set of disjoint leaves |
| Cover | A division that permits overlap | Mistaken for a partition | Set of possibly overlapping leaves |
| Segment / family / class / brick | The four GS1 Global Product Classification levels | Used as a commercial category tree | 8-digit GPC codes |
| GTIN | Global Trade Item Number, the identifier of a trade item | Treated as a product hierarchy | 8, 12, 13, or 14 digits |
| GLN | Global Location Number, the identifier of a party or place | Treated as a sales territory | 13 digits |
| UN/LOCODE | United Nations Code for Trade and Transport Locations | Treated as an address | 5 characters |
| ISO 3166-2 | The standard for country subdivision codes | Confused with ISO 3166-1 country codes | 1–3 alphanumeric characters |
| Level skip | A ragged branch that omits a level present elsewhere | Filled with an invented member | Depth difference |

Two of these deserve expansion.

**Grain versus level.** A level is a position in a hierarchy. A grain is a statement about a
fact table. They are related but not interchangeable: a fact table can be at the leaf grain of
the product hierarchy and at the month grain of the time hierarchy, which is two different
levels in two different trees. Saying "the data is at category level" is ambiguous until you say
which hierarchy.

**Allocation versus measurement.** An allocated number is a decision about who bears a shared
cost. A measured number is an observation. They can look identical in a report, and the
difference matters when somebody uses the number to price a service or judge a manager.

## 5. The mechanics, step by step

This section builds a hierarchy from source tables to a reconciled report. Each step ends with a
test, because a hierarchy that has not been tested is a hypothesis.

### Phase A — Declare the grain

**Step 1. Write the grain in one sentence, in business terms.**

Not "the primary key is order_id, line_id, sku_id". That is a list of columns, and it can be
satisfied by rows that mean nothing. Write: "one row is one shipment line on one invoice, for one
stock-keeping unit (SKU), despatched from one distribution centre (DC), on one calendar day."

Test: give the sentence to a second analyst and ask what a row with two invoices would look
like. If they can construct a plausible row that the sentence forbids, or one the sentence
permits that the business would call two events, the grain is not declared.

> **Note:** Kimball's rule is that the grain is declared before dimensions and measures are
> chosen, because every later decision must be consistent with it. Choosing the grain last
> guarantees that some measures will be true at a different grain from the one the table has.

### Phase B — Inventory the hierarchies

**Step 2. List every hierarchy that touches the measure.**

There is never one. A distributor typically has at least four product trees: the commercial
category tree used by sales, the classification tree used for regulatory or clinical reporting,
the procurement tree used for supplier agreements, and the planning tree used for forecasting.
They disagree, and each is authoritative for a different decision.

| Tree | Owner | Used for | Typically disagrees on |
|---|---|---|---|
| Commercial category | Merchandising or sales | Pricing, promotions, assortment | Where a dual-use item sits |
| Classification | Regulatory or clinical | Reporting, recall scope, tender eligibility | Granularity, not placement |
| Procurement | Category management or purchasing | Supplier terms, spend analysis | Whether an item is a component or a finished good |
| Planning | Demand planning | Forecast grouping, replenishment | Whether items are substitutable |

**Step 3. Name the hierarchy of record for each measure, and version it.**

A measure has one reporting hierarchy. Inventory value reports on the financial tree, not the
commercial one, because the financial tree determines the accounts. Units sold report on the
commercial tree. The choice is not technical and does not belong to the analytics team alone.

Test: for each measure in the report, name the tree and the version date. If two measures in one
table use different trees at the same level, the table is not internally comparable and the
column header must say so.

### Phase C — Determine the shape

**Step 4. Test for multi-parent nodes.**

Count nodes with more than one parent. This single number decides whether the hierarchy is a tree
or a DAG, and therefore whether every rollup needs a weight.

| Count of nodes with more than one parent | Shape | Consequence |
|---:|---|---|
| 0 | Tree or ragged tree | Unweighted sum is valid |
| 1 or more, weights sum to 1 per leaf | Weighted DAG | Unweighted sum overstates by the overlap |
| 1 or more, no weights | Cover | No valid total exists until a primary assignment is chosen |

**Step 5. Test for raggedness.**

Compute the depth of every leaf. If the set of depths has more than one value, the hierarchy is
ragged. Ragged hierarchies are normal in organisations, product families, and geographic
structures. They are not normal in a chart of accounts, and their presence there usually
indicates a mapping error.

Test: `min(depth) = max(depth)` over leaves. Record both numbers even when equal, so a later load
that introduces a deep branch is detected.

**Step 6. Choose a representation.**

| Representation | Shape it handles | Query pattern | Failure mode |
|---|---|---|---|
| Flattened level columns | Balanced tree | Direct `GROUP BY` on a level column | A ragged branch forces a `NULL` or an invented member |
| Parent-child plus recursive CTE | Any tree | Recursive traversal per query | Cost grows with depth; cycles hang the query |
| Closure or bridge table | Any tree or DAG | Join on `ancestor`, filter on `depth` | Table size is `nodes × average descendants`; needs rebuild on change |
| Path enumeration | Any tree | Prefix match on a string | Separator collisions; no index-friendly depth filter |
| Graph or native hierarchy type | Any shape | Vendor-specific traversal | Portability; hidden recursion limits |

The choice is a trade between rebuild cost and query cost. A closure table on a 50,000-node
product hierarchy is a few hundred thousand rows and answers any rollup in one join. A recursive
query on the same hierarchy runs the recursion every time the dashboard refreshes.

**Step 7. Decide the raggedness policy before the first query.**

Four policies are defensible, and they produce different totals:

- **Pad with a placeholder** so every leaf sits at maximum depth. Simple; invents members such as
  "Unassigned level 3".
- **Use a bridge table** and filter on depth from the leaf. No invented members; every rollup is a
  path query.
- **Use `HideMemberIf` semantics**, hiding placeholder members in drill-down while keeping them
  in the data.
- **Store depth per leaf** and aggregate at the level where the node has children. Level-free,
  but every query must handle it.

> **Note:** A padded hierarchy and a bridge table can produce the same total and different
> level counts. If a report says "three levels", ask whether that is three real levels or three
> padded ones. The distinction matters for any per-level normalisation.

### Phase D — Assign facts to leaves

**Step 8. Enforce referential integrity and measure the orphan rate.**

Every fact row must resolve to exactly one leaf. An inner join silently deletes the rows that do
not, which is worse than an error, because the report still runs.

Test: left join facts to the leaf dimension and count nulls. A stable orphan rate near zero is a
maintenance cost. A rising orphan rate is a mapping backlog, usually caused by new items
introduced without a category assignment.

**Step 9. Measure the unassigned share, not just the orphan count.**

New items are often mapped to a temporary placeholder so that the load succeeds. The resulting
report reconciles perfectly and attributes real volume to a fake member. Track the share of
volume in each placeholder, by month, and set a threshold above which the report is suppressed.

### Phase E — Define placeholder semantics

**Step 10. Separate unknown, not applicable, and not assigned.**

| Member | Meaning | Example | Report treatment |
|---|---|---|---|
| Unknown | The attribute exists but its value is not known | Supplier country not captured at onboarding | Show; investigate |
| Not applicable | The attribute cannot apply to this member | Storage temperature for a non-perishable item | Hide from temperature reports |
| Not assigned | The value is known but has not been mapped | New item awaiting category assignment | Suppress the rollup; treat as a defect |

A single "Unknown" row conflates three different operational problems and makes the residual
unattributable. It also tends to grow, because it is the path of least resistance for every
integration.

### Phase F — Classify the measures

**Step 11. Classify every measure by additivity, and record the classification next to the
column.**

Use the test in section 3.4. For each measure, write down the dimensions over which it may be
summed and the correct aggregate for the rest. Ratios are stored as numerator and denominator.
Balances are stored at a snapshot grain and aggregated with last-value, average, minimum, or
maximum, each labelled.

### Phase G — Reconcile

**Step 12. Run the four reconciliation tests, in this order.**

| Test | Statement | Pass condition | What failure means |
|---|---|---|---|
| R1 Leaf-to-total | Sum of leaves equals the independently computed total | Exact, or within stated rounding | Orphans, duplicates, or a filter applied to one side only |
| R2 Level-to-level | Each parent equals the sum of its children at every level | Exact, per parent | A cover, a multi-parent node, or a missing level |
| R3 Row arithmetic | Units × unit price equals amount, per row | Within rounding tolerance | Mixed units or a currency defect |
| R4 Path uniqueness | Every leaf has exactly one path to the root, and the path length equals its depth | One row per leaf per depth | Cycle, duplicate edge, or an incomplete closure table |

Test R2 is the one that finds the dual-listed item, because it fails only at the branch where the
overlap sits. Test R1 passes in that case if the "total" was also computed from the same
overlapping query. That is why an independent total is required, and why the independent total is
usually a distinct count or a physical measure.

### Phase H — Handle shared nodes and allocation

**Step 13. Identify shared nodes and choose a driver on causal grounds.**

A shared node serves more than one branch. A central sterilisation hub serves three regions; a
common component appears in four products; a sales team covers two channels.

Choose the driver in this order: what causally consumes the resource, then what is proportional
to that consumption, then what is available. Cost accounting has a name for the failure of
choosing convenience over causality: it produces a cross-subsidy, and the subsidised branch
looks cheaper than it is.

**Step 14. Allocate, and make the residual explicit.**

Allocate by `share = driver_i / sum(driver)`, then handle rounding with the largest-remainder
method so that the shares sum exactly to the pool. Publish the pool, the driver, and the shares.
An allocated cost that cannot be traced to a pool and a driver is a number with no provenance.

> **Note:** Allocation is not measurement. Two defensible drivers can differ by tens of percent,
> and both are correct allocations of the same pool. The number answers "who should bear this
> cost", not "what did this node consume".

### Phase I — Handle change over time

**Step 15. Decide whether the report is as-was or as-is, and store the hierarchy accordingly.**

A product moves from one category to another in March. A January report run in December can show
January sales under the new category (as-is) or the old one (as-was). Both are legitimate. They
answer different questions, and the difference is usually material around a reorganisation.

A type 2 slowly changing dimension (SCD) stores each version of the member with `valid_from` and
`valid_to`, and the fact row carries the version key that was current at the event. A type 1
overwrite loses the history and makes as-was reporting impossible.

### Phase J — Publish the specification

**Step 16. Write the hierarchy specification.**

| Field | Content | Example |
|---|---|---|
| Name and owner | Who is accountable | Commercial product hierarchy, merchandising |
| Grain | What one row of the fact table is | One shipment line per SKU per DC per day |
| Levels | Ordered list, root first | Total, category, subcategory, SKU |
| Parent rule | How a parent is chosen, including duplicates | Primary listing flag; dual listings flagged |
| Placeholder policy | Member keys and report treatment | UNK, NA, UNM; UNM suppressed |
| Measure additivity | Per measure, per dimension | Units additive; margin % non-additive |
| Allocation rule | Pool, driver, rounding | Hub cost by units, largest remainder |
| Effective dates | Version and as-of behaviour | Version 2026-02; as-was by default |

The specification is short. Its value is that a dispute about a total is resolved by reading one
page rather than by re-deriving the mapping from four systems.

## 6. Worked example

Meridian Health Supply distributes medical consumables through three distribution centres and
two customer channels. The figures are illustrative and internally consistent; every total in
this section reconciles to the table in 6.2 unless the text says it does not.

### 6.1 The three trees

| Tree | Levels | Used for | Rows at leaf |
|---|---|---|---|
| Commercial product | Total, category, subcategory, SKU | Revenue, margin, assortment | 12 SKUs |
| Location | Network, region, DC, ship-to | Inventory, cost-to-serve, service | 3 DCs, 9 ship-to points |
| Customer | Channel, account, ship-to | Pricing, terms, service | 2 channels, 7 accounts |

The product tree is the one that fails in this example, and the location tree supplies the
shared-node case.

### 6.2 One month of shipment data

| SKU | Description | Category | Subcategory | Units | Unit price (EUR) | Revenue (EUR) | Ship-from DC |
|---|---|---|---|---:|---:|---:|---|
| D1 | Gauze dressing 10 cm | Wound care | Dressings | 1,200 | 2.50 | 3,000 | North |
| D2 | Foam dressing | Wound care | Dressings | 800 | 4.00 | 3,200 | North |
| D3 | Post-operative dressing | Wound care | Dressings | 500 | 6.00 | 3,000 | Central |
| T1 | Paper tape | Wound care | Tapes | 2,000 | 1.00 | 2,000 | North |
| T2 | Silk tape | Wound care | Tapes | 1,500 | 1.20 | 1,800 | South |
| P1 | IV cannula 20G | Vascular access | Peripheral IV | 3,000 | 1.50 | 4,500 | North |
| P2 | IV cannula 22G | Vascular access | Peripheral IV | 900 | 3.00 | 2,700 | Central |
| P3 | Midline catheter | Vascular access | Peripheral IV | 600 | 5.00 | 3,000 | Central |
| C1 | Central line kit | Vascular access | Central lines | 200 | 25.00 | 5,000 | North |
| C2 | PICC kit | Vascular access | Central lines | 120 | 40.00 | 4,800 | South |
| I1 | Continence brief, medium | Continence | Briefs | 4,000 | 0.50 | 2,000 | South |
| I2 | Continence brief, large | Continence | Briefs | 2,500 | 0.80 | 2,000 | South |
| | **Total** | | | **17,320** | | **37,000** | |

Column check: 1,200 × 2.50 = 3,000, and similarly for every row. A row that fails this check
is the first sign of a unit-of-measure defect, which Topic 01 covered.

### 6.3 The correct rollup

| Level | Node | Revenue (EUR) | Units | Share of revenue |
|---|---|---:|---:|---:|
| 0 | Total | 37,000 | 17,320 | 100.0% |
| 1 | Wound care | 13,000 | 6,000 | 35.1% |
| 2 | Dressings | 9,200 | 2,500 | 24.9% |
| 2 | Tapes | 3,800 | 3,500 | 10.3% |
| 1 | Vascular access | 20,000 | 4,820 | 54.1% |
| 2 | Peripheral IV | 10,200 | 4,500 | 27.6% |
| 2 | Central lines | 9,800 | 320 | 26.5% |
| 1 | Continence | 4,000 | 6,500 | 10.8% |
| 2 | Briefs | 4,000 | 6,500 | 10.8% |

Three identities hold. Category revenue sums to 37,000. Subcategory revenue sums to 37,000.
Revenue share sums to 100.0% to one decimal. Unit share would not sum to the revenue share, and
that difference is the point of the next subsection.

### 6.4 The dual-listed SKU

D3 is sold standalone and also inside a sterile procedure pack. Merchandising carries it in two
places: Dressings, and a fourth top-level category, Surgical procedure packs. The mapping table
therefore has two rows for D3, both marked active.

| Query | Result (EUR) | Arithmetic |
|---|---:|---|
| Sum of fact rows | 37,000 | Independent total from the shipment table |
| Sum over top-level category assignments | 40,000 | 13,000 + 20,000 + 4,000 + 3,000 |
| Gap | 3,000 | The D3 revenue counted a second time |
| Gap as share of total | 8.1% | 3,000 / 37,000 |

Both queries are single `SUM`s over the same fact table. The second joins to the category
assignment table and returns one row per assignment, which is a fan-out on the dimension side.
A `COUNT(DISTINCT sku)` returns 12 in both queries and does not detect the problem. A
`SUM(DISTINCT revenue)` returns 34,000, because two SKUs happen to share a revenue value, and is
worse than the original error.

The repair is a policy, not a query: mark one assignment primary and filter on it, or allocate the
SKU's revenue across both categories with a stated weight. The first is the default. The second is
defensible when the dual listing reflects genuine demand, and must then be applied consistently at
every level.

### 6.5 The fan trap: order header joined to order lines

Order SO-4471 has three lines and one header-level freight charge.

| Element | Value |
|---|---:|
| Line 1 revenue (EUR) | 1,200 |
| Line 2 revenue (EUR) | 800 |
| Line 3 revenue (EUR) | 500 |
| Order revenue (EUR) | 2,500 |
| Header freight (EUR) | 240 |
| Rows after header-to-line join | 3 |

Summing `freight` after the join gives 240 × 3 = 720. The freight ratio reads 720 / 2,500 =
28.8%. The correct ratio is 240 / 2,500 = 9.6%. The error factor is exactly the number of lines,
so it is largest on the orders with the most lines, which are also the orders where freight
matters most.

Revenue is unaffected in this example because it is stored at line level. That is what makes the
defect hard to see: half the report is right. If the header also carried a discount or a
promotional allowance, those would inflate too, and the total would drift without any single
column looking absurd.

### 6.6 The chasm trap: sales and returns

Product D1 has three sales rows in the month and two return rows.

| Table | Rows | Sum (EUR) |
|---|---:|---:|
| Sales for D1 | 3 | 3,000 |
| Returns for D1 | 2 | 400 |
| Sales after joining both on product key | 6 | 6,000 |
| Returns after the same join | 6 | 1,200 |

The joined return rate is 1,200 / 6,000 = 20.0%. The true rate is 400 / 3,000 = 13.3%. Both the
numerator and the denominator are inflated, but by different factors — sales by the count of
returns, returns by the count of sales — so the ratio does not cancel.

The correct construction aggregates each fact table to the same headers first, then aligns them.
In structured query language (SQL) this is two common table expressions (CTEs) and a full outer
join; in a business intelligence (BI) tool it is a drill-across or multipass query. Section 7
gives the shape.

### 6.7 The semi-additive measure

Month-end on-hand inventory, in units:

| DC | January | February | March | Sum over months |
|---|---:|---:|---:|---:|
| North | 1,000 | 1,100 | 1,050 | 3,150 |
| Central | 400 | 380 | 420 | 1,200 |
| South | 900 | 950 | 880 | 2,730 |
| **Total across DCs** | **2,300** | **2,430** | **2,350** | **7,080** |

The column totals are valid: at each month-end, inventory is additive across DCs. The row totals
are not balances. 3,150 is a sum of three month-end positions and describes no moment in time.
7,080 is 3 × the three-month average of 2,360, and calling it "inventory" overstates the holding
cost by a factor of three.

Three valid aggregates answer three different questions:

- **Closing balance**: 2,350 units. What was held at the end of March.
- **Average balance**: 7,080 / 3 = 2,360 units. What was held on average, if the three
  month-ends are representative.
- **Maximum**: 2,430 units. The peak the network had to store.

Each is correct, and the specification must say which one a report column uses.

### 6.8 The non-additive measure

| Category | Revenue (EUR) | Gross margin (EUR) | Margin % |
|---|---:|---:|---:|
| Wound care | 13,000 | 3,900 | 30.0% |
| Vascular access | 20,000 | 9,000 | 45.0% |
| Continence | 4,000 | 1,000 | 25.0% |
| **Total** | **37,000** | **13,900** | **37.6%** |

The simple average of the three percentages is (30 + 45 + 25) / 3 = 33.3%. The correct total
margin is 13,900 / 37,000 = 37.6%. The gap is 4.3 percentage points, and it arises because the
simple average gives Continence the same weight as Vascular access, which is five times its
size.

This is not a rounding issue and it does not disappear with more data. It is the reason a ratio
must be stored as its numerator and denominator, and recomputed after every aggregation.

### 6.9 The shared node

A central sterile-processing hub serves all three regions. Its fixed cost is EUR 180,000 per
month. Three drivers are available.

| Region | Revenue (EUR) | Units | Orders |
|---|---:|---:|---:|
| East | 17,700 | 7,200 | 400 |
| Central | 8,700 | 2,000 | 220 |
| South | 10,600 | 8,120 | 380 |
| **Total** | **37,000** | **17,320** | **1,000** |

| Allocation basis | East (EUR) | Central (EUR) | South (EUR) |
|---|---:|---:|---:|
| Revenue | 86,108 | 42,324 | 51,568 |
| Units handled | 74,827 | 20,785 | 84,388 |
| Orders | 72,000 | 39,600 | 68,400 |
| Spread on South | -14,108 | | +32,820 |

South's allocated cost ranges from 51,568 to 84,388, a difference of 63.6% against the lower
figure. The region's apparent profitability changes by more than 30,000 EUR without a single
operational change. Every row in that table sums to 180,000 exactly, so the allocation passes
the only test that many organisations apply.

The defensible driver is the one that causes the hub's cost. If sterile units drive it, units is
causal and the other two are proxies. If instrument sets drive it, neither revenue nor units is
right, and the data for the correct driver may not exist; the honest answer is then that the
allocation is approximate, and the approximation is published with the number.

### 6.10 What an analyst should have asked

1. Which of the three product trees produced the category column, and is the dual listing marked?
2. Is freight stored at header or line level, and was it ever allocated down?
3. Are sales and returns in one fact table or two, and if two, is the report a drill-across?
4. Is the inventory column a closing balance, an average, or a sum of snapshots?
5. Does the margin percentage have a stored numerator and denominator, or only the percentage?
6. Which driver produced the allocated hub cost, and does the reader know it is an allocation?
7. Does March show a recategorised product under the old category or the new one?

None requires a model; each changes a number in the report.

## 7. Code, query, or calculation

One query, because the arithmetic in section 6 is the substance. It compares the leaf total with
the category rollup and counts orphans in a single pass, so that a failure names itself.

```sql
-- One row per SKU-month. Leaf total vs. category rollup vs. orphans.
with leaf as (
    select sum(revenue) as leaf_revenue
    from fact_shipment
    where month_id = 202602
),
rollup as (
    select sum(f.revenue) as category_revenue
    from fact_shipment f
    join bridge_category b
      on b.sku_key = f.sku_key
     and b.depth = 1                 -- one row per SKU per top-level category
    where f.month_id = 202602
),
orphan as (
    select count(*) as orphan_rows
    from fact_shipment f
    left join dim_product p on p.sku_key = f.sku_key
    where p.sku_key is null
      and f.month_id = 202602
)
select leaf_revenue, category_revenue,
       category_revenue - leaf_revenue as gap,
       orphan_rows
from leaf cross join rollup cross join orphan;
```

Expected output on the Meridian month: `37000, 40000, 3000, 0`.

Read the output as a diagnosis, not a pass or fail. A nonzero gap with zero orphans means the
category bridge contains more than one assignment per SKU. A zero gap with nonzero orphans means
the two sides were filtered identically and both silently dropped the same rows, which is the
more dangerous case because it reconciles. Change `left join` to `inner join` in the orphan CTE
and the count goes to zero; the query would then be measuring nothing.

Two further assertions are worth building: `bridge_category` has exactly one row per
`(sku_key, depth)` unless a dual listing is intended; and every leaf's `max(depth)` equals the
category depth used in the query, or a ragged branch has been silently excluded.

> **Note:** The query counts orphans but not unmapped category assignments. A SKU can resolve to
> the product dimension and still have no category row, disappearing from the rollup and not from
> the leaf. Add a CTE counting facts with no depth-1 bridge row, or the gap is misattributed.

## 8. Data, standards, and regulatory context

Hierarchies are partly standardised and partly invented. The standardised part covers
identification and classification; the invented part covers the commercial trees that most
reports actually use. Confusing the two is the source of most cross-system disagreement.

### 8.1 Product identification and classification

| Standard | Governs | What it fixes | Grade |
|---|---|---|---|
| GS1 Global Product Classification (GPC) | Product classification | A four-level hierarchy of segment, family, class, and brick, plus brick attributes | B |
| GS1 GTIN | Trade item identification | The identifier of the item, not its place in a hierarchy | B |
| GS1 Global Data Synchronisation Network (GDSN) | Master data exchange | GPC classification is mandatory for items published to the network | B |
| GS1 Digital Link | Product identification on packaging | A resolvable identifier that can carry classification | B |

GPC is the closest thing the consumer and healthcare supply chain has to a shared product
hierarchy. Its structure is specific and worth memorising:

- **Segment** — the highest level, an industry vertical. Codes are 8 digits, formed from a
  two-digit segment code followed by six zeros, for example `10000000`.
- **Family** — a broad division of a segment, for example beverages within food and beverage.
- **Class** — a group of like categories.
- **Brick** — the lowest level, a category of similar products serving a common purpose, of
  similar form or material, sharing the same attribute set. Every GTIN maps to exactly one brick.
- **Brick attributes** — typically zero to seven questions per brick, with mutually exclusive
  values below them.

Three operational facts follow. GPC is released twice a year, in May and November, with delta
files, so a stored classification is a snapshot rather than a constant. A temporary brick code
`99999999` exists for products that cannot yet be classified, and its presence in a report is a
data quality signal rather than a category. Codes are never reused, which means a retired brick
can be joined historically without ambiguity.

### 8.2 Location identification

| Standard | Governs | Structure | Grade |
|---|---|---|---|
| UN/LOCODE | Location codes for trade and transport | 5 characters: two-character ISO 3166-1 country code plus three-character location code | B |
| ISO 3166-1 | Country codes | Two-letter alpha-2, three-letter alpha-3, numeric | B |
| ISO 3166-2 | Country subdivisions | Subdivision code of one to three alphanumeric characters | B |
| GS1 GLN | Party and place identification | 13-digit location number | B |

UN/LOCODE carries the subdivision as a separate column holding the ISO 3166-2 code, with the
country portion omitted because it is already in the first two characters of the location code.
A location code can also be extended with child codes for facilities such as container
terminals and rail stations, which is how a port becomes a set of berths in a location hierarchy.

The distinction that matters for analytics: **UN/LOCODE identifies a place, GLN identifies a
party or a physical location in a commercial relationship, and neither identifies a sales
territory.** A sales territory is an internal construct that cuts across both, and it changes
when the sales organisation changes. Building a territory hierarchy on top of GLNs and expecting
it to be stable is a category error.

### 8.3 Time as a hierarchy

Time is the one hierarchy where a standard fixes the structure at some levels: ISO 8601 defines
week dates, and the fiscal and retail calendars of Topic 04 define the rest. Two consequences
carry into this topic. A time hierarchy is balanced at the calendar level and often ragged at
the fiscal level, because a 53-week year adds a week to one branch. And time is the only
hierarchy whose leaf is unambiguous, which is why the grain declaration in section 5 still comes
before the level list.

### 8.4 What the standards do not fix

Four things that look standardised and are not:

- **The commercial category tree.** GPC bricks are classification, not merchandising. A retailer
  runs planogram categories, a distributor runs buying categories, and a manufacturer runs brand
  and therapeutic categories. All three can be GPC-compliant and mutually incompatible.
- **The customer hierarchy.** There is no global standard for "account". A hospital is a ship-to,
  a bill-to, a member of a group purchasing organisation, and part of a health system, and four
  systems will model those four facts differently.
- **The location hierarchy above the site.** Regions, sales areas, service zones, and transport
  lanes are all internal, all changeable, and all overlapping.
- **The cost hierarchy.** A chart of accounts is a regulatory artefact in its top levels and a
  management invention below them.

> **Note:** A standard tells you what an identifier means. It does not tell you which identifier
> your report should group by. That decision is the hierarchy design, and it is made by the
> business with the analyst's advice, not by the standard.

## 9. Industry practice and case evidence

### 9.1 The dimensional modelling canon

The vocabulary of this topic comes almost entirely from one practitioner tradition.

Ralph Kimball's rule is that declaring the grain is the pivotal step, that it must be stated in
business terms rather than as a list of keys, and that it must precede the choice of dimensions
and facts. The grain is a binding contract: once fixed, every candidate dimension and measure can
be tested against it. Kimball also recommends the atomic grain, because it withstands queries
nobody anticipated, and one fact table per grain, because mixing grains makes every measure
unreliable for some rows.

Conformed dimensions are the second rule: two dimensions conform when attribute names and domain
contents match, which lets separate fact tables align on identical row headers. **Conformance is
about the attributes used as row headers, not about the tables being identical.** A brand
attribute that is a subset of a product attribute conforms for that report.

Drilling across is the third rule: query each fact table separately at the same grain and with
the same row headers, then align by sort-merge. Kimball's explicit warning is never to place two
fact tables in the same `FROM` clause. The join is syntactically valid, fast on small test data,
and wrong.

These are practitioner rules, not experimental findings. They are graded C: canonical and widely
adopted, definitional rather than empirical. That does not make them weak. It makes them
conventions, and a convention can be examined and departed from with reasons.

### 9.2 What practice looks like

| Observation | Evidence | Consequence for analytics |
|---|---|---|
| GPC classification is mandatory for items published through GDSN | GS1 documentation | A classification tree exists for most traded items, but it is not the commercial tree |
| GPC is released twice a year with delta files | GS1 documentation | A stored classification needs a version and a refresh date |
| Retailers and distributors run proprietary category trees | Widely observed; no public statistic | The tree a report uses is a local decision with no external reference |
| Enterprise resource planning (ERP) hierarchies are configuration, not data | Vendor documentation and practice | A hierarchy change is invisible in historical rows unless versioned |
| Ragged hierarchies are handled by tool-specific flags | Microsoft documentation for Analysis Services | Portable models cannot rely on the flag; the bridge table is the portable answer |

No public statistic establishes how often fan or chasm traps occur in production dashboards. The
claim that they are common is an inference from schema patterns, not a measured rate, and any
specific prevalence figure should be treated as unsourced.

### 9.3 Evidence on allocation

Cost allocation has a genuine empirical literature, and it supports the warning in section 5
more strongly than most analytics writing does.

The step-down method allocates support-department costs in sequence, closing each department
after it has been allocated. The sequence is usually chosen by judgement. A worked teaching
example with two support departments and two operating departments shows cost per gigabyte
ranging from about 0.588 to 0.895 USD depending on which department is allocated first, and cost
per phone ranging from about 741 to 1,793 USD. The same pool, the same drivers, a different order.

The reciprocal method solves the simultaneous equations for mutual service consumption and
produces one unit cost for all users. Collins (2003) shows that the direct and step-down methods
are special cases of the general matrix formulation, differing only in the resource-consumption
matrix, and that the step method requires an arbitrarily chosen sequence. Brown and Doran (2007)
find no single sequencing heuristic superior, which undercuts the rule of thumb that the
department serving the most others should be allocated first.

The practical reading for a supply chain analyst is narrower than the accounting debate. Most
network cost models allocate shared capacity or overhead by a driver chosen for convenience, and
present the result as a cost-to-serve. The evidence says the choice of method can move a unit
cost by tens of percent. Publish the method, and do not compare cost-to-serve figures computed
under different methods.

### 9.4 The tooling layer

Three implementation patterns recur, each encoding a different assumption.

- **Flattened level columns.** Fast, and assumes a balanced tree; a ragged branch is forced into
  a placeholder member.
- **Parent-child recursion.** Matches the source model and needs no rebuild, but query cost grows
  with depth and a source cycle hangs the query unless recursion is bounded.
- **Bridge or closure table.** Answers any rollup with one join and supports multi-parent and
  time-varying hierarchies, at the cost of a rebuild whenever the hierarchy changes. Kimball's
  bridge stores every path from a node to all descendants, with depth, and includes the
  zero-depth self row so that a node's own facts are counted.

The bridge is the only one of the three that handles a DAG without special-casing, which is why
it appears in every serious treatment of ragged and shared-node hierarchies.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dual-listed member | Category total exceeds the independent total by a stable amount | Two active assignments for one leaf | Revenue or volume double counted, usually under 10% | Primary-assignment flag; test R2 per branch |
| Fan trap | Header-level measure scales with the number of lines | One-to-many joined to one-to-many | Freight, discount, and tax inflated by the line count | Aggregate the header to line grain first, or keep measures at their own grain |
| Chasm trap | Ratio moves when a second fact table is added to the report | Two fact tables joined through a shared dimension | Every sum inflated by the other side's row count | Drill-across: aggregate each side, then align |
| Semi-additive measure summed | Inventory or headcount grows with the number of periods shown | Balance treated as a flow | Holding cost overstated by the period count | Declare the aggregate: last, average, min, max |
| Ratio averaged | Total percentage lies outside the range of the parts, or far from the weighted value | Mean of ratios instead of ratio of sums | Margin or service misstated by several points | Store numerator and denominator |
| Allocated cost presented as measured | Cost per region changes when the driver changes | Allocation rule invisible in the report | Cross-subsidy; wrong pricing and sourcing decisions | Publish pool, driver, and method with the number |
| Orphans silently dropped | Report total falls after a load with no operational change | Inner join to a dimension with missing keys | Volume disappears; no error raised | Left join and count; alert on the rate |
| Placeholder absorbs new items | "Unassigned" share rises month over month | New items mapped to a temporary member | Real volume attributed to a fake category | Track placeholder share by month; suppress above threshold |
| Hierarchy change not versioned | Last quarter's report changes after a reorganisation | Type 1 overwrite of a category attribute | As-was analysis impossible; audit trail lost | Type 2 dimension with effective dates |
| Level invented to balance a ragged branch | A member named "Other level 3" appears | Padding applied to make depths equal | Per-level metrics computed on invented members | Bridge table, or explicit level-skip policy |
| Cycle in the parent relation | Recursive query runs until it times out | A node is its own ancestor through a chain | Dashboard unavailable; no rollup possible | Detect cycles at load; bound recursion depth |
| Shared node counted in both branches | Two regions each report the same DC cost | DAG treated as a tree | Total cost exceeds the pool | Bridge with weights, or a single owning branch |
| Wrong driver for allocation | Cost per unit varies by region with no operational cause | Driver correlates with size, not with consumption | Structural cross-subsidy | Justify the driver causally; test sensitivity |
| Drill-across replaced by a join | A new fact table makes the existing report wrong | Convenience join during development | Silent inflation, often large | Code review rule: one fact table per `FROM` |
| Unit price rolled up | Average price per unit changes when the mix changes | Price treated as additive | Misleading price trend | Sum extended value and units; divide last |

**The dual-listed member is the hardest to see**, because internal consistency holds: leaves sum
to parents and parents to total. Only an independent distinct count or a physical measure exposes
it. Build that count into the reconciliation; query review will not find the problem.

**The drill-across rule is broken most often during prototyping**, when a single join is faster
to write and the test data is too small to show inflation. By the time the report reaches
production, the join is invisible inside a view. The mitigation is a review rule, not a technical
control.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Each fact row maps to exactly one leaf | Additive rollups require a function from facts to leaves | Dual listing, missing mapping, late-arriving items | Count facts per leaf; alert on zero and on more than one assignment |
| The hierarchy is a tree for additive measures | Summing without weights is only valid in a tree | Shared components, shared nodes, matrix organisations | Count nodes with more than one parent |
| Parent-child edges are stable in the analysis window | Trends compare like with like | Reorganisation, recategorisation, acquisitions | Compare hierarchy versions at the start and end of the window |
| The leaf is the finest grain available | Coarser facts cannot be decomposed | Source captures only category level | Check the source grain before promising SKU analysis |
| The allocation driver is proportional to consumption | Only then is the share defensible | Convenience drivers, correlated but not causal | Compare two drivers; report the sensitivity |
| Placeholder members are known to be unknown | Otherwise real volume hides in them | New items routed to "Unknown" | Track placeholder share by month |
| All source systems use one hierarchy version | Joins on level labels require it | Multiple ERP instances, regional variants | Join the hierarchy tables and count mismatched members |
| Measures carry one unit across all levels | A total in mixed units is meaningless | Currency mix, case versus each | Test R3 at every level |
| The reporting hierarchy is the decision hierarchy | Reports are used to decide | Finance, commercial, and planning trees diverge | Ask the decision owner which tree they recognise |
| Rounding is immaterial relative to the conclusion | Exact reconciliation is sometimes impossible | Large node counts, small values | Bound the residual; state it |

Two limits are worth stating plainly.

**The reconciliation identity is necessary, not sufficient.** A hierarchy can reconcile at every
level and still be wrong for the decision: reconciliation proves the arithmetic is consistent
with the mapping, not that the mapping reflects the business.

**Hierarchy design is a governance problem wearing a technical costume.** The analyst can build
the bridge table, run the tests, and publish the specification. The analyst cannot decide whether
a product belongs in Wound care or in Surgical procedure packs.

## 12. Exercises

### Exercise 1 — Reconcile the rollup (calculation, 40 minutes)

Using the table in section 6.3, verify the rollup at every level, then compute the effect of the
dual listing in 6.4.

**Deliverable:** a reconciliation table with leaf total, category total, subcategory total, the
gap, and the gap as a percentage of the independent total.
**Acceptance criteria:** all three totals stated; the gap attributed to a named SKU; one sentence
on why `SUM(DISTINCT revenue)` is not a valid repair.
**Time:** 40 minutes.

### Exercise 2 — Repair the fan trap (calculation, 30 minutes)

An order has five lines and a header-level discount of 150 EUR. Line revenues are 400, 300, 250,
200, and 100 EUR.

**Deliverable:** the inflated discount total, the correct total, the inflation factor, and the
discount ratio before and after the repair.
**Acceptance criteria:** factor stated as a function of line count; the ratio computed on the
correct denominator; one sentence on why line-level revenue is unaffected.
**Time:** 30 minutes.

### Exercise 3 — Repair the chasm trap (calculation, 35 minutes)

A product has four sales rows summing 8,000 EUR and three return rows summing 640 EUR.

**Deliverable:** the joined sums for both sides, the joined return rate, the correct return rate,
and the relative overstatement.
**Acceptance criteria:** both inflation factors stated separately; the reason the ratio does not
cancel explained; the drill-across shape described in two sentences.
**Time:** 35 minutes.

### Exercise 4 — Semi-additive inventory (calculation, 40 minutes)

Using the table in section 6.7, compute the closing balance, the average balance, the peak, and
the sum over months. Assume a monthly holding cost of 2% of unit value and a unit value of
12 EUR.

**Deliverable:** the four aggregates in units, and the holding cost computed from each.
**Acceptance criteria:** the sum-over-months cost identified as invalid and the factor of
overstatement stated; the three valid aggregates each matched to a decision.
**Time:** 40 minutes.

### Exercise 5 — Non-additive margin (calculation, 30 minutes)

Using the table in section 6.8, compute the simple average margin, the weighted margin, and the
gap. Then compute the margin if Continence is given double weight.

**Deliverable:** three percentages and the arithmetic behind each.
**Acceptance criteria:** the weighted figure recomputed from summed amounts, not from averaged
percentages; the weight effect explained in one sentence.
**Time:** 30 minutes.

### Exercise 6 — Build a bridge table by hand (application, 45 minutes)

Take the hierarchy: root R; children A and B; A has children A1 and A2; B has child B1; A1 has
children A1a and A1b.

**Deliverable:** the full closure table of `(ancestor, descendant, depth)` rows, including
zero-depth rows, and the row count.
**Acceptance criteria:** every pair present exactly once; depth correct; the row count equals the
sum of subtree sizes; one sentence on what a query filtering `depth = 1` returns.
**Time:** 45 minutes.

### Exercise 7 — Choose an allocation driver (design and calculation, 50 minutes)

Using the table in section 6.9, compute all three allocations, then recommend a driver for a hub
whose cost is driven by the number of instrument sets processed. Assume sets per month are 900
in East, 500 in Central, and 1,600 in South.

**Deliverable:** the three existing allocations plus the set-based allocation, and a
recommendation with a sensitivity statement.
**Acceptance criteria:** every allocation sums to 180,000; the recommendation names the causal
mechanism; the sensitivity to the driver choice is quantified in EUR for at least one region.
**Time:** 50 minutes.

### Exercise 8 — Specify a ragged hierarchy (design, 45 minutes)

A product hierarchy has four levels in most branches and two in one branch covering 6% of
revenue. Write the hierarchy specification.

**Deliverable:** the eight fields from section 5, Phase J, filled in for this case.
**Acceptance criteria:** placeholder policy distinguishes unknown, not applicable, and not
assigned; the reconciliation test names an independent total; the level-skip policy is explicit
and testable.
**Time:** 45 minutes.

### Exercise 9 — Audit a dashboard (critique, 60 minutes)

Take any dashboard you can access that reports a total and at least one ratio, and reconstruct
its grain, its hierarchy, and its aggregation rules from the documentation and the numbers.

**Deliverable:** a one-page audit stating the grain, the hierarchy of record, the additivity class
of each measure, and one defect with its numerical effect.
**Acceptance criteria:** at least one defect quantified; the defect classified as fan trap, chasm
trap, cover, additivity error, or allocation artefact; a specific repair proposed.
**Time:** 60 minutes.

### Exercise 10 — Design a drill-across (design, 45 minutes)

A report must show, by category and month, shipped revenue, returned value, and return rate. The
data has two fact tables, `fact_shipment` and `fact_return`, both keyed by product and date.

**Deliverable:** a query plan in prose or SQL showing two independent aggregations, the alignment
key, and the treatment of categories with shipments but no returns.
**Acceptance criteria:** no `FROM` clause contains both fact tables; the outer join type is
justified; the return rate is computed after alignment; the null-versus-zero decision is stated.
**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 1.** Leaf total 37,000. Subcategory totals: 9,200 + 3,800 + 10,200 + 9,800 + 4,000 =
37,000. Category totals excluding the second listing: 13,000 + 20,000 + 4,000 = 37,000. Category
totals including it: 40,000. Gap 3,000, which is 8.1% of 37,000. The gap is exactly the revenue of
D3, the post-operative dressing. `SUM(DISTINCT revenue)` returns 34,000 because two SKUs share a
revenue value of 3,000, so it removes a legitimate row and is not a repair.

**Exercise 2.** Five lines, so the header discount of 150 is counted five times: 750. Correct
total 150. Inflation factor 5, equal to the line count. Order revenue is 400 + 300 + 250 + 200 +
100 = 1,250. Inflated discount ratio 750 / 1,250 = 60.0%. Correct ratio 150 / 1,250 = 12.0%.
Line-level revenue is unaffected because it exists once per line; only the header measure is
replicated by the join.

**Exercise 3.** Four sales rows and three return rows produce twelve joined rows. Sales sum
8,000 × 3 = 24,000. Returns sum 640 × 4 = 2,560. Joined rate 2,560 / 24,000 = 10.67%. Correct rate
640 / 8,000 = 8.00%. The ratio does not cancel because numerator and denominator are inflated by
different factors — sales by the three return rows, returns by the four sales rows. Relative
overstatement 2.67 percentage points, or 33.3% against the correct rate.

**Exercise 4.** Closing balance 2,350 units. Average 7,080 / 3 = 2,360 units. Peak 2,430 units.
Sum over months 7,080 units. At 12 EUR per unit and 2% per month, the holding cost is 0.24 EUR per
unit per month.

| Aggregate | Units | Cost at 0.24 EUR per unit |
|---|---:|---:|
| Closing balance | 2,350 | 564 EUR for one month |
| Average balance | 2,360 | 566 EUR per month, 1,699 EUR over three months |
| Peak | 2,430 | 583 EUR for the peak month |
| Sum over months | 7,080 | 1,699 EUR presented as a single month's cost, a 3× overstatement |

The sum-over-months figure is three times the average monthly cost, which is exactly the number
of periods. It is not a holding cost; it is three months of holding cost added together and
labelled as one.

**Exercise 5.** Simple average (30 + 45 + 25) / 3 = 33.33%. Weighted 13,900 / 37,000 = 37.57%.
Gap 4.24 percentage points. If Continence is double-weighted, the weights become 13,000, 20,000,
and 8,000 against a total of 41,000; margin becomes (3,900 + 9,000 + 2,000) / 41,000 = 14,900 /
41,000 = 36.34%. The percentage moves because the weight moved, not because any category's margin
changed.

**Exercise 6.** Nodes and subtree sizes: R 8, A 5, B 2, A1 3, A2 1, B1 1, A1a 1, A1b 1. Total
pairs 22.

| Depth | Pairs | Count |
|---:|---|---:|
| 0 | (R,R) (A,A) (B,B) (A1,A1) (A2,A2) (B1,B1) (A1a,A1a) (A1b,A1b) | 8 |
| 1 | (R,A) (R,B) (A,A1) (A,A2) (B,B1) (A1,A1a) (A1,A1b) | 7 |
| 2 | (R,A1) (R,A2) (R,B1) (A,A1a) (A,A1b) | 5 |
| 3 | (R,A1a) (R,A1b) | 2 |

A query filtering `depth = 1` returns each node paired with its immediate children only, which is
the direct-parent view and not a rollup. Filtering `depth = 3` and grouping by ancestor returns
the root's total only for leaves at depth 3; leaves at shallower depths are excluded, which is
the ragged-hierarchy trap in its purest form.

**Exercise 7.** The three existing allocations are in section 6.9. The set-based allocation uses
900 + 500 + 1,600 = 3,000 sets.

| Region | Sets | Share | Allocation (EUR) |
|---|---:|---:|---:|
| East | 900 | 30.0% | 54,000 |
| Central | 500 | 16.7% | 30,000 |
| South | 1,600 | 53.3% | 96,000 |
| **Total** | **3,000** | **100.0%** | **180,000** |

South's allocation ranges from 51,568 under revenue to 96,000 under sets, a difference of 44,432
EUR, or 86% against the lower figure. If sets are what consume hub capacity, the set-based figure
is the defensible one and the revenue-based figure is a cross-subsidy from South to East. A
sensitivity statement should say that rank order between Central and South reverses under the
revenue driver, so any decision that depends on that ranking is driver-dependent.

**Exercise 8 rubric.** A strong specification states four real levels, notes that the two-level
branch covers 6% of revenue, and chooses between padding and a bridge table with a reason. It
distinguishes UNK (attribute not captured), NA (attribute cannot apply), and UNM (known but
unmapped), suppresses UNM from published rollups, and names a distinct SKU count as the
independent total. It states which level the ragged branch reports at. A weak specification lists
the levels and says "use a recursive query", with no placeholder policy and no test.

**Exercise 9 rubric.** A strong audit reconstructs the grain, identifies the hierarchy of record,
and classifies each measure. It quantifies at least one defect — a header freight charge inflating
a cost ratio by the line count, or a category total exceeding the independent total by the
dual-listed share — and proposes a repair that matches the mechanism. A weak audit lists data
quality complaints without connecting any of them to a number.

**Exercise 10 rubric.** A strong plan aggregates each fact table to `(category, month)`, checks
that each result is unique on those headers, and aligns with a full outer join. It uses
`COALESCE` to zero for the return value but leaves the rate null where shipments are zero,
because a rate with a zero denominator is undefined rather than zero. A weak plan joins the two
fact tables directly and then filters.

## 14. Interview questions

**1. "What is a grain, and why declare it before anything else?"**
Strong: defines it as the business meaning of one row, gives an example in business terms, and
says every dimension and measure must be consistent with it. Weak: describes it as the primary key.

**2. "Difference between a fan trap and a chasm trap?"**
Strong: fan trap is one-to-many joined to one-to-many within one fact's dimensions; chasm trap is
two fact tables joined through a shared dimension; both multiply rows and neither is fixed by
`DISTINCT` on sums. Weak: calls both "duplicate rows".

**3. "A category total exceeds the company total. How do you find the cause?"**
Strong: compares with an independent distinct count, tests R2 per branch, looks for multiple
active assignments, and proposes a primary-assignment flag. Weak: reruns the query and checks the
join condition.

**4. "How would you model a product that belongs to two categories?"**
Strong: names the choice between a primary assignment and a weighted distribution, notes that
weights must sum to one and be applied at every level, and says the choice is commercial. Weak:
adds the product twice and moves on.

**5. "What is a ragged hierarchy, and how would you represent it?"**
Strong: leaves at different depths; names flattened columns with placeholders, recursive
parent-child, bridge or closure, and path enumeration, with one trade-off each. Weak: says "use a
recursive query" without a placeholder policy.

**6. "Why can you not sum inventory over time?"**
Strong: it is a balance at a point, semi-additive, so last value or average; summing produces
unit-months and multiplies holding cost by the period count. Weak: says inventory is "not
additive" without distinguishing dimensions.

**7. "When is an average of percentages wrong?"**
Strong: when the denominators differ; recompute from summed numerator and denominator; gives a
weighted-versus-simple example. Weak: says averages are always acceptable if you use enough
decimal places.

**8. "How would you allocate a shared distribution centre cost?"**
Strong: identifies the pool, chooses a driver on causal grounds, states the shares sum to the
pool, and warns that another defensible driver moves the answer by tens of percent. Weak: picks
revenue because it is available.

**9. "How do you reconcile a rollup to its parent?"**
Strong: names leaf-to-total, level-to-level, row arithmetic, and path uniqueness, and says which
failure pattern each catches. Weak: checks that the query runs and that the total looks plausible.

**10. "What is a conformed dimension, and why does it matter for drill-across?"**
Strong: same attribute names and domains across fact tables, enabling separate queries aligned on
identical row headers; never join two fact tables directly. Weak: says it means a shared key.

## 15. Key terms

- **Hierarchy** — a containment relation used to aggregate measures.
- **Level** — a named depth; **depth** — the number of edges to the root.
- **Grain** — the business meaning of one fact table row.
- **Leaf** — a node with no children; where facts are recorded.
- **Partition** — non-overlapping division; **cover** — division that permits overlap.
- **Fan trap** — inflation from joining a one-side table to several many-side tables.
- **Chasm trap** — inflation from joining two fact tables through a shared dimension.
- **Drill-across** — separate queries aligned on conformed row headers.
- **Bridge table** — every ancestor-descendant pair with a depth.
- **Closure table** — a bridge table including the zero-depth self row.
- **Ragged hierarchy** — a hierarchy whose leaves sit at different depths.
- **DAG** — directed acyclic graph; a hierarchy where a node may have several parents.
- **Additive measure** — summable across every dimension in its fact table.
- **Semi-additive measure** — summable across some dimensions, notably not time.
- **Non-additive measure** — a ratio or price; recompute from components.
- **Allocation** — assigning a shared pool by a driver; not a measurement.
- **Shared node** — a node serving more than one branch.
- **Placeholder member** — a row for unknown, not applicable, or not assigned.
- **SCD type 2** — a dimension row versioned with effective dates.
- **As-was / as-is** — reporting with the hierarchy then, versus now.
- **Conformed dimension** — shared attribute names and domains across fact tables.
- **GPC** — GS1's four-level product classification: segment, family, class, brick.

## 16. Sources

**Peer-reviewed and academic (Grade A)**

- Collins, J. (2003). "A General Matrix Model of Support Department Cost Allocation Methods."
  *Journal of Applied Business Research*, 19(2). DOI: 10.19030/jabr.v19i2.2155.
- Brown, R. and Doran, D. (2007). "The Relative Accuracy of Allocating Service Departments' Cost
  to Production Departments under the Step Method." *Journal of Cost Analysis and Management*,
  9(1). DOI: 10.1080/15411656.2007.10462259.
- Lowenthal, F. and Malek, M. (2005). "Reciprocal Service Department Cost Allocation as a Markov
  Process." *Journal of Applied Mathematics and Decision Sciences*, 2005(3).
- Togo, D. (2014). "Improve Accounting for Cost Allocations from Support to Revenue-Generating
  Departments with the Reciprocal Method." *American Journal of Health Sciences*, 5(1).
  DOI: 10.19030/ajhs.v5i1.8611.

**Standards and official (Grade B)**

- GS1, "How GPC works," gs1.org/standards/gpc/how-gpc-works.
- GS1, *GPC Development and Implementation Guide*, gs1.org/docs/gpc.
- GS1, "What is the GPC schema?" and "What are the GPC Schema Principles?" support articles.
- UNECE, Recommendation No. 16, *UN/LOCODE*, Revision 4 (2020), ECE/TRADE/459.
- UNECE, *UN/LOCODE Manual* and the UN/LOCODE column description.
- ISO 3166-1 (country codes) and ISO 3166-2 (country subdivision codes).

**Practitioner canon (Grade C)**

- Kimball, R. (2003). "Declaring the Grain." Kimball Group.
- Kimball Group, "Grain" and "Conformed Dimensions," dimensional modeling technique pages.
- Kimball Group, "Ragged/Variable Depth Hierarchies."
- Kimball Group, "Additive, Semi-Additive, and Non-Additive Facts."
- Kimball, R. "Building the Hierarchy Bridge Table," Kimball Group technical PDF.
- Kimball, R. (2003). "The Soul of the Data Warehouse, Part Two: Drilling Across."
- Ross, M. (2009). "The 10 Essential Rules of Dimensional Modeling." Kimball Group.

**Vendor and tutorial documentation (Grade D)**

- Microsoft Learn, "Ragged Hierarchies (Analysis Services multidimensional)."
- Honeydew, "Handling Fan-Out and Chasm Joins."
- ThoughtSpot, "Schemas at Scale: How to Avoid Common Data Modeling Traps."
- Coginiti glossary, "Fan and Chasm Traps."
- DataDank, "Bridge Tables, Many-to-Many and Hierarchies"; "Star, Snowflake and Galaxy Schemas."
- KEVOS/READOS, "When Reporting Measures Cannot Be Added."
- A. H. Morris, "Proper Cost Allocation of Support Departments," teaching slides, acct3210;
  source of the step-down figures cited in section 9.3.

**Note on citations.** The cost-allocation percentages in section 9.3 are reported as the cited
material states them and are not independently verified. The Meridian figures in section 6 are
constructed for teaching, not observed. No public statistic on trap prevalence was found.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Predecessor: [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md)
- Predecessor: [`03_case-pack-moq-rounding.md`](03_case-pack-moq-rounding.md)
- Unit foundations: [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md)
- Currency and reporting hierarchies: [`06_currency-and-fx.md`](06_currency-and-fx.md)
- Next topic: `08_measurement-error-bias.md` (planned)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Hierarchical forecasting: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- Network design and shared capacity: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
- Category and assortment hierarchies: [`../../../course/17_assortment_category_retail_analytics.md`](../../../course/17_assortment_category_retail_analytics.md)
- Data engineering and interoperability: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
