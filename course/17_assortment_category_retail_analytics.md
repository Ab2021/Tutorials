# Module 17 — Assortment, Category, and Retail Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 28 hours (10 lecture, 10 lab, 8 project)
> **Track fit:** Retail & E-commerce core; elective for Consumer Goods commercial roles
> **Prerequisites:** Module 09 (demand sensing, promotion lift); Module 16 (price and elasticity); Module 04 (causal inference); basic SQL
> **Tools:** Python 3.12, pandas, NumPy, scikit-learn, matplotlib; a planogram tool or spreadsheet for space allocation; optional Pyomo for the assortment MILP
> **Industry question:** "We carry 42,000 SKUs. Which 6,000 of them earn their shelf space, and what happens to the other 36,000 customers if we drop them?"

---

## 1. Why industry cares

Assortment is the largest single decision a retailer makes, and the one with the least analytical support. A grocery superstore carries tens of thousands of items; a convenience store carries a few thousand; an online marketplace carries millions. Every item consumes working capital, shelf space, replenishment attention, and the customer's limited attention. Adding one more variant costs almost nothing in isolation and is nearly impossible to remove once a buyer has sponsored it.

The economics run in both directions. Too narrow, and the customer leaves for the store that had the item — the "one missing item" problem, where a basket of twenty items is abandoned because the twenty-first was absent. Too broad, and the shelf fragments: each facing shrinks, replenishment thins, slow movers occupy space that a fast mover would use better, and the customer faces a wall of near-identical options that suppresses rather than aids the decision.

Three forces changed the problem after 2020. **Data**: basket-level and loyalty-level data made cross-item effects measurable, so the question moved from "how many units does this SKU sell" to "what does this SKU do for the rest of the basket". **Channel**: buy-online-pickup-in-store and same-day delivery made the store an e-commerce fulfilment node, so the assortment must serve two demand patterns from one inventory. **Automation**: planogram generation, assortment optimisation, and shelf-image recognition became commodity capabilities, which moved the bottleneck from computing an answer to defending one.

The characteristic failure is not a bad clustering. It is **the delisting decision made on the item's own sales**. A slow-moving SKU that is the only gluten-free option in its subcategory, or the trigger item for a high-value basket, looks like a deletion candidate in an item-level report and is a revenue asset in a basket-level one. Almost every expensive assortment mistake is a misspecified objective, not a bad algorithm.

## 2. Learning outcomes

By the end of this module, you can:

- Define assortment by breadth, depth, and variety, and state which one a given decision changes.
- Explain choice overload and its boundary conditions, and design an assortment that reduces decision cost without removing needed variety.
- Assign category roles and apply the role-specific decision tree to a rationalisation question.
- Compute support, confidence, lift, and conviction, and distinguish a statistically interesting rule from a commercially actionable one.
- Build a basket affinity matrix and use it to defend or challenge a delisting proposal.
- Model space elasticity and construct a planogram that respects both space and fixture constraints.
- Segment customers and estimate CLV, and connect the segment structure to assortment decisions.
- Cluster stores into assortment groups, choose `k` on an explicit criterion, and translate clusters into localised planograms.
- Measure cannibalization and halo effects, and compute the true incremental contribution of a new item.
- Design a causal evaluation of an assortment change and state the identifying assumption.

## 3. Mental model

### 3.1 The assortment decision stack

```
 customer need state (mission, occasion, basket context)
                    │
        which items satisfy it, and at what cost in attention
                    ▼
   category structure: roles, subcategories, price tiers, brands
                    │
   breadth (how many subcategories) × depth (how many items within)
                    ▼
   space allocation: facings, shelf position, planogram
                    │
   inventory and replenishment: service level per item
                    ▼
   measured outcome: sales, margin, basket size, retention
```

Each layer constrains the one below it. A category role determines how much breadth is defensible; breadth determines how much depth each subcategory can carry; depth determines whether the planogram can give anything a second facing.

### 3.2 The four assortment questions

| Question | Unit of analysis | Method | Failure if wrong |
|---|---|---|---|
| How much variety? | Category | Choice modelling, assortment optimisation | Choice overload or missed trips |
| Which items? | SKU | Basket analysis, elasticity, space | Deletion of a basket-critical item |
| Where on the shelf? | Facing / position | Space elasticity, planogram rules | Fast movers starved of space |
| For which store? | Store cluster | Clustering, localisation | Uniform planogram in heterogeneous estate |

### 3.3 The basket-level accounting identity

```
category margin = Σ_items (units_i × margin_i)
item value to the category = own margin + effect on other items' margin
                              (halo, cannibalization, traffic)
```

The second term is the one missing from most item-level reports, and it is where the analytical work lives. A delisting changes both terms, and the sign of the total is not obvious from either alone.

## 4. Core concepts

### 4.1 Assortment planning: the decision and its constraints

| Dimension | Definition | Example | Constraint that binds |
|---|---|---|---|
| Breadth | Number of subcategories or segments served | 12 types of coffee | Customer mission and store format |
| Depth | Number of items within a subcategory | 40 SKUs of ground coffee | Shelf space and replenishment cost |
| Variety | Perceived differentiation among items | Roast, origin, pack size, brand | Attention and comparison cost |
| Availability | Probability the item is in stock when demanded | 97% in-stock | Working capital and lead time |

Assortment planning is a constrained selection problem: choose a set of items to maximise category objective subject to space, budget, and a minimum service requirement for each customer need. The constraints are what make it hard, and they are usually undocumented — the real constraint is often a buyer's relationship with a supplier rather than the planogram's linear metres.

**Decision rule.** State the objective before the method. "Maximise category profit" and "maximise category sales" produce different assortments, and "maximise profit subject to no customer segment losing its only option" produces a third. A model without a stated objective cannot be defended when a buyer disagrees with it.

### 4.2 Variety versus choice overload

The intuition that more choice is better has a well-known limit. Iyengar and Lepper (2000) found that a display of 24 jams attracted more browsing but produced far fewer purchases than a display of 6 — the classic demonstration that assortment size can raise evaluation cost faster than it raises match probability. Subsequent work qualified the result heavily: the effect depends on expertise, on whether the choice is hedonic or utilitarian, on whether the customer has a clear prior preference, and on time pressure.

| Condition | More variety helps | More variety hurts |
|---|---|---|
| Customer has a defined need | Strongly helps | — |
| Hedonic, low-stakes purchase | Helps | — |
| Utilitarian, high-comparison purchase | — | Increases deferral |
| Low category expertise | — | Increases deferral and dissatisfaction |
| Repeat purchase, known preference | — | Adds clutter only |
| High time pressure | — | Increases abandonment |

The commercially useful formulation is not "reduce variety" but **"reduce irrelevant variety"**. Sixty mustard SKUs where forty are near-duplicates in the same price tier produce overload without matching benefit. The same sixty, structured by use occasion and price tier, produce a navigable set. Structure, not count, is the lever.

### 4.3 Category roles and decision trees

| Role | Purpose | Typical metrics | Assortment posture |
|---|---|---|---|
| Destination | Defines the store's reason to visit | Trips, market share, basket entry | Deep, broad, never out of stock |
| Routine | Regular fill-in purchase | Sales, margin, share of basket | Broad enough to complete the mission |
| Occasional / seasonal | Mission-specific | Seasonal sales, attach rate | Deep in season, shallow outside |
| Convenience | Captures impulse and fill-in | Margin rate, basket add-on | Narrow, high-margin, impulse-positioned |

**Decision tree for a rationalisation question.**

```
1. Is the item the only option for a distinct customer need?
       yes → retain (or replace with a like-for-like)
       no  → continue
2. Is the item a basket entry or trigger item?
       yes → retain; quantify the basket it anchors
       no  → continue
3. Does the item carry a role-defining brand or price tier?
       yes → retain, or re-evaluate the tier
       no  → continue
4. Is contribution below the cost of the space it occupies?
       no  → retain
       yes → candidate for delisting, subject to test
```

The tree is deliberately ordered: need coverage and basket effects come before item economics, because item economics is the number that produced the wrong answer in the first place.

### 4.4 Market basket analysis and association rules

For items `A` and `B` across `N` baskets:

```
support(A)        = baskets containing A / N
support(A → B)    = baskets containing both / N
confidence(A → B) = support(A ∪ B) / support(A)
lift(A → B)       = confidence(A → B) / support(B)
conviction(A → B) = (1 − support(B)) / (1 − confidence(A → B))
```

**Worked example.** 1,000 baskets. Bread appears in 100. Butter appears in 200. Both appear in 40.

```
support(bread)          = 0.10
support(bread, butter)  = 0.04
confidence(bread → butter) = 0.04 / 0.10 = 0.40
lift                    = 0.40 / 0.20 = 2.00
conviction              = (1 − 0.20) / (1 − 0.40) = 0.80 / 0.60 = 1.33
```

Interpretation: a bread basket is twice as likely as a random basket to contain butter. Lift above 1 means positive association; below 1 means the items substitute or repel; equal to 1 means independence.

| Measure | What it answers | Trap |
|---|---|---|
| Support | How common is the combination? | Rare rules can have huge lift and no value |
| Confidence | Given A, how often B? | Ignores how common B already is |
| Lift | How much does A raise B's probability? | Noisy for low-support items |
| Conviction | How strongly does A imply B? | Undefined at confidence 1 |

**The commercial filter matters more than the algorithm.** A rule with support 0.1% and lift 12 will surface in every Apriori run and will never justify a shelf change. Apply a minimum support floor tied to the decision: an item needs enough baskets to move a planogram only if it appears in a meaningful share of trips. Apriori and FP-Growth produce the same rules; the difference is runtime, not insight.

### 4.5 Basket affinity and cross-selling

Affinity analysis generalises pairwise rules to a full item-item matrix, and is the workhorse for delisting defence and cross-sell design.

```python
"""Basket affinity from transactions, with a delisting check.
Run: pip install pandas ; python affinity.py
Input: one row per (basket_id, item). Uses lift and a support floor."""
import itertools
from collections import Counter
import pandas as pd

TX = [("b1", "bread"), ("b1", "butter"), ("b1", "milk"),
      ("b2", "bread"), ("b2", "jam"),
      ("b3", "butter"), ("b3", "milk"), ("b3", "eggs"),
      ("b4", "bread"), ("b4", "butter"), ("b4", "eggs"),
      ("b5", "milk"), ("b5", "eggs"),
      ("b6", "bread"), ("b6", "butter"), ("b6", "milk"), ("b6", "eggs")]

df = pd.DataFrame(TX, columns=["basket", "item"])
n = df["basket"].nunique()
baskets = df.groupby("basket")["item"].apply(set)
item_support = df["item"].value_counts() / n
pair = Counter()
for items in baskets:
    for a, b in itertools.combinations(sorted(items), 2):
        pair[a, b] += 1
        pair[b, a] += 1

rows = []
for (a, b), c in pair.items():
    sup_ab = c / n
    conf = sup_ab / item_support[a]
    rows.append({"antecedent": a, "consequent": b, "support": round(sup_ab, 3),
                 "confidence": round(conf, 3),
                 "lift": round(conf / item_support[b], 3)})
rules = pd.DataFrame(rows).query("support >= 0.15").sort_values("lift", ascending=False)
print(f"{n} baskets, {df['item'].nunique()} items, {len(rules)} rules at support >= 0.15")
print(rules.head(8).to_string(index=False))

# Delisting check: what fraction of the victim item's baskets co-contain the anchor?
anchor, victim = "bread", "eggs"
co = (baskets.apply(lambda s: anchor in s and victim in s)).sum()
print(f"\nBaskets with {victim}: {(baskets.apply(lambda s: victim in s)).sum()}"
      f" | also with {anchor}: {co}")
```

The last block is the pattern to internalise: before deleting an item, compute how many of its baskets contain the category's anchor items. An item in few baskets but mostly anchor baskets is a basket component, not a slow mover.

**Cross-selling** uses the same matrix in the other direction: given what is in the basket, rank the items with the highest expected incremental margin, not the highest lift. A rule with lift 8 and a 4% margin is worth less than a rule with lift 1.5 and a 40% margin.

### 4.6 Space elasticity and planograms

Space elasticity relates sales to allocated space, usually facings:

```
sales ∝ (facings)^α        α typically 0.1 to 0.3 for grocery
```

**Worked example.** At 4 facings an item sells 100 units per week. Doubling to 8 facings with `α = 0.2`:

```
8/4 = 2 → 2^0.2 = 1.149 → 115 units, a 15% increase
```

A 15% sales gain for 100% more space is a poor trade unless the item's margin per facing exceeds the displaced item's. That comparison — **marginal margin per facing**, not sales per facing — is the correct allocation rule.

| Planogram element | Rule of thumb | Analytical check |
|---|---|---|
| Vertical position | Eye-level for highest-margin, not highest-volume | Test position effects; they are large |
| Blocking | Brand blocks for destination, colour blocks for impulse | Measure switch rates |
| Facings | Proportional to velocity, with a minimum of 2 for visibility | Compare marginal margin per facing |
| Adjacency | Complement adjacency for basket pairs | Use lift, filtered by margin |
| Shelf height | Heavy and bulk at the bottom | Ergonomic, not analytical |

Planograms are also a **compliance** problem: the share of stores executing the planogram as designed is usually well below 100%, so the measured effect of a planogram change is diluted by non-compliance. Measure compliance before concluding the planogram did not work.

### 4.7 Private label and brand architecture

| Tier | Position | Role | Assortment implication |
|---|---|---|---|
| Value / entry | Lowest price | Price anchor, traffic | Narrow, high-velocity |
| Standard / own brand | Price and quality parity | Margin, loyalty | Broadest tier |
| Premium / specialty | Above national brand quality | Differentiation | Narrow, curated |
| Exclusive / licensed | Unique to the retailer | Destination, defence | Very narrow |

Private label changes the assortment calculus in two ways. It raises the retailer's margin per unit, so a private-label item can justify space at lower velocity than a branded one. And it changes cross-price elasticity: a strong private label makes the national brand more elastic, because the customer now has a close substitute inside the store.

**Brand architecture** determines how many items are genuinely distinct. Ten SKUs that are the same product in different pack sizes are one product with ten price points. Counting them as variety overstates breadth, which is why a rationalisation that removes pack-size duplication often raises perceived variety per facing.

### 4.8 Segmentation and CLV

| Segmentation basis | Example | Assortment use |
|---|---|---|
| Behavioural / RFM | Recency, frequency, monetary | Depth for high-value segments |
| Basket / mission | Quick trip, weekly shop, stock-up | Breadth for mission coverage |
| Demographic | Household composition | Pack size, dietary variants |
| Value | CLV decile | Protect the items the top decile buys |
| Channel | Store, online, delivery | Channel-specific assortment |

Customer lifetime value, in its simplest defensible form:

```
CLV = (average basket margin × visits per year × expected retention years) − acquisition cost
```

**Worked example.** Basket margin $12, 24 visits per year, 4-year expected life, acquisition cost $40:

```
CLV = 12 × 24 × 4 − 40 = $1,152 − $40 = $1,112
```

Model-based CLV (Pareto/NBD, BG/NBD with a gamma-gamma spend model) improves the retention and frequency estimates for non-contractual settings, but the simple formula is usually enough to rank customers for an assortment decision. The trap is using CLV as a segmentation input when the segment is defined by the outcome — a "high-CLV" segment will by construction buy the high-margin items, and an assortment built for it will simply reinforce the status quo.

### 4.9 Store clustering and localisation

Stores differ in format, catchment, competition, and customer mix. A single planogram across a heterogeneous estate is either over-ranged in small stores or under-ranged in large ones. Clustering groups stores with similar demand structure so that each group can carry its own assortment.

**Worked example.** 60 stores, eight category-share features, k-means on standardised features.

| k | Within-cluster SS | R² = 1 − WSS/TSS | Stores per cluster |
|---:|---:|---:|---|
| 1 | 100.0 | 0.000 | 60 |
| 2 | 24.0 | 0.760 | 31 / 29 |
| 3 | 9.0 | 0.910 | 22 / 19 / 19 |
| **4** | **5.4** | **0.946** | **17 / 15 / 14 / 14** |
| 5 | 4.6 | 0.954 | 14 / 13 / 12 / 11 / 10 |

With total sum of squares 100, `R² = 1 − WSS/TSS`. The gain from k=4 to k=5 is 0.008, below a 0.01 stopping threshold, so k=4 is chosen.

**The clustering is the easy part; the naming is the work.** A cluster is only usable if it can be described in operational language — "urban convenience, high fresh share, low household penetration" — and if the buyer can act on the description. A cluster defined only by a centroid vector will not survive the first merchandising meeting.

| Validation check | Question | Failure signal |
|---|---|---|
| Stability | Do clusters persist across bootstrap samples? | Store membership flips between runs |
| Separation | Is between-cluster distance large relative to within? | Silhouette near zero |
| Actionability | Does each cluster imply a different planogram? | Two clusters with identical action |
| Coverage | Does every cluster have enough stores to range? | Singleton clusters |
| Face validity | Do the clusters match known store types? | Clusters that contradict operations |

### 4.10 Omnichannel assortment and channel conflict

| Model | Assortment implication | Conflict |
|---|---|---|
| Store-only | Space-constrained, deep in fast movers | None |
| Ship-from-store | Store carries items it would not stock for local demand | Store inventory committed to online orders |
| Endless aisle | Long tail online, core in store | Cannibalisation of store long tail; supplier terms |
| BOPIS | Store is a pick location | Pick accuracy, out-of-stocks for walk-in customers |
| Marketplace | Third-party breadth | Price and brand control |

The central tension is that online assortments are effectively unconstrained by shelf space, while store assortments are not, and both draw on the same inventory. A long-tail item that sells twice a year nationally is viable online and impossible in 400 stores. The analytics question is which items earn physical space: the answer is not sales rank but **basket completion and mission coverage per square metre**.

**Channel conflict** arises when the online assortment undercuts the store's price or offers items the store's buyer was told not to stock. The control is a single assortment governance process with explicit channel rules, not a reconciliation after the fact.

### 4.11 New item acceptance and cannibalization

Cannibalization is the share of a new item's volume that comes from existing items rather than from new demand.

```
cannibalization rate = units lost by existing items / units sold by the new item
incremental units    = new item units − units lost by existing items
net category effect  = new item margin − margin lost on cannibalized units
```

**Worked example.** Category sells 20,000 units per week at $3.00 margin. A new item launches and sells 5,000 units per week; existing items fall to 17,800.

```
Units lost              = 20,000 − 17,800 = 2,200
Cannibalization rate    = 2,200 / 5,000 = 44%
Margin lost             = 2,200 × $3.00 = $6,600
New item margin         = 5,000 × $3.50 = $17,500
Net category margin     = $17,500 − $6,600 = $10,900 per week
Incremental units       = 5,000 − 2,200 = 2,800
Incremental margin      = 2,800 × $3.50 = $9,800
```

The two margin figures differ by $1,100 because the 2,200 cannibalized units each gained $0.50 of margin: `2,200 × 0.50 = 1,100`. Both are correct, and a report that quotes only one is easy to attack. Quote the net category effect and show the decomposition.

| Effect | Definition | Measurement |
|---|---|---|
| Cannibalization | New item takes volume from own items | Matched-control DiD on existing items |
| Halo | New item lifts adjacent categories | Cross-category basket analysis |
| Market expansion | New item brings new customers or trips | Penetration and trip-frequency change |
| Switching | Volume from competitor items | Share change in the category |

**Beware the naive comparison.** Comparing post-launch category sales with pre-launch sales confounds the launch with seasonality, price changes, and distribution gains. The defensible estimate uses a control store set that did not receive the item, and reports the difference-in-differences.

### 4.12 Causal assortment evaluation

| Design | When to use | Assumption |
|---|---|---|
| Store-level A/B | Enough stores per arm | Random assignment, no cross-store spillover |
| Matched-pair DiD | Limited store count | Parallel trends in the matched pairs |
| Synthetic control | One treated market | Donor pool reproduces pre-period trend |
| Regression discontinuity | A threshold drove the change | No manipulation of the running variable |
| Switchback | One store, time-varying assortment | No carryover between periods |

The recurring threat is **spillover**: customers in a control store shop at a treated store when their item is missing. That contaminates the control and biases the effect toward zero. Check for it by measuring control-store sales of items that were removed only in treated stores; a rise in the control is the signature.

### 4.13 Assortment optimisation

The formal problem, for items `i` with space requirement `a_i` and total space `S`:

```
maximise   Σ_i x_i × (own margin_i + basket effect_i)
subject to Σ_i a_i x_i ≤ S ;  Σ_i x_i ≥ 1 per need ;  x_i ∈ {0, 1}
```

The basket effect is what makes this different from a knapsack: it is not additive, because item `i`'s value depends on which other items are selected. In practice, solve the linear problem, test borderline items with the basket matrix, then validate the set in a store test. The optimiser proposes; the test disposes.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Transaction line | Basket × item × timestamp | POS | Basket ID preserved; returns netted |
| Item master | SKU | PIM / ERP | Hierarchy: department, category, subcategory, brand, pack |
| Store master | Store | Site system | Format, size, catchment, competition |
| Planogram | SKU × store × position × facings | Space planning | Effective dates; compliance audit |
| Loyalty basket | Customer × basket | CRM | Consent and pseudonymisation |
| Inventory and in-stock | SKU × store × day | WMS | Distinguishes delisted from out-of-stock |
| Cost and margin | SKU × period | Finance | Net of trade terms and shrink |

**Standards.** GS1 GTIN for item identity and GS1 GPC for category classification; ISO 3166 for country and ISO 4217 for currency in multi-market estates; EPCIS for supply-chain event traceability where item-level serialisation applies. Planogram exchange has no single open standard — most retailers exchange via vendor-specific formats or a space-planning tool's native file, so version and effective date must be carried explicitly.

**Workflow.** Reconcile basket data to the financial sales ledger before any analysis; a category that does not tie to the P&L will not be trusted. Distinguish permanent delisting from a temporary out-of-stock, because the two look identical in sales data and only the first is an assortment decision. Record every assortment change with a date, a store list, and a reason code, because the change log is the input to every causal evaluation you will later want to run.

## 6. Industry case studies

### 6.1 Tesco, Dunnhumby, and the loyalty basket (1995 onward)

**Context and intervention.** Tesco faced a competitive disadvantage against Sainsbury's in the 1990s and had no systematic view of what individual customers bought. Tesco launched the Clubcard in 1995 and built a joint venture with Dunnhumby to analyse basket data at customer level, using it to segment customers, target offers, and inform range and space decisions.

**Evidence and outcome.** Company accounts and the customer-analytics literature describe the programme; Tesco credits Clubcard with a material role in its late-1990s market-share gain, though specific uplift figures are company claims. Loyalty-based basket analytics became a standard retail capability, and the customer-level basket replaced the store-level aggregate as the unit of merchandising analysis.

**What transferred.** The shift of unit of analysis from the item to the basket and the customer, which is the conceptual foundation of this module. **What did not.** The assumption that loyalty data is representative. Clubcard penetration, and the systematic differences between members and non-members, mean that basket conclusions drawn from loyalty data need a coverage check before they are applied to the whole estate.

### 6.2 Walmart's "Project Impact" SKU reduction (2009)

**Context and intervention.** Walmart stores had accumulated a very large SKU count, with reports of clutter and slow replenishment. Under Project Impact, the company removed a large number of SKUs from stores — widely reported at roughly 8,500 per store — to reduce clutter and improve in-stock on the remaining items.

**Evidence and outcome.** Company statements and trade press document the programme and its reversal; attributed sales declines are company and analyst estimates rather than controlled measurements. The company restored a substantial number of the removed items after sales fell, and later messaging emphasised assortment discipline rather than pure reduction.

**What transferred.** The mechanism: deleting items that look slow at the item level can remove items that complete baskets, and the loss shows up in category and basket metrics rather than in the deleted item's own line. **What did not.** The inference that SKU reduction is inherently harmful. The failure was in *which* items were removed and how the effect was measured, not in the concept of reducing duplication.

### 6.3 Aldi and the limited-assortment model (1990s onward)

**Context and intervention.** Aldi built its business on a deliberately narrow assortment — commonly cited at roughly 1,000–1,500 core lines where full-line supermarkets carry tens of thousands. The company concentrated volume in a small number of own-brand items, rotated weekly specials, and kept store format and replenishment simple as a consequence of the narrow range.

**Evidence and outcome.** Company materials and the retail-strategy literature document the model; operating-margin comparisons are company-reported or analyst estimates. Aldi and comparable limited-assortment retailers grew share in several markets, showing that narrow assortment can be a competitive position rather than a constraint.

**What transferred.** The economics: fewer SKUs mean higher volume per SKU, better buying terms, simpler replenishment, and lower working capital — and customers accept the narrowness when the price and quality position is credible. **What did not.** Transferability to a full-line format. The model works because the customer's mission is a limited weekly shop; applied to a store whose value proposition is one-stop completeness, the same narrowness drives trips elsewhere.

### 6.4 7-Eleven Japan and item-by-item merchandising (1970s onward)

**Context and intervention.** Convenience stores have very limited space and a demand pattern that changes daily by store, weather, and time of day. 7-Eleven Japan developed *tanpin kanri* — item-by-item merchandising — in which each store's assortment and order quantities are adjusted on short cycles using POS data, weather, and local events, supported by a distribution system that delivers small quantities frequently.

**Evidence.** Documented in the operations and retail literature and in company publications; the mechanism is well described, while the performance figures are company-reported.

**Outcome.** The chain became the reference model for store-level assortment and replenishment responsiveness, and the approach spread across the convenience and small-format sector.

**What transferred.** The principle that assortment is a store-level, short-cycle decision rather than an annual category-level one. **What did not.** The cost. Frequent small deliveries and per-store ordering carry a logistics cost that only works at high store density and high gross margin; a low-density chain that copies the merchandising without the distribution network gets the cost and not the responsiveness.

## 7. Labs

### Lab 17.1 — Market basket analysis with a commercial filter

**Objective.** Mine association rules and separate statistically interesting rules from actionable ones.

**Data.** Six months of basket transactions from one store cluster, with item hierarchy and margin.

**Steps.** (1) Compute support for all items and set a floor tied to the decision. (2) Mine rules with Apriori and FP-Growth and confirm identical rule sets. (3) Rank by lift and by expected margin per basket. (4) Identify rules that would justify an adjacency change. (5) Identify rules that are artefacts of category structure. (6) Write a one-page merchandising recommendation with the rules it relies on.

**Deliverable.** A rule table, a filtered shortlist, and a recommendation.

**Acceptance criteria.** Support floor justified; lift and confidence both reported; margin weighting applied; artefacts identified. **Stretch.** Add a temporal filter and show how rules change by daypart.

### Lab 17.2 — Assortment rationalisation with cannibalization

**Objective.** Rationalise a category without destroying basket value.

**Data.** Category item-level sales, margins, planogram space, and basket co-occurrence.

**Steps.** (1) Rank items by contribution per facing. (2) Apply the decision tree in §4.3. (3) For each candidate, measure anchor-basket dependence. (4) Estimate cannibalization to remaining items. (5) Build the reduced assortment and compute the net category margin. (6) Recommend a store test for the borderline items.

**Deliverable.** A rationalisation proposal with a net-margin calculation and a test plan.

**Acceptance criteria.** Tree applied in order; basket dependence measured, not assumed; cannibalization rate computed and decomposed; test proposed for ambiguous cases. **Stretch.** Add a space constraint and solve the selection as a MILP.

### Lab 17.3 — Store clustering and localised planogram

**Objective.** Build actionable store clusters and translate them into planogram differences.

**Data.** Store-level category shares, format, size, and catchment variables for 60–200 stores.

**Steps.** (1) Standardise features and check for collinearity. (2) Cluster across a range of `k` and compute R² and silhouette. (3) Choose `k` on an explicit stopping rule. (4) Validate stability by bootstrap. (5) Name each cluster in operational language. (6) Specify the planogram difference per cluster.

**Deliverable.** Cluster profile table, validation output, and a planogram specification per cluster.

**Acceptance criteria.** `k` chosen by a stated criterion, not by eye; stability tested; clusters named actionably; differences specified concretely. **Stretch.** Estimate the sales effect of localisation with a DiD across pilot stores.

### Lab 17.4 — Causal evaluation of an assortment change

**Objective.** Estimate the effect of a delisting on category and basket outcomes.

**Data.** A pilot store set, a matched control set, and 52 weeks of pre-period data.

**Steps.** (1) Match stores on pre-period trends and category structure. (2) Check parallel trends visually and with a pre-period placebo. (3) Estimate DiD on category sales, units, and basket size. (4) Test for spillover into control stores. (5) Decompose into cannibalization and market expansion. (6) State the decision and its confidence.

**Deliverable.** An evaluation report with a pre-trend check, effect estimates, and a spillover test.

**Acceptance criteria.** Parallel trends tested rather than asserted; spillover checked; effect reported with an interval; decision rule stated in advance. **Stretch.** Repeat with synthetic control and compare.

## 8. Assignment — Assortment plan

Produce an **assortment plan** for one category in a multi-store retailer.

| Section | Content |
|---|---|
| 1. Decision | Which category, which stores, which objective, and who approves |
| 2. Current state | Breadth, depth, facings, contribution, in-stock, and duplication |
| 3. Customer | Segments, missions, and basket structure |
| 4. Basket evidence | Key rules, anchor items, and delisting risk |
| 5. Space | Space elasticity, marginal margin per facing, planogram constraint |
| 6. Clusters | Store groups, validation, and localisation rules |
| 7. Proposal | Item list by cluster, with the rationale per deletion and addition |
| 8. Economics | Net category margin, cannibalization, and working-capital effect |
| 9. Test | Pilot design, metrics, decision rule, and rollback condition |
| 10. Reproduction | Code, data snapshot, and one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Basket analysis | 20% | Lists top rules by lift | Filters by support and margin | Distinguishes artefacts; connects rules to shelf decisions |
| Rationalisation logic | 25% | Deletes lowest-selling items | Applies contribution-per-facing | Applies the full tree with basket dependence measured |
| Space modelling | 15% | Proportional facings | Space elasticity applied | Marginal margin per facing with constraint |
| Clustering | 15% | `k` chosen by eye | R² and silhouette reported | Stopping rule, stability test, actionable naming |
| Causal evidence | 15% | Before/after comparison | Matched control DiD | Parallel trends tested, spillover checked, interval reported |
| Communication | 10% | Item list | Buyer-ready proposal | A merchandising committee document with a test and rollback |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| GDPR / CCPA-CPRA | Basket and loyalty data are personal data | Consent, purpose limitation, pseudonymisation, retention limits |
| Antitrust (Sherman Act §1, EU Art. 101) | Category captaincy can become information sharing | Written data-sharing limits with supplier partners |
| Slotting and listing fees | Fees can exclude smaller suppliers | Document objective listing criteria; review for discriminatory terms |
| Private-label copying | Trade dress and design infringement | Legal review before packaging imitation |
| Fairness in assortment | Underserved communities can lose access to staple ranges | Equity check on delisting by store demographics |
| Supplier dependency | Delisting can eliminate a small supplier | Notice periods and transition planning per contract |
| Consumer protection | "Was/now" and availability claims must be truthful | Substantiate availability and price claims |
| Accessibility | Planogram and store layout affect disabled customers | Reach-range and aisle-width checks |

The fairness question is the one most often skipped. Delisting decisions concentrate in stores with the lowest sales per square metre, which are disproportionately in lower-income catchments. A rationalisation that is neutral on average can still remove the only affordable staple option from the stores that need it most. Audit delisting proposals by store demographics before approving them.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Delisting on item sales alone | Slow movers flagged for deletion | Basket effects omitted | Category and basket revenue loss | Measure anchor-basket dependence |
| Lift without support floor | Rules with huge lift, tiny support | Rare items dominate | Unactionable shelf changes | Support floor tied to the decision |
| Rule mining on non-comparable baskets | Mixed store formats in one dataset | Store heterogeneity | Rules describe the mix, not behaviour | Segment before mining |
| Out-of-stock read as delisting | Item disappears from sales | Data cannot distinguish | Wrong causal conclusion | Use in-stock and item status flags |
| Choice overload overgeneralised | "Cut the range" applied broadly | Context dependence ignored | Loss of needed variety | Structure variety; test before cutting |
| Uniform planogram across estate | Small stores under-ranged | Heterogeneity ignored | Missed sales, poor space use | Cluster and localise |
| `k` chosen by eye | Cluster count follows the slide deck | No criterion | Unstable, unusable segments | Stopping rule plus stability test |
| Cannibalization ignored | New item measured on own sales | Substitution unmeasured | Overstated launch value | DiD on existing items |
| Halo assumed, not measured | "It brings traffic" with no evidence | Untested claim | Over-investment in the item | Measure penetration and trips |
| Planogram non-compliance | Measured effect smaller than expected | Stores not executing | Wrong conclusion about the planogram | Audit compliance first |
| Spillover into controls | Control stores rise during the test | Customers travel | Effect biased to zero | Test control-store demand for removed items |
| Loyalty data treated as the population | Members differ systematically | Coverage bias | Wrong estate-wide decision | Compare member and non-member baskets |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Breadth or depth — which would you cut first, and why? | Depends on role: destination categories protect depth; convenience categories cut depth first. State the objective and the role. |
| 2 | A SKU sells 3 units a week. Argue for keeping it. | Sole option for a need, anchor-basket dependence, price-tier or brand role, or halo. Quantify each. |
| 3 | Explain lift to a buyer who thinks confidence is enough. | Confidence ignores base rate; bread→butter at 40% confidence is meaningless if 40% of all baskets have butter. Lift is 2.0 here. |
| 4 | Why is a rule with lift 12 often useless? | Support too low; not enough baskets to change a shelf, and the estimate is noisy. |
| 5 | How do you measure cannibalization? | DiD on existing items against control stores; rate = units lost / new item units; decompose into net category effect. |
| 6 | Your new item shows 44% cannibalization. Good or bad? | Depends on margin: net = new margin − lost margin on cannibalized units. Here it is positive ($10,900/wk). |
| 7 | How would you choose the number of store clusters? | R² gain against a stopping threshold, plus silhouette and bootstrap stability, plus actionability. |
| 8 | What makes a cluster useless? | Two clusters with the same action, or membership that flips between runs. |
| 9 | Explain space elasticity and its limits. | Sales ∝ facings^α, α≈0.1–0.3; diminishing and only valid near observed ranges; use marginal margin per facing. |
| 10 | What is the difference between assortment breadth and perceived variety? | Breadth counts items; perceived variety counts distinguishable options. Duplicates add breadth without variety. |
| 11 | How does private label change elasticities? | Gives an in-store substitute, making the national brand more price-elastic and shifting the category margin mix. |
| 12 | Design an assortment test in 30 stores. | 15/15 matched split or matched pairs, pre-period parallel-trends check, DiD, spillover test, pre-registered decision rule. |

## 13. Key terms

| Term | Definition |
|---|---|
| Assortment | The set of items a store or channel offers |
| Breadth | Number of subcategories or segments served |
| Depth | Number of items within a subcategory |
| Variety | Perceived differentiation among items |
| Choice overload | Deferral caused by too many comparable options |
| Category role | The strategic purpose of a category: destination, routine, occasional, convenience |
| Planogram | The visual specification of item placement, facings, and shelf position |
| Facing | One visible unit of an item on the shelf |
| Space elasticity | Sensitivity of sales to allocated shelf space |
| Market basket analysis | Mining co-occurrence patterns across transactions |
| Support | Share of baskets containing an itemset |
| Confidence | Probability of the consequent given the antecedent |
| Lift | Ratio of observed co-occurrence to independence |
| Conviction | Strength of implication, `(1−sup(B))/(1−conf)` |
| Apriori | Frequent-itemset mining algorithm using the anti-monotone property |
| FP-Growth | Frequent-pattern mining without candidate generation |
| Basket affinity | Item-item association strength used for cross-sell and delisting defence |
| Cross-selling | Recommending complementary items to increase basket value |
| Private label | Retailer-owned brand |
| Brand architecture | The tier and relationship structure across a portfolio of brands |
| Customer lifetime value | Expected margin from a customer over the relationship |
| Pareto/NBD | Probabilistic model of purchase frequency and dropout |
| Store clustering | Grouping stores with similar demand structure |
| Localisation | Adapting assortment and planogram to a store cluster |
| Omnichannel | Integrated assortment and fulfilment across store and digital |
| Channel conflict | Tension when channels compete for the same demand or margin |
| Cannibalization | Volume a new item takes from existing items |
| Halo effect | Lift in adjacent categories attributed to an item |
| Slotting fee | Payment for shelf or warehouse space |
| Difference-in-differences | Causal design comparing change in treated and control units |

## 14. Sources and further reading

**Standards and regulation**

- GS1 General Specifications and GS1 Global Product Classification. https://www.gs1.org/standards — accessed 27 October 2026.
- GS1 EPCIS 2.0 standard for supply-chain event traceability. https://www.gs1.org/standards/epcis — accessed 27 October 2026.
- Regulation (EU) 2016/679 (GDPR) and California Consumer Privacy Act as amended by CPRA.
- Sherman Antitrust Act, 15 U.S.C. §1; Consolidated TFEU, Article 101. EUR-Lex.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex.

**Peer-reviewed and foundational**

- Iyengar, S. S., and Lepper, M. R. "When Choice Is Demotivating: Can One Desire Too Much of a Good Thing?" *Journal of Personality and Social Psychology*, 79(6), 2000.
- Scheibehenne, B., Greifeneder, R., and Todd, P. M. "Can There Ever Be Too Many Options? A Meta-Analytic Review of Choice Overload." *Journal of Consumer Research*, 37(3), 2010.
- Agrawal, R., and Srikant, R. "Fast Algorithms for Mining Association Rules." *Proceedings of the 20th International Conference on Very Large Data Bases*, 1994.
- Han, J., Pei, J., and Yin, Y. "Mining Frequent Patterns without Candidate Generation." *ACM SIGMOD Record*, 29(2), 2000.
- Fader, P. S., Hardie, B. G. S., and Lee, K. L. "Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model." *Marketing Science*, 24(2), 2005.
- Schmittlein, D. C., Morrison, D. G., and Colombo, R. "Counting Your Customers: Who Are They and What Will They Do Next?" *Management Science*, 33(1), 1987.
- Kök, A. G., Fisher, M. L., and Vaidyanathan, R. "Assortment Planning: Review of Literature and Industry Practice." In *Retail Supply Chain Management*, Springer, 2008.
- Fisher, M., and Vaidyanathan, R. "A Demand Estimation Procedure for Retail Assortment Optimization with Results from Implementations." *Management Science*, 60(10), 2014.
- van Ryzin, G., and Mahajan, S. "On the Relationship Between Inventory Costs and Variety Benefits in Retail Assortments." *Management Science*, 45(11), 1999.
- Dreze, X., Hoch, S. J., and Purk, M. E. "Shelf Management and Space Elasticity." *Journal of Retailing*, 70(4), 1994.

**Industry**

- Tesco PLC annual reports and Clubcard programme disclosures; Dunnhumby company materials. Company claims on programme contribution.
- Walmart Stores, Inc., merchandising statements and trade press on Project Impact, 2009–2011. Company and analyst reports; SKU counts are estimates.
- Aldi Süd and Aldi Nord company materials and the limited-assortment retail literature. Store range figures are commonly cited estimates `[VERIFY: current core range count per market before quoting]`.
- Seven & i Holdings, *Seven-Eleven Japan* operational disclosures on tanpin kanri and combined distribution. Company-reported.
- McKinsey & Company, retail assortment and category-management practice publications. Consultant view.

**Tools and datasets**

- `mlxtend` frequent-pattern mining (`apriori`, `fpgrowth`, `association_rules`). https://rasbt.github.io/mlxtend/ — accessed 27 October 2026.
- scikit-learn clustering and metrics (`KMeans`, `silhouette_score`). https://scikit-learn.org — accessed 27 October 2026.
- Dunnhumby Complete Journey and Corporación Favorita grocery datasets for basket labs. Accessed 27 October 2026.
- Google OR-Tools for the assortment MILP. https://developers.google.com/optimization — accessed 27 October 2026.

> **Note:** An assortment is an argument about what a customer is trying to do, expressed as a list of items. The list is only as good as the argument, and basket data is the evidence that the argument is either right or expensive.

---

> **Next:** [Module 18](18_deep_learning_foundation_models_llm_agents.md) — from structured baskets to unstructured text and images: deep learning, foundation models, and the agents that now sit between the planner and the decision.