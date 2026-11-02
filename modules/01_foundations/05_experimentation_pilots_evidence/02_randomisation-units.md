# Randomisation Units — customer, store, SKU, lane, DC, and the unit that matches the decision

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 04 topics 01–05; Module 05 topic 01
> Version 2026.5

---

## 1. Why this topic exists

Every pilot in a supply chain has to answer one question before it can answer any other: **what
gets randomised?** The answer is not obvious, it is not dictated by the software, and it is not the
same as "what gets measured". Getting it wrong produces one of the few outcomes in analytics worse
than a wide confidence interval — a narrow one attached to the wrong quantity.

Four failures recur in commercial supply chain work, and all four are unit failures.

**One. The store pilot analysed at basket level.** A grocery chain tests a new front-of-store
display in 48 of its 96 stores. The analyst pulls loyalty transactions, gets 710,400 rows, runs a
t-test on basket spend, and reports a standard error of GBP 0.033. The test was assigned to 96
stores. The 710,400 baskets are 96 facts observed about 7,400 times each. The reported interval is
roughly twenty times too narrow, and it is reported to two decimal places.

**Two. The SKU test that leaked through substitution.** A retailer randomises 14 household-paper
stock-keeping units (SKUs) to a promotional price inside every store, with 22 SKUs as control.
Shoppers switch between a promoted and an unpromoted brand on the same shelf, in the same trip. The
control SKUs lose volume to the treated ones, so the treatment effect is measured against a
counterfactual that the treatment itself damaged.

**Three. The lane pilot that moved equipment.** A logistics team tests a new carrier on 30 of its
140 lanes. The carrier reallocates trailers across its own network, so some control lanes improve
because treated lanes absorbed spare capacity. The control group is contaminated by a mechanism
nobody modelled, and the direction of the bias depends on the carrier's allocation rule.

**Four. The distribution centre (DC) pilot with no replication.** A new slotting algorithm is
installed in one DC and compared with the other nine. There is one treated unit, so no
between-cluster variance can be estimated, and the comparison is a before-and-after confounded with
a seasonal peak that began the same week.

> **Note:** None of the four is a statistical failure in the narrow sense. In each case the
> arithmetic is right and the estimand is wrong, because the unit randomised, the unit analysed, and
> the unit the decision applies to were three different things.

**Why this is the second topic, not the fifth.** Topic 01 turned a claim into a testable estimand, and
an estimand is not fully specified until it names its unit: "spend per transaction" and "mean spend
per store" are different quantities with different variances, and only one is what the category
director will act on.

**The claim of this topic.** There are three units in any experiment — assignment, analysis, and
decision — and a test is interpretable only when the first two match and the third is served by
them. When assignment and analysis differ, the analysis understates uncertainty by a factor that
can be computed in advance. When assignment and decision differ, the test answers a question nobody
asked.

**Where this sits.** Topic 03 turns the design effect into full cluster-design arithmetic. Topic 04
handles the case where the only feasible unit is a market. Topic 05 handles the case where even the
market is too coarse and the unit becomes a time block. Topic 06 turns the variance structure into a
minimum detectable effect. Topic 07 treats interference formally. This topic is the vocabulary and
the choice.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** the unit of assignment, of analysis, of decision, and of measurement, and
   **identify** all four in a described pilot within five minutes.
2. **Draw** the nesting ladder for a named supply chain — customer, household, account; shipment,
   order, lane; SKU, product family, category; store, district, region; DC, facility group — and
   **state** which levels are nested and which are crossed.
3. **Compute** the design effect and effective sample size for a cluster-assigned test, and
   **explain** why effective sample size converges to `clusters / ICC` as cluster size grows.
4. **Diagnose** treatment leakage from a described operational process, and **name** the mechanism
   by which control units are partly treated.
5. **Choose** an assignment unit for a stated decision, and **defend** it against two alternatives on
   contamination, sample size, and operational feasibility.
6. **Recognise** a split-plot structure, **identify** whole-plot and sub-plot factors, and **state**
   which factor is estimated with which error term.
7. **Estimate** the cost of a unit choice in stores, weeks, or currency, and **state** whether the
   proposed pilot can detect the effect the business hopes for.
8. **Critique** a test report for unit mismatch, and **rewrite** its claim so that it is defensible
   at the unit that was actually randomised.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three units in your head at once, and one rule for choosing among them.

### 3.1 The three units

```text
   THE THREE UNITS, AND WHAT HAPPENS WHEN THEY DISAGREE

     ASSIGNMENT              ANALYSIS                DECISION
     (where the coin lands)  (what the SE assumes)   (who acts, on what)

     +-------------+         +-------------+         +-------------+
     |    STORE    |  ---->  |   BASKET    |  ---->  |  CATEGORY   |
     |   n = 96    |         | n = 710,400 |         |  DIRECTOR   |
     +-------------+         +-------------+         +-------------+
           |                       |                       |
     one coin flip per      every row treated as     the rollout happens
     store, 48 per arm      an independent fact       store by store

     ASSIGNMENT = STORE, ANALYSIS = BASKET
        -> the standard error is too small by sqrt(DEFF)
        -> in the worked example, by a factor of 21

     ASSIGNMENT = STORE, DECISION = STORE
        -> coherent. The test answers the question the director asks.

     ASSIGNMENT = SKU, DECISION = CATEGORY
        -> incoherent. The test measures a within-shelf substitution
           effect; the decision is about the whole shelf.
```

### 3.2 The ladder

Every supply chain has a hierarchy of possible units, and the levels are not interchangeable. Two
properties matter: whether a level is **nested** inside another (a store is inside a region) or
**crossed** with it (an SKU is sold in many stores, and a store sells many SKUs).

```text
   THE NESTING LADDER FOR A TYPICAL RETAIL SUPPLY CHAIN

   DEMAND SIDE                          SUPPLY SIDE
   -----------                          -----------

   region                               facility group
     |                                    |
     +-- district                         +-- distribution centre (DC)
           |                                    |
           +-- store                             +-- dock / zone
                 |                                  |
                 +-- household                        +-- pick face
                       |                                  |
                       +-- customer account                +-- SKU slot
                             |
                             +-- basket / transaction
                                   |
                                   +-- line item  ---- crosses to ----> SKU
                                                                          |
                                                                     product family
                                                                          |
                                                                     category

   LANE is not on either ladder. A lane is an origin-destination pair plus a
   mode: (DC 4 -> Store 117, road). It is crossed with everything above,
   because one DC serves many stores and one store receives from many lanes.
```

**Nested levels share variance; crossed levels share exposure.** Two stores in a district share a
manager, a delivery schedule, and a local labour market, so their outcomes are correlated before
treatment. Two SKUs in a category share a shelf, a shopper, and a substitution budget, so treating
one changes the other. Nesting creates correlation; crossing creates contamination. They need
different remedies: nesting is handled by the design effect, crossing by changing the unit.

**The lane is the level practitioners most often forget.** A lane is not a place, so it appears on no
org chart, yet it is the natural unit for carrier tests, transit-time interventions, and routing
changes, and it is crossed with both ladders above.

### 3.3 The leak

Contamination, also called spillover or interference, is the property that a control unit's outcome
depends on the treatment assigned to some other unit. When it is present, the control group is no
longer a counterfactual.

```text
   TREATMENT LEAK WHEN THE UNIT IS TOO SMALL

   +----------------+                       +----------------+
   |   STORE 12     |                       |   STORE 13     |
   |   TREATMENT    |                       |    CONTROL     |
   | new shelf block|                       |  old layout    |
   +----------------+                       +----------------+
          ^                                        ^
          |          same household                |
          +--------------- visits ---------------->+
                     both stores in one week

   The household is the unit whose behaviour the intervention is meant to
   change. Assignment was at store. The household saw both conditions, so
   its "control" behaviour is partly a response to the treatment.

   Remedy, in order of preference:
     1. assign at the household (if delivery allows it)
     2. assign at a geography that contains whole households
     3. measure the leak and report the effect as a bound, not a point
```

The rule that follows is blunt: **never randomise at a level that lets the treatment reach a unit
assigned to the other arm.** A shelf layout reaches every shopper who enters the store, so no unit
below the store is safe unless shoppers are confined to one store. A coupon mailed to a household is
safe only if the same person cannot receive the control coupon through another account.

### 3.4 Why the design effect exists, in one line

Correlation inside a cluster means the second observation carries less new information than the
first. The design effect converts a row count into an information count: `DEFF = 1 + (m − 1) × ICC`,
where `m` is average cluster size, so `effective n = n / DEFF`, and as `m` grows the effective count
tends to `clusters / ICC`.

That limit is the most useful number in this topic. With 96 stores and an ICC of 0.06, effective `n`
tends to `96 / 0.06 = 1,600` no matter how many rows you collect. Adding transactions, weeks, or
loyalty households inside those 96 stores cannot move it, because information is capped by the number
of independent coin flips, not by the measurements taken after each flip.

> **Note:** The limit assumes equal cluster sizes and one variance component. Unequal sizes make it
> worse: a few very large clusters carry less information than the formula suggests. Topic 03 gives
> the correction.

### 3.5 The rule for choosing

Three questions, in this order. The order matters.

1. **What behaviour must change, and at what level is that behaviour decided?** A category director
   decides a planogram for all stores. A replenishment planner decides a parameter per item-location.
   A carrier decides a route per lane. That level is the unit of decision.
2. **What is the smallest unit at which the intervention can be delivered without crossing into
   another unit?** This is the unit of assignment, usually the unit of decision or one level
   coarser.
3. **What will be measured, and at what level?** Measurement can be finer than assignment, and
   often should be, but the analysis must aggregate to the assignment level or model the clustering.

The practical rule, stated once: **assign at the level closest to the thing whose behaviour you are
trying to change, and never below the level at which the treatment can travel.**

### 3.6 Split-plot: when two factors must live at two levels

Some pilots vary two things at once, and one cannot be varied at the same granularity as the other.
The warehouse management system (WMS) release is installed DC by DC. The pick-path rule is a setting
inside a zone and can differ zone by zone within a DC.

```text
   SPLIT-PLOT: ONE FACTOR AT DC LEVEL, ONE AT ZONE LEVEL

   DC 1  [ WMS = new ]                    DC 2  [ WMS = old ]
   +--------------------------+           +--------------------------+
   | zone A   zone B          |           | zone A   zone B          |
   | path=NEW path=OLD        |           | path=NEW path=OLD        |
   |                          |           |                          |
   | zone C   zone D          |           | zone C   zone D          |
   | path=OLD path=NEW        |           | path=OLD path=NEW        |
   +--------------------------+           +--------------------------+

   WHOLE PLOT  = DC     -> whole-plot factor = WMS version
   SUB-PLOT    = zone   -> sub-plot factor   = pick-path rule

   The WMS effect is estimated against DC-to-DC variation (few df).
   The path effect is estimated against zone-to-zone variation within DC
   (many df). The two effects have different precision, and the WMS
   comparison is the weaker one even though it is the larger question.
```

The name is agricultural: a whole field is irrigated one way, and seed varieties are planted in
strips inside it. The structure survives in industrial experimentation because a hard-to-change
factor lives at a coarse level and an easy-to-change factor at a fine level. The statistical
consequence is two error strata; the practical one is that the hard-to-change factor always has
fewer degrees of freedom. Analysing such a design as a single-error factorial is the inadvertent
split-plot, and it inflates the significance of the whole-plot factor — usually the expensive one.

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention; a definition without one is the defect this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Unit of assignment | The object to which treatment is randomised; the thing that receives a coin flip | Called "the unit of analysis" | Identifier, such as `store_id` |
| Unit of analysis | The object the statistical model treats as an independent observation | Assumed to equal the unit of assignment | Identifier, one row per object |
| Unit of decision | The object the decision applies to, and the level at which the deciding role acts | Never stated explicitly | Identifier plus role |
| Unit of measurement | The object recorded in a row of the source table, before aggregation | Confused with the unit of analysis | Table grain |
| Unit of inference | The population the claim generalises to; Donner and Klar argue this is clearer than "unit of analysis" | Used as a synonym for assignment | Population definition |
| Cluster | A group of units assigned together, so units within it share a treatment | Used to mean "segment" or "cohort" | Group identifier |
| Cluster randomised trial (CRT) | A trial in which whole clusters, not individuals, are randomised | Assumed to be weaker in every respect | Design label |
| Intra-cluster correlation (ICC) | Correlation between two observations in the same cluster; equivalently the share of variance that is between clusters | Assumed zero by default | Dimensionless, 0 to 1 in practice |
| Design effect (DEFF) | Variance under the actual design divided by variance under simple random sampling of the same rows | Reported as a variance ratio when a standard-error ratio is meant | Dimensionless, at least 1 |
| Effective sample size | `n / DEFF`; the number of independent observations carrying equivalent information | Reported as the raw row count | Count of units |
| Variance component | The between- or within-cluster variance in a hierarchical model | Estimated on one dataset and reused without checking | Units squared |
| Minimum detectable effect (MDE) | Smallest true effect a design detects with stated power at a stated significance level | Quoted without stating power or alpha | Units of the outcome |
| Stable unit treatment value assumption (SUTVA) | A unit's outcome depends only on its own assignment, not on anyone else's | Asserted rather than argued | Assumption |
| Interference | The general condition in which one unit's outcome depends on another's treatment | Used interchangeably with contamination | Mechanism |
| Spillover | Interference crossing the boundary of an assigned unit | Treated as nuisance rather than bias | Mechanism |
| Contamination | The specific case where control units receive part of the treatment | Used loosely for any noise | Mechanism |
| Carryover | Interference along time: a unit's current outcome depends on its own past treatment | Ignored in switchback designs | Mechanism |
| Treatment leakage | The operational route by which treatment reaches a control unit | Treated as statistical when it is operational | Process description |
| Substitution | Cross-unit competition in which treating one unit diverts demand from another in the same cluster | Confused with genuine lift | Cross-elasticity |
| Cannibalisation | The portion of a measured gain that comes from another unit's loss inside the same decision boundary | Reported as growth | Units or currency |
| Nesting | A hierarchy in which each lower unit belongs to exactly one higher unit | Confused with crossing | Hierarchy relation |
| Crossing | A structure in which every level of one factor co-occurs with every level of another | Confused with nesting | Factorial relation |
| Blocking | Grouping similar units before randomisation, then randomising within groups | Confused with analysis-time stratification | Block identifier |
| Stratification | Partitioning the sample into homogeneous strata before randomisation or estimation | Used to mean "segmentation" | Stratum identifier |
| Matched pairs | A design pairing units on pre-period similarity with one of each pair treated | Treated as a substitute for randomisation everywhere | Pair identifier |
| Cluster-level analysis | Computing one summary per cluster and analysing the summaries | Used without noting the loss of within-cluster information | One row per cluster |
| Multilevel model | A model with random effects at more than one level | Called "hierarchical" when only standard errors are clustered | Level identifiers |
| Generalised estimating equations (GEE) | A marginal approach handling clustering through a working correlation structure | Reported as if it were a random-effects model | Cluster identifier |
| Cluster-robust standard error | A standard error allowing arbitrary correlation within a cluster | Used with 10 clusters and no small-sample correction | Cluster identifier |
| Degrees-of-freedom penalty | Precision lost from estimating between-cluster variance on few clusters | Omitted from power calculations | Clusters minus parameters |
| Whole plot | The larger experimental unit in a split-plot design, carrying the hard-to-change factor | Called "the cluster" when the design is really split-plot | Plot identifier |
| Sub-plot | The smaller unit nested in a whole plot, carrying the easy-to-change factor | Randomised as if independent of the whole plot | Plot identifier |
| Split-plot design | Two randomisation stages at two levels, producing two error strata | Analysed as a single-error factorial | Design label |
| Switchback | A design in which one unit alternates between arms over time blocks | Called a within-store A/B test | Time block identifier |
| Geo | A geographic randomisation unit used when individual assignment is impossible | Called "a market" with no boundary definition | Postal code, DMA, or polygon |
| Household | A group of people sharing a dwelling and often a loyalty account | Treated as equal to "customer" | Household identifier |
| Customer account | A billing or login identity that may cover several people or households | Treated as equal to "person" | Account identifier |
| Basket | One shopping transaction, containing one or more line items | Called "a customer" | Transaction identifier |
| Lane | A directional origin-destination pair with a mode and a service | Called "a route" | Origin, destination, mode |
| Stock-keeping unit (SKU) | The finest saleable item definition, including pack size and variant | Used interchangeably with "product" | GTIN plus pack configuration |
| Product family | A group of SKUs customers treat as substitutes | Defined by the merchandise hierarchy rather than by behaviour | Family identifier |
| Holdout | Units deliberately excluded from a change, used as a comparison group | Assumed to remain uncontaminated | Unit identifier and period |

Five of these carry most of the practical damage.

**Assignment versus analysis versus inference.** Donner and Klar (2004) argue that "unit of analysis
problem" is misleading and that "unit of inference" is the useful phrase, because an individual-level
analysis that properly accounts for clustering is equivalent to a correctly weighted cluster-level
analysis. That is correct, and most practitioners do not need it. What they need is operational:
assignment is where the coin lands, analysis is what the model treats as independent, and inference
is the population you may generalise to.

**ICC versus design effect.** The ICC is a property of the data-generating process; the design effect
is a property of the design. An ICC of 0.06 is a fact about stores; a design effect of 445 is a fact
about a study that put 7,400 baskets into each store. The same ICC with 200 baskets per store gives a
design effect of 13. The ICC is what you learn; the design effect is what you compute.

**Contamination versus interference.** Interference is the general condition; contamination is the
case where control units get some treatment; spillover is interference crossing a boundary. The words
matter because the remedies differ. Contamination is usually an operations problem solved by
geography or delivery mechanism; interference inside the treated group is a market-equilibrium
problem solved by changing the estimand or the design.

**Nesting versus crossing.** Nesting is handled by a variance component and a design effect; crossing
by choosing a different unit or modelling the cross-unit effect. A regional shock hits every store in
a region, and the design effect absorbs it. A promotion on one SKU steals from another in the same
store, and no design effect repairs that, because the interference is the effect you are measuring.

**Substitution, cannibalisation, pull-forward.** Substitution is a shopper buying a different item in
the same category; cannibalisation is the accounting consequence for a portfolio; pull-forward is
buying the same item earlier. All three inflate a measured lift against a naive baseline, and all
three are invisible in a test randomised at too fine a level.

## 5. The mechanics, step by step

Five phases, fifteen steps. Each step names what you do, what you check, and what fails if the check
is skipped.

### Phase A — Fix the decision and its unit

**Step 1. Write the decision as "who will do what, to how many units, when".**

"The category director will roll out the new front-of-store block to all 96 stores in Q3, or not."
That fixes the unit of decision as the store, the population as the 96 stores, and the timing.
Without it you cannot choose anything else.

*Output:* one sentence with a role, an action, a unit, and a date. *Fails if skipped:* the test
measures basket-level behaviour while the decision is store-level rollout, so the measured effect is
not the decided effect.

**Step 2. Ask whether the decision is made at that level, or only announced there.**

A planogram is announced centrally and executed store by store, but compliance varies. If store
managers can decline or adapt, the effective decision unit is managerial discretion, and the test
should measure compliance as an intermediate outcome. This is where the stated decision unit and the
real one most often diverge.

*Check:* ask the person who will execute it, not the person who will approve it. *Fails if skipped:*
an intention-to-treat effect diluted by non-compliance, reported as the effect of the intervention.

**Step 3. State the population of units and whether the sample covers all of them.**

Ninety-six stores exist. If all are in the test, the finite population correction applies and the
target is these stores, not stores in general. If 40 are chosen, the selection rule determines
whether the result generalises.

*Record:* frame size `N`, assigned count `n`, selection rule. *Fails if skipped:* a result from the 40
highest-volume stores presented as a chain-wide effect.

### Phase B — Draw the hierarchy and find the crossing points

**Step 4. Draw the nesting ladder for both demand and supply side.**

Use section 3.2 as a template. Name the actual levels in the actual business, with the identifier
that represents each level in the warehouse. *Check:* can every outcome row be traced to exactly one
node at each level?

**Step 5. Mark the levels that are crossed rather than nested.**

| Crossing | What flows across | Consequence for assignment |
|---|---|---|
| SKU between stores | Shelf space, substitution, shopper budget | Randomising SKUs within a store leaks through substitution |
| Shoppers between stores | Households visit several stores in a week | Randomising stores in one catchment leaks at the household |
| Lanes within a carrier | Equipment, crews, network reallocation | Randomising lanes to a carrier option leaks through capacity |
| DCs within a territory | Overlapping delivery postcodes, shared inbound freight | Randomising DCs leaks where service areas overlap |
| Accounts within a household | Shared cards, addresses, logins | Randomising accounts leaks at the person |

*Fails if skipped:* a control group that is partly treated, and an effect that is a difference between
two mixtures.

**Step 6. Identify the level at which the behaviour you want to change is decided.**

For a shopper, the decision is made in the aisle, which makes the store the smallest safe unit because
the shopper is exposed to the whole store. For a replenishment planner, the decision is made per
item-location, so item-location is legitimate if the planner's workload is not shared. For a carrier,
the decision is made per lane by a routing engine, so the lane is legitimate unless the carrier
reallocates capacity across lanes.

### Phase C — Test for leakage before choosing

**Step 7. Describe the treatment delivery mechanism in physical terms.**

Not "the store gets the new layout" but "a merchandising team visits on Monday, removes four gondola
ends, installs a display unit, and leaves signage for six weeks". Physical description reveals the
leak paths: which stores the team visits, which vans it uses, which shoppers see the signage. *Fails
if skipped:* a leak path everyone in the room knows about but nobody wrote down.

**Step 8. For each leak path, ask whether it can reach a control unit.**

| Leak path | Reaches control if | Typical fix |
|---|---|---|
| Shared shopper | Catchments overlap | Assign by catchment, or use a geography |
| Shared shelf | Both SKUs sit on the same fixture | Assign at category or fixture level |
| Shared carrier capacity | The carrier serves both arms | Assign at carrier level |
| Shared inbound freight | The DCs share trucks | Assign at facility-group level |
| Shared staff | One manager runs two units | Assign at the manager's span |
| Shared marketing budget | Budget is reallocated across arms | Hold budget fixed, or assign at budget holder |
| Shared learning | Control staff see the treatment | Stagger rollout, or accept and report |

*Fails if skipped:* the "we controlled for store" defence, which controls for the level of assignment
but not for the leak across it.

**Step 9. Decide whether to eliminate the leak, model it, or bound it.**

Eliminate: change the unit or the geography so the leak cannot occur. Model: measure the leak path and
include it as a covariate or spillover term. Bound: report an interval whose width reflects the
plausible leak, and say so. *Fails if skipped:* a leak acknowledged in the appendix and absent from
the headline number.

### Phase D — Choose the assignment unit, and price it

**Step 10. Enumerate at least three candidate units and score them on four criteria.**

| Criterion | Question | Typical answer shape |
|---|---|---|
| Contamination | Can treatment reach a control unit? | Low, medium, high, with the path named |
| Sample size | How many independent units exist? | A count, and the resulting MDE |
| Feasibility | Can operations deliver it without a system change? | Yes, with cost, or no |
| Decision proximity | Is this the level the decision applies to? | Same level, one coarser, or unrelated |

A four-column table with three rows takes ten minutes and prevents most of the argument that follows.

**Step 11. Compute the design effect and effective sample size for each candidate.**

This is arithmetic, not judgement, and it usually settles the argument. Compare candidates with each
other, not with a textbook threshold. *Check:* does effective `n` change when you add rows inside
clusters? If not, you have understood the design.

**Step 12. Compute what the candidate costs, and whether the effect is detectable.**

A pilot that cannot detect the effect the business hopes for is not a smaller version of the
experiment; it is a different activity. Say which it is. *Output:* stores, weeks, and currency per
candidate, beside the MDE each achieves. *Fails if skipped:* a pilot launched to "learn" that reports
a null result, read as evidence of no effect.

**Step 13. Check the degrees-of-freedom penalty.**

Below about 40 clusters the cluster-robust variance estimator is biased downward and the normal
approximation is poor, so effective alpha exceeds the stated level and power falls short of the
calculation. Use a small-sample correction, a wild cluster bootstrap, or a cluster-level `t` test on
`k − 2` degrees of freedom. Below 20 clusters per arm, treat every interval as approximate.

### Phase E — Fix the analysis unit and report

**Step 14. Aggregate to the assignment level, or model the clustering explicitly.**

| Analysis | What you compute | Answers | Weakness |
|---|---|---|---|
| Cluster-level summary | One mean per cluster, then compare `k` values | "What happened to the average store?" | Loses within-cluster precision; needs `k` large |
| Multilevel or GEE model | Row-level model with cluster random effect or working correlation | "What happened to the average row, given clustering?" | Needs assumptions about correlation structure |
| Cluster-robust standard errors | Row-level model, variance corrected at cluster level | Similar to multilevel for large `k` | Biased with few clusters |

*Check:* the degrees of freedom in the reported interval. If it is `n − 2` where `n` is 710,400, the
analysis is at the wrong level.

**Step 15. Report the four units explicitly.**

State what was randomised, how many units per arm, what the analysis treated as independent, what the
decision applies to, and what the ICC estimate was. Five fields, one sentence each. *Fails if
skipped:* the number travels without its provenance and the next team reuses it as a chain-wide
effect.

## 6. Worked example

Harlow and Finch is a grocery chain with 96 stores in four regions. The category team proposes a new
front-of-store display block for the household-paper category, replacing two end-of-aisle displays.
The proposal is to test it in half the estate for four weeks and roll out if it produces a 6% lift in
category spend per transaction. All figures are constructed to be internally consistent and checkable
by hand.

### 6.1 The decision, the candidates, and the data

**The decision.** The category director will roll the fixture out to all 96 stores or to none. The
unit of decision is the store; the population is the 96 stores.

Five candidate units exist. The first is the one the analyst will use by default, because the data
arrives at transaction grain and the software does not ask.

| Candidate | Count | Why it might be chosen |
|---|---:|---|
| Transaction (basket) | 710,400 | Most rows, tightest naive interval |
| Loyalty household | 59,520 | Closer to the shopper |
| Store-week | 384 | Natural operations unit |
| Store | 96 | The decision unit |
| Region | 4 | Feasible, far too few |

```text
  stores 96; stores per arm 48; test 4 weeks
  transactions per store-week 1,850; per store 7,400; total 710,400
  category spend per transaction: mean GBP 9.20, SD GBP 14.00, variance 196.00
  intra-cluster correlation at store level, ICC = 0.06

  between-store variance = 0.06 x 196.00 = 11.76 GBP^2   (SD 3.43)
  within-store variance  = 0.94 x 196.00 = 184.24 GBP^2
```

Six per cent of the variation in basket spend is a store-level difference that persists across
shoppers. Six per cent sounds small. It is not.

### 6.2 The naive analysis

```text
  n per arm        355,200 transactions
  SE of the mean   sqrt(196.00 / 355,200) = 0.02349 GBP
  SE of the diff   sqrt(2) x 0.02349 = 0.03322 GBP
  MDE (80%, 5%)    2.80 x 0.03322 = 0.0930 GBP = 1.01% of 9.20
```

The naive analysis says the test detects a 1.0% change. The business hopes for 6%, so on this
arithmetic the pilot looks comfortably overpowered.

### 6.3 The honest analysis

```text
  clusters per arm, k = 48;  cluster size, m = 7,400

  variance of a store mean = 11.76 + 184.24 / 7,400
                           = 11.76 + 0.0249 = 11.785 GBP^2

  SE of the difference = sqrt(2 x 11.785 / 48) = 0.7007 GBP
  MDE = 2.80 x 0.7007 = 1.962 GBP = 21.3% of 9.20

  DEFF = 1 + (m - 1) x ICC = 1 + 7,399 x 0.06 = 444.94
  sqrt(444.94) = 21.09
```

The honest minimum detectable effect is **21.3%**, not 1.0%. The ratio is 21.1, exactly `sqrt(DEFF)`.
That is not a rounding difference; it is the difference between a test that can see a 6% effect and
one that can see nothing smaller than a fifth of the category's value.

### 6.4 Effective sample size

```text
  effective n = 710,400 / 444.94 = 1,596.6   (0.22% of the rows)
  ceiling as m grows:  k / ICC = 96 / 0.06 = 1,600
  households, m = 620:  DEFF = 1 + 619 x 0.06 = 38.14
  effective n = 59,520 / 38.14 = 1,560.6
```

Seven hundred thousand transactions carry about 1,600 transactions' worth of information, and the
design is already at its ceiling. At 620 loyalty households per store the rows fall by 91.6% and the
information falls by 2.3%. The rows were never the constraint.

### 6.5 What the design would actually take

**Stores needed for the 6% effect.** A 6% effect on GBP 9.20 is GBP 0.552.

```text
  required SE = 0.552 / 2.80 = 0.1971;  variance = 0.03887
  per arm = 0.019434;  k = 11.785 / 0.019434 = 606.5
  -> 607 stores per arm, 1,214 stores in total
```

The chain has 96. No analysis changes that.

**Duration.** Doubling to eight weeks doubles `m` to 14,800: `11.76 + 184.24 / 14,800 = 11.772`
against 11.785. The standard error falls 0.055%, and the MDE moves from 21.33% to 21.31%. Duration
buys nothing, because the within-store term was already negligible at `m = 7,400`. **Duration only
helps when clusters are small enough for the within-cluster term to matter**, which is why teams that
extend pilots instead of adding stores usually waste calendar time.

**Blocking.** Blocks on region and prior-year volume that explain 45% of between-store variance give
`6.468 + 0.0249 = 6.493`, `SE = 0.5201`, `MDE = 15.8%`. Reaching 6% with blocking still takes 335
stores per arm, 670 in total.

### 6.6 What the team did instead

Faced with these numbers the team made three changes, each defensible alone, and together
uninterpretable.

**Randomise SKUs rather than stores.** Every store carried the fixture, but only 14 of 36
household-paper SKUs were merchandised on it. Shoppers substitute between household-paper SKUs on the
same trip, so control SKUs lose volume to treated SKUs inside the same store. The measured lift is a
substitution effect plus a category-expansion effect, and the decision is about the whole shelf.

> **Note:** Roemheld and Rao (2024) report that cross-price interference in SKU-level tests can
> inflate estimated effects by up to about 100% relative to a global rollout effect. That is a
> working paper, not a refereed publication; read the figure as an order of magnitude.

**Analyse at household level to "capture the shopper".** Households are not randomly assigned; they
are exposed to both arms by construction. The change gave a finer measurement of a contaminated
comparison.

**Extend to eight weeks because the first four looked noisy.** Section 6.5 shows why this could not
have worked.

### 6.7 What a coherent design looks like

| Element | Choice | Reason |
|---|---|---|
| Unit of decision | Store | Rollout is store by store |
| Unit of assignment | Store, blocked on region and prior-year volume | Matches decision; blocking cuts between-store variance |
| Leak check | Catchment overlap; household card usage across stores | Measured, not assumed |
| Unit of analysis | Store mean over the test period, `t` on 94 df | Matches assignment |
| Power | 96 stores gives 15.8% MDE after blocking | Stated up front, and stated to the business |
| Decision rule | Roll out only if the estimate exceeds 15.8% and the interval excludes zero | Matches what the design can see |
| What it cannot answer | A 6% effect | Declared, not discovered later |

The design is not the one the business wanted. It is the one the business can have with 96 stores,
and the difference is the finding.

### 6.8 The reporting line

```text
  DESIGN    96 stores, 48 per arm, blocked on region and prior-year volume,
            4-week test, store-level assignment.
  RESULT    Mean category spend per store-week: +11.4% (95% CI +2.1% to +21.9%).
  UNIT      Analysis at store level, n = 96, t(93). Transactions were used only
            to compute store means; they are not independent observations.
  POWER     MDE at 80% power was 15.8%. The design could not have detected a
            6% effect, and the interval is consistent with both 6% and 21%.
  LIMIT     Catchment overlap between five store pairs means some control
            stores were partly exposed. For a positive effect that bias is
            toward zero, so 11.4% is if anything conservative.
```

The three sentences that make this useful are the unit, the power, and the limit. Without the power
line a reader concludes the effect is 11.4%; with it, the reader concludes it is somewhere between
2% and 22%, and that a larger test is needed before a rollout decision.

## 7. Code, query, or calculation

One block. It reproduces sections 6.3 to 6.5 so the numbers can be checked and the parameters
changed.

```python
# Design-effect arithmetic for a store-randomised retail pilot.
# Currency GBP; variances GBP^2.
mean_spend, sd_spend = 9.20, 14.00
icc, m, k = 0.06, 7400, 48          # cluster size, clusters per arm
z = 2.80                            # 80% power, 5% two-sided

var = sd_spend ** 2
deff = 1 + (m - 1) * icc
n_eff = (2 * k * m) / deff
var_store_mean = icc * var + (1 - icc) * var / m
se_diff = (2 * var_store_mean / k) ** 0.5
mde = z * se_diff

print(f"DEFF            {deff:8.2f}")
print(f"effective n     {n_eff:8.0f}  of {2 * k * m:,} rows")
print(f"ceiling k/ICC   {2 * k / icc:8.0f}")
print(f"SE of diff      {se_diff:8.3f} GBP")
print(f"MDE             {mde:8.3f} GBP  = {100 * mde / mean_spend:.1f}%")

target = 0.06 * mean_spend
print(f"stores for 6%   {2 * (2 * var_store_mean * z ** 2 / target ** 2):8.0f}")
```

Expected output, in order: `DEFF` 444.94; effective `n` 1,597 of 710,400 rows; ceiling `k / ICC`
1,600; standard error of the difference 0.701 GBP; MDE 1.962 GBP, which is 21.3% of mean spend; and
1,213 stores in total for a 6% effect.

Four checks before trusting any such block.

- **The design effect must exceed 1 whenever the ICC is positive and the cluster size exceeds 1.**
  A value below 1 means the cluster size is wrong, or a variance was entered as a correlation.
- **Effective sample size must be below the row count by roughly the design effect.** 710,400 / 445
  is about 1,600, which is 0.22% of the rows. An effective `n` above a few per cent of the rows
  means the ICC was too small for the cluster size.
- **The ceiling `k / ICC` must be close to the computed effective `n` for large `m`.** A gap above
  5% means the cluster size is small enough that the within-cluster term still matters.
- **The store requirement must scale as `1 / effect^2`.** Halving the target effect should quadruple
  the requirement. If it does not, the formula was mis-transcribed.

> **Note:** `icc * var + (1 - icc) * var / m` is the variance of a cluster mean under a one-way
> random-effects model with equal cluster sizes. Unequal sizes require `sum(m_i^2) / sum(m_i)` in
> place of `m`, and the result is always worse. Topic 03 gives the correction; the printed `1213`
> differs from the hand figure of 1,214 by rounding in `z`.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain before a unit can be chosen

| Field | Why the unit depends on it | What goes wrong without it |
|---|---|---|
| A stable unit identifier | The coin flip is applied to it and the analysis groups by it | Units cannot be held fixed across the test window |
| The parent identifier at each level | Nesting must be reconstructable | Clusters cannot be defined after the fact |
| An assignment table with a timestamp | The assignment must be auditable and immutable | Assignment drift: units change arm mid-test |
| An exposure log | Contamination is measured, not assumed | Leak paths are invisible |
| A pre-period outcome of the same definition | Blocking and variance estimation need it | Between-cluster variance cannot be estimated in advance |
| The geography of each unit | Catchment overlap is a spatial join | Leak through shared shoppers is undetectable |

The third row fails most often. If the assignment table is a mutable column in an operational table,
a store can change arm when the table is refreshed. The discipline is the same as in clinical
trials: the randomisation list is written once, versioned, and joined to outcomes, never recomputed.

### 8.2 Standards that define the units

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GLN (Global Location Number) | Location identity | Whether a store, DC, or dock has a stable identifier across systems | B |
| GS1 GTIN (Global Trade Item Number) | Trade item identity | Whether an SKU is the same SKU across retailer, supplier, and distributor | B |
| GS1 SSCC (Serial Shipping Container Code) | Logistics unit identity | Whether a pallet or case can be tracked as a unit | B |
| GS1 EPCIS 2.0 | Event data | Whether exposure events exist at a stated location and time | B |
| ISO 8601-1:2019 | Date and time representation | What a test window or time block is measured on | B |
| ISO 3534-1:2016 | Statistical vocabulary | What "cluster", "sample", and "population" mean | B |

The consequence is direct. A retailer that maps stores to GLNs can join treatment assignment to
point-of-sale (POS), warehouse, and delivery data without ambiguity. A retailer using internal store
numbers has three store lists in three systems, and the randomisation is applied to whichever list
the analyst happened to open. That is not a statistical problem, and it produces statistical
failures.

> **Note:** The GTIN defines a trade item, not a product family. Two pack sizes are two GTINs and
> one family. If the decision is a family-level decision, the assignment unit must be defined by a
> family key the business maintains, not by a prefix rule invented in the analysis.

### 8.3 Reporting standards that force the unit to be declared

| Standard | Applies to | What it requires | Grade |
|---|---|---|---|
| CONSORT extension for cluster trials (Campbell et al., 2012) | Health and public-health trials | Unit of randomisation, number of clusters, ICC, and whether analysis allowed for clustering | A |
| Cochrane Handbook, chapter 23 | Reviews of cluster trials | Unit-of-analysis error identified and corrected by effective `n` or standard-error inflation | B |
| ICH E9, *Statistical Principles for Clinical Trials* | Regulated clinical development | The unit of analysis pre-specified and justified | B |
| ICH E9(R1) addendum on estimands | Regulated clinical development | The estimand states population, variable, treatment, and handling of intercurrent events | B |
| AEA RCT Registry | Social-science field experiments | Pre-registration of the unit of randomisation and the analysis plan | B |
| ClinicalTrials.gov | Regulated and academic trials | Registration of design, including allocation unit | B |

None of these binds a commercial supply chain pilot. They matter for two reasons. First, they are
where the vocabulary came from: the design effect, the unit-of-analysis error, and the small-cluster
corrections were developed and validated in clinical and public-health trials. Second, they are a
template. A retailer that adopted the CONSORT cluster checklist for its store pilots would eliminate
most of the failures in section 10 at the cost of one page per test.

### 8.4 Privacy law as a constraint on the unit

Assigning at household or individual level requires personal data. In the European Union the General
Data Protection Regulation (GDPR) constrains what may be processed and for which purpose; loyalty data
linked to individuals needs a lawful basis and typically a privacy notice covering analytics and
experimentation, and comparable regimes apply in the United Kingdom, Brazil, and several United States
states.

Two consequences push the unit coarser. Pseudonymised identifiers are usually sufficient but not
always available, and a household key that is not linkable to a name is the thing to ask the data
protection officer for. And some attributes cannot be used for assignment even where they exist:
assigning on a protected characteristic is unlawful in many jurisdictions, and assigning on a proxy
that correlates with one is a discrimination risk even where it is legal. This is one reason
store-level and market-level designs remain common in retail even when individual assignment is
technically possible.

### 8.5 What is convention rather than requirement

- **"The customer is the natural unit"** comes from digital experimentation, where the user
  identifier is stable and treatment is delivered by a web server. In a physical store, the natural
  unit is whatever the treatment can be confined to.
- **"Always randomise at the finest possible level"** maximises unit count and maximises leakage
  risk. The two pull in opposite directions; the leak test resolves them.
- **"Twenty clusters is too few"** is a rule of thumb from cluster-robust inference. It is
  directionally right; the correct number depends on the ICC, cluster-size variation, and method.
- **"Pre/post is not an experiment"** is true and unhelpful when no concurrent control exists. The
  honest statement is that a pre/post design identifies an effect only under assumptions that must
  be named.

An analyst who can distinguish a requirement from a convention can challenge a convention; one who
cannot will treat every inherited design template as a law.

## 9. Industry practice and case evidence

### 9.1 The methodology, and where it comes from

| Contribution | Source | What it established | Grade |
|---|---|---|---|
| Randomisation by group, formalised | Cornfield (1978), *American Journal of Epidemiology* 108, 100–102, as cited in Chakraborty and Lyons (2015) | Group randomisation has a variance cost that must be computed, not assumed away | A |
| The unit-of-analysis error | Whiting-O'Keefe, Henke and Simborg (1984), cited in the Cochrane Handbook, chapter 23 | Analysing below the assignment level produces artificially narrow intervals and inflated significance | A |
| Design effect and sample size for cluster trials | Donner, Birkett and Buck (1981), cited in Chakraborty and Lyons (2015) | The `1 + (m − 1)ICC` inflation factor and its use in planning | A |
| Randomisation, delivery, and analysis levels can all differ | Hemming and Taljaard (2023), *International Journal of Epidemiology*, doi:10.1093/ije/dyad064 | Design effect and a degrees-of-freedom correction are both required | A |
| Unit-of-analysis error in practice | Chaplin and Dwan (2023), *Cochrane Evidence Synthesis and Methods*, doi:10.1002/cesm.12024 | The two remedies and worked examples | A |
| Terminology: "unit of inference" | Donner and Klar (2004), *American Journal of Public Health* 94(3), 416 | A correctly clustered individual-level analysis is equivalent to a weighted cluster-level analysis | A |

These results are forty years old, and are ignored in commercial pilots while mandatory in clinical
ones: one field has a regulator checking the analysis plan; the other has a deadline.

### 9.2 What online experimentation established

| Finding | Source | Implication | Grade |
|---|---|---|---|
| The randomisation unit determines which metrics are valid, not only how precise they are | Kohavi, Tang and Xu (2020), *Trustworthy Online Controlled Experiments*, Cambridge, ch. 14, pp. 166–170 | A metric defined below the assignment unit needs the delta method or a cluster bootstrap | A |
| Analysis below the randomisation unit requires a variance correction | Deng, Longbotham, Walker and Xu (2011), JSM proceedings | The delta-method variance for user-randomised, page-view-analysed experiments | C |
| Ramp-up creates Simpson's paradox; the analysis unit must not change mid-test | Kohavi et al. (2009), KDD, "Seven Pitfalls to Avoid when Running Controlled Experiments on the Web" | Assignment tables must be fixed at launch | C |
| Network effects violate SUTVA; cluster randomisation is the standard remedy | Microsoft Research, "Patterns of Trustworthy Experimentation: Pre-Experiment Stage" | Cluster randomisation is the tool; its cost is power | C |

The last three are conference and technical material rather than refereed articles, and are graded
accordingly. They remain the canonical practitioner references, and the distinction they draw —
between the unit randomised and the unit analysed — is the one this topic is built on.

### 9.3 Marketplace evidence: the clearest measured case

| Finding | Source | Detail | Grade |
|---|---|---|---|
| Cluster randomisation removed about 20% of the measured effect | Holtz, Lobel, Lobel, Liskovich and Aral (2025), *Management Science* 71(1), 390–406, doi:10.1287/mnsc.2020.01157 | Airbnb pricing meta-experiment, March 2019; about 2.6 million listings, of which about 1.96 million in 5,960 clusters and about 647,000 individually randomised | A |
| Naive estimate −0.345 bookings per listing; cluster-randomised −0.277 | Same | The difference implies 19.76% (±9.06%) of the individual-level estimate was interference bias | A |
| Bias larger in demand-constrained geographies | Same | 28.65% (±14.91%) against 12.05% (±11.55%) — suggestive, not significant | A |
| Cluster randomisation reduces power substantially | Same | The authors name power-preserving cluster designs as the main open problem | A |
| Which side of a two-sided market is randomised determines the direction of bias | Johari, Li, Liskovich and Weintraub (2022), *Management Science*, arXiv:2002.05670 | In demand-constrained markets customer randomisation is unbiased and listing randomisation is biased; the reverse holds in supply-constrained markets | A |

That last row is the most transferable idea here, because a supply chain is a two-sided market in
miniature. A promotion tested on the demand side is a customer-side randomisation. A carrier or
supplier intervention is a listing-side randomisation. Which one is biased depends on whether the
binding constraint that season is demand or capacity. A test that was unbiased in a slack quarter can
be biased in a tight one.

> **Note:** The Airbnb figures come from one company's meta-experiment and should not be
> generalised as a typical bias magnitude. What generalises is the existence and direction of the
> bias, and the fact that it was measurable in advance.

### 9.4 What retail practice looks like

| Practice | Evidence | What it shows | Grade |
|---|---|---|---|
| Matched-pair store randomisation | Glanz et al., Healthy Food Marketing Strategies Study, 33 supermarkets (16 intervention, 17 control), matched on store size, chain, region, and benefit-programme share | Pairing is the standard way to buy power in small store samples | A |
| Geographic separation to control contamination | Ayala et al. (2013), *Public Health Nutrition*, 4 tiendas, matched pairs ordered by distance, shoppers screened for cross-visits | Contamination control is done operationally, by distance and screening | A |
| Store-product-week randomisation without clustering | Bray, Sanders and Stamatopoulos (2024), AEA RCT Registry AEARCTR-0014114, 82 stores, 409 products, 35 weeks | The unit can be finer than the store, at the cost of within-store interference | B |
| Geo tests with few, heterogeneous units | Vaver and Koehler (2011), Google technical report | Constrained randomisation, ranking geos by size and sampling within blocks, reduced interval width by about 10% | C |
| Matching instead of randomising when geos are few | Wayfair engineering blog (2023) | 210 geos with market shares from about 6.5% to about 0.013%; integer optimisation used to balance arms | C |

The last two rows are company and vendor material, dated 2011 and 2023. They are graded C because
they are first-party claims about method, not audited results. The mechanism they describe — that a
small number of heterogeneous units makes simple randomisation unreliable, and that constrained
assignment recovers some balance — is corroborated by the refereed geo-experiment literature and by
the general result that blocking reduces variance when block membership predicts the outcome.

### 9.5 What is missing from the public record

There is no public audit of unit-choice practice in commercial supply chain pilots. Retailers do not
publish their test designs, and the failures in section 10 appear in conversation far more often than
in print. What the record supports is narrower: the arithmetic is settled, the failure modes are
documented in the clinical literature, and the one public meta-experiment that measured the bias
found it to be about a fifth of the effect.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Analysis below the assignment level | Degrees of freedom equal the row count, not the cluster count | Within-cluster correlation ignored | Standard error too small by `sqrt(DEFF)`; false positives | Aggregate to cluster, or model the clustering |
| Assignment below the leak boundary | Control units show a partial response in week one | Treatment reaches control units through a shared path | Attenuated or inflated effect, direction unknown | Leak map before launch; assign at the leak boundary |
| One treated cluster | The treated group has a single unit | No between-cluster variance can be estimated | Before-and-after disguised as a controlled test | Do not run; use a switchback in time |
| Too few clusters | Fewer than about 20 per arm | Cluster-robust variance biased down; normal approximation poor | Nominal 95% behaves like 85–90% | Small-sample correction, wild bootstrap, or more clusters |
| Extending duration instead of adding clusters | Pilot extended twice with no gain in precision | Between-cluster variance dominates; duration shrinks only the within term | Calendar time lost; decision delayed for nothing | Compute the variance decomposition first |
| SKU-level assignment on a shared shelf | Control SKUs decline while treated SKUs rise | Substitution within the category | Substitution reported as category growth | Assign at category or store level |
| Household treated as independent of store | The same household appears in both arms | Households shop across stores | Contaminated control; uninterpretable contrast | Assign by catchment, or screen cross-shoppers |
| Lane assignment ignoring carrier capacity | Control lanes improve in the same weeks | Carrier reallocates equipment across its network | Bias of unknown sign | Assign at carrier level, or hold capacity fixed |
| Assignment table is mutable | Arm membership changes between refreshes | Assignment recomputed from a changing field | Analysis is not of the design that was run | Write the list once and version it |
| Split-plot analysed as a single-error factorial | The hard-to-change factor has suspiciously tight intervals | Whole-plot error used as residual error | Overstated significance for the expensive factor | Model two error strata; report both |
| ICC borrowed without checking | The ICC comes from another category or country | ICC is category- and context-specific | Power calculation wrong by a large factor | Estimate from a pre-period, or report a range |
| Blocking defined on post-treatment data | Blocks absorb part of the effect | Blocks defined on outcomes rather than pre-period covariates | Effect understated, sometimes to zero | Define blocks on pre-period data only |

Three deserve expansion.

**Analysis below the assignment level is the most common**, and it survives review most easily
because the number looks like a number. The tell is in the degrees of freedom, and the degrees of
freedom are almost never printed beside the estimate. Ask for them.

**One treated cluster is the most damaging**, because the result is not merely imprecise but
uninterpretable, and it is often the only feasible design in a small network. The honest response is
to change the design — a switchback in time, a stepped rollout.

**Extending duration instead of adding clusters is the most expensive in calendar time**, because it
feels like rigour. Section 6.5 shows the arithmetic: doubling the window moved the MDE by 0.02
percentage points.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| SUTVA holds at the chosen unit | The control group must be an untreated counterfactual | Shared shoppers, shelf, capacity, or staff | Exposure log; catchment overlap; control-group trend check |
| The ICC is stable over the test window | Design effects and power depend on it | Seasonal change, competitor entry, supply disruption | Estimate ICC in pre-period and test period; compare |
| Clusters are exchangeable within arm after blocking | The comparison of means must be unbiased | Selection on the outcome; voluntary participation | Compare pre-period outcomes across arms |
| Cluster size is roughly equal | `1 + (m − 1)ICC` assumes it | A few very large stores or lanes | Compute `sum(m_i^2)/sum(m_i)`; correct if far above the mean |
| The unit of decision does not change mid-test | The estimand is defined by the decision | Reorganisation, scope change, new director | Written decision statement, signed before launch |
| Assignment is independent of potential outcomes | Randomisation must be the only systematic difference | Assigning on a variable that also affects the outcome | Audit the assignment algorithm for covariate rules |
| Outcome definition is stable across arms and time | The comparison must be like with like | Metric change mid-test; POS system change | Freeze the metric dictionary before launch |
| The measured unit proxies the behavioural unit | The outcome must reflect the behaviour targeted | Basket spend proxies category choice; on-time delivery proxies reliability | State the proxy; triangulate with a second measure |
| No concurrent experiment on the same units | Two treatments on one unit cannot be separated | Overlapping pilots run by different teams | Experiment registry with unit-level collision checks |
| Cluster means are approximately normal | `t` intervals at cluster level assume it | Few clusters, skewed means, small counts | Plot cluster means; consider a bootstrap over clusters |

**What no unit choice fixes.** Confounding from a non-random assignment mechanism, a metric that does
not measure the behaviour, and a decision that changes after the result is known.

**The hard limit is the number of independent units.** A chain with 96 stores has at most 96
independent observations of a store-level intervention, and no analysis changes that. It is why the
field has developed paired designs, switchbacks, and geo matching: all extract more information from
a fixed number of clusters rather than more clusters. The second hard limit is that the unit of
decision is not negotiable: an analyst chooses the assignment and analysis units, the organisation
sets the decision unit.

## 12. Exercises

### Exercise 1 — Name the four units (recall, 20 minutes)

For each scenario, state the unit of assignment, the unit of analysis as described, the unit of
decision, and whether the design is coherent.

(a) A retailer tests a new checkout layout in 30 stores and analyses 90,000 transactions.
(b) A third-party logistics provider (3PL) tests a routing engine on 200 lanes, analysing on-time
performance per lane.
(c) A manufacturer tests a supplier scorecard with 12 buyers, analysing 4,000 purchase orders.
(d) A grocer tests a loyalty coupon on 40,000 randomly selected households, analysing household
spend.

**Deliverable:** a four-row table with the four units named and a coherent/not-coherent verdict.
**Acceptance criteria:** every cell names a specific identifier type, not a generic level; each
incoherent verdict states the consequence in one sentence.

### Exercise 2 — Draw the ladder (application, 30 minutes)

Draw the nesting ladder for a business you know, covering demand and supply side, and mark every
crossing point you can identify.

**Deliverable:** one hand-drawn diagram with at least five levels per side and at least three marked
crossings, plus a two-sentence note on which crossing is most dangerous and why.
**Acceptance criteria:** nesting drawn as vertical containment and crossing as horizontal links;
identifiers named, not generic; the dangerous crossing justified by a mechanism, not intuition.

### Exercise 3 — Compute the design effect (calculation, 30 minutes)

A test assigns 60 stores, 30 per arm. Each store contributes 5,200 transactions. The ICC is 0.09.
Compute the design effect, the effective sample size, the ceiling `k / ICC`, and the ratio by which a
naive transaction-level standard error is too small.

**Deliverable:** four quantities with formulas and units, and one sentence interpreting each.
**Acceptance criteria:** `DEFF = 1 + (m − 1)ICC` written explicitly; the standard-error ratio given as
a square root; the ceiling compared with the effective sample size and the gap explained.

### Exercise 4 — Find the leak (diagnosis, 40 minutes)

A fashion retailer tests a new fitting-room layout in 25 of its 80 stores. The treatment includes new
signage, additional mirrors, and a staff training session delivered to all managers in the treated
group. The retailer also runs a national online campaign at the same time. Identify every leak path,
rank them by likely magnitude, and propose a mitigation for each.

**Deliverable:** a leak table with path, mechanism, ranking, and mitigation.
**Acceptance criteria:** at least five distinct paths; at least one organisational rather than
physical; each mitigation states whether it eliminates, models, or bounds the leak.

### Exercise 5 — Choose the unit (design, 45 minutes)

A distributor wants to know whether a new order-consolidation rule reduces cost-to-serve. It has 14
DCs, 380 lanes, 9,000 customer accounts, and 240,000 orders a year. The rule is configured in the
warehouse management system per DC. Propose three candidate assignment units, score them on the four
criteria in section 5 step 10, and recommend one.

**Deliverable:** a three-by-four scoring table plus a recommendation with a one-paragraph defence.
**Acceptance criteria:** each score justified in the cell, not merely assigned; the recommendation
names what it gives up; at least one candidate rejected for a reason other than sample size.

### Exercise 6 — Price the pilot (calculation, 40 minutes)

Using the parameters in section 6.1 but with an ICC of 0.12 and a hoped-for effect of 10%, compute the
design effect, the effective sample size, the MDE with 48 stores per arm, and the number of stores
needed to detect the 10% effect.

**Deliverable:** the four quantities with the arithmetic shown.
**Acceptance criteria:** design effect computed with `m = 7,400`; required store count computed from
the variance formula, not scaled from the worked example; the answer stated in a sentence a
commercial director would understand.

### Exercise 7 — Split-plot or not (critique, 35 minutes)

A manufacturer installs a new planning system at four plants, and within each plant runs a new
scheduling heuristic on two of six production lines. The analyst reports both effects with a single
pooled error term and 180 residual degrees of freedom.

**Deliverable:** a written critique identifying the two randomisation stages, the correct error term
for each factor, and what the reported degrees of freedom should be.
**Acceptance criteria:** whole-plot and sub-plot factors named; the direction of the bias in the
reported significance stated; a corrected analysis recommended by name.

### Exercise 8 — Design a coherent pilot (design, 60 minutes)

Design a store-level pilot for a same-day delivery promise change, for a retailer with 120 stores,
where the promise is fulfilled from store stock. Address the unit of assignment, the leak paths, the
analysis, the power, and the decision rule.

**Deliverable:** a two-page design note with a unit table, a leak table, a power calculation, and a
pre-registered decision rule.
**Acceptance criteria:** the assignment unit matches the decision unit or the mismatch is justified;
at least two leak paths identified; the power calculation states the ICC source; the decision rule is
stated before the result is known.

## 13. Self-check answers

### Exercise 1

| Scenario | Assignment | Analysis | Decision | Verdict |
|---|---|---|---|---|
| (a) Store layout | Store (30) | Transaction (90,000) | Store rollout | Not coherent: the interval is too narrow by `sqrt(1 + (m − 1)ICC)` |
| (b) Lane routing | Lane (200) | Lane | Lane or network | Coherent; cluster by carrier if the engine is carrier-specific |
| (c) Buyer scorecard | Buyer (12) | Purchase order (4,000) | Buying-team policy | Not coherent: 12 buyers are 12 independent behaviours |
| (d) Household coupon | Household (40,000) | Household | Household targeting | Coherent; check for households holding several accounts |

The incoherent cases are the ones where analysis is finer than assignment. None is incoherent because
the decision unit differs — (b) and (d) both have a decision unit at or above the assignment unit and
remain interpretable.

### Exercise 2 rubric

A strong ladder has at least five levels per side with named identifiers, draws nesting as vertical
containment and crossing as horizontal links, and justifies the most dangerous crossing by a
mechanism rather than by intuition. A weak ladder stops at store and SKU, or marks a crossing without
saying what flows across it.

### Exercise 3

```text
  k per arm = 30, m = 5,200, ICC = 0.09
  total rows = 60 x 5,200 = 312,000

  DEFF = 1 + (5,200 - 1) x 0.09 = 1 + 467.91 = 468.91
  effective n = 312,000 / 468.91 = 665.4
  ceiling = 60 / 0.09 = 666.7
  standard-error ratio = sqrt(468.91) = 21.65
```

The naive standard error is 21.65 times too small, so a naive 95% interval is about 4.6% of the width
it should be. Effective `n` of 665 is 0.21% of the 312,000 rows and within 0.2% of the ceiling of
667 — the design is saturated, and adding transactions inside these stores would buy nothing.

A common wrong answer omits the `− 1` and computes `1 + 5,200 × 0.09 = 469.0`, a negligible
difference here. It matters at small `m`: with `m = 2` the correct value is `1 + ICC` and the wrong
one is `1 + 2 × ICC`, a 100% error in the inflation factor.

### Exercise 4

Strong answers find at least these paths.

1. **Shared shoppers across stores.** A customer visits two branches in a month. High for fashion.
   Mitigation: assign by catchment, or screen loyalty records for cross-store visits.
2. **Shared staff.** Training went to treated managers, who talk to control managers at regional
   meetings. Medium. Mitigation: measure managerial proximity, or compare early and late periods.
3. **Shared stock.** If e-commerce fulfilment draws on store stock, a treated store's improved
   conversion changes online availability. Medium. Mitigation: assign the fulfilment pool at the same
   level, or exclude pool stock from the window.
4. **The national online campaign.** It runs across all stores, so it is an interaction rather than a
   leak: the effect may depend on campaign exposure. High for interpretability. Mitigation: measure
   exposure per store and report the effect with and without it.
5. **Signage visible from outside, and control managers changing layout unaided.** Both low.
   Mitigation: bound the first, measure compliance for the second.

The organisational path is (2); an answer where every path is physical is weak.

### Exercise 5

| Candidate | Contamination | Sample size | Feasibility | Decision proximity |
|---|---|---|---|---|
| DC | Low — configured per DC, and DCs have distinct delivery territories | 14 units; MDE will be very large | Yes, a configuration change | Same level as the configuration, one below the network decision |
| Lane | High — the rule pools orders across lanes, so a treated lane changes pooling for control lanes | 380 units; attractive | No — the rule cannot be applied per lane | Unrelated to the decision |
| Customer account | High — accounts are served by whichever DC covers them, so assignment collides with DC configuration | 9,000 units; very attractive | No, without a system change | Below the decision level |

The recommendation is DC, with the honest statement that 14 units is not enough for a cost-to-serve
effect of the size the business hopes to see, so the design should be a stepped rollout across DCs in
time — which converts it into something closer to a switchback and belongs to Topic 05. A weak answer
picks customer account because it has the most units.

### Exercise 6

```text
  mean 9.20, sd 14.00, var 196.00; ICC 0.12; m 7,400; k 48

  between-store variance = 0.12 x 196.00 = 23.52
  within-store variance  = 0.88 x 196.00 = 172.48

  DEFF = 1 + 7,399 x 0.12 = 888.88
  total rows = 710,400;  effective n = 710,400 / 888.88 = 799.2

  variance of a store mean = 23.52 + 172.48 / 7,400 = 23.543
  SE of the difference = sqrt(2 x 23.543 / 48) = 0.9904 GBP
  MDE = 2.80 x 0.9904 = 2.773 GBP = 30.1% of 9.20

  target 10% of 9.20 = 0.920 GBP
  required SE = 0.920 / 2.80 = 0.3286;  variance 0.10796
  per arm 0.05398;  k = 23.543 / 0.05398 = 436.1
  -> 437 per arm, 874 stores in total
```

For the commercial director: "At this ICC a 10% effect needs about 870 stores. We have 96, so this
pilot can only tell you whether the effect is larger than about 30%; anything smaller is
indistinguishable from noise."

### Exercise 7

The design has two randomisation stages. The planning system is applied to whole plants — four
plants, so two treated and two control if balanced — and is the whole-plot factor. The scheduling
heuristic is applied to lines within plants and is the sub-plot factor.

The planning-system effect is tested against plant-to-plant variation, which with four plants gives at
most 2 degrees of freedom. The heuristic effect is tested against the line-within-plant residual, which
has substantially more. Reporting both with a pooled error term and 180 residual degrees of freedom
uses the sub-plot error for the whole-plot factor, overstating the significance of the planning system
— a false positive on the expensive factor, which is exactly the one the business wants to be
significant. The corrected analysis is a mixed model with a random plant effect, or an analysis of
variance with plant as the whole-plot stratum and line within plant as the sub-plot stratum.

### Exercise 8 rubric

The assignment unit is the store, because the promise is fulfilled from store stock and the customer's
experience of it is determined by the store they order from. At least two leak paths are identified:
customers who order from one store and collect at another, and the shared inventory pool if online
orders are fulfilled from store stock. Power is computed at store level with a stated ICC source —
ideally from pre-period variation in on-time rate across stores — and the MDE is stated as a
percentage of the base on-time rate. The decision rule is pre-registered, for example "roll out if the
lower bound of the 95% interval exceeds the incremental fulfilment cost per order".

A weak note randomises orders rather than stores, or randomises stores and analyses orders, or states
a decision rule that depends on the result.

## 14. Interview questions

**1. "What is the difference between the unit of assignment and the unit of analysis?"**

Strong: assignment is where the coin lands; analysis is what the model treats as independent; when
they differ the standard error is understated by `sqrt(DEFF)`, and the fix is to aggregate to the
assignment level or model the clustering. Weak: says they should always be the same, which is a
reasonable default but not an answer.

**2. "You have a 40-store test and 200,000 transactions. How many observations do you have?"**

Strong: 40 independent observations, or a number near `k / ICC` in effective terms, and explains
that the rows carry the information of roughly the cluster count divided by the ICC. Weak: says
200,000.

**3. "Why does adding weeks to a store pilot often fail to improve precision?"**

Strong: once within-store variance divided by cluster size is small relative to between-store
variance, the cluster mean's variance is dominated by the between-store term, and duration shrinks
only the already-small term; cites the 0.05% reduction in the worked example. Weak: says more data is
always better.

**4. "When would you randomise at SKU level rather than store level?"**

Strong: only when the treatment cannot substitute across SKUs — an isolated category, a display that
shares no fixture, a price change on a product with no close substitutes — and names substitution as
the mechanism being tested for. Weak: says SKU level gives more units, which is true and is exactly
the trap.

**6. "What is a split-plot design and why does it matter?"**

Strong: two randomisation stages at two levels, a hard-to-change factor at whole-plot level and an
easy-to-change factor at sub-plot level, two error strata, and the whole-plot factor estimated with
far fewer degrees of freedom; names the inadvertent split-plot as the common error. Weak: describes
it as a factorial with two factors.

**7. "How would you detect contamination in a store test?"**

Strong: an exposure log, catchment overlap analysis, a check that control-group trends are flat
before and after launch, and a comparison of control behaviour across the launch date; notes that
contamination often biases toward zero and therefore looks like a null result. Weak: says "compare
control stores to treated stores", which is the test that is already contaminated.

**8. "You have 12 DCs. Can you run a controlled experiment?"**

Strong: yes, but not a parallel-arm one with adequate power for small effects; discusses stepped
rollout, switchback in time, paired designs, and the fact that 12 units supports only large-effect
detection; states the MDE. Weak: says no, or says yes and proposes a 6-versus-6 split without
computing power.

**9. "What is SUTVA and when does a supply chain violate it?"**

Strong: a unit's outcome depends only on its own assignment; violated by shared shoppers, shelf
space, carrier capacity, staff, and market equilibrium effects; notes that marketplaces violate it
structurally and that cluster randomisation is the usual remedy. Weak: defines it without an example.

**10. "A vendor says their pilot produced a 12% lift with p < 0.01 across 400 stores. What do you
ask?"**

Strong: what was randomised, how many units per arm, what the analysis treated as independent, what
the ICC was, whether the interval was computed at the assignment level, and whether the decision unit
matches. Weak: asks for the sample size, which is the one number that does not settle it.

**11. "How do you decide the unit when the decision unit and the feasible assignment unit differ?"**

Strong: states that the decision unit cannot be changed, prices the mismatch by computing what the
feasible unit can detect, and either renegotiates the decision rule or declares the pilot descriptive
rather than confirmatory. Weak: picks the feasible unit and reports the result as if it answered the
decision.

## 15. Key terms

- **Unit of assignment** — the object that receives the randomisation.
- **Unit of analysis** — the object the model treats as an independent observation.
- **Unit of decision** — the object the resulting decision applies to.
- **Unit of inference** — the population the claim generalises to.
- **Cluster** — a group assigned together, sharing a treatment.
- **Cluster randomised trial (CRT)** — a trial randomising whole clusters.
- **Intra-cluster correlation (ICC)** — the share of variance that is between clusters.
- **Design effect (DEFF)** — `1 + (m − 1)ICC`; variance inflation from clustering.
- **Effective sample size** — `n / DEFF`; the information-equivalent independent count.
- **Minimum detectable effect (MDE)** — the smallest effect a design detects at stated power.
- **SUTVA** — no interference: a unit's outcome depends only on its own assignment.
- **Interference** — one unit's outcome depending on another's treatment.
- **Spillover** — interference crossing an assigned unit's boundary.
- **Contamination** — control units receiving part of the treatment.
- **Substitution** — cross-unit competition within a cluster, as between SKUs on a shelf.
- **Nesting** — each lower unit belongs to exactly one higher unit.
- **Crossing** — every level of one factor co-occurs with every level of another.
- **Blocking** — grouping similar units before randomisation and randomising within groups.
- **Cluster-level analysis** — one summary per cluster, analysed as `k` observations.
- **Cluster-robust standard error** — a variance estimate allowing within-cluster correlation.
- **Degrees-of-freedom penalty** — precision lost from estimating between-cluster variance on few clusters.
- **Split-plot design** — two randomisation stages, producing two error strata.
- **Whole plot / sub-plot** — the coarser and finer experimental units in a split-plot design.
- **Switchback** — one unit alternating between arms over time blocks.
- **Lane** — an origin-destination pair plus mode; the natural unit for carrier tests.

## 16. Sources

**Grading convention.** A peer-reviewed or audited, B regulatory or official, C company case, investor claim, or unrefereed technical report, D vendor marketing, E press or anecdote. Preprints and conference proceedings are graded C regardless of quality, because the grade records the venue, not the merit. Where a claim rests on a single company experiment, the text says so.

**Peer-reviewed methodology**

- Cornfield, J. (1978). "Randomization by group: A formal analysis." *American Journal of Epidemiology*, 108, 100–102; cited as reported in Chakraborty and Lyons (2015). Grade A.
- Whiting-O'Keefe, Q. E., Henke, C., and Simborg, D. W. (1984). Work on the unit-of-analysis problem in medical care experiments, as cited in the Cochrane Handbook, chapter 23. Journal details were not confirmed from a primary source for this topic; consult the Cochrane chapter's reference list before citing. Grade A for the finding, unverified for the full citation.
- Donner, A., Birkett, N., and Buck, C. (1981). Sample size requirements for cluster randomisation; cited as reported in Chakraborty and Lyons (2015). Grade A.
- Donner, A., and Klar, N. (2004). "Pitfalls of and Controversies in Cluster Randomization Trials." *American Journal of Public Health*, 94(3), 416–422. Grade A.
- Hemming, K., and Taljaard, M. (2023). "Key considerations for designing, conducting and analysing a cluster randomized trial." *International Journal of Epidemiology*. doi:10.1093/ije/dyad064. Grade A.
- Chaplin, M., and Dwan, K. (2023). "Cluster-randomized controlled trials: A tutorial." *Cochrane Evidence Synthesis and Methods*. doi:10.1002/cesm.12024. Grade A.
- Chakraborty, H., and Lyons, G. (2015). "Cluster Randomized Trials: Considerations for Design and Analysis." *Journal of Statistical Theory and Practice*. doi:10.1080/15598608.2014.992081. Grade A.
- Campbell, M. K., et al. (2012). "CONSORT 2010 statement: extension to cluster randomised trials." *BMJ*. Grade A.
- Johari, R., Li, H., Liskovich, I., and Weintraub, G. Y. (2022). "Experimental Design in Two-Sided Platforms: An Analysis of Bias." *Management Science*. arXiv:2002.05670. Grade A.
- Holtz, D., Lobel, F., Lobel, R., Liskovich, I., and Aral, S. (2025). "Reducing Interference Bias in Online Marketplace Experiments Using Cluster Randomization: Evidence from a Pricing Meta-experiment on Airbnb." *Management Science*, 71(1), 390–406. doi:10.1287/mnsc.2020.01157. Grade A.
- Glanz, K., et al. "The Healthy Food Marketing Strategies Study: design, baseline characteristics, and supermarket compliance." PMC7796717. Grade A.
- Ayala, G. X., et al. (2013). "Efficacy of a store-based environmental change intervention compared with a delayed treatment control condition on store customers' intake of fruits and vegetables." *Public Health Nutrition*. doi:10.1017/s1368980013000955. Grade A.

**Books and reference works**

- Kohavi, R., Tang, D., and Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press, chapter 14, "Choosing a Randomization Unit", pp. 166–170. Grade A.
- Donner, A., and Klar, N. (2000). *Design and Analysis of Cluster Randomization Trials in Health Research*. Arnold. Grade A.

**Official and standards**

- Cochrane Handbook for Systematic Reviews of Interventions, chapter 23, "Including variants on randomized trials": design effect, unit-of-analysis error, and the two correction approaches. Grade B.
- ICH E9, *Statistical Principles for Clinical Trials*, and the E9(R1) addendum on estimands. Grade B.
- GS1 General Specifications: GLN, GTIN, SSCC, and EPCIS 2.0. Grade B.
- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 8601-1:2019, *Date and time — Representations for information interchange*. Grade B.
- Regulation (EU) 2016/679 (General Data Protection Regulation). Grade B.

**Company, technical, and unrefereed**

- Vaver, J., and Koehler, J. (2011). "Measuring Ad Effectiveness Using Geo Experiments." Google technical report. Grade C. Constrained randomisation and the ~10% interval-width reduction.
- Wayfair Engineering (2023). "How Wayfair Uses Geo Experiments to Measure Incrementality." Company blog. Grade C.
- Deng, A., Longbotham, R., Walker, T., and Xu, Y. (2011). "Choice of the Randomization Unit in Online Controlled Experiment." JSM proceedings. Grade C.
- Kohavi, R., et al. (2009). "Seven Pitfalls to Avoid when Running Controlled Experiments on the Web." KDD. Grade C.
- Microsoft Research, "Patterns of Trustworthy Experimentation: Pre-Experiment Stage." Grade C.
- Roemheld, L., and Rao, J. (2024). "Interference Produces False-Positive Pricing Experiments." arXiv:2402.14538. Preprint. Grade C. The "up to about 100%" figure is the authors' reported estimate and is not independently verified.
- Bray, R., Sanders, N., and Stamatopoulos, I. (2024). "Large Midwestern Grocery Chain Price Randomization, First and Second Waves." AEA RCT Registry AEARCTR-0014114. Grade B for the registered design; no results are claimed here.

**Note on citations.** Findings are reported as the cited works state them. The worked example in section 6 is constructed for teaching, and its arithmetic was checked against the code block in section 7. Where a bibliographic detail could not be confirmed from a primary source, the entry says so, and no citation is invented to fill the gap.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, which fixes the estimand this topic attaches a unit to: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- Next topic, which turns the design effect into full cluster-design arithmetic: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Market-level units when store-level assignment is impossible: [`04_geo-experiments.md`](04_geo-experiments.md)
- Time as the unit when even a store cannot be split: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Turning the variance structure into a detectable effect: [`06_power-mde.md`](06_power-mde.md)
- Interference treated formally: [`07_interference-spillover.md`](07_interference-spillover.md)
- The unit hierarchy in measurement terms:
  [`../../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
- Hierarchy design for products and locations:
  [`../../01A_foundations_units_time_assumptions/07_hierarchy-design.md`](../01A_foundations_units_time_assumptions/07_hierarchy-design.md)
- The clustering problem in interval estimation, which this topic shares:
  [`../../04_statistics_causal_inference/02_probability-bootstrap.md`](../04_statistics_causal_inference/02_probability-bootstrap.md)
- Quasi-experimental designs when randomisation at the right unit is impossible:
  [`../../04_statistics_causal_inference/07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Module survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Promotion and cannibalisation measurement: [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Return to the part map: [`../../_index.md`](../_index.md)
