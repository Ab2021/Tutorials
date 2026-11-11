---
title: "Store KPI Tree"
topic: "Store KPI Tree"
collection: "retail-01"
file_id: "01"
level: 2
evidence: "DERIVED"
created: "2026-11-09"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["ratio arithmetic", "event time vs processing time", "proportion sampling error"]
pattern_tags: ["retail-operations", "metric-hierarchy", "variance-decomposition"]
source_status: "OPEN"
owner_role: "retail analytics practitioner"
source_index: "INDEX.md"
review_trigger: "Change to traffic counter, POS basket rules, shelf-audit route, or labour booking"
---

# Store KPI Tree

> **Purpose:** Give one store-week a defensible metric hierarchy, so that a sales move can be split into traffic, conversion, and basket components without double counting, and so that each node has a named owner and an action.
> **Prerequisites:** You can compute a ratio, you can separate event time from processing time, and you know that a proportion estimated from a sample carries error.
> **Key outcome:** Given two consecutive store-weeks you can state which node moved, by how much in dollars, which attribution rule you used, and what the movement does *not* license you to claim.
> **This file does NOT cover:** Company-specific POS, counter, or labour systems; legal or accounting treatment of shrink; any claim that a described result was measured in a live retailer.
> **Level:** 2 — core methods.
> **Evidence:** Every formula is algebraically derived here (DERIVED). All store numbers are ILLUSTRATIVE constructions, arithmetically checked but not observed anywhere.
> **Retrieval cue:** If two people disagree about whether conversion "really" rose, start at Section 4.2 (weighting) and Section 8 (denominator drift).

## 1. File Header

A store KPI tree is an ordered set of ratios, each with a fixed grain, a fixed denominator rule, and one owner, arranged so that the top node decomposes into the nodes below it by an identity rather than by correlation. The tree is a measurement instrument, not a causal model: it tells you *where* a number moved, never *why* until you add an experiment or a mechanism.

This sheet covers four node families that a physical store actually acts on. Demand capture: footfall and conversion. Basket value: units per transaction and average selling price. Service and stock: on-shelf availability and its gap rate. Cost and loss: labour hours, sales per labour hour, earned hours, and shrink. Contribution sits at the top because it is the only node that combines revenue and cost in one unit (currency per store-week).

Three things this sheet deliberately refuses to do. It does not treat the tree as a dashboard layout — dashboards show nodes side by side, a tree states which node is a product of which. It does not give a single "correct" attribution of a sales change, because sequential, Shapley, and log-share attributions each answer a different question and give different dollars. And it does not label any node "predictive"; a tree of ratios is descriptive arithmetic and inherits every weakness of its inputs.

The unit of analysis throughout is the **store-week**: one store, seven consecutive trading days, cut at local midnight on the closing day, with a stated lag before the numbers are final. Nothing in the tree is meaningful at a finer grain unless the denominator grain moves with it — a store-day conversion rate computed against a weekly counter reset is nonsense, and a store-hour conversion rate needs an hourly counter feed that most estates do not have.

The tree has five artefacts, in order of creation. (1) A **node contract**: grain, formula, source system, event-time stamp, refresh lag, owner, action. (2) An **identity**: the algebraic statement that makes the node a product of its children. (3) An **attribution rule**: how a change at the top is split among children, chosen before the data arrives. (4) A **guardrail**: the value of a sibling node that must hold for the attribution to be interpreted. (5) An **action**: the decision the owner is allowed to take when the node breaches its band.

| Node | Formula | Grain | Primary source | Refresh | Owner | Action it authorises |
|---|---|---|---|---|---|---|
| Store contribution | Net sales × gross margin − labour − shrink − occupancy | store-week | finance close | weekly, 5-day lag | store manager | escalate to district review |
| Net sales | Gross sales − returns − markdown given | store-week | POS | daily, 1-day lag | commercial lead | price or promotion change |
| Footfall | count of entry events after de-duplication | store-day | door counter | hourly, 15-min lag | facilities | door or layout change |
| Conversion | transactions ÷ footfall | store-week | POS + counter | daily | store manager | staffing, service, queue fix |
| Average transaction value | net sales ÷ transactions | store-week | POS | daily | commercial lead | bundle, upsell, price ladder |
| Units per transaction | units ÷ transactions | store-week | POS line items | daily | category lead | cross-merchandising, add-on placement |
| Average selling price | net sales ÷ units | store-week | POS line items | daily | pricing analyst | markdown depth, mix correction |
| On-shelf availability | filled positions ÷ audited positions | store-route | shelf audit | weekly sample | replenishment lead | gap-fill task, backroom sweep |
| Labour hours | clocked hours net of unpaid breaks | store-week | time and attendance | weekly | store manager | schedule shape, task reallocation |
| Shrink rate | (book value − counted value) ÷ sales base | store-period | stock count | 4-weekly | loss-prevention lead | count cadence, security spend |

The table is the contract. If a store cannot name the owner of a node, that node cannot appear in a performance conversation: an unowned ratio is a ratio nobody is allowed to act on, and acting on it anyway produces the gaming described in Section 8.

Two grain conventions keep the tree consistent across an estate. First, week numbering is the retailer's fiscal week, not ISO week, and the cut is at local midnight so that a Saturday-night trade is not split across two weeks by a timezone conversion. Second, the counter and the POS are both stamped with *event* time; the load timestamp in the warehouse is *processing* time and is never used for the cut. A late-arriving correction therefore restates history rather than being pinned to the week it arrived in.

Version 2.0 replaces an earlier draft that repeated a fixed answer template across its question bank. The substance added here is the identity and attribution algebra of Section 4, the double-counting boundary in Section 4.4, and the discriminating checks in Section 8, each of which resolves a specific disagreement that arises when a store's sales move.

Reading path by role. A store manager reads Section 1, Section 7, and the card in Section 12. An analyst building the numbers reads Section 4 in full, then Section 5 to check the arithmetic against a worked case. A loss-prevention or replenishment lead reads Section 8 first, because the failure modes for shelf and stock metrics are different from the failure modes for demand metrics, and Section 8 marks which is which.

## 2. Concept Map / Mental Model

```text
                       [store contribution]  currency / store-week
                                 |
             +-------------------+-------------------+
             |                   |                   |
      [net sales]          [labour cost]        [shrink $]
             |                   |                   |
   +---------+---------+     [hours x rate]     [book - counted]
   |         |         |          |                   |
[footfall] [conv.] [ATV]     [task hours]      [count coverage]
   |         |         |          |                   |
 [door     [POS txn  [UPT]     [schedule]        [receiving]
  counter]  /footfall]  |          |             [transfer]
             |        [ASP]        |
             |          |          |
             +----------+----------+
                        |
             [on-shelf availability]  <- cross edge 1
                        |
             [checkout service level] <- cross edge 2
```

Read the diagram as a product structure on the revenue side and an additive structure on the cost side. Net sales is the product of footfall, conversion, and average transaction value; average transaction value is in turn the product of units per transaction and average selling price. Labour cost and shrink are subtracted, so the top node is a mixed expression: multiplicative above the revenue line, additive below it.

The diagram is drawn as a tree but the real object is a directed acyclic graph, and the two cross edges are where most analytical mistakes happen. **Cross edge 1**: on-shelf availability feeds *both* conversion and units per transaction. An empty facing causes a walkout (conversion) or a substitution to a cheaper item (average selling price) or an extra trip to another aisle (units up, price down). Because availability influences two children of the same parent, a naive split of a conversion change will over-attribute to "service" what was actually a stock-out. **Cross edge 2**: checkout service level feeds conversion, because a customer who abandons a basket is not a transaction but was counted as footfall. Labour hours feed both availability (replenishment is a task performed by hours) and checkout service (lanes are staffed by hours), which is why cutting hours shows up as a sales effect before it shows up as a cost saving.

Three uses justify the tree's existence, and they need different subsets of it. **Variance explanation**: a sales move is decomposed into contributions whose sum equals the move, using Section 4.2 or 4.3. **Ownership**: each node has exactly one accountable role, so a breach produces one conversation rather than a meeting. **Instrumentation design**: because each node names a source system, the tree tells you what to instrument next — if units per transaction cannot be computed, the gap is POS line-item capture, not analytics capability.

The tree is not a causal graph. Two children of net sales are multiplied, so their contributions are not independent: a traffic gain and a conversion gain interact, and the interaction has to be assigned by an explicit rule or it silently lands wherever the analyst's loop happens to put it. This is the whole content of Section 4.3, and it is the reason two analysts can both be arithmetically correct and disagree by twenty dollars on a small move.

One number per node, one direction per node. Every ratio in the tree has a defined "good" direction, and that direction is fixed in the node contract, not chosen at review time. Availability is good when high; gap rate is good when low; shrink rate is good when low; units per transaction is good when high *only if* margin holds, which is why average selling price is a sibling guardrail rather than an optional extra. A node whose good direction flips with context cannot carry an owner, because the owner cannot be measured against it.

A signal trace shows how a physical event becomes a currency number. Suppose the weekly shelf audit finds 16 empty facings on a 420-position route, a gap rate of 3.81%. With an illustrative lost-sales coefficient of 0.40 — meaning four in ten gap-exposed shopping occasions end without the item — the implied sales loss is 1.524% of net sales, about \$384.64 on a \$25,242 week. That number enters the tree twice: once as a downward pressure on conversion (walkouts) and once as a downward pressure on units per transaction (basket truncation). It must be counted once at the sales line, and Section 4.4 gives the boundary rule for doing so.

## 3. Foundational Definitions

Each definition below is written as: formal statement, unit and valid range, the system that can actually produce it, and the trap that makes the naive version wrong. A definition without a source system is a wish; a definition without a trap is a definition nobody has used.

**Store-week (the grain).** One store identifier crossed with one fiscal week of seven trading days, cut at local midnight. Unit: a row. Valid range: the store must be open for all seven days or the row must carry an `open_days` field, otherwise its weekly totals are not comparable to a full week. Trap: a store closed for two days has lower sales, lower footfall, and a *higher* sales-per-labour-hour if hours were also cut, so all four nodes move together and the tree looks healthy while the store lost trade.

**Footfall (E).** The count of entry events at the customer door after de-duplication, excluding staff and contractors who enter through the same door. Unit: visits per store-day. Valid range: non-negative integer. Trap: a single person re-entering after a car trip is two visits, and a group of three entering together is one door event on a beam counter — the counter measures *door crossings*, not people, so footfall is a proxy whose bias is stable but not zero.

**De-duplication window.** The minimum interval, typically 3 to 10 seconds, inside which repeated beam interruptions are collapsed into one entry. Unit: seconds. Trap: widening the window to suppress noise also suppresses genuine fast re-entries, and the window is usually set once at installation and never re-validated, so it drifts as door hardware changes.

**Transaction (X).** A completed POS basket that took money, counted once. Unit: transactions per store-week. Valid range: non-negative integer. Trap: a basket split across two tenders is one transaction; a basket voided and re-rung is two transactions unless voids are netted, so transaction count is sensitive to cashier behaviour and to till software version.

**Units (U).** The sum of line-item quantities on completed baskets, net of returns processed on the same basket. Unit: units per store-week. Valid range: non-negative. Trap: weighed goods and services lines inflate or deflate units depending on how the scale item is configured, which is why units per transaction is comparable across stores only within a category set.

**Net sales (S).** Gross sales at till, minus returns, minus markdown given at the till, excluding tax and excluding gift-card load. Unit: currency per store-week. Trap: the three plausible denominators for shrink and for margin are gross sales, net sales, and net sales plus markdown. They differ by several percent, so a shrink rate quoted without its base cannot be compared to another store's.

**Conversion (c).** Transactions divided by footfall, both over the same seven days and the same trading hours. Unit: dimensionless fraction, typically 0.05 to 0.45 for general merchandise and higher for grocery top-up formats. Valid range: (0, 1] in principle; above 1 means the counter is undercounting or the transaction is being counted at a different door. Trap: conversion is a ratio of two aggregates, so it is *not* the average of daily conversion rates unless daily footfall is constant — Section 4.2 makes this precise.

**Average transaction value (ATV).** Net sales divided by transactions. Unit: currency per transaction. Valid range: positive. Trap: ATV mixes price and mix, so a promotional week can raise units per transaction and lower ATV at the same time; reporting ATV alone hides which happened.

**Units per transaction (UPT).** Units divided by transactions. Unit: items per transaction. Valid range: positive, typically 1.2 to 5.0. Trap: a "buy one get one" mechanic mechanically doubles UPT and halves average selling price while leaving ATV unchanged, so a UPT target without an ASP guardrail is trivially gameable.

**Average selling price (ASP).** Net sales divided by units. Unit: currency per unit. Valid range: positive. Trap: ASP falls when a cheap item is added to a basket even if every price is unchanged — it is a mix statistic first and a price statistic second.

**On-shelf availability (OSA).** Filled shelf positions divided by audited shelf positions, where a position is a planogram slot on the audit route. Unit: dimensionless fraction. Valid range: [0, 1]. Trap: availability measured only on the audit route measures the route, not the store; a route chosen for convenient walking over-samples front-of-store gondolas and under-samples the back aisles where gaps persist.

**Gap rate (g).** One minus availability, reported per audit rather than per week when audits are less frequent than weekly. Unit: dimensionless fraction. Trap: mixing a weekly availability average computed from three audits with a weekly sales total double-weights whichever days were audited, and audit days are rarely representative.

**Phantom inventory.** Stock that the system believes is on the shelf but is not — mis-scanned deliveries, unrecorded damages, or theft not yet counted. Unit: units or currency. Trap: phantom stock makes availability look fine in the system while the shelf is empty, so an availability metric built from the inventory system rather than a physical audit measures the system's optimism.

**Lost-sales coefficient (κ).** The share of gap-exposed shopping occasions that end without a substitute purchase. Unit: dimensionless, range [0, 1]. It is a *behavioural* parameter, not an arithmetic one, and it must be estimated from an experiment or stated as an assumption. Trap: κ is often assumed at 0.3 to 0.5 by convention; using a convention as if it were a measurement turns an availability target into a number with no error bar.

**Labour hours (H).** Clocked hours net of unpaid breaks, including hours booked to the store for tasks performed for other channels. Unit: hours per store-week. Trap: if e-commerce picking hours are booked to the store but the picked sales are booked to the online channel, sales per labour hour is understated for the store and the error grows with online penetration.

**Sales per labour hour (SPLH).** Net sales divided by labour hours. Unit: currency per hour. Trap: SPLH rises when a store cuts hours during a quiet week and falls when it staffs up for a promotion; it is a productivity ratio, not an efficiency verdict, and it can improve while service degrades.

**Earned hours.** A labour standard: net sales divided by a task rate expressed as currency per hour. Unit: hours. Trap: earned hours inherit every error in the task rate, and the task rate is usually an engineered standard set years ago, so the variance against actual hours mixes schedule error with standard error.

**Shrink.** Book value of stock minus counted value at a stock count, split into known causes (damage, markdown not recorded, supplier short-delivery) and unknown causes (theft, admin error). Unit: currency per count period. Trap: known and unknown shrink have different owners, so a single shrink number with a single owner guarantees that neither cause gets fixed.

**Contribution.** Net sales times gross margin rate, minus labour cost, minus shrink, minus occupancy. Unit: currency per store-week. Trap: occupancy is fixed within a lease period, so including it in a weekly node makes small sales moves look catastrophic; many estates report contribution *before* occupancy and state that explicitly.

**Mix weight.** The share of a parent total contributed by a subgroup, for example the share of footfall coming from one store format. Unit: dimensionless, sums to one. Trap: mix weights are the mechanism behind the pooled-rate paradox in Section 8, and they are almost never reported next to the pooled rate they distort.

**Guardrail node.** A sibling node whose value must be checked before a movement in the target node is interpreted. Unit: whatever the sibling measures. Trap: a guardrail that is only checked after a bad quarter is a post-mortem, not a guardrail; it has to be evaluated in the same refresh as the node it guards.
