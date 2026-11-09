---
collection: "retail-ecommerce-media-and-personalisation"
file_id: "01"
topic: "Commerce metric tree and order reconciliation"
level: 1
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Order-state semantics, accounting policy, source availability, or decision grain changes"
notation_style: "plain-text"
prerequisites:
  - "Arithmetic with counts, currency, ratios, and units"
  - "Basic event-time and database-join concepts"
pattern_tags:
  - "commerce-metrics"
  - "order-lifecycle"
  - "reconciliation"
  - "decision-boundaries"
source_status: "OPEN for the unavailable original source and accounting interpretation; examples are ILLUSTRATIVE"
owner_role: "commerce analytics and decision-systems curriculum"
source_index: "./INDEX.md"
---

# Commerce Metric Tree and Order Reconciliation

## 1. File Header

> **Purpose:** Build a commerce metric tree whose order states, denominators, time windows, currencies, and contribution components reconcile before a funnel result informs a product or operations decision.
> **Prerequisites:** Compute ratios with nonzero denominators; distinguish an event from an entity; read a simple order ledger.
> **Key outcome:** Given order, visit, availability, refund, and cost records, derive mutually consistent funnel and unit-economics measures and reject mixed-grain comparisons.
> **This file does NOT cover:** Recommender model training, price optimization, statutory revenue recognition, customer-level causal identification, or autonomous order execution.
> **Level:** 1 Foundations, with production reconciliation practice.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** Arithmetic is DERIVED and smoke-checked; every workload and receipt is ILLUSTRATIVE; external accounting and unavailable original-source claims remain OPEN.
> **Retrieval cue:** If revenue rises while contribution falls, start at Section 4.3; if dashboards disagree, start at Sections 4.4 and 8.

This sheet treats a metric as a contract, not as a dashboard label.

The contract names the population, event grain, observation window, inclusion rule, numerator, denominator, currency, and update time.

A metric without those fields can be arithmetically correct and still answer the wrong business question.

The running example is a small retail order workflow with visits, paid orders, cancellation, shipment, refunds, merchandise costs, and fulfillment costs.

The same reasoning applies to media only after its events and economics are redefined; a play is not a paid retail order.

No value in this file is an observed company result.

No threshold is a universal operating target.

No formula establishes that a recommendation, promotion, or agent caused an outcome.

The original source file named in the collection's focused plan was not supplied.

Its title establishes a planned commerce-agent topic, not a source-backed architecture or a performance claim.

The related availability exercise is retained here: among ten recommended stock keeping units, three unavailable units leave seven eligible candidates.

That count constrains ranking; it does not measure demand, conversion, or customer satisfaction.

A local accounting policy must determine formal revenue recognition.

This sheet instead defines analytical merchandise and contribution measures explicitly so their components can be audited.

### 1.1 Decision contract

Start with the decision owner and the action under review.

A merchandising owner may change eligible assortment.

A fulfillment owner may investigate cancellations.

A product owner may change checkout friction.

A finance owner may reconcile contribution components.

Those decisions require different populations and time boundaries.

Use this minimum metric specification before computing anything:

| Field | Required content | Example in this sheet |
|---|---|---|
| Decision | Action the analysis may inform | Inspect availability-filter behavior |
| Owner | Role able to act | Merchandising operations |
| Entity | Stable unit of analysis | Order, visit, customer, SKU |
| Grain | One row or event per what | One state transition per order |
| Population | Included and excluded entities | Authorized paid orders in the stated window |
| Window | Event-time or cohort-time interval | Paid-at time in one UTC day |
| Numerator | Quantity and units | Completed orders [orders] |
| Denominator | Quantity and units | Paid orders [orders] |
| Comparator | Prior period, baseline, or experiment | Defined separately, never implied |
| Guardrails | Conditions that block action | Unreconciled refunds or stale inventory |
| Uncertainty | Sampling, late data, or model uncertainty | Not measured in the constructed example |
| Evidence | Observed, derived, illustrative, or open | ILLUSTRATIVE receipts |

A conversion metric cannot repair an ambiguous entity.

A contribution metric cannot repair an omitted cost.

A causal claim cannot be rescued by adding more funnel stages.

The metric tree is an accounting and diagnostic structure before it is an optimization target.

### 1.2 Evidence and maintenance boundary

`DERIVED` identifies consequences of definitions stated here.

`VERIFIED` is used only for arithmetic and state traces actually exercised by the smoke check recorded in Section 5.

`ILLUSTRATIVE` identifies constructed receipts, sessions, prices, and costs.

`SOURCED` is reserved for a proposition linked to a primary document that was actually consulted.

`OPEN` identifies missing source content, unsettled accounting treatment, or unmeasured deployment behavior.

The IFRS 15 landing page identifies the standard's title but did not expose its substantive recognition text in this authoring session.

Therefore this file does not claim an IFRS interpretation.

RFC 3339 supplies an Internet timestamp representation, not a complete order-event timezone policy or clock-synchronization guarantee.

W3C PROV-O supplies provenance concepts, not a commerce ledger implementation.

These source boundaries are deliberate and should survive later revisions.

**Takeaway:** A useful commerce metric is a reproducible decision contract with explicit evidence limits, not a bare ratio.

## 2. Concept Map / Mental Model

The central mental model is a stateful order ledger joined to separately defined demand and cost streams.

The arrows below are dataflow, not proof of causation.

```text
Eligible visit events                 Catalog and stock snapshots
          |                                    |
          v                                    v
[Visit eligibility contract]       [Candidate eligibility filter]
          |                                    |
          |                                    v
          |                         Eligible SKU set for ranking
          |                                    |
          v                                    v
Paid-order events ------------------> [Order-state ledger]
                                             |
                    +------------------------+----------------------------+
                    |                        |                            |
                    v                        v                            v
             Cancellation facts      Shipment/completion facts     Refund/return facts
                    |                        |                            |
                    +------------------------+-----------------------------+
                                             |
                                             v
                              [Time-safe order snapshot]
                                             |
                    +------------------------+----------------------------+
                    |                        |                            |
                    v                        v                            v
             Funnel measures         Net merchandise bridge       Contribution bridge
                    |                        |                            |
                    +------------------------+-----------------------------+
                                             |
                                             v
                              Reconciliation and diagnostics
                                             |
                                             v
                       Bounded human or policy-owned decision
```

Three separations keep the tree coherent.

First, exposure and visit denominators describe opportunities, not orders.

Second, order states describe lifecycle progress, not customer intent.

Third, contribution describes defined economic components, not statutory profit.

### 2.1 The central invariant

For any disjoint, exhaustive order-state partition at a fixed snapshot time, counts across states sum to the number of distinct orders in the population.

For any monetary bridge, the opening amount plus signed adjustments must equal the closing amount within an explicitly declared rounding tolerance.

Duplicate event identifiers must not create duplicate orders.

A late event may revise a snapshot, but it must be visible as a revision rather than silently rewriting history.

An unavailable SKU may remain in a raw candidate list but must not enter the eligible ranked list under the stated availability rule.

### 2.2 What the tree does not mean

A path from visit to payment does not show that a page element caused payment.

A path from refund to lower contribution does not identify why the refund happened.

A correlation between stock availability and conversion does not separate assortment, traffic, price, or fulfillment effects.

The diagram supports reconciliation and diagnosis.

Causal estimation requires a separate assignment mechanism and estimand.

### 2.3 Operating loop

Define the metric before extraction.

Freeze the decision-time snapshot.

Validate entity and currency keys.

Apply lifecycle state transitions.

Aggregate each measure at its own grain.

Reconcile counts and currency bridges.

Only then compare cohorts or propose an action.

After action, retain the policy and input versions so later results can be interpreted.

**Takeaway:** Keep opportunities, order states, and money bridges separate until their explicitly keyed aggregation points.

## 3. Foundational Definitions

The definitions below are local analytical contracts.

They are not substitutes for legal, tax, accounting, or customer-policy definitions.

### 3.1 Core notation

| Term/symbol | Formal meaning | Units / shape | Valid domain | Common confusion | INDEX glossary entry |
|---|---|---|---|---|---|
| `V` | Eligible visits in window `W` | visits; nonnegative integer | One eligible visit per defined visit identifier | Not unique customers or page views | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `N_paid` | Distinct orders reaching paid state in `W` | orders; nonnegative integer | Paid-at event determines window membership | Not completed orders | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `N_done` | Paid cohort orders completed by snapshot | orders; nonnegative integer | `0 <= N_done <= N_paid` | Not all orders shipped in the calendar window | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `CVR_paid` | Paid-order conversion at visit grain | dimensionless | `N_paid / V`, with `V>0` | Not customer conversion | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `G_item` | Item amounts for included orders before cancellation/refund adjustments | currency; scalar | Declared currency and tax treatment | Not cash or accounting revenue | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `R_cancel` | Item amount removed by cancellation | currency; nonnegative scalar | Same included-order population as `G_item` | Not a refund after completion | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `R_refund` | Item refunds against included orders | currency; nonnegative scalar | Refund window explicitly stated | Not total return shipping expense | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `M_net` | Analytical net merchandise amount | currency; scalar | `G_item-R_cancel-R_refund` under local bridge | Not recognized revenue | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `AOV_net` | Net merchandise per completed included order | currency/order | `N_done>0` | Not basket size at payment | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `CM` | Defined contribution after listed variable components | currency; scalar | Local bridge only; omitted costs listed | Not operating profit | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `K_raw` | Raw recommended SKU count | SKUs; nonnegative integer | Before eligibility filtering | Not impressions | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `K_unavail` | Raw candidates unavailable at decision snapshot | SKUs; integer | `0 <= K_unavail <= K_raw` | Not stockout demand | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `K_eligible` | Candidates remaining after eligibility rules | SKUs; nonnegative integer | `K_raw-K_unavail` when no other filter applies | Not number purchased | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `tau` | Snapshot or decision timestamp | instant | Fully qualified timezone or UTC | Not event date alone | [Glossary](./INDEX.md#7-master-symbol-glossary) |
| `epsilon` | Allowed reconciliation residual | currency; nonnegative scalar | Rounding policy declared | Not permission to ignore material mismatch | [Glossary](./INDEX.md#7-master-symbol-glossary) |

### 3.2 Visit

A visit is an identifier-delimited interaction episode under a declared sessionization rule.

The rule must specify inactivity timeout, identity changes, bot exclusion, consent restrictions, and cross-device handling.

`V` counts eligible visits, not cookies, people, or page views.

A nonexample is summing daily unique visitors and calling the sum unique visitors for the month.

Operationally, test whether the same visit identifier can legitimately appear in multiple windows.

If it can, document the split rule before using visit conversion.

### 3.3 Order and order state

An order is a stable commercial entity identified by an order key.

An order state is a fact about that entity at a snapshot or after an event.

A useful state set separates created, paid, cancelled, shipped, completed, and returned-or-refunded facts without assuming that every jurisdiction or merchant uses the same names.

Some states can coexist as dimensions rather than a single linear enum.

For example, a shipped order can later have a partial refund.

A nonexample is overwriting the paid timestamp with a refund timestamp.

Operationally, retain immutable events and derive a versioned state snapshot.

### 3.4 Item amount and analytical net merchandise

`G_item` is the sum of included order-line item amounts under a stated price and tax convention.

This sheet excludes separately identified taxes from its illustrative item amounts.

It treats shipping revenue and shipping cost as separate contribution components.

`R_cancel` removes item amounts for cancelled lines.

`R_refund` removes refunded item amounts, including partial refunds when present.

`M_net` is the result of this analytical bridge, not an assertion about accounting revenue.

A nonexample is subtracting product cost from `G_item` and calling the result net merchandise.

### 3.5 Contribution

`CM` is a deliberately named analytical measure: net merchandise plus included shipping income minus included merchandise, fulfillment, shipping, and other listed variable costs.

It is not gross margin unless the local definition matches.

It is not profit because fixed labor, marketing, overhead, and other costs may be absent.

The exact component list is part of the metric name and version.

A nonexample is comparing a contribution measure that includes returns with one that excludes returns.

Operationally, publish a machine-readable component registry with signs and currency.

### 3.6 Cohort window and calendar window

A paid cohort groups orders by paid-at time.

Its completion and refund outcomes may arrive after the cohort window closes.

A calendar-window shipment measure groups shipment events by shipment time.

These measures answer different questions and need not have matching denominators.

A nonexample is dividing shipments in May by orders paid in May without declaring a cohort maturity rule.

Operationally, store both event timestamps and the snapshot time used to derive the metric.

### 3.7 Eligible candidate

An eligible candidate satisfies the decision-time constraints required for the next action.

Availability is one such constraint.

Customer region, rights, policy, and authorization may impose additional filters.

`K_eligible=K_raw-K_unavail` only when no other exclusions occur.

A nonexample is recommending an unavailable item and counting a later substitution as conversion of the original item.

Operationally, retain raw and excluded candidate identifiers with reason codes.

### 3.8 Reconciliation residual

A residual is the difference between two quantities that should agree under the declared mapping.

A zero residual supports arithmetic consistency, not semantic correctness.

Both sides could share the same wrong join or omitted population.

A nonzero residual localizes a mismatch but does not identify its root cause by itself.

A nonexample is forcing the residual to zero with an unexplained plug.

Operationally, report residual magnitude, sign, affected keys, and late-data status.

**Takeaway:** Precise grains, state timestamps, money signs, and candidate eligibility prevent most metric-tree ambiguity.

## 4. Core Formulas with Full Derivation

Every formula below is a definition or an algebraic consequence of one.

Currency examples use a generic `CU` so no national tax or accounting treatment is implied.

All numerical inputs are ILLUSTRATIVE.

Rounding is to two decimal places only when explicitly reported; internal calculations retain exact values where possible.

### 4.1 Simplest / Base Case

#### CM-01: Paid-visit conversion

Validity before use: every eligible visit has a defined opportunity to produce an attributed paid order under the declared attribution rule.

A paid order may map to at most one visit in this base case.

The denominator must be nonzero.

Derivation: paid conversion is the fraction of eligible visit opportunities whose attributed outcome is a paid order.

Counting those visits is equivalent to counting paid orders only under the one-order-per-visit base assumption.

```text
FORMULA [CM-01]: CVR_paid = N_paid / V
WHERE: CVR_paid is paid-order conversion [dimensionless];
       N_paid is distinct paid orders attributed to eligible visits [orders];
       V is eligible visits [visits];
       both use the same population and paid-attribution window.
CONSTRAINT: V>0; one attributed paid order per eligible visit;
            visit eligibility and attribution are fixed before computation.
NUMERICAL EXAMPLE: N_paid=4 orders and V=100 visits;
                   CVR_paid=4/100=0.04=4%.
INTERPRETATION: four percent of eligible visits have an attributed paid order;
                this says nothing about completion, refunds, or causation.
```

Dimensional check: orders and visits are entity counts, so the reported ratio is a rate per visit rather than a physical dimensionless probability unless the sampling model is stated.

Boundary: if `V=0`, conversion is undefined rather than zero.

Counterexample: a visit with two paid orders violates the base one-order attribution assumption.

Use an order-per-visit measure or a many-outcome attribution model instead.

#### CM-02: Candidate eligibility count

Validity before use: raw candidates are distinct and unavailable candidates are a subset removed before ranking.

No additional eligibility filter is active in the base case.

Derivation: set subtraction gives eligible candidates as raw candidates minus unavailable raw candidates; cardinalities therefore subtract.

```text
FORMULA [CM-02]: K_eligible = K_raw - K_unavail
WHERE: K_raw is raw distinct recommended SKUs [SKUs];
       K_unavail is distinct unavailable raw candidates [SKUs];
       K_eligible is candidates permitted for ranking [SKUs].
CONSTRAINT: unavailable candidates belong to the raw set;
            availability is evaluated at one declared snapshot;
            no rights, region, or policy exclusions beyond availability.
NUMERICAL EXAMPLE: K_raw=10 SKUs and K_unavail=3 SKUs;
                   K_eligible=10-3=7 SKUs.
INTERPRETATION: ranking has seven eligible candidates, not ten;
                availability changes the action set, not the relevance scores by itself.
```

Boundary: if all ten candidates are unavailable, the eligible set is empty and the policy needs a no-result or fallback response.

Repeated raw SKU identifiers require deduplication before this arithmetic.

#### CM-03: Lifecycle completion ratio for a paid cohort

Validity before use: the denominator is a fixed paid cohort and the numerator is those same orders completed by a stated snapshot.

Derivation: completion proportion is completed cohort members divided by all cohort members.

```text
FORMULA [CM-03]: F_done = N_done / N_paid
WHERE: F_done is cohort completion proportion [dimensionless];
       N_done is completed orders from the paid cohort [orders];
       N_paid is the fixed paid cohort [orders].
CONSTRAINT: N_paid>0; 0<=N_done<=N_paid; snapshot and maturity are stated.
NUMERICAL EXAMPLE: N_done=3 and N_paid=4;
                   F_done=3/4=0.75.
INTERPRETATION: three quarters of the paid cohort have completed by the snapshot;
                the remaining order need not be permanently lost.
```

Boundary: an immature cohort can understate eventual completion because later events have not arrived.

#### CM-04: Net merchandise bridge

Validity before use: item amount, cancellation, and refund components cover the same order population and currency with consistent signs.

Derivation: conservation of the analytical item balance subtracts cancellations and refunds from the starting item amount.

```text
FORMULA [CM-04]: M_net = G_item - R_cancel - R_refund
WHERE: M_net is analytical net merchandise [CU];
       G_item is included item amount before these adjustments [CU];
       R_cancel is cancelled item amount [CU];
       R_refund is refunded item amount [CU].
CONSTRAINT: same population/currency/tax convention; nonnegative adjustments;
            no adjustment is counted twice; this is not statutory revenue recognition.
NUMERICAL EXAMPLE: G_item=360 CU, R_cancel=60 CU, R_refund=120 CU;
                   M_net=360-60-120=180 CU.
INTERPRETATION: the constructed ledger leaves 180 CU of net merchandise
                before merchandise and fulfillment costs.
```

Boundary: a refund larger than the eligible remaining item balance indicates a data or policy exception, not a routine negative sales adjustment.

### 4.2 General Case

#### CM-05: Order-line summation

The general base replaces pre-aggregated amounts with order-line records.

Let `q_i` be included units on line `i`, `p_i` the unit item price, and `d_i` the line discount in currency.

This formula assumes `d_i` is already the total line discount, not a percentage.

Taxes are excluded because the constructed receipts identify them separately.

```text
FORMULA [CM-05]: G_item = sum_i (q_i * p_i - d_i)
WHERE: i indexes included order lines;
       q_i is item quantity [items/line];
       p_i is unit item price [CU/item];
       d_i is total line discount [CU/line];
       G_item is item amount [CU].
CONSTRAINT: one currency; consistent price/discount basis;
            cancelled lines are included here only if later represented in R_cancel.
NUMERICAL EXAMPLE: lines (2*50-0) and (1*80-0) give
                   G_item=(100+80)=180 CU before any later adjustments.
INTERPRETATION: line-level quantity and discount semantics reproduce the
                aggregate bridge and expose mixed discount interpretations.
```

Derivation: currency per item multiplied by items, less line currency, gives currency; summation preserves the population.

Boundary: a percentage discount of `10` cannot be inserted as `d_i=10 CU`; convert it using the declared pre-discount or post-discount base.

#### CM-06: General contribution bridge

Validity before use: all included components share currency and population and each has a versioned definition.

Derivation: start with net merchandise, add included shipping income, then subtract the named variable costs.

```text
FORMULA [CM-06]: CM = M_net + S_in - C_goods - C_fulfill - C_ship - C_other
WHERE: CM is defined contribution [CU];
       M_net is analytical net merchandise [CU];
       S_in is included shipping income [CU];
       C_goods is merchandise cost for included net units [CU];
       C_fulfill is variable fulfillment cost [CU];
       C_ship is shipping/delivery cost [CU];
       C_other is other explicitly listed variable cost [CU].
CONSTRAINT: costs are neither omitted silently nor double counted;
            fixed costs and taxes are outside this constructed bridge unless named;
            refund cost reversals require an explicit policy.
NUMERICAL EXAMPLE: M_net=180, S_in=30, C_goods=115,
                   C_fulfill=25, C_ship=33, C_other=0, all CU;
                   CM=180+30-115-25-33=37 CU.
INTERPRETATION: after the listed components, the illustrative ledger has
                 37 CU contribution, not 180 CU profit.
```

The cost figures here correspond to the three completed orders in Section 5, not to cancelled-order cost treatment.

Boundary: if return shipping is incurred but absent from every component, `CM=37 CU` overstates the defined complete contribution.

Use a new named component or a documented inclusion in `C_ship`.

#### CM-07: Net merchandise per completed order

Validity before use: net merchandise and completed orders have the same population and snapshot.

Derivation: a per-order average divides total currency by the corresponding order count.

```text
FORMULA [CM-07]: AOV_net = M_net / N_done
WHERE: AOV_net is net merchandise per completed order [CU/order];
       M_net is net merchandise for included completed orders [CU];
       N_done is included completed orders [orders].
CONSTRAINT: N_done>0; partial-refund and cancellation treatment matches CM-04;
            this is not a paid-basket AOV.
NUMERICAL EXAMPLE: M_net=180 CU and N_done=3 orders;
                   AOV_net=180/3=60 CU/order.
INTERPRETATION: completed orders average 60 CU net merchandise;
                refunds have already changed this average.
```

Boundary: zero completed orders makes the measure undefined, not infinite and not zero.

#### CM-08: Weighted aggregation of ratios

Validity before use: groups partition the population and share the same ratio definition.

Derivation: if `r_g=n_g/d_g`, then total numerator is `sum_g n_g` and total denominator is `sum_g d_g`.

```text
FORMULA [CM-08]: r_all = (sum_g n_g) / (sum_g d_g)
                        = sum_g (d_g / sum_h d_h) * r_g
WHERE: g,h index disjoint groups;
       n_g is group numerator [events];
       d_g is group denominator [eligible units];
       r_g=n_g/d_g with d_g>0;
       r_all is pooled ratio.
CONSTRAINT: same metric/population rules in every group;
            denominator weights, not visitor counts unless those are the denominator.
NUMERICAL EXAMPLE: group A has n=2,d=20 and group B has n=8,d=80;
                   r_all=(2+8)/(20+80)=10/100=0.10;
                   denominator-weighted form=0.2*0.10+0.8*0.10=0.10.
INTERPRETATION: pooling reconstructs counts; an unweighted mean of group
                ratios is valid only for a different, explicitly group-weighted estimand.
```

Boundary: empty-group denominators require exclusion under a declared rule, not replacement with zero conversion.

#### CM-09: Two-period change bridge

A difference in contribution is a descriptive bridge, not automatically a causal attribution.

Let subscripts `1` and `0` denote two comparable periods.

```text
FORMULA [CM-09]: Delta_CM = CM_1 - CM_0
WHERE: Delta_CM is contribution change [CU];
       CM_1 and CM_0 use the same versioned bridge [CU].
CONSTRAINT: comparable currency basis and component coverage;
            changes in mix and late adjustments remain visible.
NUMERICAL EXAMPLE: CM_0=50 CU and CM_1=37 CU;
                   Delta_CM=37-50=-13 CU.
INTERPRETATION: defined contribution fell by 13 CU;
                the sign is negative even if gross booked item amount rose.
```

Derivation is direct subtraction.

Boundary: if refund policy or cost coverage changes between periods, the difference mixes definition change with operating change.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | Link/equation ID |
|---|---|---|---|---|---|
| Paid-basket AOV | Denominator is paid orders and numerator uses paid item amount | Fixed paid cohort; pre-refund view | Checkout basket diagnosis | Refund-adjusted unit economics | [CM-07](#cm-07-net-merchandise-per-completed-order) with redefined numerator/denominator |
| Completion funnel | Paid cohort is followed to snapshot | Maturity and late events recorded | Fulfillment diagnosis | Calendar-shipment denominator mixed with paid cohort | [CM-03](#cm-03-lifecycle-completion-ratio-for-a-paid-cohort) |
| Calendar refund rate | Refund events use refund-time window | Numerator/denominator both refund-time compatible | Refund operations workload | Cohort refund propensity | Separate window contract required |
| Cohort refund rate | Paid cohort outcomes observed to maturity | Fixed cohort and right-censoring stated | Product quality/return analysis | Immature cohorts compared without warning | [CM-04](#cm-04-net-merchandise-bridge) |
| Visit conversion | Visit eligibility and attribution are fixed | One attributed paid order per visit in base case | Journey diagnostics | Unique-customer interpretation | [CM-01](#cm-01-paid-visit-conversion) |
| Customer conversion | Distinct eligible customers form denominator | Identity resolution and privacy basis declared | Customer-level analysis | Visit and customer denominators interchanged | New grain, not CM-01 |
| Availability-constrained ranking | Candidate set excludes unavailable SKUs | Snapshot availability and fallback rule | Recommendation/action gating | Scores treated as permission | [CM-02](#cm-02-candidate-eligibility-count) |
| Contribution per order | Divide CM by matching completed orders | `N_done>0`; same population | Unit-economics comparison | Cross-cohort denominator | [CM-06](#cm-06-general-contribution-bridge) and [CM-07](#cm-07-net-merchandise-per-completed-order) |

#### Multiplicative funnel identity

When the same paid cohort is retained, a two-stage identity is available:

```text
FORMULA [CM-10]: N_done / V = (N_paid / V) * (N_done / N_paid)
WHERE: V is eligible visits [visits]; N_paid is paid orders [orders];
       N_done is completed orders from the same paid cohort [orders].
CONSTRAINT: V>0 and N_paid>0; identical attribution and cohort definitions;
            no orders enter or leave between stages except by declared state transitions.
NUMERICAL EXAMPLE: V=100, N_paid=4, N_done=3;
                   left side=3/100=0.03;
                   right side=(4/100)*(3/4)=0.04*0.75=0.03.
INTERPRETATION: paid conversion and cohort completion are separate diagnostic
                stages whose product reproduces visit-to-completion proportion.
```

Derivation: cancel the common nonzero `N_paid` factor.

This is an identity, not an independence assumption and not a causal mediation analysis.

Boundary: calendar-window shipped orders in `N_done` break the cohort identity.

#### Contribution sensitivity to an additional refund

Suppose an additional refund of `r` currency units removes net merchandise and reverses `c` currency units of merchandise cost, with all other components fixed.

```text
FORMULA [CM-11]: Delta_CM_refund = -r + c
WHERE: r is additional item refund [CU];
       c is merchandise cost reversed under the stated policy [CU];
       Delta_CM_refund is change in defined contribution [CU].
CONSTRAINT: no additional shipping, fulfillment, fee, or tax change;
            0<=c<=r is common but not assumed as a law;
            the refund is incremental and not already in the ledger.
NUMERICAL EXAMPLE: r=40 CU and c=20 CU;
                   Delta_CM_refund=-40+20=-20 CU;
                   prior CM=37 CU becomes 17 CU.
INTERPRETATION: a cost reversal offsets only part of the item refund;
                omitted return costs would make the result still lower.
```

Derivation: apply CM-06 before and after and subtract; unchanged terms cancel.

Boundary: if the returned item cannot be restocked, `c=0` under this analytical policy and the full 40 CU reduces contribution.

This sensitivity is checked arithmetic, not a refund forecast.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

| Result/invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| Conversion is defined | Positive eligible denominator | Inspect `V` before division | Zero consented eligible visits | Report undefined with zero exposure |
| Candidate count subtracts cleanly | Raw set distinct; unavailable subset | Compare key sets | Duplicate SKU appears twice | Deduplicate and record exclusion reasons |
| Cohort completion is bounded by one | Numerator is subset of fixed cohort | Check `N_done<=N_paid` | Calendar completions joined to paid cohort | Separate cohort and calendar metrics |
| Net merchandise bridge reconciles | Same population and no duplicate adjustments | Sum signed ledger components | Refund applied both to line and order | Use one adjustment owner |
| AOV uses meaningful denominator | Net amount and completed orders align | Join key and snapshot audit | Paid orders used with refunded completed sales | Publish paid AOV separately |
| Pooled ratio uses denominator weights | Common definition across groups | Reconstruct totals from counts | Average of 1% and 50% reported as pooled 25.5% | Sum numerators and denominators |
| Funnel factors multiply | Same cohort across stages | Compare left and right sides | Mixed visitor and customer grain | Redefine stages at one grain |
| Contribution sign is interpretable | Complete component registry | List missing costs and reversals | Return freight omitted | Add/version the component |
| Late refunds preserve auditability | Snapshot and revision retained | Recompute prior snapshot | Historical dashboard overwritten | Bitemporal or versioned snapshot |

As traffic grows with a fixed numerator-to-denominator ratio, the ratio remains unchanged; scale alone does not improve conversion.

As order count approaches zero, per-order ratios become unstable and eventually undefined.

As refund maturity extends, a paid-cohort contribution estimate can change without any new payment event.

As the raw candidate set becomes empty, no ranking model can produce an eligible recommendation from that set.

As currencies multiply, conversion requires a dated exchange-rate and rounding policy; simple summation is invalid.

A material counterexample to “rising AOV is good” is a refund-heavy mix with high item value and negative contribution after costs.

A material counterexample to “more candidates improve choice” is a raw list dominated by unavailable or unauthorized items.

**Takeaway:** The formulas are ledger identities with strict grains; changing a denominator, maturity, currency, or cost registry changes the question.

## 5. Worked Numerical Example

### 5.1 Scenario and evidence

**Example title:** Four-order ledger with availability filtering, cancellation, and partial refund.

**Evidence:** ILLUSTRATIVE; all receipts, visits, candidates, prices, and costs are constructed.

**Workflow and decision:** Determine whether a checkout/availability review should use paid conversion, completion, net merchandise, or contribution, and reconcile all four.

**Intentional simplifications:** one currency, no tax in item amounts, no shipping refund, no return freight, no payment fee, no marketing cost, no fixed overhead, and immediate complete data.

The example does not establish accounting revenue, causal lift, customer behavior, or production performance.

### 5.2 Input data

There are 100 eligible visits in the constructed window.

Exactly four distinct orders receive an attributed paid event, one per visit.

Ten raw recommended SKUs enter the availability filter.

Three are unavailable at the decision snapshot.

No other eligibility filter removes a candidate.

| Order | Paid item amount [CU] | Shipping income [CU] | Tax excluded [CU] | State by snapshot | Item refund [CU] | Goods cost after refund policy [CU] | Fulfillment [CU] | Shipping cost [CU] |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| A | 100 | 10 | 8 | completed | 0 | 60 | 8 | 11 |
| B | 80 | 8 | 6.40 | completed, fully refunded | 80 | 0 | 7 | 10 |
| C | 60 | 0 | 0 | cancelled before shipment | 0 | 0 | 0 | 0 |
| D | 120 | 12 | 9.60 | completed, partially refunded | 40 | 55 | 10 | 12 |
| Total | 360 | 30 | 24 | 3 completed, 1 cancelled | 120 | 115 | 25 | 33 |

For B, the illustrative goods-cost reversal is the full original 45 CU because the item returns to available stock under the constructed policy.

For D, the original goods cost was 75 CU; the refund reverses 20 CU, leaving 55 CU.

These reversals are policy inputs, not natural laws.

The cancelled C order contributes its item amount to `G_item` and the same amount to `R_cancel`.

Its shipping income and variable costs are zero in this constructed snapshot.

Shipping income for completed A, B, and D remains 30 CU under the stated no-shipping-refund simplification.

### 5.3 Availability and funnel steps

| Step | Equation/rule | Inputs with units | Intermediate | Reported result | Independent check |
|---|---|---|---|---|---|
| 1 | CM-02 | 10 raw SKUs; 3 unavailable SKUs | `10-3` | 7 eligible SKUs | Set sizes: `7+3=10` |
| 2 | CM-01 | 4 paid orders; 100 visits | `4/100` | 0.04 paid conversion | `0.04*100=4 orders` |
| 3 | CM-03 | 3 completed orders; 4 paid orders | `3/4` | 0.75 completion | `3+1 cancelled=4` |
| 4 | CM-10 | 100 visits; 4 paid; 3 completed | `0.04*0.75` | 0.03 visit-to-completion | Direct `3/100=0.03` |

The availability answer constrains a recommendation action.

The paid conversion answer describes checkout attribution.

The completion answer describes the paid cohort at the snapshot.

They are not interchangeable success metrics.

### 5.4 Merchandise and contribution steps

| Step | Equation/rule | Inputs with units | Intermediate | Reported result | Independent check |
|---|---|---|---|---|---|
| 5 | CM-04 | `G=360`, cancel `60`, refund `120` CU | `360-60-120` | `M_net=180 CU` | Completed item total `100+80+120=300`; less refunds `120=180` |
| 6 | CM-06 | `180+30-115-25-33` CU | `210-173` | `CM=37 CU` | Sum costs `115+25+33=173`; `37+173=210` |
| 7 | CM-07 | `180 CU`, 3 completed orders | `180/3` | `60 CU/order` | `60*3=180 CU` |
| 8 | CM-09 | prior illustrative `CM_0=50`, current `37` CU | `37-50` | `-13 CU` | `50-13=37 CU` |

Tax totals 24 CU are deliberately excluded from every item and contribution input.

Adding tax to merchandise would double count a separately identified pass-through in this constructed convention.

The contribution bridge includes shipping income and shipping cost separately so their net effect is visible.

Their net effect is `30-33=-3 CU`.

Merchandise after refunds less goods cost is `180-115=65 CU`.

Fulfillment reduces this by 25 CU and net shipping by 3 CU.

The alternate decomposition is `65-25-3=37 CU`.

That independently reproduces Step 6 without reusing its exact grouping.

### 5.5 Sensitivity and wrong path

Apply CM-11 to an additional 40 CU refund with a 20 CU goods-cost reversal.

The change is `-40+20=-20 CU`.

New contribution is `37-20=17 CU`.

New net merchandise is `180-40=140 CU`.

If there is no cost reversal because the item cannot be restocked, new contribution is `37-40=-3 CU`.

That boundary changes the sign of the constructed result.

A plausible wrong path divides `G_item=360 CU` by four paid orders and calls the resulting 90 CU per paid order “net AOV.”

That number ignores cancellation and refunds and uses a different denominator.

It can be a valid paid-basket amount only under a new name and contract.

Another wrong path sums taxes into contribution, producing `37+24=61 CU` under this example's convention.

That is a sign and scope error, not a conservative adjustment.

### 5.6 Smoke-check record

The arithmetic and lifecycle oracle were executed with Python 3 using exact integer arithmetic for these CU values and rational checks for the funnel.

The check asserted `10-3=7`, `4+0=100*0.04` through fraction equivalence, `3+1=4`, `360-60-120=180`, `180+30-115-25-33=37`, and `37-40+20=17`.

It independently reconstructed the state count from the receipt rows and rejected a duplicate paid event for order A.

It also rejected `N_done=4` with this three-completion ledger.

The oracle result was `PASS` for the stated inputs and invalid-transition rejection.

The check covers constructed arithmetic and state rules only.

It did not test a database, production event stream, financial reporting policy, or user behavior.

No latency, availability, conversion lift, or financial outcome was measured.

### 5.7 Production-scale interpretation

At larger scale, duplicate events, partial lines, multi-currency orders, late refunds, and identity resolution dominate simple arithmetic.

A daily dashboard may need incremental state maintenance plus periodic full reconciliation.

An immature paid cohort needs a maturity label or survival-style outcome analysis rather than a final completion claim.

A high-volume ranking system must retain availability snapshot versions to explain why candidates were excluded.

A contribution dashboard needs cost-effective coverage tests, not just higher decimal precision.

A useful next test is replaying the same receipts with delayed refund events and verifying that each snapshot remains reconstructible.

**Takeaway:** The four-order ledger yields 7 eligible SKUs, 4% paid conversion, 75% completion, 180 CU net merchandise, and 37 CU defined contribution; each number answers a different question.

## 6. Visual / Geometric Intuition

### 6.1 Cohort funnel as nested sets

The schematic below shows entity counts, not elapsed time and not causal flow.

```text
Eligible visits, V=100
+--------------------------------------------------------------------------------+
| Visits with attributed paid order, N_paid=4                                    |
| +----------------------------------------------------------------------------+ |
| | Paid-cohort orders completed by snapshot, N_done=3                        | |
| | +----------------------------------------------------------------------+  | |
| | | A, B, D                                                             |  | |
| | +----------------------------------------------------------------------+  | |
| | C: paid, then cancelled before completion                              | |
| +----------------------------------------------------------------------------+ |
| 96 visits without an attributed paid order in this constructed attribution     |
+--------------------------------------------------------------------------------+
```

Caption: the paid set has four orders and its completed subset has three.

The visit-to-completion area ratio is `3/100`, while the paid conversion ratio is `4/100`.

The identity in CM-10 follows from this nesting.

If completed orders from another paid cohort entered the inner set, the diagram would no longer support CM-10.

### 6.2 Money bridge

The diagram uses signed components and is not to scale.

```text
CU
360 |  G_item
    |    |
    |    | -60 cancellation
300 |    v
    |    |
    |    | -120 item refunds
180 |    v  M_net
    |    +30 shipping income
210 |    |
    |    -115 goods cost
 95 |    |
    |    -25 fulfillment
 70 |    |
    |    -33 shipping cost
 37 |    v  CM
  0 +--------------------------------------------------> ordered bridge steps
```

Caption: booked item amount and final defined contribution occupy different vertical positions because adjustments and costs have explicit signs.

The bridge makes a negative shipping margin of 3 CU visible.

It also makes omitted return freight an open component rather than an invisible assumption.

### 6.3 Event time versus snapshot time

```text
Event time ->       Monday             Tuesday             Wednesday
Order A             paid               shipped             completed
Order B             paid               shipped             refunded
Order C             paid               cancelled           no new event
Order D             paid               shipped             partial refund

Snapshot tau_1: end Monday       -> four paid; zero completed in this trace
Snapshot tau_2: end Wednesday    -> three completed; one cancelled; refunds applied
```

Caption: cohort membership is fixed by Monday payment, while observed state depends on snapshot time.

A Monday completion ratio of zero would be immature, not evidence that none will complete.

Retaining both axes prevents silent revision and denominator drift.

**Takeaway:** Nested entity sets, signed money bridges, and separate event/snapshot axes expose errors that a single funnel percentage hides.

## 7. Comparison / Selection Table

### 7.1 Choose a measurement approach

| Method/variant | Preconditions | Quality/guarantee | Time/memory/cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Spreadsheet receipt baseline | Small stable ledger; manual definitions | Transparent row-by-row arithmetic, not scalability | `O(N)` scan for `N` rows; manual effort grows | Reconciling a few orders and validating semantics | High event volume or concurrent revisions | DERIVED operation count; NOT MEASURED operationally |
| Immutable events plus derived snapshots | Stable IDs; event schema; replay support | Reconstructable states if log and code versions retained | `O(E)` replay for `E` events; state roughly `O(N)` entities | Late/duplicate events and audit | Missing event history or ambiguous identity | DERIVED design property; deployment NOT MEASURED |
| Mutable current-state table only | Current question needs latest state | Fast current lookup if updates correct | `O(1)` indexed lookup under key assumptions; history extra | Operational current status | Reproducing old cohort snapshots | DERIVED limitation |
| Paid-cohort analysis | Fixed paid-at cohort and snapshot | Completion/refund maturity is explicit | Joins by order key; late outcomes retained | Product and fulfillment cohort comparison | Calendar workload is the actual question | DERIVED metric contract |
| Calendar-event analysis | Event-time windows consistently applied | Measures activity occurring in window | Partition by event time | Operational shipment/refund workload | Claiming cohort completion propensity | DERIVED metric contract |
| Experiment with causal estimand | Defined assignment, exposure, outcomes, interference assumptions | Addresses only specified causal effect under design assumptions | Requires assignment/outcome logs and analysis | Estimating effect of a change | Reconciling ledger alone or post-hoc cherry-picking | OPEN for any actual commerce effect |

The simplest credible baseline is a small reconciled ledger.

More infrastructure is justified by replay, volume, lateness, or access requirements, not by a preference for architectural complexity.

No option guarantees business improvement.

### 7.2 Choose the primary decision metric

| Decision | Primary measure | Guardrails | Do not substitute |
|---|---|---|---|
| Are raw recommendations actionable? | Eligible candidate count and exclusion reasons | Snapshot freshness; policy/region filters | Raw candidate count |
| Did paid attribution change? | Paid-visit conversion | Visit eligibility; attribution; maturity | Revenue per visitor without same grain |
| Did paid orders complete? | Cohort completion at fixed maturity | Late events; cancellation semantics | Calendar shipments divided by unrelated paid orders |
| What merchandise balance remains? | Net merchandise bridge | Refund/cancellation ownership; currency | Booked item amount |
| What listed variable economics remain? | Versioned contribution | Missing costs; reversals; return freight | Net merchandise or formal profit |
| Did an intervention cause a change? | Predeclared causal estimand | Assignment validity; interference; uncertainty | Before/after difference alone |

### 7.3 Prioritized selection guide

```text
IF entity identity, currency, or order-state ownership is unresolved
THEN stop metric comparison and repair the ledger contract.
ELSE IF the question concerns an executable recommendation or order action
THEN apply eligibility, inventory, customer-policy, and authorization gates first.
ELSE IF the question concerns operational workload in a calendar window
THEN use event-time counts with late/duplicate handling.
ELSE IF the question concerns eventual outcomes of paid orders
THEN use a fixed paid cohort with a stated maturity/snapshot.
ELSE IF the question concerns economics
THEN use the versioned contribution bridge and disclose omitted costs.
ELSE
THEN retain the simplest reconciled descriptive metric and avoid a causal claim.
If a hard customer policy conflicts with a metric objective, policy takes priority.
```

Conflicts between conversion and contribution require an explicit objective and guardrails, not automatic selection of the larger conversion number.

A proposed launch needs a comparator, uncertainty, and rollback criterion.

Those elements are not supplied by this sheet's illustrative arithmetic.

**Takeaway:** Select state representation and metric window from the decision; do not let an available dashboard field choose the question.

## 8. Failure Modes & Edge Cases

All failure entries below are HYPOTHETICAL teaching scenarios, not observed merchant incidents.

The Common Issues collection artifact is planned, so these IDs are local to this draft rather than claimed completed issue links.

### F-01: Paid conversion and completion use incompatible populations

**Trigger and symptom:** A dashboard reports completion above 100% or a sudden discontinuity after a reporting change.

**Competing causes:** Calendar completions may be divided by a paid cohort; a late-arriving cohort may also be legitimately maturing; duplicate paid IDs can distort the denominator.

**Discriminating check:** Reproduce numerator and denominator order-key sets at a fixed snapshot.

Expected signal for grain mismatch: numerator contains orders paid outside the denominator window.

Expected signal for maturity: keys align but recent cohorts have systematically shorter observation time.

Expected signal for duplication: distinct event count exceeds distinct order count.

**Immediate mitigation:** Mark the ratio invalid and suppress automated decisions based on it.

**Correction:** Choose either fixed-cohort completion or calendar activity and rebuild the contract.

**Mitigation risk:** Restricting to mature cohorts delays visibility and can change mix.

**Prevention:** Schema-enforce window type, cohort key, snapshot, and maturity.

**Verification:** Assert `N_done<=N_paid` for every cohort and recompute CM-10 on a hand-checked sample.

**Escalation:** If keys still differ, involve the order-ledger owner rather than forcing a join.

**Provenance:** HYPOTHETICAL; derived set-boundary invariant.

### F-02: Duplicate or out-of-order events change order economics

**Trigger and symptom:** Replaying a topic changes paid counts or a cancelled order becomes paid again.

**Competing causes:** Duplicate delivery, unstable order keys, missing sequence/version, or an actual second payment.

**Discriminating check:** Group by order key, event ID, event type, and source version.

Expected signal for duplicate delivery: identical event IDs or identical idempotency keys repeat.

Expected signal for stale transition: an older source version arrives after a newer state.

Expected signal for real second payment: a distinct authorized payment and order-line relationship exists.

**Immediate mitigation:** Quarantine ambiguous transitions; do not simply take latest arrival time.

**Correction:** Define event identity, source ordering where available, and a state-transition table.

**Mitigation risk:** Over-aggressive deduplication can merge legitimate partial payments or refunds.

**Prevention:** Stable order/event IDs, idempotent projections, and replay tests.

**Verification:** Replay permutations and duplicates; final state and monetary bridge must match the specified oracle.

**Escalation:** Reconcile against the authoritative payment/order system when identity remains ambiguous.

**Provenance:** HYPOTHETICAL; state-oracle smoke check covers one duplicate-paid rejection only.

### F-03: Stockout censoring is interpreted as low demand

**Trigger and symptom:** An item shows few purchases and the ranker reduces exposure, while availability records show repeated stockouts.

**Competing causes:** Weak customer interest, unavailable inventory, regional eligibility, price change, or missing exposure logs.

**Discriminating check:** Align impressions, eligibility, stock snapshots, price, and purchases at decision time.

Expected signal for censoring: purchases fall during intervals with zero eligible candidates while qualified exposure also falls.

Expected signal for weak interest: eligible impressions remain present but downstream response is low.

Expected signal for logging error: availability is present but exposure IDs are missing.

**Immediate mitigation:** Do not label zero purchases during no-exposure intervals as a revealed preference.

**Correction:** Preserve censoring indicators and train/evaluate with availability-aware samples.

**Mitigation risk:** Forcing exposure of unavailable items creates customer frustration and invalid actions.

**Prevention:** Log raw candidates, eligibility reasons, actual impressions, and stock version.

**Verification:** CM-02 must hold on sampled decision traces; evaluation must report eligible exposure coverage.

**Escalation:** If inventory snapshots are unreliable, fix the inventory source before interpreting demand.

**Provenance:** HYPOTHETICAL; no demand estimate is supplied.

### F-04: Refund and cancellation adjustments are double counted or omitted

**Trigger and symptom:** Net merchandise becomes unexpectedly negative or differs between finance and product reports.

**Competing causes:** Line-level and order-level refunds both subtracted, cancelled amounts omitted, shipping refunds confused with item refunds, or late refunds not yet loaded.

**Discriminating check:** Reconcile each adjustment to a unique line/action ID and sign.

Expected signal for duplication: two adjustments reference the same economic event without a reversal relationship.

Expected signal for lateness: ledger keys reconcile but source high-water marks differ.

Expected signal for component confusion: item totals reconcile while shipping components do not.

**Immediate mitigation:** Freeze the affected contribution metric and label the unresolved bridge.

**Correction:** Assign one owner to each economic adjustment and version reversal policy.

**Mitigation risk:** Suppressing all late refunds improves apparent contribution while making it less accurate.

**Prevention:** Component registry, signed ledger entries, and automated CM-04 reconciliation.

**Verification:** Rebuild Section 5 from line rows and verify each refund is applied once.

**Escalation:** Formal recognition questions go to accounting; this analytical bridge does not decide them.

**Provenance:** HYPOTHETICAL; constructed arithmetic demonstrates sign sensitivity.

### F-05: Group ratios are averaged without denominator weights

**Trigger and symptom:** The overall conversion rate lies outside the range supported by reconstructed counts or changes when groups are split.

**Competing causes:** Unweighted averaging, different eligibility definitions, zero denominators, or genuine mix shift.

**Discriminating check:** Recover `n_g` and `d_g` for every group and apply CM-08.

Expected signal for averaging error: pooled counts differ from the displayed average.

Expected signal for definition mismatch: identical raw visits receive different eligibility status by group.

Expected signal for mix shift: both subgroup rates are stable but denominator shares change.

**Immediate mitigation:** Publish counts alongside rates and withdraw the pooled average.

**Correction:** Use denominator-weighted aggregation or explicitly define an equal-group-weight estimand.

**Mitigation risk:** Pooling can conceal a harmful subgroup result even when arithmetic is correct.

**Prevention:** Store numerator and denominator, not only rounded percentages.

**Verification:** Recompute pooled and subgroup metrics; inspect both mix and within-group changes.

**Escalation:** If denominator definitions differ, harmonize before aggregation.

**Provenance:** HYPOTHETICAL; CM-08 is derived arithmetic.

### F-06: Conversion optimization overrides contribution or customer policy

**Trigger and symptom:** A proposed recommendation increases paid orders in an evaluation but uses unavailable items, unauthorized substitutions, or refund-heavy products.

**Competing causes:** Wrong objective, stale availability, missing authorization gate, incomplete cost registry, or biased evaluation.

**Discriminating check:** Trace candidate eligibility and authorization independently of model score, then reconcile downstream contribution by cohort.

Expected signal for policy failure: an excluded SKU reaches an executed action.

Expected signal for economics failure: net orders rise but CM-06 falls after costs and refunds mature.

Expected signal for evaluation bias: assigned and analyzed populations differ systematically.

**Immediate mitigation:** Block unauthorized actions and retain a safe no-action or approved fallback.

**Correction:** Put hard policy gates outside ranking and define multi-metric evaluation with maturity.

**Mitigation risk:** A blanket fallback can reduce useful availability even when only one policy source is stale.

**Prevention:** Versioned policy, immutable decision traces, and guardrail tests before launch.

**Verification:** No executed item violates eligibility; metric comparisons use identical cohorts and cost versions.

**Escalation:** Customer disputes or policy ambiguity go to the accountable human owner.

**Provenance:** HYPOTHETICAL; no uplift or deployment is claimed.

### F-07: Timestamp and timezone handling changes the cohort

**Trigger and symptom:** Daily order totals move across midnight or differ between regional reports.

**Competing causes:** Local timestamps without offsets, daylight-saving transitions, event-time versus processing-time fields, or late corrections.

**Discriminating check:** Inspect raw timestamp strings, offsets, event type, and ingestion time for boundary orders.

Expected signal for timezone error: a fixed hourly offset explains discrepancies.

Expected signal for field misuse: processing timestamps fall in a different window than payment timestamps.

Expected signal for late correction: the same event-time record changes across snapshots.

**Immediate mitigation:** Stop cross-region aggregation until the time basis is declared.

**Correction:** Store qualified instants and preserve source event time, processing time, and snapshot.

**Mitigation risk:** Converting historical local times without historical offset rules can introduce new errors.

**Prevention:** Timestamp schema validation and boundary tests around midnight and daylight-saving changes.

**Verification:** Reassign a small set of boundary events manually and compare window membership.

**Escalation:** Seek source-system ownership for ambiguous unqualified timestamps rather than guessing.

**Provenance:** HYPOTHETICAL; RFC 3339 supports timestamp representation, not this merchant's clock correctness.

**Takeaway:** Diagnose entity sets, event identity, eligibility, component signs, denominator weights, policy gates, and timestamps before interpreting movement.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let `E` be order events, `N` distinct orders, `L` order lines, `V` visits, `K` raw candidates, and `G` reporting groups.

Assume hash-based indexes with expected constant-time lookup under a suitable hash distribution and bounded key size.

Sorting-based alternatives have different bounds and may be preferable for deterministic ordered output.

| Operation | Input sizes | Time | Space | I/O/communication | Assumptions / dominant bottleneck |
|---|---|---|---|---|---|
| Validate candidate eligibility | `K` candidates | Expected `O(K)` lookups | `O(K)` result/exclusion set | Inventory/policy reads | Availability snapshot and distinct SKU keys |
| Build order states | `E` events, `N` orders | Expected `O(E)` hash updates | `O(N)` current state plus retained events | Event scan; replay reads | Transition logic is constant work per event |
| Sum line amounts | `L` lines | `O(L)` | `O(1)` auxiliary if streamed | Line scan | Exact decimal representation may increase per-op cost |
| Join visits to orders | `V+N` entities | Expected `O(V+N)` | `O(V)` or `O(N)` index | Attribution logs | One-order base attribution; many-to-many adds edges |
| Aggregate groups | `G` groups over `N` orders | Expected `O(N+G)` | `O(G)` aggregates | Usually local after join | Shared metric definition |
| Full replay | `E` events | Expected `O(E)` | `O(N)` state; durable log extra | Full history read | Event log completeness |
| Sort ledger for audit | `E` events | `O(E log E)` comparison sort | Implementation-dependent | Random/sequential I/O | Bounded key comparisons and no presorted index |

These are operation-count bounds, not observed throughput or latency.

A join's expected bound does not cover pathological hash collisions.

Decimal currency arithmetic and external inventory calls can dominate the constant factors.

Incremental snapshots reduce repeated full scans but add state-version and recovery obligations.

### 9b. Back-of-Envelope Sizing

**Evidence:** ILLUSTRATIVE workload; no infrastructure benchmark or price is used.

Assume 2,000,000 order events per day and an average stored event envelope of 1,200 bytes.

The envelope is a constructed storage assumption, not a message-format guarantee.

```text
FORMULA [CM-12]: S_day = E_day * b_event
WHERE: S_day is payload/envelope bytes per day [bytes/day];
       E_day is events per day [events/day];
       b_event is average stored bytes per event [bytes/event].
CONSTRAINT: constant average size; one stored copy;
            indexes, replication, compression, logs, backups, and metadata excluded.
NUMERICAL EXAMPLE: E_day=2,000,000 events/day; b_event=1,200 bytes/event;
                   S_day=2,000,000*1,200=2,400,000,000 bytes/day.
                   In binary units, divide by 1,048,576 bytes/MiB:
                   2,400,000,000/1,048,576=2288.818359375 MiB/day,
                   approximately 2288.82 MiB/day.
INTERPRETATION: this is a single-copy envelope estimate, not a storage quote.
```

Derivation: events multiplied by bytes per event cancels events.

Independent check: `2,288.818359375*1,048,576=2,400,000,000` bytes.

For thirty days with no deletion and the same rate, the estimate is 72,000,000,000 bytes.

This is 72 GB in decimal units, before overhead.

Sensitivity: if the average envelope doubles to 2,400 bytes, the one-day estimate doubles to 4,800,000,000 bytes.

Boundary: retention, replication, compression, or compaction invalidates simple linear extrapolation unless their factor is explicitly added.

### 9c. Economic cost boundary

No currency-per-event price is supplied.

A complete operating-cost estimate would need dated storage, compute, network, engineering, support, and reconciliation-labor inputs.

The metric tree itself can reduce ambiguity but that does not prove a financial saving.

An evaluation should count analyst hours spent resolving residuals and operational cost of incorrect actions.

Those values are NOT MEASURED here.

A useful cost per reconciled order would divide total measured reconciliation cost by distinct reconciled orders, not by event count.

A useful cost per eligible recommendation would exclude unavailable raw candidates from successful-action counts.

Both require actual workload and cost evidence.

**Takeaway:** Linear scans and envelope sizing provide reproducible planning bounds; real latency, storage bills, and savings require measured deployments.

## 10. Key Theorems / Results Reference

This sheet relies primarily on finite-set and accounting identities rather than advanced statistical theorems.

| Theorem/result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Primary/textbook source |
|---|---|---|---|---|---|
| Disjoint finite-set addition | Cardinalities add for disjoint finite sets | Every order belongs to exactly one partition cell | State counts reconcile to distinct orders | States are causally ordered or mutually exclusive in every domain | DERIVED set identity |
| Ratio aggregation identity | Pooled ratio equals sum of numerators over sum of denominators | Common definitions; positive group denominators | Reconstruct pooled rates from counts | Equal weighting of groups has the same estimand | DERIVED in CM-08 |
| Funnel factor identity | A three-stage nested ratio factors through an intermediate count | Positive intermediate; same cohort | Conversion and completion can be diagnosed separately | Independence or causal mediation | DERIVED in CM-10 |
| Signed bridge conservation | Opening amount plus signed adjustments equals closing amount | Complete, same-grain components | Residuals expose omissions or duplicates | Zero residual proves semantic correctness | DERIVED in CM-04/CM-06 |
| Timestamp representation | RFC 3339 defines a profile for Internet date/time representation | Representation rules are applied correctly | Qualified timestamps reduce representation ambiguity | Clock synchronization, event ordering, or correct merchant timezone history | [RFC 3339, July 2002](https://www.rfc-editor.org/rfc/rfc3339.html), consulted 2026-11-09 |
| Provenance vocabulary | PROV-O is a W3C Recommendation for representing the PROV data model in OWL2 | A provenance model is designed and populated | Supports explicit entity/activity/agent modeling choices | A commerce ledger is automatically complete or correct | [W3C PROV-O, 2013-04-30](https://www.w3.org/TR/2013/REC-prov-o-20130430/), consulted 2026-11-09 |

### 10.1 Source and evidence register

| Source ID | Primary title/link | Supported claim | Publication/version | Reviewed on | Limitation | Recheck trigger |
|---|---|---|---|---|---|---|
| SRC-COM-01 | [RFC 3339](https://www.rfc-editor.org/rfc/rfc3339.html) | Internet timestamp format and qualified-offset concepts | July 2002; updated by RFC 9557 | 2026-11-09 | Not a clock-synchronization or cohort policy | Timestamp protocol or policy changes |
| SRC-COM-02 | [W3C PROV-O](https://www.w3.org/TR/2013/REC-prov-o-20130430/) | Normative OWL2 representation of PROV data model | W3C Recommendation 2013-04-30 | 2026-11-09 | No proof of ledger quality or commerce suitability | Provenance architecture changes |
| SRC-COM-03 | [IFRS 15 landing page](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/) | Identifies the standard title only | Current recognition text not reviewed | 2026-11-09 | OPEN; substantive text unavailable in this session; no accounting claim made | Before any formal revenue-recognition instruction |
| SRC-COM-04 | Original retail/e-commerce agent source named in focused plan | Planned topic provenance only | Unavailable | 2026-11-09 | OPEN; source body not supplied and not reconstructed | If original source becomes available |
| NUM-COM-01 | Section 5 and 9 arithmetic | Constructed funnel, ledger, sensitivity, and envelope arithmetic | Python 3 exact/rational smoke check | 2026-11-09 | VERIFIED arithmetic only; not a production experiment | Any changed input, sign, or formula |

### 10.2 Statistical caution

A sample proportion has uncertainty under a specified sampling or assignment model.

Visit dependence, repeat customers, bots, and adaptive policies can invalidate a simple independent-Bernoulli calculation.

This sheet therefore does not attach a confidence interval to the four-order example.

If inference is needed, declare the sampling unit, estimand, dependence, missingness, and multiple-comparison policy.

A zero residual is not statistical validation.

A large sample does not fix selection bias or incorrect event identity.

**Takeaway:** The strong results here are finite reconciliation identities; time, provenance, accounting, and causal claims need separate evidence.

## 11. Interview / Exam Questions

Each answer below is checked against the local derivation or smoke-checked example.

The scoring scale is `0` incorrect, `1` partial, `2` correct with assumptions, and `3` correct plus a defended tradeoff and verification.

No question is attributed to an employer.

### 11.1 Level 1 — Definition / Recall

#### Q-COM-01 / Level 1 / metric contracts

**Prompt:** Why is “conversion rate” an incomplete metric name?

**Clarifications:** Use a retail visit example and one nonexample.

**Answer:** The name omits numerator entity, denominator entity, eligibility, attribution, window, and snapshot. Paid orders per eligible visit differs from paying customers per eligible customer and from completed orders per paid cohort. For example, `4/100=4%` is paid-visit conversion in CM-01, not completion. A ratio without a denominator contract is not reproducible.

**Alternative answers:** A precise organization-specific definition is acceptable if all relevant fields are named.

**Follow-up:** Which fields change if a visitor can place two orders? The one-order attribution assumption changes; use a many-outcome rule or orders per visit.

**Trap/red flag:** “It is purchases divided by users” without identity or window.

**Scoring:** 0 wrong denominator; 1 mentions numerator only; 2 specifies entities/window; 3 includes attribution and maturity.

**Answer source:** Sections 3.2 and 4.1, CM-01.

#### Q-COM-02 / Level 1 / money boundaries

**Prompt:** Distinguish analytical net merchandise from defined contribution.

**Clarifications:** Refer to the four-order example.

**Answer:** Net merchandise is `360-60-120=180 CU`. Defined contribution additionally includes shipping income and subtracts listed goods, fulfillment, and shipping costs: `180+30-115-25-33=37 CU`. Neither is automatically statutory revenue or operating profit.

**Alternative answers:** Other local contribution definitions are defensible if their components and signs are explicit and consistently applied.

**Follow-up:** What happens if return freight is omitted? The stated 37 CU excludes it and can overstate a more complete contribution measure.

**Trap/red flag:** Equating revenue, cash, merchandise, and profit.

**Scoring:** 0 conflates terms; 1 defines one; 2 computes both; 3 identifies omitted-cost and accounting boundaries.

**Answer source:** CM-04, CM-06, Section 5.4.

#### Q-COM-03 / Level 1 / cohort semantics

**Prompt:** Why can a paid cohort and a calendar shipment report disagree without either being wrong?

**Clarifications:** Assume orders paid near month end.

**Answer:** A paid cohort fixes orders by paid-at time and follows later shipments; a calendar report fixes shipments by shipped-at time and can include earlier-paid orders. Their key sets differ. Late-month payments make the difference material.

**Alternative answers:** A bitemporal design may support both; that does not make their denominators interchangeable.

**Follow-up:** Which is better for daily shipping workload? Calendar shipment activity.

**Trap/red flag:** Forcing one report to match by deleting late orders.

**Scoring:** 0 claims one is always correct; 1 mentions timing; 2 explains key sets; 3 selects based on decision and discusses maturity.

**Answer source:** Sections 3.6 and 6.3.

### 11.2 Level 2 — Derivation / Apply

#### Q-COM-04 / Level 2 / availability calculation

**Prompt:** Ten distinct recommended SKUs are returned and three are unavailable. How many are eligible before ranking, and what additional information could change the answer?

**Clarifications:** Availability is the only stated filter initially.

**Answer:** `K_eligible=10-3=7 SKUs` by CM-02. Regional restrictions, rights, customer policy, duplicates, or other exclusions could further reduce the set. Availability must be tied to a decision snapshot.

**Alternative answers:** If unavailable IDs are not a subset of raw distinct candidates, first reconcile keys rather than subtracting blindly.

**Follow-up:** What if all ten are unavailable? Return a governed no-result/fallback, not a ranked unavailable item.

**Trap/red flag:** Reporting seven purchases or seven impressions.

**Scoring:** 0 wrong arithmetic; 1 gives seven only; 2 states snapshot/subset; 3 adds policy and empty-set behavior.

**Answer source:** CM-02; smoke check NUM-COM-01.

#### Q-COM-05 / Level 2 / funnel derivation

**Prompt:** Given 100 eligible visits, four paid orders, and three completed orders from that paid cohort, derive visit-to-completion proportion in two ways.

**Clarifications:** All denominators are nonzero and one paid order maps to one visit.

**Answer:** Directly, `3/100=0.03`. Through stages, `(4/100)*(3/4)=0.04*0.75=0.03`. The intermediate factor cancels algebraically. This requires the same cohort and does not establish independence.

**Alternative answers:** A percentage response of 3% is equivalent.

**Follow-up:** If `N_paid=0`, can the factored form be used? No; direct completion is also zero only if no completed cohort orders exist, while the intermediate ratio is undefined.

**Trap/red flag:** Adding 4% and 75%.

**Scoring:** 0 wrong operation; 1 one method; 2 both and assumptions; 3 explains zero/cohort boundary.

**Answer source:** CM-10 and Section 5.3.

#### Q-COM-06 / Level 2 / aggregation

**Prompt:** Group A has 2 conversions from 20 eligible visits; group B has 8 from 80. Compute the pooled conversion and the equal-group arithmetic mean. Explain the difference.

**Clarifications:** Both use the same conversion definition.

**Answer:** Pooled is `(2+8)/(20+80)=10/100=0.10`. The equal-group mean is `(0.10+0.10)/2=0.10`; they coincide here because rates are equal. They need not coincide generally. Pooled weights groups by denominator, while equal-group mean represents a different group-weighted estimand.

**Alternative answers:** Either may be reported if its weighting objective is explicit, but pooled population conversion requires counts.

**Follow-up:** If A were 1/20 and B 8/80, pooled would be 9/100=0.09 while equal-group mean would be 0.075.

**Trap/red flag:** Averaging percentages without stating weights.

**Scoring:** 0 arithmetic error; 1 pooled only; 2 both and weighting; 3 identifies changed estimand and follow-up values.

**Answer source:** CM-08.

#### Q-COM-07 / Level 2 / refund sensitivity

**Prompt:** Current contribution is 37 CU. An additional refund returns 40 CU to the customer and reverses 20 CU of goods cost. Compute the result, then remove the cost reversal.

**Clarifications:** No other component changes.

**Answer:** Change is `-40+20=-20 CU`; new contribution is 17 CU. Without reversal, change is -40 CU and new contribution is -3 CU. The sign flips, so restocking policy is material.

**Alternative answers:** Including return freight is defensible only as an added negative component and would lower both results.

**Follow-up:** What evidence is needed before forecasting? Actual refund incidence, cost reversal, freight, and cohort maturity.

**Trap/red flag:** Subtracting the refund and adding its cost as a new expense.

**Scoring:** 0 sign error; 1 first result; 2 both results; 3 explains components and evidence.

**Answer source:** CM-11 and Section 5.5; smoke check NUM-COM-01.

### 11.3 Level 3 — Judgement / Design / Debug

#### Q-COM-08 / Level 3 / ambiguous diagnosis

**Prompt:** Completion rises from 70% to 95% overnight with no fulfillment change. Give competing causes and discriminating checks.

**Clarifications:** You have event and cohort data.

**Answer:** Possible causes include denominator loss, calendar/cohort mixing, duplicate or reclassified events, and a snapshot/maturity bug. Compare distinct order-key sets, paid cohort definitions, event IDs, source versions, and snapshot times. If numerator keys include orders paid outside the denominator, grain mismatch is supported; if keys align but source versions changed, investigate state projection. A ratio alone cannot distinguish them.

**Alternative answers:** A real operational improvement is possible only after ruling out definition changes and verifying comparable mature cohorts.

**Follow-up:** What is the safe immediate action? Suppress automated action on the metric and retain the prior reconciled report with an incident note.

**Trap/red flag:** Declaring an operations success because the percentage improved.

**Scoring:** 0 unsupported conclusion; 1 generic “data issue”; 2 causes plus checks; 3 containment and negative-test interpretation.

**Answer source:** F-01, F-02, Section 6.3.

#### Q-COM-09 / Level 3 / selection tradeoff

**Prompt:** A ranking variant has higher paid conversion but lower mature contribution. Should it launch?

**Clarifications:** Eligibility and policy tests pass; no causal design details are supplied.

**Answer:** Not on that evidence alone. First establish whether the conversion difference is causal and whether contribution is measured on comparable mature cohorts with complete costs. The decision requires the predeclared objective, guardrails, uncertainty, and downside tolerance. Lower contribution can disqualify a variant if economics is a hard guardrail.

**Alternative answers:** A bounded experiment may be justified if guardrails and customer protections hold, but it is not a full launch decision.

**Follow-up:** If contribution data are immature, defer the economic conclusion and prespecify a maturity window rather than selecting the favorable early metric.

**Trap/red flag:** Choosing the larger top-of-funnel percentage or treating before/after as lift.

**Scoring:** 0 automatic launch; 1 mentions tradeoff; 2 requires comparable evidence; 3 defines guardrails, maturity, and bounded next step.

**Answer source:** Sections 7.2–7.3 and F-06.

#### Q-COM-10 / Level 3 / recovery design

**Prompt:** A replay duplicates every paid event. How should the projection behave, and what could legitimate deduplication accidentally remove?

**Clarifications:** Order IDs and event IDs are available but partial payments exist in another workflow.

**Answer:** The projection should key event identity and order identity, reject already-applied event IDs, and validate legal transitions. Replaying identical events must leave state and money unchanged. Deduplicating solely on order ID and event type could incorrectly merge legitimate partial payments or refunds. Use the stable economic-event/idempotency identity.

**Alternative answers:** Source sequence numbers can help when their scope and reset behavior are documented.

**Follow-up:** After a crash following an external refund but before checkpoint, reconcile the refund operation ID with the authoritative system before retrying.

**Trap/red flag:** Keeping only the latest arrival by processing timestamp.

**Scoring:** 0 recounts events; 1 mentions dedupe; 2 stable identity and replay invariant; 3 handles legitimate duplicates and crash recovery.

**Answer source:** F-02 and the Section 5.6 state-oracle boundary.

#### Q-COM-11 / Level 3 / evidence verification

**Prompt:** What does a zero reconciliation residual prove, and how would you verify a claimed improvement in conversion?

**Clarifications:** Two pipelines share some upstream code.

**Answer:** Zero residual proves only that compared quantities agree under the implemented mapping. Shared incorrect joins or omitted populations can still reconcile. A claimed conversion improvement requires independent raw-count reconstruction, fixed eligibility and attribution, comparator/assignment validity, uncertainty, and maturity checks. No such improvement is asserted in this sheet.

**Alternative answers:** Cross-checking against an authoritative independent ledger strengthens evidence but does not establish causality.

**Follow-up:** If both pipelines share the same wrong eligibility flag, introduce source-level eligibility traces and sampled manual reconstruction.

**Trap/red flag:** “The dashboard balances, therefore the experiment is valid.”

**Scoring:** 0 overclaims; 1 says arithmetic only; 2 independent reconstruction and design; 3 shared-failure and causal boundary.

**Answer source:** Sections 3.8, 7.1, and 10.2.

#### Q-COM-12 / Level 3 / changed constraint

**Prompt:** A single order now contains two currencies and a partial line refund. Redesign the calculation without forcing it into CM-04 unchanged.

**Clarifications:** The business wants both transactional audit and a consolidated report.

**Answer:** Preserve original-currency line and adjustment amounts first; apply partial refunds to the correct line and adjustment identity. For consolidation, use a dated, authorized conversion rate and rounding policy, recording rate source and conversion timestamp. Separate exchange differences from merchandise adjustments. CM-04 can be applied within each currency or after explicit conversion, but not by naively summing unlike units.

**Alternative answers:** Reporting only original currencies is valid when consolidation is unnecessary.

**Follow-up:** If the rate source is unavailable, leave consolidated contribution OPEN rather than inventing a rate.

**Trap/red flag:** Converting at an undocumented current rate and rewriting historical cohort results.

**Scoring:** 0 sums currencies; 1 mentions conversion; 2 line identity and rate policy; 3 preserves audit, rounding, and unavailable-evidence behavior.

**Answer source:** Sections 4.4 and 9b; CM-04 constraint extension.

**Takeaway:** Strong answers reconcile counts and money first, then defend attribution, maturity, policy, and evidence before recommending action.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Commerce metric tree with visit, candidate, order-state, merchandise, and contribution contracts |
| Core formula | `CVR=N_paid/V`; `M_net=G_item-R_cancel-R_refund`; `CM=M_net+S_in-C_goods-C_fulfill-C_ship-C_other`; all require same declared population/window/currency |
| Key assumption | Entity identity, denominator eligibility, cohort snapshot, state transitions, and component signs are fixed and auditable |
| Complexity | Expected `O(E+V+K+L)` for separate hash-based event, visit, candidate, and line scans; storage includes retained events and indexes beyond the illustrative envelope |
| Use when | Reconciling funnels and unit economics before a merchandising, fulfillment, checkout, or ranking decision |
| Avoid when | Denominators are zero or mixed, refunds/costs are immature, currencies are unconverted, or a descriptive ratio is being sold as causal lift |
| Go deeper | Planned next local sheet: `02_recsys-launch-checklist.md`; recommendation training and causal experiment design remain separate planned topics |
| Diagnostic | Reconstruct key sets, replay duplicates, test CM-04/CM-06 bridges, and inspect availability/policy exclusion traces |
| Evidence | ILLUSTRATIVE values; arithmetic/state oracle PASS for stated cases; original source and formal accounting treatment OPEN; no performance claim |
