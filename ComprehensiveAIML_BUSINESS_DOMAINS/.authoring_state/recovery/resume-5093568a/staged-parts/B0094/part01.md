---
case_id: "CASE-04-02-05"
case_title: "Omnichannel Fulfilment Allocation"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "constructed teaching inputs; arithmetic recomputed in-file; no company data"
as_of_date: "2026-11-10"
owner_role: "order-routing manager"
source_index: "INDEX.md"
review_trigger: "Node cost structure, promise threshold, late-penalty value, or store pick-rate changes"
---

# Omnichannel Fulfilment Allocation

**Evidence disclaimer.** Every quantity in this case is a constructed teaching input. No retailer,
warehouse, carrier, store fleet, or vendor is described, and no measured business result is claimed.
Node counts, wage rates, pick productivities, lead-time distributions, and the pilot outcome are
invented to make the arithmetic inspectable; they are labelled `ILLUSTRATIVE` where they first
appear. The derivations, the probability arithmetic, the break-even solutions, and the assignment
counterexample are recomputed in §6 and are internally consistent, so a reader can re-derive any
printed figure from the inputs listed in §5a. What this case cannot offer is external validity: no
statement here should be read as "stores beat distribution centres" or "splitting pays". Those
conclusions are conditional on the cost and reliability parameters printed below, and §6 shows
exactly where each conclusion flips.

## 1. The One-Line Framing

An order-routing manager must decide, for each open order at the moment the promise clock starts,
which fulfilment node — national distribution centre, one of eleven participating stores, or one of
three dark stores — ships each line, and whether to split the order across two nodes, so as to
minimise economic cost per fulfilled line subject to a stock-available-to-promise rule, a per-node
daily pick capacity, and a 48-hour delivery promise.

The trap is the choice of objective. The routing system already had a score: predicted freight
dollars per order. It is cheap to compute, it is the line item the finance team sees, and it moved
in the right direction whenever a store was chosen. But freight is one of four cost terms. It
excludes pick labour, which is far more expensive per line inside a store than inside a distribution
centre, and it excludes the expected late-delivery penalty, which is where stores earn most of their
advantage. Optimising the visible term and ignoring the two terms that dominate in opposite
directions produces an allocation that looks cheaper and is not.

The reusable pattern is evaluation-mismatch: the convenient score is not the decision objective.
The binding constraint is that any rule the router applies must be computable from data available at
the promise clock, must respect an inventory reservation that stops the same unit being promised
twice, and must degrade to a defensible default when a node's capacity is exhausted.

Takeaway in one sentence: allocation is a constrained economic assignment, and the economic part
must include the expected cost of the promise you are about to make.

## 2. Context & Background

The retailer in this exercise runs a single national distribution centre, eleven stores that
participate in ship-from-store, and three dark stores opened to absorb urban volume. Orders arrive
through one web storefront and one mobile app, with a small marketplace tail that is out of scope
here because its inventory is not owned.

Prior to the change, routing was a fixed rule cascade: if the order's lines were all in stock at the
distribution centre, ship from the distribution centre; otherwise attempt the nearest store with the
full basket; otherwise split the basket across the two nodes with the largest partial coverage.
The cascade was written when there was one distribution centre and no dark stores. It encodes
geography as a proxy for delivery speed, which was approximately true when a single facility served
the whole country on a two-day ground network and became false once urban dark stores could deliver
next day at a lower parcel rate.

Three business pressures made the cascade untenable.

First, the delivery promise moved from a vague three-to-five-day band to a published 48-hour
commitment with a goodwill credit when missed. A promise with a published penalty converts an
operations metric into a financial one: lateness is no longer a service-quality footnote, it is a
line in the profit-and-loss statement, and it must appear in the objective the router optimises.

Second, store labour became measurable. Participating stores clock pick tasks against an order-line
identifier, so lines picked per paid hour is now observable per store per shift. Before that, store
fulfilment cost was an allocated overhead and the router could plausibly pretend it was free. Once
the rate is measurable, the store-versus-centre comparison stops being a matter of policy preference
and becomes arithmetic — the arithmetic in §6, derivation D2.

Third, inventory accuracy diverged across nodes. Store on-hand records drift because of customer
handling, mis-scans, and shrink that is only reconciled at cycle count. A routing decision that
trusts a store's on-hand figure without a reserve is a promise made against inventory that may not
exist, and a substituted or cancelled line costs more than the freight it saved.

The decision cadence is continuous: routing runs on every order at promise creation, with a nightly
re-planning pass that re-solves the next day's committed volume against refreshed capacities. The
owner is the order-routing manager, who also owns the override queue and the escalation path when a
node fails mid-shift.

The pre-existing baseline is not a straw man. It is the cascade described above, it has run for
years, it is understood by every store manager, and it can be reinstated by flipping one
configuration flag. Any replacement must beat it on the economic objective and must not degrade the
promise-hit rate, or the replacement is not worth the operational risk of a new failure surface.

One structural fact shapes everything that follows: the store fleet is small in capacity relative to
demand. Eleven stores at 210 picked lines per day each give 2,310 store lines of daily capacity
against 5,520 lines of daily demand — about 42%. The three dark stores add 840 lines. Allocation is
therefore a rationing problem, not a matching problem: most lines must go to the distribution
centre, and the interesting question is which minority of lines stores should take.

## 3. The Business Question

**Question.** For the set of order lines open at a promise clock *t*, which lines should be assigned
to which node, and which orders should be split across two nodes, so that total economic cost per
fulfilled line is minimised while no node exceeds its remaining capacity for the day and no line is
promised against unreserved inventory?

**Decision owner.** The order-routing manager, who signs the policy and holds the override budget.
The manager is accountable for three numbers: economic cost per fulfilled line, promise-hit rate,
and the fraction of lines routed to a node whose recorded on-hand later proved wrong.

**Trigger.** Continuous, once per order at promise creation; plus a nightly re-plan of the following
day's committed volume; plus an exception trigger whenever a node's remaining capacity falls below
the volume already committed to it, which happens when a store's pick rate degrades mid-shift.

**Available actions.** Four, and they must be genuinely available, not decorative:

1. Hold the baseline cascade for this order — the default and always legal action.
2. Assign all lines of the order to a single node from the eligible set.
3. Split the order across two nodes when the split's extra parcel cost is exceeded by the lateness
   value of moving the covered lines to a faster node (the threshold is derived in §6, D4).
4. Defer the promise by one day, or hold the order for manual review, when no eligible node has
   both stock-available-to-promise cover and remaining capacity.

**Objective.** Minimise the sum over lines of economic cost per line, where economic cost has four
terms: pick labour at the node's own wage and productivity, per-line handling consumables, the
node's allocated freight, and the node's expected late-penalty contribution. The last term is a
probability times a penalty, not a service-level constraint, because the penalty is paid in cash.

**Hard constraints.** Four, each of which can veto an otherwise attractive assignment:

- Inventory reserve: a line may be promised only against on-hand minus reservations minus safety
  stock. Violating this converts a cheap assignment into an expensive cancellation.
- Capacity: each node has a remaining daily pick capacity; assignments compete for it, so the
  problem is a packing problem and the greedy nearest-node rule is not optimal (§6, D6).
- Data clock: no field timestamped after the promise clock may enter the decision, including the
  current shift's pick-completion record and any same-day inventory adjustment.
- Explainability and reversal: every assignment must name the node, the binding constraint that
  allowed it, and the alternative that was rejected; a supervisor must be able to re-route an order
  within the promise window without breaching another promise.

**Scope.** One decision window, one day of orders at 2,400 orders and 2.30 lines per order, eleven
stores, three dark stores, one distribution centre, one carrier contract with two service tiers.

**Out of scope.** A claim of deployed financial impact; carrier renegotiation; store labour
scheduling; the marketplace channel; any legal determination about promise wording. The case
evaluates a routing policy, not a company.

**Why the question is hard.** Three reasons, each of which produces a different kind of error.

The first is that the objective's largest term is a probability-weighted penalty, so the router must
carry a distributional belief about pick-completion time per node, not a point estimate. A node with
a faster mean but a fat right tail can lose to a node with a slower mean and a tight tail.

The second is that capacity couples decisions that look independent. Assigning one more line to a
store consumes a scarce slot that a higher-margin order might need two hours later. Without a shadow
price on capacity, the router systematically spends the fleet's best slots on the first orders it
sees rather than the best orders.

The third is that the visible metric and the decision objective disagree in sign for specific order
shapes. For a one-line order, splitting loses money at any realistic pick rate (§6, D4 boundary).
For a six-line order containing two heavy items and four light ones, splitting can win because the
light parcel moves to a fast node while the heavy items stay at the centre where handling is cheap.
A single average-cost model cannot express that difference, which is why §8 routes on order shape.

Takeaway: the answer is a policy, not a per-order optimum — a policy is what survives the capacity
constraint, the inventory clock, and the shape dependence simultaneously.
