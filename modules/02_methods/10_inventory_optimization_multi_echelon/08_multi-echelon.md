# Multi-Echelon Inventory, MEIO, and DRP — echelon stock, deployment, and where the buffer belongs

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 10 topics 04, 06, and 07
> Version 2026.5

---

## 1. Why this topic exists

Every other topic in this module optimises one item in one place. This one optimises the same item
in several places at once, and that change of scope is not an increment. It changes what the
numbers mean, what a service target refers to, who may decide, and what counts as a good answer.

The gap is visible in a live planning team. Each regional distribution centre (RDC) has an
inventory policy, computed correctly, reviewed monthly, signed off by a planner. The central
distribution centre (DC) has one too. Nobody has a policy for the network. The consequences do not
appear on any single item card, which is why they survive for years.

Three failures recur, and all three are network failures rather than calculation failures.

1. **The compounding target.** Management sets a 95 per cent cycle service level (CSL) at every
   stocking location, because 95 per cent is the contract number. Three stages later the customer
   experiences something well below it, while every stage shows a report proving it hit its own.
2. **The double buffer.** The RDC buffers against its lead time and the central DC against its own,
   both covering the same uncertainty at different points in one pipeline, so the network holds
   roughly twice the protection and justifies it twice.
3. **The deployment argument.** A container arrives with 2,000 packs against requests of 2,100. The
   split is decided in a meeting, by seniority, and next month every RDC orders more than it needs.

This topic builds the vocabulary and the mechanics to see all three. It is the first topic in the
module where the unit of analysis is a graph rather than a stock-keeping unit (SKU) at a location.

**What the topic is not.** It does not design the network: Module 11 owns where facilities should
be and how many. This topic takes the node set and the lane set as given and asks how much
inventory to hold at each node and how to move it. It does not derive pooling: Module 10 topic 09
owns risk pooling and postponement, and section 5.6 uses that theory rather than re-deriving it.
And it does not re-derive policy families; topic 06 established the four families, and this topic
composes them across levels.

**Why an analytics team gets asked.** The question arrives as "tell us how much inventory to hold
across the network", which sounds like a forecasting question and is not. It is a decomposition
question. The data required is a graph, a lead-time distribution per arc, a demand distribution at
the leaves, a cost or service target at the boundary, and the policy families already running. Four
of those five are usually wrong or missing, and the analyst who cannot name which one is wrong will
spend a quarter fitting a model to a network that does not exist.

> **Note:** This file uses "service level" for three distinct objects: the external measure at the
> customer boundary, the internal measure a stage promises the stage below it, and the target
> written into a contract. Topic 05 defines the measures. Here the discipline is only that every
> number is labelled with which of the three it is.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** the end-to-end availability implied by stage-by-stage service targets, **state** the
   assumption that makes the multiplication valid, and **explain** where it fails.
2. **Convert** a physical inventory position into installation and echelon stock for every node in a
   two- or three-level network, and **show** one shelf producing two numbers.
3. **State** the Clark–Scarf result at its actual strength, assumptions named, and **list** what
   fails in a distribution network.
4. **Draw** a network as a directed graph with a leaf set, a boundary measure, and a lead time per
   arc, and **identify** the decomposition boundaries.
5. **Explain** what multi-echelon inventory optimisation (MEIO) buys, what it cannot buy, and why
   vendor and textbook descriptions of the acronym differ.
6. **Build** a distribution requirements planning (DRP) table by hand across three levels and four
   buckets, and **distinguish** what it computes from what an optimisation decides.
7. **Choose** a deployment rule and **predict** how it changes downstream ordering behaviour.
8. **Recommend** a decoupling-point placement, trading response time against pooling benefit, and
   **justify** it by mechanism rather than outcome.
9. **Audit** a network buffer to the four-part standard and **identify** which element is missing.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures: the same physical inventory counted two ways, and a ladder of service targets
multiplying down to a customer who experiences none of them.

### 3.1 The same units, two numbers

A central DC supplies one RDC. The central DC has 620 packs on hand, 180 are in transit, and the
RDC has 240 on hand. Nothing else exists in this item.

```text
   INSTALLATION VIEW — each location reports only what it owns

        +----------------------+   180    +----------------------+
        |     CENTRAL DC       |  transit |      REGIONAL DC     |
        |     on hand   620    | -------> |      on hand   240   |
        +----------------------+          +----------------------+

        central DC reports ........  620 packs
        regional DC reports .......  240 packs
        sum of the reports ........  860 packs   (plus 180 in transit, unowned)


   ECHELON VIEW — each location reports what it owns plus everything downstream

        +----------------------+   180    +----------------------+
        |     CENTRAL DC       |  transit |      REGIONAL DC     |
        |  echelon stock 1,040 | -------> |  echelon stock  240  |
        +----------------------+          +----------------------+
                    ^                              ^
              620 + 180 + 240                  240 + 0 downstream

        central DC reports ...... 1,040 packs   (the whole network below it)
        regional DC reports .....   240 packs   (its own, nothing below)
        network total ........... 1,040 packs   (counted once, at the top)
```

The physical inventory is 1,040 packs in both pictures. What changes is the allocation of that
1,040 across the two reporting lines: installation stock says 620 and 240, echelon stock says 1,040
and 240. Ask "how much stock is at the central DC" and the two conventions differ by 68 per cent,
and both are defensible statements about the same shelf.

Three properties follow. **Echelon stock is additive up the chain and installation stock is not**:
sum installation stocks and you get the network total only if you also add every in-transit unit at
the right owner, while summing echelon stocks over-counts by the number of levels. The network total
is the top node's echelon stock. **The policy a node can run depends on which number it can see**: an
installation policy compares its own position to its own trigger, an echelon policy compares its
echelon position to an echelon trigger, which requires knowing what is downstream. That is an
information requirement before it is a modelling choice. **The lead time that matters changes with
the convention**: installation exposure includes upstream waiting while that node is out of stock,
echelon exposure is transport time, and the difference is the effective lead time.

### 3.2 The service ladder

```text
   stage          its own CSL        cumulative probability that
                                    every stage so far served
   ------------   -----------        --------------------------
   plant / CDC       0.95            0.95
   regional DC       0.95            0.95 x 0.95 = 0.9025
   local DC          0.95            0.95 x 0.95 x 0.95 = 0.8574

   customer sees ....  85.7 per cent, while every report shows 95 per cent
```

That multiplication is correct arithmetic and a correct answer to one question: the probability
that three independent stages each avoid a stock-out in the same cycle. It is not the end-to-end
fill rate of a network, and treating it as one is the most common conceptual error in this topic.
Where a downstream node holds stock that can absorb an upstream delay, a stage-level shortage does
not automatically reach the customer; the customer experiences a delay only if the cumulative
shortfall exceeds every buffer between the failure and the delivery. The end-to-end measure is then
a function of the whole vector of stock levels, not a product of three numbers. Diks, de Kok and
Lagodimos (1996) review service measures across echelons and argue that internal upstream service
levels are, in their phrase, in some sense irrelevant, because the external fill rate is determined
jointly by all echelon stock levels. Van Houtum, Inderfurth and Zijm (1996) give the serial-system
fill rate as a convolution over all echelons rather than a product.

So the honest reading is that the product is a **lower bound on the failure probability under
independent local buffering** — the case where no stage can draw on another's stock. It is the
right arithmetic when you set an internal promise between two stages and the receiving stage holds
no slack to absorb a breach. It is the wrong arithmetic when you use it to conclude that the only
fix is raising every stage's target, which is what organisations usually do, at three times the
necessary cost.

### 3.3 The network as a graph

```text
                     leaves = demand points, where the service measure lives

        [ P1 ] --3wk--> [ CDC ] --1wk--> [ RDC-N ] --2d--> [ store cluster N ]
                          |    \
                          |     --2wk--> [ RDC-S ] --2d--> [ store cluster S ]
                          |
                          --3wk--> [ RDC-W ] --3d--> [ store cluster W ]

        nodes: P1, CDC, RDC-N, RDC-S, RDC-W, three store clusters   (7)
        arcs:  six replenishment lanes, each with a lead time       (6)
        leaves: the three store clusters; demand originates only there
        boundary: the service measure at the leaves is what the network is judged on
        internal: every other service measure is a construction of the model
```

Five things decide whether a MEIO project will work. **Direction**: demand is only at leaves, and
demand recorded at an internal node means the graph is wrong or the node is a leaf in disguise.
**Acyclicity**: a returns lane, an inter-RDC transfer, or a supplier who is also a customer creates
a cycle the optimiser cannot represent. **The lead time on each arc is a distribution, not a
number**, which is where most projects die. **Which node holds the buffer is a decision variable**,
not a given; section 5.6 addresses it. **The graph changes**: a solution is valid only for the graph
it was solved on, which is why every buffer report carries a graph version.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Multi-echelon inventory | Inventory held at two or more levels of one network, where a decision at one level affects stock at another | Used to mean "inventory in several warehouses" | Level count plus network identifier |
| Echelon | A level of the network relative to the customer; level 0 is closest to demand | Confused with supplier tier | Ordinal, counted upstream from the leaf |
| Node | A stocking or transforming location in the network graph | Treated as a warehouse only | Location code, typically a GLN |
| Arc | A directed replenishment lane between two nodes, with a lead time | Treated as a transport route | Origin–destination pair |
| Leaf | A node with demand and no downstream node | Assumed to be every node | Node identifier |
| Installation stock | Inventory physically at one node: on hand there, plus on order to there | Reported as the network position | Units |
| Echelon stock | Inventory at a node plus all inventory downstream of it, in transit and on hand | Computed as on hand only, omitting transit | Units |
| Installation inventory position | Installation on-hand plus on-order minus backorders at that node | Used as the trigger input for an echelon policy | Units |
| Echelon inventory position | Echelon stock plus on-order to that echelon minus customer backorders | Computed without the in-transit term | Units |
| In-transit stock | Units shipped from one node and not yet received at the next | Omitted from echelon stock | Units, with a lane identifier |
| Pipeline stock | All units in motion plus all buffers along a lane | Confused with in-transit stock | Units |
| Installation lead time | Time from placing an order at a node to receiving it, including upstream queuing | Assumed equal to transport time | Days or weeks |
| Echelon lead time | Time from releasing an order into an echelon to its availability at the leaf | Confused with installation lead time | Days or weeks |
| Effective lead time | Installation lead time plus expected waiting caused by upstream stock-outs | Ignored in installation models | Days, as a random variable |
| MEIO | Multi-echelon inventory optimisation: choosing stock levels across a network against a cost or service objective | Used to mean any multi-location calculation | Method, with a named objective |
| DRP | Distribution requirements planning: time-phased netting and offsetting of requirements across distribution levels | Used to mean the deployment decision | Method |
| DRP table | The time-phased record of gross requirement, scheduled receipt, projected on hand, and planned order release | Treated as an optimisation output | Table, one row per bucket |
| Gross requirement | Demand or dependent requirement in a bucket, before netting | Confused with net requirement | Units per bucket |
| Scheduled receipt | Firm supply already in the pipeline, dated to arrive in a bucket | Confused with planned order receipt | Units per bucket |
| Projected available balance | On hand at the start of a bucket, rolled forward by receipts minus requirements | Reported as current on hand | Units |
| Net requirement | The shortfall after netting available stock and scheduled receipts, including safety stock | Omitted when safety stock is a separate field | Units |
| Planned order receipt | The quantity the plan intends to arrive in a bucket | Treated as a commitment | Units per bucket |
| Planned order release | The planned receipt offset backwards by the lead time | Confused with the purchase order | Units per bucket |
| Lead-time offset | Placing a requirement `L` buckets earlier than its need date | Applied to the receipt instead of the release | Buckets |
| Deployment | Deciding how available supply is split across downstream nodes | Confused with allocation under shortage | Units by destination |
| Allocation | Splitting a scarce quantity across competing requests | Used for both ample and scarce supply | Units by destination |
| Fair share allocation | Allocation proportional to each destination's projected need | Used loosely for any proportional rule | Units |
| Priority allocation | Allocation by a rank order such as contract or customer tier | Assumed to be fair | Rank order |
| Demand-weighted allocation | Allocation proportional to forecast demand over the deployment window | Confused with fair share | Units |
| Turn-and-earn | Allocation proportional to each destination's past sales share | Assumed to reward performance | Percentage share |
| Uniform allocation | Equal split of available supply up to each request | Confused with fair share | Units |
| Proportional allocation | Each request scaled by the same fill factor | Assumed to be truth-inducing | Ratio |
| Shortage gaming | Inflating orders to protect allocation under a rationing rule | Diagnosed as forecast bias | Units of inflation |
| Rationing rule | The published rule for splitting scarce supply | Left undocumented | Rule name plus parameters |
| Internal service level | The service a stage promises the stage below it | Reported as customer service | Proportion |
| External service level | The service measured at the leaf, against end demand | Reported per stage | Proportion |
| End-to-end fill rate | Fraction of leaf demand served without delay, through all echelons | Computed as a product of stage rates | Proportion |
| Guaranteed service time | A promised maximum replenishment time from a stage, used as a modelling device | Treated as a physical constraint | Days |
| Net replenishment lead time | Guaranteed inbound time plus processing time minus guaranteed outbound time | Set negative by mistake | Days, must be non-negative |
| Decomposition | Splitting a network problem into per-stage problems linked by service times or echelon penalties | Treated as an approximation without loss | Method |
| Serial system | Nodes in a single chain, one supplier each | Confused with a distribution system | Topology |
| Distribution system | One upstream node serving several downstream nodes | Confused with an assembly system | Topology |
| Assembly system | Several upstream nodes feeding one downstream node | Confused with a distribution system | Topology |
| Imbalance | A node's echelon position falling after rationing because upstream could not fill | Invisible in installation reports | Units |
| Decoupling point | The node where stock breaks the dependency between downstream demand and an upstream lead time | Confused with the push–pull boundary | Node identifier |
| Postponement | Delaying differentiation until closer to demand | Confused with delaying delivery | Stage |
| Pooling benefit | The reduction in required buffer from aggregating independent demand | Assumed to be free | Units or percentage |
| Network graph version | The dated identifier of the node, arc, and lead-time set a solution was computed on | Omitted from reports | Version string |
| Buffer echelon | The level at which a stated buffer physically sits and is counted | Left implicit | Echelon number |
| Protection period | The span a buffer covers at its echelon | Assumed equal to the lead time | Days |

Four rows carry most of the confusion. **Echelon stock versus installation stock** is a choice
between two accounting conventions over one physical reality, and it is not cosmetic: a planner who
sees 620 packs will re-order for demand the planner who sees 1,040 knows is already covered, and the
double order is invisible because each order is locally justified. **Fair share versus
proportional**: fair share allocates against need, so a destination already holding stock receives
less; proportional scales every request by one factor, so a destination that over-ordered receives
more, and only one of the two is stable against gaming. **Deployment versus allocation**: deployment
places ample supply where it will be consumed; allocation distributes a loss. Systems often use one
rule for both, which is how a fair deployment rule becomes an unfair rationing rule.
**Internal versus external service level**: an internal level is a construction of the model, not a
customer promise, and collapsing the two pays for protection at every level while still missing
end-to-end.

## 5. The mechanics, step by step

The procedure is written as a project you run, in the order that keeps the project alive. Each step
states its input, its output, and the branch where projects go wrong.

### Phase A — Frame the network

**Step 1. Fix the graph and give it a version.** Input: the node list with location codes and the
lane list with origin, destination, and effective date. Output: a directed acyclic graph, a leaf
set, and a version string such as `NET-2026-Q4-r3`. Branch: **is any node both a source and a
destination for the same item?** A transfer between RDCs, a return lane, or a repair loop creates a
cycle. Exclude it, model it as a separate flow, or accept that the optimiser cannot represent it,
but do not silently drop the arc: buffers will be sized for a network that does not exist.

**Step 2. Classify the topology.** Output: serial (a chain), distribution (one source, many sinks),
assembly (many sources, one sink), or general. Branch: **which topology is it really?** Most real
networks are general, and most solvers are exact only on trees. A general network is solved by
enumeration, coordinate descent, or a mathematical programme, and the quality of the answer depends
on the method. Say which. "MEIO" is not a statement about the algorithm.

**Step 3. Put demand only at the leaves.** Output: a demand series per leaf and a statement of
what each series measures. Branch: **is demand recorded at an internal node?** If a central DC has
"demand", it is either a leaf with its own customers or it is recording shipments to downstream
nodes, which are derived demand and must not be forecast as independent. The distinction changes
the variance, and therefore every buffer downstream.

**Step 4. Name the boundary and internal service measures separately.** Output: one external
measure at the leaves with a target and a measurement window, and a separate internal target per
internal arc. Branch: **is the external measure a CSL, a fill rate, or a time-based measure?**
Topic 05 owns the definitions. The point here is that internal targets are derived from the
external one rather than copied from it, and the derivation is the part most projects skip.

### Phase B — Choose the accounting convention and the policy composition

**Step 5. Decide whether each level's policy is installation-based or echelon-based.** Output: a
convention per node with its reason. Branch: **can the node see downstream stock?** If the RDC's
stock is not visible to the central DC's planning engine in the same run, an echelon policy cannot
be executed there, whatever the design says. The practical answer is usually a hybrid: echelon
logic at the top, installation logic at the leaves, documented as such.

**Step 6. Compose the policy families across levels.** Input: the family chosen at each node from
topic 06. Output: a per-node family and a statement of which node's order is the input to which
other node's demand. Branch: **does the upstream protection period depend on the downstream review
period?** It does, always. A weekly review at three RDCs produces a weekly, lumpy order stream at
the central DC, so its demand is not the sum of leaf demand but the sum of reviews. Its buffer must
be sized on the order stream it receives, or the network must move to an echelon policy.

**Step 7. Fix the protection period at each echelon.** Output: `P = L` under continuous review,
`P = R + L` under periodic, per node. Branch: **is the lead time the graph's or the receipts'?**
Use the realised distribution. A published lead time of 14 days with a 90th percentile of 26 days
is a 26-day network with a 14-day document.

### Phase C — Validate the inputs before solving

**Step 8. Build the lead-time distribution per arc.** Input: order and receipt timestamps per lane,
at least twelve months, excluding or flagging known disruption. Output: mean, standard deviation,
and the 50th, 90th, and 95th percentiles per arc. Branch: **is the mean being used as if it were
the lead time?** A buffer sized on the mean covers half the arrivals. Topic 07 derives the correct
treatment; the requirement here is that the distribution exists and is per arc, not per node.

**Step 9. Check for demand censoring.** Output: a demand series corrected for the days on which
the item was unavailable. Branch: **were leaf sales limited by stock?** If so, the recorded series
understates demand and its variance, and every buffer computed from it is too small. This is the
most common silent error in network models, because the data looks clean.

**Step 10. Check the cost and service inputs at the boundary.** Output: a stated objective with
its units. Branch: **is the objective cost or service?** A cost-minimising model needs a shortage
penalty the business will defend. Most organisations cannot produce one, so the model runs as
service-constrained cost minimisation with the target at the leaves. Say which, because the two
produce different buffers at the same nominal target.

### Phase D — Decompose and solve

**Step 11. Decompose the network into stage problems.** Output: one subproblem per stage, linked
either by an internal service time (guaranteed-service style) or by an echelon penalty passed
downstream (Clark–Scarf style). Branch: **which decomposition?** The guaranteed-service form gives
each stage a promised outbound time and computes a net replenishment lead time as inbound promise
plus processing minus outbound promise. The Clark–Scarf form passes a penalty for downstream
shortage upstream and solves the stages in sequence. Both are exact only where their assumptions
hold, and section 5.2 states what Clark and Scarf assumed.

**Step 12. Solve stage by stage, from the leaves upstream.** Output: a stock level per node and an
internal service time or penalty per arc. Branch: **did the solve converge or enumerate?** On a
tree, dynamic programming is exact. On a general network, coordinate descent can stall at a local
optimum. Report the method and whether the objective is a bound or an achieved value.

**Step 13. Re-express the solution in the units the business uses.** Output: a target per node, in
units, with its echelon, its protection period, its service measure, and the graph version.
Branch: **is the target stated in echelon or installation units?** A target of 1,040 at the
central DC means one thing under echelon accounting and something else under installation
accounting. The report must say which, and a second analyst must be able to reproduce the number.

### Phase E — Deploy through DRP

**Step 14. Convert network targets into time-phased requirements.** Input: the per-node targets,
opening inventory, firm receipts, and the leaf forecast. Output: a DRP table per node, as in
section 7. Branch: **is the table being used as a plan or as an optimiser?** DRP nets and offsets;
it does not choose the target, which comes from step 13. A DRP table run against levels nobody
optimised is a very precise execution of a bad policy.

**Step 15. Establish the deployment rule before the first shortage.** Output: one published rule
with its parameters and an exception path. Branch: **is the rule the one the system implements or
the one the meeting uses?** Measure the last twelve shortage events and reconstruct the realised
rule from the quantities. That is the rule the RDCs are responding to, and therefore the rule
driving their orders.

**Step 16. Publish the rule to the destinations.** Output: a one-page statement of how scarce
supply will be split, sent to every RDC planner. Branch: **is the rule truth-inducing?** If a
destination can improve its allocation by inflating its order, it will, and the inflation appears
next quarter as forecast bias. The rule is a behavioural instrument, not only a fair-division one.

### Phase F — Maintain and report

**Step 17. Set the re-solve cadence and the trigger events.** Output: a scheduled interval plus
the events that force an out-of-cycle re-solve. Branch: **what counts as a network change?** A new
lane, a closed node, a lead-time shift beyond one standard deviation, a change in the leaf set, or
a change in the contractual service measure. Each invalidates the solution, and none announces
itself.

**Step 18. Write the buffer record in the four-part standard.** Output: for every buffered node, a
line stating the echelon, the protection period, the service measure with its target and window,
and the graph version. Branch: **can a reader reproduce the number?** Without the protection period
the buffer is unfalsifiable. Without the graph version nobody can tell whether it is stale. Both
omissions are normal and both are defects.

**Step 19. Instrument the end-to-end measure.** Output: a monthly end-to-end fill rate with lost
sales decomposed by the stage that caused them. Branch: **can a miss be attributed to a stage?**
Without the decomposition, every stage reports success and the network reports failure, which is
exactly the condition described in section 1.

### 5.1 Why stage-by-stage optimisation is not a network optimum

Each node optimising its own buffer against its own lead time and demand is locally correct. The
network is not optimal for three reasons: the same leaf uncertainty is buffered twice, once by the
downstream node and again by the node above it; the order stream an upstream node sees has higher
variance than the demand it faces, because the downstream review cycle amplifies it; and the lead
times are not independent, since an upstream stock-out lengthens the downstream lead time beyond
transport. The decomposition in step 11 addresses all three by keeping the coupling in the
objective, so the stages can be solved separately without discarding it.

### 5.2 What Clark and Scarf actually proved

The result is often stated as "echelon stock policies are optimal in a serial system". That is
close but misses the conditions, and the conditions are the useful part.

Clark and Scarf (1960), "Optimal Policies for a Multi-Echelon Inventory Problem", *Management
Science* 6(4), 475–490, considered a serial system numbered from the customer end, where the
delivery lead time at an installation depends not only on transport but on whether the upstream
installation has stock. That second dependency is why the problem is hard and why the paper matters.
They worked under four assumptions:

1. Demand originates only at the lowest installation.
2. Purchase and shipping costs are linear, with a fixed setup cost permitted only at the highest
   installation.
3. The one-period holding and shortage cost at each level is a function of **echelon** stock.
4. Each echelon backlogs excess demand rather than losing it.

Under those assumptions they showed that an echelon-stock (s,S)-type policy is optimal and — the
part that made the paper foundational — that the multi-installation problem decomposes into a
sequence of single-installation problems solved recursively from the bottom up. Each stage's
problem is a standard single-location problem with one extra term: a convex penalty charged to the
upstream stage for the expected cost its own shortage imposes downstream. That penalty carries the
coupling a stage-by-stage optimisation omits.

What the result does **not** say, and what is routinely over-claimed:

- It does not say echelon policies are optimal in a distribution network. The paper's Section IV
  discusses shared-supplier systems and calls the results there less satisfactory. The distribution
  case has a rationing problem the serial case does not: a short central node must split what it
  has, and that decision feeds back into the downstream policies.
- It does not say echelon policies dominate installation policies. Axsäter and Rosling (1993)
  showed that for order-up-to policies the two classes are equivalent, while for (s,nQ) the
  conversion works one way only; Axsäter and Juntti (1996) report echelon policies performing worse
  in distribution systems, for rationing and imbalance reasons.
- It does not survive a non-convex cost structure, such as a large fixed stock-out penalty or a
  quantity discount, nor non-stationary demand, nor a graph that changes during the horizon.

> **Note:** The four assumptions are reported as they appear in the paper and in secondary
> summaries returned by search. The decomposition result is Theorem 1 and Theorem 2 in the
> original. Where this file says "optimal", it means optimal within the model, not for the business.

### 5.3 MEIO as it is practised

The acronym covers at least three computational objects, and the difference matters when you buy
one.

| What is being solved | Method | Exact on | Output |
|---|---|---|---|
| Safety stock placement under service-time promises | Guaranteed-service model: each stage promises an outbound time; buffer scales with the square root of net lead time | Trees: serial, assembly, distribution | A service time and a buffer per stage |
| Base-stock levels under holding and shortage costs | Stochastic dynamic programme with echelon decomposition | Serial systems, under the Clark–Scarf assumptions | An echelon (s,S) per stage |
| Levels under a fill-rate constraint at the leaves | Constrained optimisation, often a quadratic or mixed-integer reformulation | Divergent networks with the right structure | Stock levels meeting a boundary target |

The guaranteed-service family is what most commercial tools implement, because it turns a stochastic
problem into a deterministic optimisation over service times. A stage's buffer depends on its **net
replenishment lead time**: guaranteed inbound time plus processing time minus guaranteed outbound
time. If that is negative the model is infeasible, because the stage promises to deliver faster than
it can be replenished, and setting those promises is the optimisation. Two tools that both say
"MEIO" may therefore solve different problems, one exact only on a topology your network lacks.

### 5.4 DRP as the execution engine

DRP is material requirements planning logic applied to a distribution network. The arithmetic is the
same at every level: roll the projected available balance forward one bucket, net the gross
requirement against it, add the safety stock target, size by the lot rule, and offset the receipt
backwards by the lead time to get a release date. The release at one level becomes a requirement at
the level above.

Four properties distinguish DRP from an optimiser. It is **deterministic**: one number per bucket,
so it executes a target rather than choosing one. It is **time-phased**: a requirement in week 9 is
released in week 3, and the past-due release appears when cumulative lead time exceeds the horizon.
It is **auditable by hand**: every number is one addition or subtraction from the number to its
left. And it **exposes the cascade**: when the number at the plant looks wrong, you can follow it
down in four steps.

The complementary weakness is equally clear. DRP will plan to a target that no cost or service
calculation supports, with complete internal consistency. It answers "given these targets, what
should ship when", not "what should the targets be".

### 5.5 The deployment rules, compared

When supply is adequate, deployment is a placement problem. When it is short, the same mechanics
become rationing, and the rule chosen changes behaviour. Six rules cover almost all practice.

| Rule | Allocation basis | Stable against order inflation | Who gains | Where it fails |
|---|---|---|---|---|
| Proportional to request | Every request scaled by one factor | No | Whoever orders most | Rewards inflation directly |
| Uniform | Equal split, capped at each request | Yes | Small, honest destinations | Perceived as unfair; ignores need |
| Fair share | Proportional to projected need after netting | Partly | Destinations that sold through | Needs accurate positions |
| Priority | Rank order by contract or customer tier | Yes, if the rank is fixed | Contracted customers | Politically expensive; ignores demand |
| Turn-and-earn | Proportional to past sales share | Partly | Historically large destinations | Punishes destinations censored by stock-outs |
| Demand-weighted | Proportional to forecast over the window | No | Destinations with optimistic forecasts | Rewards forecast inflation |

Cachon and Lariviere (1999) show analytically that proportional and linear rules give each retailer
less than it ordered when capacity binds, which makes order inflation individually rational and can
leave the game without an equilibrium, while a uniform rule supports a unique equilibrium with
truthful ordering. Spiliotopoulou, Donohue and Gürbüz (2022) find experimentally that only the
uniform rule induces truthful orders, with proportional and linear rules producing larger and more
frequent order adjustments. Spiliotopoulou and Conte (2022) add the fairness dimension: the most
prevalent fairness ideal under shortage is equal fill rates, which is the proportional rule, and
retailers rate proportional and linear allocations as fairer than uniform ones. The two findings
sit together uncomfortably and both are true: the rule that stops gaming is the rule the
destinations consider unfair.

### 5.6 Where the buffer belongs

The decoupling-point question at network level has three canonical answers, and they trade response
time against pooling.

```text
   MORE POOLING                                        MORE RESPONSE
   cheaper buffer,                                  shorter promise,
   longer promise,                                  buffer held per variant,
   variant risk pooled                              variant risk carried

   [ component ] ---------> [ sub-assembly ] -------> [ finished goods ]
   at plant or CDC           at CDC or RDC             at RDC or store
        |                         |                          |
   one buffer covers          buffer covers              buffer covers
   many variants              families                   one variant
        |                         |                          |
   response = full             response =                  response =
   manufacturing +             assembly +                  pick and ship
   transport                   transport                   only
```

| Placement | Response time | Pooling benefit | Obsolescence exposure | Typical fit |
|---|---|---|---|---|
| Component at the plant or central DC | Longest: make plus transport | Highest: one buffer across all variants | Lowest: components age slowly | Long lead times, many variants, high configuration risk |
| Sub-assembly at the central DC | Medium: assembly plus transport | High: buffer across a family | Medium | Configure-to-order, modular products |
| Finished goods at the RDC | Short: pick and ship | Low: one buffer per variant per region | Highest: variant-specific | Short promises, stable assortment |
| Finished goods at the store | Shortest: immediate | None | Highest, plus shelf cost | Immediate consumption, low value density |

The trade is a ratio, not a preference. Moving the buffer upstream multiplies the pooling benefit
by the number of variants it covers and adds the downstream lead time to the customer promise.
Moving it downstream divides the pooling benefit and subtracts that lead time. The correct
placement is where the marginal pooling saving equals the marginal cost of the longer promise.
Module 10 topic 09 derives the pooling arithmetic; this file uses it.

Two network-level complications are absent from the single-item version. First, the decoupling point
can differ by echelon within one network: components at the plant, finished goods at the central DC,
and a small assortment at the RDCs is three decoupling points for one item, each needing its own
buffer record. Second, the placement interacts with the allocation rule. With the buffer at the
central DC every shortage becomes an allocation decision across RDCs; with the buffer at the RDC the
central DC is mostly a pass-through, the allocation question is smaller, and the pooling benefit is
gone. The decoupling decision and the allocation decision are the same decision viewed from two ends.

## 6. Worked example

A synthetic but internally consistent case. Halden Medical Distribution supplies sterile procedure
packs to hospital customers through three RDCs fed by one central DC, which a plant replenishes.
All figures are constructed for teaching; none is a claim about a real company.

### 6.1 The network and its inputs

One SKU, `4410-B`, unit of measure: pack.

| Node | Role | Lead time | Review | Weekly demand mean | Weekly demand SD |
|---|---|---|---:|---:|---:|
| Plant | Makes to stock | 1 week | — | — | — |
| Central DC | National buffer | 2 weeks from plant | Weekly | — | — |
| RDC North | Regional | 5 days from CDC | Weekly | 420 | 105 |
| RDC South | Regional | 7 days from CDC | Weekly | 300 | 90 |
| RDC West | Regional | 10 days from CDC | Weekly | 180 | 72 |

Aggregate leaf demand is 900 packs per week. The aggregate standard deviation, assuming
independence across regions:

```text
   sqrt(105^2 + 90^2 + 72^2) = sqrt(11,025 + 8,100 + 5,184)
                             = sqrt(24,309) = 155.9 packs per week
```

Service is a 95 per cent CSL at the leaves, so `z = 1.645`. No shortage penalty was available, so the
model is service-constrained cost minimisation and the service target is the input.

### 6.2 Stage-by-stage buffering

Each node sizes its own buffer against its own protection period and its own observed demand. For
the RDCs that is leaf demand. For the central DC it is the order stream from the RDCs, which is not
the same series.

| Node | `P = R + L` (weeks) | Sigma used (packs/wk) | Buffer (packs) |
|---|---:|---:|---:|
| RDC North | (7 + 5) / 7 = 1.714 | 105 | 226 |
| RDC South | (7 + 7) / 7 = 2.000 | 90 | 209 |
| RDC West | (7 + 10) / 7 = 2.429 | 72 | 185 |
| Central DC | (7 + 14) / 7 = 3.000 | 230 (order stream) | 655 |
| Network total | | | **1,275** |

The arithmetic for two of them, so the method is reproducible:

```text
   RDC North:  sigma over P = 105 x sqrt(1.714) = 105 x 1.309 = 137.4
               buffer       = 1.645 x 137.4 = 226.0 packs

   Central DC: sigma over P = 230 x sqrt(3.000) = 230 x 1.732 = 398.4
               buffer       = 1.645 x 398.4 = 655.4 packs
```

The central DC's sigma of 230 is an **observed** order-stream standard deviation, not the aggregate
of 155.9: three weekly review cycles turn smooth demand lumpy, an amplification of 1.48. The
655-pack central buffer and the 620 packs of RDC buffer below it protect the same uncertainty twice.

### 6.3 The echelon allocation

Under an echelon policy the central DC's position includes RDC stock and in-transit stock, so the
RDC need not buffer against its full replenishment lead time. It buffers against the risk that the
central DC cannot fill an order during transit, which is a much smaller exposure when the central
DC holds stock. The RDC buffer becomes a transit buffer; the central DC's becomes the network
buffer.

| Node | Exposure covered | Sigma used | Buffer (packs) |
|---|---|---:|---:|
| RDC North | transit only, 5 days | 105 x sqrt(0.714) = 88.7 | 146 |
| RDC South | transit only, 7 days | 90 x sqrt(1.000) = 90.0 | 148 |
| RDC West | transit only, 10 days | 72 x sqrt(1.429) = 86.1 | 142 |
| Central DC | pooled demand over `R + L`, 98% internal | 155.9 | 555 |
| Network total | | | **991** |

The central DC buffer uses `z = 2.054` for 98 per cent on the pooled leaf sigma of 155.9 rather than
the amplified 230, because an echelon policy sees through the RDC reviews instead of being driven
by them:

```text
   Central DC: z(0.98) x 155.9 x sqrt(3) = 2.054 x 155.9 x 1.732 = 554.7 packs
```

That is 991 against 1,275, a reduction of 284 packs or 22.3 per cent. It is a typical magnitude for
a small network and it is the number a vendor would quote. It is also **wrong**, in an instructive
way.

### 6.4 The trap in the internal service level

The 991-pack solution set the central DC's internal target at 98 per cent and the RDC's external
target at 95 per cent, then reported 95 per cent end-to-end. That is not what it delivers. The RDC
buffer was sized assuming the central DC fills its orders. When the central DC is out of stock, the
RDC buffer must cover the extra wait and was not sized for it. The end-to-end probability is the
probability the central DC served **and** the RDC served given that it did:

```text
   end-to-end = 0.98 x 0.95 = 0.931
```

Ninety-three per cent, not ninety-five. This is one of the few places where the multiplication is
the right arithmetic, and it is the opposite of the section 3.2 case: here it is correct because the
downstream buffer is conditional on the upstream delivery and has no slack to absorb a breach. The
fix is to set the downstream target so the product hits the external target:

```text
   required downstream CSL = 0.95 / 0.98 = 0.9694        z = 1.873
```

Recomputing the three RDC buffers at `z = 1.873`:

| Node | Sigma over transit | Buffer at `z = 1.873` |
|---|---:|---:|
| RDC North | 88.7 | 166 |
| RDC South | 90.0 | 169 |
| RDC West | 86.1 | 161 |
| RDC subtotal | | **496** |
| Central DC at 98% | 155.9 | 555 |
| Network total | | **1,051** |

Against 1,275, that is a reduction of 224 packs, or 17.6 per cent, at a stated end-to-end CSL of
95.0 per cent. The smaller number is the honest one: the 22.3 per cent compared two solutions that
did not deliver the same service.

> **Note:** The 17.6 per cent is a constructed figure for a three-leaf network with one central
> node. It is not a benchmark. Vendor claims in the 10 to 30 per cent range are plausible for
> networks with this structure and are not evidence that any particular project will achieve them.

Every number above needs the four-part record from step 18. Central DC: 555 packs, echelon network,
protection `R + L` = 7 + 14 = 21 days, internal CSL 98 per cent per cycle over 52 weeks measured on
RDC order fulfilment, graph `NET-2026-Q4-r3`. RDC North: 166 packs, echelon installation, protection
12 days, external CSL 96.9 per cent per cycle over 52 weeks, same graph. The top holds the network
buffer over three weeks at a high internal target; the bottom holds a transit buffer at a slightly
higher external target; the two multiply to the contract.

### 6.5 The deployment decision

A container arrives at the central DC with 2,000 packs. The RDCs have requested 900, 700, and 500.
Total requests exceed supply by 100, so this is an allocation decision, not a deployment one,
whatever the meeting is called.

| Rule | North | South | West | Total | Note |
|---|---:|---:|---:|---:|---|
| Proportional to request | 857 | 667 | 476 | 2,000 | Factor 2,000 / 2,100 = 0.952 |
| Uniform | 667 | 667 | 667 | 2,001 | West capped at its 500 request |
| Fair share on need | 780 | 680 | 540 | 2,000 | Net of each RDC's projected position |
| Priority: North contract first | 900 | 700 | 400 | 2,000 | West absorbs the shortfall |
| Turn-and-earn, 8-week shipments | 933 | 667 | 400 | 2,000 | Shares 46.7 / 33.3 / 20.0 per cent |
| Demand-weighted forecast | 933 | 667 | 400 | 2,000 | Shares 46.7 / 33.3 / 20.0 per cent |

Two rows look identical and are not. **Turn-and-earn and demand-weighted agree here by
coincidence**, because West's eight-week shipments happened to track its forecast. Had West stocked
out in three of those weeks, its recorded shipments would be 1,200 rather than 1,440, its
turn-and-earn share would fall to 16.7 per cent, and its allocation to 333 packs, 67 fewer than its
demand-weighted entitlement. **Uniform gives West more than fair share and North much less**: the
rule that is stable against gaming is the rule that misplaces the most stock.

### 6.6 What the deployment does to next quarter's orders

Under proportional allocation North learns, correctly, that its allocation is a fixed fraction of
what it asks for, so its optimal response is to ask for more than it needs. If North expects a fill
factor near 0.95, asking for 950 rather than 900 recovers the 50 packs it expects to lose. South and
West learn the same thing. Next quarter requests total 2,850 against supply of 2,000, the fill factor
falls to 0.70, and every destination is worse off than under an honest rule at the same supply.

The diagnosis is visible in the numbers. The observed order-stream sigma at the central DC was 230
against a pooled leaf sigma of 155.9, an amplification of 1.48. Some of that is the weekly review
cycle and some is the allocation rule. A forecast improvement programme touches neither, and a
change of allocation rule might remove a third of the amplification at no inventory cost.

## 7. Code, query, or calculation

The DRP table below is the whole cascade in nineteen lines, hand-computable, and the artifact a
planner will actually check. Buckets are weeks. The RDC replenishes from the central DC with a
one-week lead time, the central DC from the plant with two weeks, and the plant makes in one week.
The convention here is that a release in bucket `t` becomes a gross requirement one level up in
bucket `t + 1`, because the order is transmitted at the end of the bucket and dated by the receiving
planner at the start of the next one.

```text
SKU 4410-B | weekly buckets | RDC lead 1 wk | CDC lead 2 wk | plant lead 1 wk
bucket                          1     2     3     4
Level 2 - RDC North (safety stock 150, lot multiple 50)
gross requirement             420   420   420   420
scheduled receipt             300     0     0     0
projected on hand             280   160   190   170
planned order release         300   450   400   400

Level 1 - central DC (safety stock 400, lot multiple 100, all RDCs)
gross requirement               0 1,040 1,180 1,120
scheduled receipt               0     0     0     0
projected on hand           2,500 1,460   480   460
planned order release           0   200 1,100     0

Level 0 - plant (safety stock 200, lot-for-lot)
gross requirement               0   200 1,100   900
scheduled receipt               0     0     0     0
projected on hand             600   400   200   200
planned order release           0   900   900     0
```

Read it down, then up.

**Down, within a level.** At RDC North in bucket 2 the projected balance would be 280 minus 420,
which is negative, so a net requirement of `420 + 150 − 280 = 290` is raised and rounded up to the
300 lot multiple, arriving in bucket 2 and released in bucket 1. In bucket 3 the balance of 160
produces `420 + 150 − 160 = 410`, rounded to 450. In bucket 4 the balance of 190 produces 380,
rounded to 400.

**Up, between levels.** North's 300 release in bucket 1 is part of the central DC's 1,040 gross
requirement in bucket 2. The central DC's balance rolls 2,500 to 1,460 to 480, breaches its 400
safety stock in bucket 3, and raises `1,180 + 400 − 1,460 = 120`, rounded to 200, released in bucket
1 against a two-week lead. Its bucket 4 requirement is larger because the balance is thin:
`1,120 + 400 − 480 = 1,040`, rounded to 1,100 and released in bucket 2.

**The one-bucket shift the eye should catch.** The central DC releases 200 in bucket 1 and the plant
sees that as a gross requirement of 200 in bucket 2; it releases 1,100 in bucket 2 and the plant sees
1,100 in bucket 3. Note the convention explicitly: many DRP texts place the requirement in the same
bucket as the release, with the offset applied only at the level that consumes it. The arithmetic is
identical if you remove the lag; what changes is which bucket the requirement lands in, and that is
a full bucket of protection. State which convention your system uses.

Expected output: RDC North releases 300, 450, 400, 400; the central DC releases 0, 200, 1,100, 0;
the plant releases 0, 900, 900, 0. Any submission producing a central DC release in bucket 4 has
applied the lead-time offset twice.

> **Note:** The plant's gross requirement in bucket 4 is 900 rather than 1,100 because the central
> DC's bucket 4 release is zero — the balance of 460 covers that bucket and the next receipt was
> already released. That lumpiness is the bullwhip, visible in three rows of arithmetic.

## 8. Data, standards, and regulatory context

No standard defines an echelon stock figure, a MEIO objective, or a deployment rule. What the
standards constrain is the identity of the nodes and items the graph is built from, the event data
that produces lead times, and the auditability of the parameters. Those are the parts usually
missing.

### 8.1 What governs the graph and the money

| Standard or instrument | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GLN | Location identity | The node key. A re-coded warehouse splits one node into two and silently doubles a buffer | B |
| GS1 GTIN | Product identity | The item key at every node. A pack-size change creates a new item and a new buffer | B |
| GS1 SSCC | Logistics unit identity | The handling unit for a deployment and the grain at which a receipt is posted | B |
| GS1 EPCIS 2.0 | Event data: what, where, when, why | The receipt and shipment timestamps that produce the arc lead-time distributions | B |
| GS1 EDI (EANCOM, or ANSI X12 in North America) | Business document exchange | The order, the advance ship notice, and the product activity data DRP consumes | B |
| ISO 8000 series | Data quality | Completeness and consistency of the keys the graph is built on | B |
| ISO 22400-2 | Manufacturing operations KPIs | How service and inventory indicators are defined; does not choose the network policy | B |
| IAS 2 / ASC 330 | Inventory measurement | Lower of cost and net realisable value. An echelon buffer is a policy quantity, not an accounting category | B |
| IFRS 15 / ASC 606 | Revenue recognition | Control transfer, which determines when an in-transit unit leaves the seller's balance sheet | B |
| Incoterms 2020 | Delivery responsibility | Who owns goods on an arc, and therefore which node's installation stock contains them | B |
| Internal control over financial reporting | Listed-company governance | Change control over parameters that materially move reported inventory | B |
| Transfer pricing rules | Multinational groups | How margin is allocated across the legal entities that own the nodes | B |

The Incoterms row is the one that bites network models. A unit in transit is nobody's installation
stock under some terms and the buyer's under others. If the model assigns in-transit units
inconsistently, the echelon stocks do not reconcile, and the failure looks like a data quality
problem rather than a definitional one.

### 8.2 What is convention rather than requirement

- **The multiplicative service ladder.** Three stages at 95 per cent producing 85.7 per cent is
  arithmetic, not a standard, and section 3.2 explains why it is the wrong model for a network with
  absorbent downstream stock.
- **The 10 to 30 per cent inventory reduction.** A vendor range repeated in every MEIO business
  case. No audited benchmark establishes it, and the figure depends on how bad the starting point
  was.
- **A weekly planning bucket.** Inherited from a paper-based era; some networks need it and most are
  not required to be limited by it.
- **"Safety stock at every level"** is a habit, and **a single decoupling point** holds for a
  single-product line and fails for a multi-echelon assortment.

### 8.3 The audit trail a network model needs

A network solution is a controlled record in any listed company, because it sets parameters that
determine reported inventory at several entities at once. Three artifacts are needed and two are
usually missing. **The graph version** dates the node, arc, and lead-time set, so a March buffer is
distinguishable from a September one computed on a network with one fewer DC. **The solve record**
states the objective, constraints, method, and date, without which a solution can be accepted but
not reviewed. **The buffer register** holds one line per buffered node in the four-part standard.

## 9. Industry practice and case evidence

### 9.1 What the research literature establishes

| Study | Design | Finding | Grade |
|---|---|---|---|
| Clark and Scarf (1960) | Analytical, serial system, dynamic programme | Echelon (s,S) policies optimal under four stated assumptions; decomposition into single-stage problems with a convex penalty passed upstream | A |
| Axsäter and Rosling (1993) | Analytical, two-level systems | For order-up-to policies, installation and echelon policies are equivalent; for (s,nQ) the conversion is one-directional | A |
| Axsäter and Juntti (1996) | Analytical and numerical, distribution systems | Echelon policies can perform worse than installation policies in divergent systems, for rationing and imbalance reasons | A |
| Diks, de Kok and Lagodimos (1996) | Review, service measure perspective | Service measures are not interchangeable across echelons; internal upstream service levels are in some sense irrelevant | A |
| van Houtum, Inderfurth and Zijm (1996) | Analytical, N-stage serial | End-to-end fill rate is a convolution over all echelon stock levels, not a product of stage rates | A |
| Chen (1998) | Analytical, serial systems | Quantifies the cost gap between echelon and installation reorder points against stages, lead times, batch sizes, variability, and service level | A |
| Cachon and Lariviere (1999) | Game-theoretic | Proportional and linear allocation make order inflation rational and can leave no equilibrium; uniform supports a truthful equilibrium | A |
| Spiliotopoulou, Donohue and Gürbüz (2022) | Experimental | Only uniform allocation induces truthful orders; proportional and linear produce larger and more frequent order adjustments | A |
| Spiliotopoulou and Conte (2022) | Experimental, fairness | Equal fill rates is the prevalent fairness ideal under shortage; proportional and linear are rated fairer than uniform | A |
| Li, Yu, Liu and Cai (2017) | Game-theoretic | Allocation with demand competition across uniform, proportional, and lexicographic mechanisms | A |
| Villa (2022) | Experimental | Allocation mechanism choice changes order variability; turn-and-earn outperforms proportional | A |

Three readings matter more than the individual rows. **The Clark–Scarf result is stronger than
practitioners think and narrower than vendors imply**: within its four assumptions it is a theorem;
outside them it is the starting point for an approximation whose error is not usually quantified.
**The echelon-versus-installation question has no universal answer**: Axsäter and Rosling remove the
information argument for order-up-to policies by showing equivalence, and Axsäter and Juntti show
the distribution case can go the other way. **The allocation rule is a behavioural instrument**:
Villa reports that changing it reduced order variability by up to half in an experimental setting,
larger than most forecasting programmes deliver and at no inventory cost.

### 9.2 What the practitioner literature reports

The vendor and consultancy material is consistent about failure modes and inconsistent about
benefits, which is the usual pattern.

| Source | Type | What it reports | Grade |
|---|---|---|---|
| Metreecs, "MEIO: a practical guide" | Consultancy blog | Three failure modes: category-level rather than item-to-location forecasts; flat average lead times instead of ranges; data silos. Cites a 22-store case where fixing lead times, not the algorithm, resolved the allocation problem | D |
| Umbrex, multi-echelon inventory playbook | Consultancy resource | Seven-step workflow; typical 10–30 per cent inventory reduction; pitfalls including skipped lead-time quality checks, ignored MOQ and pallet constraints, and treating MEIO as one-time | D |
| MCP Analytics, MEIO whitepaper | Vendor whitepaper | Lead-time variance term can dominate; inputs must use observed rather than planned lead times; stockout-censored demand understates demand; "implementation is mostly an organizational problem" | D |
| Hayden (2014), MIT thesis on the Johnson & Johnson SmartOps deployment | Thesis documenting a company case | 16+ spreadsheet files per affiliate per month, 50+ bills of material, files over 200,000 rows, all manually validated monthly; data cleansing critical; decentralised organisations take longer | C |
| Consultancy blog summary of a P&G programme | Second-hand vendor claim | 7 per cent inventory reduction and USD 1.5bn cash release, with no primary reference retrieved here | E |

The Hayden thesis is the most useful item because it is the least promotional: the binding
constraint was not the optimiser but the monthly manual assembly of sixteen spreadsheets per
affiliate. That is the modal MEIO project, and a plan that assumes a clean data feed is planning
for a different company.

> **Note:** The P&G figures are reported as a second-hand vendor claim and are not verified here.
> The underlying case was cited but not retrieved. Treat the number as an order of magnitude at
> best and do not repeat it as an audited result.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Every stage set to the contractual service level | The same 95 per cent appears in every node's parameter file | Stage targets copied, not derived | Over-buffered network that still misses end-to-end | Derive internal targets from the external one; publish both |
| Lead time entered as a single average | One number per lane in the item master, no variance field | Buffer sized on a mean, covering half of arrivals | Stock-outs concentrated on long lanes | Build the distribution per arc; store the 90th percentile beside the mean |
| Network graph not maintained | The model's node list has a DC that closed last quarter | Solutions computed on a stale topology | Buffers placed at nodes that cannot receive | Version the graph; force a re-solve on any node or lane change |
| Demand censored by stock-outs | Sales series has no zeros and no stock-out flag | Variance understated by the censoring | Buffers too small, repeated failure on the same items | Reconstruct demand for unavailable days; test before and after |
| Echelon and installation numbers mixed in one report | Total inventory does not equal the sum of the levels, or equals it twice | Two conventions in one table | Targets that cannot be reconciled or audited | State the convention on every row; reconcile to the top node's echelon stock |
| Deployment rule rewards inflation | Order-to-shipment ratio above 1 and rising quarter on quarter | Proportional allocation under shortage | Amplified order stream, inflated buffers, worse service | Publish the rule; measure the ratio by destination |
| Allocation rule differs from the documented one | System setting says fair share; the last three splits match priority | The meeting overrides the engine silently | Destinations respond to the meeting, not the system | Reconstruct the realised rule from the last twelve shortage events |
| DRP run to unoptimised targets | The table is internally consistent and service is wrong | Execution engine with no policy design | Precise execution of a bad policy | Separate target-setting from netting |
| Scope set to the whole catalogue at once | Phase one has 40,000 item-location pairs | Data quality collapses under volume | Project stalls after the pilot | Start on a bounded slice with a known problem |
| Internal service level set to 100 per cent "to be safe" | Central DC target is 99.5 per cent and holding cost is unexplained | Buffer for a certainty that cannot exist | Central inventory disproportionate to the benefit | Price the last points of internal service; they are expensive |
| Decoupling point described in the singular | One slide says "our decoupling point is the RDC" while components are buffered at the plant | Multiple buffers, one narrative | Nobody owns the total; each owner defends a local number | Enumerate every buffer with its echelon in the register |
| Re-solve treated as a one-off project | The model has a delivery date and no owner afterwards | Parameters drift with demand and cost | Solution decays within two quarters | Assign ownership, cadence, and trigger events |

Two deserve a note. **Demand censoring** survives longest because the data looks complete: a leaf
that stocked out for six days shows six days of low sales, a forecast model learns the low level as
real, every buffer computed from that series is too small, and the resulting stock-outs create more
censoring. **Mixing conventions in one report** is a presentation failure with audit consequences: a
slide showing "central DC 620, RDC 240, total 1,040" is arithmetically wrong by 180 units, because
in-transit stock appears in neither installation number, yet the total is right. State the
convention per row and show the reconciliation line.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Demand originates only at the leaves | Every decomposition takes leaf demand as primitive | Internal "demand" double-counts derived requirements | Check whether internal shipments correlate perfectly with downstream orders |
| Each arc's lead time is a stationary distribution | Buffers are computed from it | Buffer sized for a lead time that no longer exists | Control chart the realised lead time per lane; check for trend |
| The network graph is acyclic | Solvers assume no loops | Returns, transfers, or repair loops cannot be represented | Trace every arc; count cycles |
| The graph is current | A solution is valid only for the network it was solved on | Buffers at closed nodes, missing lanes | Diff the model's node list against the warehouse location master |
| Holding cost is linear in quantity | Makes the objective well-behaved and the decomposition valid | Step costs or discounts break the single-interval structure | Check for storage step costs and price breaks |
| Shortage cost or service target is defensible | The objective needs one of the two | Optimisation of an arbitrary number | Ask the business to state the cost of a lost pack |
| Demand is uncensored in the history | Variance estimation depends on it | Understated variance and undersized buffers | Compare sales with availability by day |
| Conventions are applied consistently | The reconciliation depends on it | Targets that cannot be audited or summed | Reconcile the register to the top node's echelon stock |
| The allocation rule is the one published | Destinations respond to the realised rule | Order inflation, amplification, higher buffers | Reconstruct the last twelve splits against each candidate rule |
| Downstream stock can absorb upstream delay | Justifies lower RDC buffers in an echelon design | If it cannot, the internal target must be met almost always | Measure the fill rate the central DC achieves to each RDC |
| One bucket convention at every interface | The cascade arithmetic depends on it | Requirements land one bucket early or late | Compare two consecutive DRP runs with a known release |

Two limits deserve statement. **MEIO cannot fix a wrong demand distribution or a mis-specified lead
time.** It is a placement algorithm: given the same wrong inputs it will place buffers optimally
against a fiction, and the result will look more defensible than the naive answer it replaced. The
practitioner sources in section 9.2 say so in almost identical words. **The optimum is conditional
on the topology and the objective**, so no network-wide "right" level survives a redesign.

## 12. Exercises

### Exercise 1 — Two numbers for one shelf (recall and application, 30 minutes)

A three-level chain: plant, central DC, RDC. Central DC on hand 800, in transit to RDC 250, RDC on
hand 300, in transit to stores 90, store on hand 60.

**Deliverable:** installation stock and echelon stock for each node, and the network total under
each convention.
**Acceptance criteria:** every number in packs with its node named; the two network totals are equal
and the reconciliation is shown; one sentence on which number a central planner should use and why.

### Exercise 2 — The service ladder (calculation, 25 minutes)

A four-stage chain sets 97 per cent CSL at each stage.

**Deliverable:** the product under independence, the end-to-end figure you would actually claim, and
the assumption that separates them.
**Acceptance criteria:** the arithmetic shown; the claim stated as a bound rather than a point
estimate; one condition under which the product is the right answer.

### Exercise 3 — Clark–Scarf at its actual strength (critique, 40 minutes)

A vendor slide states: "Echelon stock policies are proven optimal for multi-echelon networks."

**Deliverable:** a corrected statement in no more than four sentences, with assumptions named and
the distribution-network caveat included.
**Acceptance criteria:** the topology is named; at least three of the four assumptions appear; the
word "optimal" is qualified; no claim is made that the 1960 paper did not support.

### Exercise 4 — Build the DRP table (application, 50 minutes)

One item, three levels, four weekly buckets. RDC opening 500, gross requirement 380 per week,
safety stock 120, lot multiple 50, lead time 1 week. Central DC opening 2,200, safety stock 350, lot
multiple 100, lead time 2 weeks, receiving the RDC's releases plus 600 per week from two other RDCs.
Plant opening 700, safety stock 150, lot-for-lot, lead time 1 week.

**Deliverable:** the full table with all four rows at all three levels, plus the bucket in which
each level's first release occurs.
**Acceptance criteria:** the projected balance is consistent left to right at every level; the
cascade is visible with one bucket of transmission lag; no release is past-due within the horizon;
the arithmetic for one net requirement is shown in full.

### Exercise 5 — Allocation rules, priced in behaviour (design, 45 minutes)

A central DC has 1,500 packs against requests of 800, 600, and 400 from three RDCs of equal contract
status. Last quarter the requests were 700, 550, and 350 against the same supply, and the RDCs know
the fill factor.

**Deliverable:** the split under proportional, uniform, fair share on need, and turn-and-earn, plus
a prediction of next quarter's requests under each.
**Acceptance criteria:** all four splits sum to 1,500; the behavioural prediction names the
mechanism, not just the direction; one rule is recommended with its fairness cost stated.

### Exercise 6 — Where the buffer belongs (design, 50 minutes)

A company sells 40 variants of a configurable instrument from four RDCs. Component lead time is 16
weeks, assembly is 1 week, RDC transit is 3 days, and customers accept a 2-week promise on 60 per
cent of volume and demand same-week on the rest.

**Deliverable:** a recommended decoupling placement per volume segment, with pooling and response
consequences quantified where the inputs allow.
**Acceptance criteria:** the recommendation distinguishes the two segments; the promise arithmetic
is shown for each; the pooling argument cites topic 09 rather than re-deriving it; one condition
under which the recommendation reverses.

### Exercise 7 — Audit a buffer register (critique, 35 minutes)

You are given three lines from a buffer register: "CDC 4,200 units", "RDC North 900 units, 95 per
cent", "RDC South 1,100 units, 21 days".

**Deliverable:** a defect list, the missing elements per line, and the questions you would ask
before using any of the numbers.
**Acceptance criteria:** all four standard elements checked per line; at least four defects found;
each question answerable by a specific person or system.

### Exercise 8 — The stale graph (diagnosis, 30 minutes)

A model solved in March has a 22 per cent buffer reduction. Between March and September one RDC
closed, a lane moved from road to rail, and two SKUs were consolidated. Nobody re-ran the model.

**Deliverable:** the specific parameters invalidated by each change, and the order in which you
would rebuild.
**Acceptance criteria:** each change mapped to the parameters it invalidates; the rebuild order
justified; one change identified that invalidates the objective rather than the parameters.

### Exercise 9 — The end-to-end claim (design, 45 minutes)

A programme claims a 25 per cent inventory reduction at unchanged service, across a network where
every node was previously set to 95 per cent.

**Deliverable:** a validation plan that would confirm or refute the claim, with specific measurements
and confounders.
**Acceptance criteria:** at least four measurements named with their sources; at least two
confounders identified; the plan distinguishes an inventory reduction from a service reduction.

## 13. Self-check answers

**Exercise 1.** Installation stock: plant 0, central DC 800, RDC 300, store 60. Sum 1,160, plus the
250 and 90 in transit, giving 1,500. Echelon stock: store 60; RDC 300 + 90 + 60 = 450; central DC
800 + 250 + 450 = 1,500; plant 1,500 if the plant holds nothing. Network total 1,500, which is the
central DC's echelon stock. The totals agree, which is the reconciliation test. A central planner
should use the echelon number, because it is the only figure containing the units already committed
downstream and therefore the only one that prevents a duplicate order.

**Exercise 2.** `0.97^4 = 0.8853`. Claim 88.5 per cent as a lower bound under independent local
buffering, not as the end-to-end fill rate. The separating assumption is that no stage can draw on
another's stock, so a stock-out at any stage reaches the customer. If a downstream stage holds
buffer that absorbs an upstream delay, the end-to-end figure is higher and is computed from the
whole vector of stock levels. The product is the right answer when you are setting an internal
promise between two stages and the receiving stage has no slack.

**Exercise 3.** A corrected statement: "Clark and Scarf (1960) proved that echelon-stock (s,S)
policies are optimal in a **serial** system under four assumptions — demand only at the lowest
installation, linear purchase and shipping costs with a setup cost permitted only at the top,
holding and shortage costs that are functions of echelon stock, and backlogging at every echelon.
The result decomposes the problem into single-stage problems solved bottom-up. It does not extend to
distribution networks, where the paper describes the results as less satisfactory, and later work
shows echelon policies can underperform installation policies there." A submission that keeps the
word "proven" without the topology has not done the exercise.

**Exercise 4.** RDC: bucket 1 balance `500 − 380 = 120`, equal to safety stock, no order. Bucket 2
`120 − 380 = −260`, net `380 + 120 − 120 = 380`, rounded to 400, released in bucket 1. Bucket 3
balance `120 + 400 − 380 = 140`, no order. Bucket 4 `140 − 380 = −240`, net 360, rounded to 400,
released in bucket 3. Central DC gross requirement is the RDC release plus 600: bucket 2 = 1,000,
bucket 3 = 600, bucket 4 = 1,000. With opening 2,200 and safety stock 350 the balance runs 2,200,
1,200, 600, and bucket 4 gives `600 − 1,000 = −400`, net `1,000 + 350 − 600 = 750`, rounded to 800,
released in bucket 2 against the two-week lead. Plant gross requirement, one bucket later: bucket 3
= 800. Plant balance 700 then `700 − 800 = −100`, net `800 + 150 − 700 = 250`, released in bucket 2
against a one-week lead. First releases: RDC bucket 1, central DC bucket 2, plant bucket 2. A
submission placing the central DC release in bucket 4 has applied the offset twice.

**Exercise 5.** Proportional: factor `1,500 / 1,800 = 0.833`, giving 667, 500, 333. Uniform: 500
each. Fair share on need requires each RDC's projected position, so the answer must state the assumed
positions; if need is 800, 600, 400, fair share equals proportional. Turn-and-earn on last quarter's
requests of 700, 550, 350 gives shares of 43.75, 34.38, 21.88 per cent, so 656, 516, 328.
Behavioural prediction: under proportional each RDC raises its request by roughly the reciprocal of
0.833, so next quarter's requests rise about 20 per cent and the fill factor falls. Under uniform,
requests converge toward true need because asking for more returns nothing until everyone is served
equally, but the third RDC loses 100 packs relative to proportional and will escalate commercially.
Turn-and-earn rewards whoever asked most last quarter, the same failure as proportional with a lag.
The recommendation is normally uniform with a published exception path for contracted customers, and
the fairness cost is that the largest destination visibly subsidises the smallest.

**Exercise 6.** For the 60 per cent with a two-week promise: assembly at the central DC, one week,
plus three days transit, leaves four days of slack, so a sub-assembly buffer at the central DC works
and pools across 40 variants. For the same-week segment the buffer must be finished goods at the
RDC, because assembly plus transit alone exceeds the promise. The pooling consequence is that the
central buffer covers 40 variants with one risk pool while the RDC buffers cover one variant each,
so the RDC buffer per variant is larger per unit of demand. The recommendation reverses if
variant-specific volatility falls far enough that postponement buys little, or if a component lead
time is short enough that the component buffer no longer pools meaningfully. Module 10 topic 09 owns
the pooling arithmetic.

**Exercise 7.** Line 1 has one element of four: no echelon, no protection period, no service
measure, no graph version. Line 2 has a service measure but no target window, no echelon, no
protection period, no graph version. Line 3 has a protection period but no service measure, no
echelon, no graph version. At least four defects follow: a number with no echelon cannot be compared
to any other number; a protection period with no service measure cannot be reproduced; a missing
graph version means none can be dated; and the service measure on line 2 does not say whether it is
internal or external. Questions: which echelon, what is the protection period and its inputs, which
service measure and window, which graph version and solve date, and who owns recomputation.

**Exercise 8.** The RDC closure invalidates every downstream buffer, the allocation denominators, and
the leaf set; the lane mode change invalidates that arc's lead-time distribution and every buffer
whose protection period includes it; the SKU consolidation invalidates the item dimension of the
whole model and the demand history used to fit it. Rebuild order: item consolidation first, because
it changes the demand series everything else is computed from; then the graph with the closed RDC;
then the lane distributions; then re-solve. The change that invalidates the objective rather than
the parameters is the modal shift, if it changes the cost structure enough that a cost-minimising
objective no longer reflects the trade the business faces.

**Exercise 9.** Measurements: end-to-end fill rate at the leaves, month by month, before and after,
with definition and window stated; inventory in units and value at each node, on both conventions;
the realised allocation rule from the shortage events; the order-to-shipment ratio by destination;
and the central DC's fill rate to each RDC. Confounders: demand level change between periods; the
service definition changing with the programme; a network change during the period; and inventory
shifted to a supplier or a third-party logistics provider outside the measured boundary. The plan
distinguishes a genuine reduction from a service reduction by requiring the end-to-end measure to
hold at the same definition, at the same demand level, on a stable graph.

## 14. Interview questions

**1. "What is the difference between installation stock and echelon stock?"**

A strong answer defines both, gives a numeric example where the same shelf reports two numbers, and
says which one a network target should be expressed in. A weak answer says one is local, one total.

**2. "Why do stage-by-stage service targets compound?"**

A strong answer gives the multiplication and immediately qualifies it: in a network with absorbent
downstream stock the end-to-end measure is not the product. A weak answer gives 0.95 cubed and stops.

**3. "What did Clark and Scarf actually prove?"**

A strong answer names the serial topology, the four assumptions, and the upstream penalty, and says
what the paper does not cover. A weak answer says "echelon stock is optimal", full stop.

**4. "Where would you place the buffer with 40 variants and a 16-week component lead time?"**

A strong answer separates the volume segments by promise, places the buffer where the promise allows,
and quantifies the pooling gain against the added response time. A weak answer says "at the DC".

**5. "What does DRP do that an optimiser does not?"**

A strong answer says DRP nets and offsets in time and exposes the cascade bucket by bucket, so a
planner can audit it by hand, while an optimiser chooses a target, not a schedule. A weak answer
says DRP is the older method.

**6. "A container arrives short. How do you split it?"**

A strong answer names at least three rules, states what behaviour each rewards, and says the rule
must be published before the shortage. A weak answer says "fairly" or "by priority".

**7. "Why does the allocation rule affect the bullwhip?"**

A strong answer explains that a proportional rule makes inflation individually rational and that
inflated orders enter the upstream demand series as variance. A weak answer says customers panic.

**8. "What are the failure modes of a MEIO project?"**

A strong answer names scope, lead-time data quality, and graph maintenance, and adds that the
optimiser cannot fix a wrong demand distribution. A weak answer says "bad data".

**9. "How would you audit a network buffer?"**

A strong answer asks for the echelon, the protection period with its inputs, the service measure
with its target and window, and the graph version. A weak answer asks whether the number looks
reasonable.

**10. "Why is demand at internal nodes dangerous?"**

A strong answer says internal shipments are derived demand and already contain the downstream review
cycle and allocation behaviour. A weak answer says internal demand is less accurate.

**11. "When would you not use echelon stock policies?"**

A strong answer names the information requirement, the distribution-system rationing problem, and
the documented cases where echelon policies underperform. A weak answer says echelon is always
better because it is centralised.

**12. "What would make you distrust a 25 per cent inventory reduction claim?"**

A strong answer asks whether the end-to-end measure held at the same definition, whether demand and
graph were stable, and whether inventory moved outside the measured boundary. A weak answer asks for
reference customers.

## 15. Key terms

- **Multi-echelon inventory** — stock at two or more levels of one network, where decisions couple.
- **Installation stock** — what one node owns: on hand there, plus on order to there.
- **Echelon stock** — a node's own stock plus everything downstream, in transit and on hand.
- **Installation inventory position** — installation on-hand plus on-order minus backorders.
- **Echelon inventory position** — echelon stock plus on-order to the echelon minus backorders.
- **Effective lead time** — installation lead time including expected upstream waiting.
- **Leaf** — a node with demand and nothing downstream.
- **MEIO** — multi-echelon inventory optimisation; the placement of stock across a network.
- **DRP** — distribution requirements planning; time-phased netting and offsetting across levels.
- **Gross requirement** — demand in a bucket before netting.
- **Projected available balance** — on hand rolled forward by receipts minus requirements.
- **Planned order release** — a planned receipt offset backwards by the lead time.
- **Lead-time offset** — dating a requirement `L` buckets before its need.
- **Deployment** — splitting available supply across destinations when supply is adequate.
- **Allocation** — splitting scarce supply across competing requests.
- **Fair share** — allocation proportional to projected need after netting positions.
- **Turn-and-earn** — allocation proportional to a destination's past sales share.
- **Proportional allocation** — every request scaled by one fill factor; gaming-prone.
- **Uniform allocation** — equal split up to each request; truth-inducing, perceived as unfair.
- **Shortage gaming** — inflating orders to protect an allocation under a rationing rule.
- **Internal service level** — the promise a stage makes to the stage below it.
- **End-to-end fill rate** — leaf demand served without delay through all echelons.
- **Decomposition** — splitting a network problem into stages linked by service times or penalties.
- **Net replenishment lead time** — inbound promise plus processing minus outbound promise.
- **Imbalance** — a node's echelon position falling after rationing because upstream could not fill.
- **Decoupling point** — the node where stock breaks the dependency on an upstream lead time.
- **Pooling benefit** — the buffer reduction from aggregating independent demand.
- **Graph version** — the dated identifier of the node, arc, and lead-time set a solution used.
- **Buffer register** — one line per buffered node in the four-part reporting standard.

Three do most of the work: **echelon stock** makes a network target expressible, **the allocation
rule** changes behaviour at no inventory cost, and **the graph version** makes the output auditable.

## 16. Sources

Grades follow Module 28: A peer-reviewed or audited, B regulatory or official, C company case or
investor claim, D vendor marketing or product documentation, E press summary or anecdote.

### 16.1 Peer-reviewed literature — grade A

| Source | What it supports |
|---|---|
| Clark, A. J. and Scarf, H. (1960), "Optimal Policies for a Multi-Echelon Inventory Problem", *Management Science* 6(4), 475–490. DOI: 10.1287/mnsc.6.4.475 | Echelon (s,S) optimality in a serial system; the bottom-up decomposition with an upstream penalty; the four assumptions |
| Scarf, H. (2004), "Optimal Policies for a Multi-Echelon Inventory Problem: A Comment", *Management Science*. DOI: 10.1287/mnsc.1040.0265 | The author's retrospective on the 1960 result |
| Axsäter, S. and Rosling, K. (1993), *Management Science* 39, 1274–1280 | Equivalence of installation and echelon policies for order-up-to policies; one-directional conversion for (s,nQ) |
| Axsäter, S. and Juntti, L. (1996), *International Journal of Production Economics* 45, 303–310 | Echelon policies underperforming in distribution systems |
| Diks, E. B., de Kok, A. G. and Lagodimos, A. G. (1996), "Multi-echelon systems: A service measure perspective", *European Journal of Operational Research* 95(2), 241–263 | Service measures across echelons; internal service levels are in some sense irrelevant |
| van Houtum, G. J., Inderfurth, K. and Zijm, W. H. M. (1996), "Materials coordination in stochastic multi-echelon systems", *EJOR* 95, 1–23 | End-to-end fill rate as a convolution over all echelon stock levels |
| Chen, F. (1998), "Echelon Reorder Points, Installation Reorder Points, and the Value of Centralized Demand Information", *Management Science* 44, S221–S234 | The cost gap between echelon and installation policies as a function of network parameters |
| Cachon, G. P. and Lariviere, M. A. (1999), "An equilibrium analysis of linear, proportional and uniform allocation of scarce capacity", *IIE Transactions* 31(9) | Inflation incentives under proportional and linear rules; truthful equilibrium under uniform |
| Spiliotopoulou, E., Donohue, K. and Gürbüz, M. Ç. (2022), "Ordering Behavior and the Impact of Allocation Mechanisms in an Integrated Distribution System", *Production and Operations Management* 31(2), 422–441 | Experimental evidence that only uniform allocation induces truthful orders |
| Spiliotopoulou, E. and Conte, A. (2022), "Fairness ideals in inventory allocation", *Decision Sciences* | Equal fill rates as the prevalent fairness ideal under shortage |
| Li, J., Yu, Y., Liu, Z. and Cai, X. (2017), "Allocation with demand competition: Uniform, proportional, and lexicographic mechanisms", *Naval Research Logistics* | Allocation mechanisms and their link to the bullwhip |
| Villa, S. (2022), "Competing for supply and demand: Understanding retailers' ordering decisions", *International Journal of Production Economics* | Order variability reduced by changing the allocation mechanism |

> **Note:** Volume and page details are reported as they appeared in the search results consulted.
> Where a search returned a title, journal, and year but not a full citation, the entry is given
> without volume and page numbers rather than reconstructed. The Clark–Scarf page range and DOI
> were returned directly. Nothing here is a claim about a source not read at least in abstract.

### 16.2 Technical and implementation references — grade A to D

| Source | What it supports | Grade |
|---|---|---|
| Graves–Willems guaranteed-service model, as described in MetricGate documentation and the Stockpyl library tutorial | The net replenishment lead-time identity and the tree decomposition | D |
| Achkar, V., Brunaud, B., Musa, E. and Grossmann, I. E., work on computational strategies for MEIO under the guaranteed-service approach, *EJOR* and *Computers and Chemical Engineering* (2023–2024) | Reformulation for divergent networks; added review periods, MOQs, and fill-rate targets | A |
| Rossetti, M. D., *Analysis of Inventory Systems*, Chapter 6 | DRP as MRP with every quantity-per set to one; the projected-balance recursion and the release offset | A |
| Hayden, C. (2014), MIT thesis on the Johnson and Johnson SmartOps deployment | A documented company deployment, including the manual data assembly burden | C |

### 16.3 Standards and official instruments — grade B

| Instrument | What it supports |
|---|---|
| GS1 GLN, GTIN, SSCC | Node, item, and handling-unit identity for the graph |
| GS1 EPCIS 2.0 | Event data supplying receipt and shipment timestamps |
| GS1 EDI / EANCOM / ANSI X12 | The order, advance ship notice, and product activity documents DRP consumes |
| ISO 8000 series | Data quality of the keys the graph is built on |
| ISO 22400-2 | KPI definitions for service and inventory reporting |
| IAS 2 and ASC 330 | Inventory measurement at the lower of cost and net realisable value |
| Incoterms 2020 | Ownership during transit, which determines which installation stock contains a unit |
| IFRS 15 and ASC 606 | Control transfer and revenue recognition |

No standard consulted prescribes an echelon policy, a MEIO objective, a deployment rule, or a
decoupling point. None should be cited as if it did.

### 16.4 Practitioner and vendor material — grade D to E

| Source | Used only for | Grade |
|---|---|---|
| Metreecs, "Multi-echelon inventory optimization: a practical guide" | Reported failure modes and a case where lead-time data quality, not the algorithm, was the problem | D |
| Umbrex, multi-echelon inventory management and network optimization playbook | Reported benefit ranges and implementation pitfalls | D |
| MCP Analytics, MEIO whitepaper | The lead-time variance term, demand censoring, and the organisational framing | D |
| r4 Technologies, inventory optimization services page | The three inputs determining model quality | D |
| A consultancy blog's summary of a P&G programme | A second-hand claim of a 7 per cent inventory reduction and USD 1.5bn cash release, not verified here | E |

### 16.5 What is inference rather than finding

Three statements here are inference, not finding: that stage-by-stage optimisation typically
produces network buffers materially above a decomposed optimum, for the reasons in section 5.1;
that most organisations cannot produce a defensible shortage cost and therefore run
service-constrained models; and that the realised allocation rule differs from the documented one.
No source consulted measures any of these. They are patterns worth testing, and section 12 supplies
the tests.

## 17. Links

Within this module:

- Module index: [`_index.md`](_index.md)
- The cost-service frontier the network objective is drawn on: [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- EOQ and the lot sizing DRP applies: [`02_eoq-limits.md`](02_eoq-limits.md)
- Newsvendor, the single-period limit of the buffer problem: [`03_newsvendor.md`](03_newsvendor.md)
- Safety stock, which supplies every buffer this topic places: [`04_safety-stock.md`](04_safety-stock.md)
- Service metrics, which define the measures the boundary is stated in: [`05_service-metrics.md`](05_service-metrics.md)
- Policy families, which this topic composes across levels: [`06_policy-families.md`](06_policy-families.md)
- Lead-time demand distributions, which supply the arc distributions: [`07_lead-time-demand.md`](07_lead-time-demand.md)
- Pooling and postponement, which own the decoupling-point theory in section 5.6: [`09_pooling-postponement.md`](09_pooling-postponement.md)
- Segmentation and DDMRP, which revisit decoupling points operationally: [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md)

Elsewhere in the program:

- Part II map: [`../_index.md`](../_index.md)
- Network design, where facilities should be: [`../11_network_design_optimization/_index.md`](../11_network_design_optimization/_index.md)
- Aggregation consistency, which governs how leaf demand rolls up: [`../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md`](../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
