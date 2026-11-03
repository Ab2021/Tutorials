# Decision Use of Quantiles — choosing a quantile from service level, cost, and risk appetite

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topics 07 and 09; Module 08 topics 07–09
> Version 2026.5

---

## 1. Why this topic exists

A service review opens with a single sentence: "The target for this line is 95 per cent." Eleven people
nod. Six of them believe six different things. One believes that 95 per cent of replenishment cycles will
end with stock on the shelf. One believes that 95 per cent of the units customers ask for will be
shipped. One believes that stock will be available 95 per cent of the time. One believes that the
order quantity has been set at a point that is 95 per cent likely to be enough. One believes a vendor
guaranteed it. One has not thought about it and is waiting for the number that goes into the system.

The sentence is not wrong. It is under-specified, and the under-specification is expensive, because the
same distribution of demand produces order quantities that differ by 40 to 70 per cent depending on
which of those six readings governs. On a normal-demand item with a coefficient of variation of about
0.25 over the protection period, a 95 per cent cycle service level delivers a fill rate near 99.5 per
cent, while a 95 per cent fill rate delivers a cycle service level near 69 per cent. Both statements
use the number 95. Neither is a stronger or weaker version of the other; they are answers to different
questions, and the quantities that satisfy them differ by tens of units per cycle.

Three failures recur, and this topic exists to name and fix them.

**The definition failure.** The organisation adopts "95 per cent service" without deciding which family
of measure it means. A cycle service level (CSL) is a probability attached to an event — no stock-out
during a replenishment cycle. A fill rate is a ratio of quantities — the share of demanded units served
from stock. A ready rate is a ratio of time. A cost-optimal fractile is a ratio of costs. Four different
objects, four different numbers, one phrase. Every downstream system inherits whichever one the
implementer happened to code.

**The translation failure.** Even when the family is fixed, the probability has to become a quantity.
That conversion passes through a distribution, a horizon, a lead time, a review period, a supply
process, and a rounding rule. Each step moves the number. Most organisations perform the conversion
inside a black box and then argue about the output, which is like arguing about a total without seeing
the line items.

**The ownership failure.** A quantile is a policy parameter. It encodes a promise and a cost. When
nobody owns it, it drifts: raised after every stock-out, never lowered after a surplus, re-tuned by
whoever is closest to the pain, and eventually defended by nobody because nobody chose it. The drift is
not sabotage. It is what happens to an unowned parameter in an organisation that experiences stock-outs
more vividly than it experiences excess.

**Who is blocked.** The demand planner who is asked for "the 95th percentile" and cannot say which
distribution it belongs to. The inventory analyst who computes a reorder point that satisfies the
contract but not the cost model, or the cost model but not the contract. The finance partner who is
asked to fund safety stock without a statement of what risk it buys. The data scientist who produces a
well-calibrated predictive distribution and watches it be consumed as a point forecast plus a
percentage. The supply chain director who has to defend a service promise to a customer and a working
capital number to a board, using the same word for both.

**What this topic is not.** The predictive distribution itself is Module 08 topic 07. The mechanics of
aggregation, intermittency, and calibration are topics 01 to 09 of this module. The derivation of the
newsvendor critical fractile at the level of a single point forecast is Module 07 topic 09, and that
derivation is not repeated here. What this topic adds is the layer above it: how a probability becomes
an order quantity inside a real policy, how the answer moves when the policy has a lead time and a
review period and an unreliable supplier, and how an organisation keeps the resulting number stable and
defensible over time.

> **Note:** The inventory policy design itself — reorder points, order-up-to levels, multi-echelon
> allocation, service-constrained optimisation — is Module 10. This topic stops at the translation: it
> produces the quantile, states what it means, and hands it over. Where a sentence here sounds like
> policy design, read it as the boundary condition the policy will inherit.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** cycle service level, fill rate, ready rate, and cost-optimal fractile, and state for
   each the numerator, the denominator, and the unit in which it is expressed.
2. **Compute** the fill rate implied by a stated cycle service level, and the cycle service level
   implied by a stated fill rate, for a normal-demand item, and explain why the two numbers diverge.
3. **Derive** the order quantity from a service target using the correct family, including the case
   where the family is a fill rate and no single quantile exists.
4. **Translate** a predictive distribution over a forecast horizon into a policy distribution over a
   protection period, incorporating lead-time variability, review period, and supply yield.
5. **Choose** between the cost-optimal quantile and a risk-appetite quantile, and **document** the
   difference in expected cost that the choice buys.
6. **Diagnose** the multi-period and multi-echelon cases where the single-period fractile does not
   transfer, and state what replaces it.
7. **Design** a governance arrangement for a service parameter: owner, register, review calendar,
   evidence required for change, and drift audit.
8. **Synthesise** the module: explain how hierarchy, intermittency, and uncertainty are three views of
   one problem, and why each ends at a quantile somebody has to defend.

Each outcome maps to an exercise in section 12 and to an answer or rubric in section 13.

## 3. The core mental model

Hold one picture: a distribution on the left, a quantity in the warehouse on the right, and a chain of
four transformations between them. The chain is where the money is lost, because each link is owned by
a different function and none of them sees the whole line.

```text
   THE TRANSLATION CHAIN, ONE ITEM, ONE PROTECTION PERIOD

      (1) DISTRIBUTION        (2) LEVEL          (3) QUANTITY      (4) SHORTFALL
          F(d)                    q                  Q             E[(D-Q)+]

   density
     ^
     |          ####
     |        #######
     |      ###########
     |    ###############
     |  ###################     read F^-1(q)       round, add        expected unmet
     | #####################    at the chosen      review, add       demand per cycle
     +--|---|----|----|----|--- level             yield, add MOQ
       200 400  480  560  700
                    ^    ^    ^
                    |    |    |
                    |    |    +---- CSL 0.95       -> 700 units   fill 97.9%
                    |    +--------- FILL RATE 0.95 -> 603 units   CSL 80.0%
                    +-------------- COST 0.678     -> 560 units   fill 93.7%

   the vertical line at step (2) is where the family decides the point.
   move the family, and steps (3) and (4) move with it; move the level,
   and the shortfall moves in the opposite direction.
```

The chain has four properties that organise the rest of the topic.

**Property 1 — Every service target is a functional, and four families cover the field.** A cycle
service level reads one quantile. A fill rate reads an expectation over the shortfall and depends on the
order quantity as well as the distribution. A ready rate reads a probability over time rather than over
cycles. A cost-optimal fractile reads the ratio of the two unit costs. The first and the fourth both
produce a quantile; the second and third do not, and treating them as though they do is the single most
common translation error in inventory analytics.

| Family | What it measures | Mathematical object | Depends on `Q`? |
|---|---|---|---|
| Cycle service level (CSL) | Probability that a cycle ends with no stock-out | `F(Q)`, one quantile | No |
| Fill rate (β) | Share of demanded units served from stock | `1 − E[(D−Q)⁺]/E[D]` | Yes, strongly |
| Ready rate | Share of time with stock on hand | `P(net stock ≥ 0)` at an arbitrary instant | Weakly, through the policy |
| Cost-optimal fractile | Order quantity minimising expected underage plus overage cost | `C_u/(C_u+C_o)` | No, in the single-period case |

**Property 2 — The quantile is a decision; the quantity is an instruction.** A planner can act on "the
0.678 quantile of four-week demand is 560 units". A warehouse cannot. Between the two sits rounding to
a case pack, a minimum order quantity (MOQ), a pallet quantity, a capacity cap, and a contract. Each of
those transformations changes the effective probability actually delivered. An organisation that
governs the quantile but not the transformations governs a number that never reaches a shelf.

**Property 3 — The same item deserves different quantiles in different contexts.** Underage and overage
costs are properties of the commercial relationship, not of the product. A seal sold on contract with a
per-unit shortfall penalty, a seal sold on the spot market, and a seal that stops an aircraft are the
same stock-keeping unit (SKU) with three different cost ratios and therefore three different target
quantiles. A single service target applied across all three is a target that is wrong for at least two
of them. The asymmetry is not a modelling nicety; it is the reason a service policy cannot be inherited
from a category rule.

**Property 4 — The tail is where the money is and where the data is thinnest.** The decision reads the
upper tail: the 0.9, the 0.95, the 0.98. That is exactly the region where a demand history of two years
contains a handful of observations, where a fitted normal or gamma family is least constrained by data,
and where a single promotional spike or supplier failure dominates the empirical estimate. The
practical consequence is not that tail estimation is hopeless. It is that tail estimates must be
reported with their sample size, tested for calibration at the level actually used, and widened by a
stated mechanism where the history cannot support the claim.

**Property 5 — The forecast quantile is not the policy quantile.** A predictive distribution is usually
produced at the forecast horizon — a week, a day, a month. A policy operates over a protection period:
lead time plus review period, inflated by lead-time variability, supply yield, and information delay.
The translation from one to the other is arithmetic, and it is arithmetic almost nobody shows. Section
6.7 walks a one-week 0.678 quantile of 148 units to a purchase quantity of 600 units, every step
justified.

**Property 6 — An unowned quantile drifts.** The parameter is invisible in normal operation and becomes
visible only when it fails. That asymmetry produces a ratchet: every stock-out generates a review, and
every surplus generates silence. Left alone, the target rises until the working capital objection
arrives, at which point it is cut by a percentage nobody can trace. Governance is the mechanism that
replaces the ratchet with a calendar.

> **Name the family. Fix the horizon. State the level. Compute the quantity. Price the choice.
> Register the parameter. Review on a calendar, not on an incident.**

Every failure mode in section 10 violates one of those seven clauses, and every exercise in section 12
practises one of them.

## 4. Definitions and vocabulary

The distribution vocabulary is defined in Module 08 topic 07. What is given here is the service and
policy vocabulary, with the unit and the coding case that decides whether the number you compute is the
number you meant.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Service level | A stated probability, proportion, or ratio describing how well demand is met | Used as though it had one agreed meaning | Must be qualified by family, horizon, and population |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as a fill rate; called "type 1" without saying which type-1 | Proportion, per cycle |
| Fill rate (β service level) | Fraction of demanded units served from stock within a period | Treated as a quantile at the same numeric level | Proportion of units, over a stated window |
| Ready rate | Fraction of time that stock is available on the shelf | Treated as identical to CSL | Proportion of time |
| Period service level | Probability that demand in an arbitrary period is fully met from stock | Conflated with the cycle version when cycles equal periods | Proportion, per period |
| γ service level | Time-weighted or backorder-weighted service measure | Assumed to be the same as fill rate | Proportion, weighting stated |
| On-time in full (OTIF) | Fraction of orders delivered complete and on the promised date | Quoted without the definition of "on time" | Proportion of orders |
| Perfect order fulfilment | Share of orders meeting all of completeness, timeliness, condition, and documentation | Defined differently by every organisation | Proportion of orders |
| Critical fractile | `C_u/(C_u+C_o)`, the cost-minimising quantile of a single-period order | Confused with a service-level target | Dimensionless, 0 to 1 |
| Underage cost (`C_u`) | Cost of being one unit short at the moment of demand | Stated as gross margin alone | Currency per unit, price year stated |
| Overage cost (`C_o`) | Cost of one unit left when its demand window closes | Stated as full unit cost, ignoring recovery | Currency per unit, price year stated |
| Cost-optimal quantile | The quantile level that minimises expected underage plus overage cost | Presented as the "correct" service level regardless of policy | Proportion, 0 to 1 |
| Policy quantile | The quantile level actually encoded in the ordering rule | Never written down anywhere | Proportion, 0 to 1 |
| Forecast quantile | The quantile of the predictive distribution at the forecast horizon | Used directly as a policy parameter | Proportion at a stated horizon |
| Protection period | Lead time plus review period over which stock must cover demand | Replaced by lead time alone in periodic review | Days or weeks |
| Lead-time demand (LTD) | Total demand realised between order placement and receipt | Computed on the planning bucket rather than the lead time | Units over `L` periods |
| Review period | Interval between successive replenishment decisions | Omitted from the protection period | Days |
| Safety stock | Buffer above expected protection-period demand | Treated as a policy constant rather than a quantile offset | Units |
| Reorder point | Inventory position at which a replenishment is triggered | Set as a percentage uplift on the forecast | Units |
| Order-up-to level | Target inventory position in a periodic-review policy | Confused with the order quantity | Units |
| Base-stock level | The order-up-to level in a continuous-review or unit-review policy | Used interchangeably with reorder point | Units |
| Expected shortage per cycle | `E[(D−Q)⁺]`, mean unmet demand at level `Q` | Omitted when fill rate is calculated | Units per cycle |
| Standard normal loss function | `L(z) = φ(z) − z(1−Φ(z))` | Unknown outside specialist teams | Dimensionless |
| Safety factor (`z`) | Number of standard deviations between the policy level and the mean | Quoted without the probability it encodes | Dimensionless |
| Service-level premium | Extra expected cost of a chosen quantile over the cost-optimal one | Never computed, so never defended | Currency per period |
| Risk appetite | The organisation's willingness to accept shortfall risk for a stated reason | Confused with a cost optimum | Policy statement plus a number |
| Chance constraint | A requirement that a probability or quantile stay inside a limit | Modelled as a mean constraint by mistake | Probability threshold |
| Conditional value at risk (CVaR) | Expected cost in the worst `α` fraction of outcomes | Reported as though it were a mean | Currency per period, at level `α` |
| Quantile crossing | Estimated quantiles out of order across levels | Silently sorted by the planning system | Data-quality defect |
| Tail calibration | Agreement between stated tail probabilities and observed frequencies | Assumed from central-mass calibration | Diagnostic with sample size |
| Parameter register | The record of every policy parameter, its value, owner, and history | Does not exist | Table keyed by item and version |
| Change control | The rule governing who may alter a parameter, with what evidence | Replaced by "the planner fixed it" | Process plus approval record |
| Ratchet effect | Asymmetric adjustment: rises after failures, never falls after surpluses | Described as "conservatism" | Trend in parameter values over time |
| Drift | Slow unlogged movement of a parameter until nobody can defend it | Discovered at audit | Difference between register and system |
| Make-to-order (MTO) | Production triggered by a confirmed customer order | Treated as a service-level problem when it is a lead-time promise | Order type |
| Assemble-to-order (ATO) | Final assembly triggered by order, from held components | Buffered at the wrong level | Order type |
| Stock-keeping unit (SKU) | The item at the finest level at which it is planned and stocked | Aggregated into a family and given the family target | Identifier |
| Service level agreement (SLA) | A contract fixing a service measure, threshold, window, and consequence | Assumed to use the same measure as the internal target | Contract |
| Minimum order quantity (MOQ) | Supplier-imposed floor on an order | Ignored when converting a quantile into an order | Units |
| Case pack | The indivisible multiple in which an item is ordered | Rounds away the whole decision at low volume | Units per case |
| Supply yield | Fraction of ordered quantity actually received | Assumed to be one | Proportion |
| Multi-echelon | A network in which stock is held at more than one level | Treated as a set of independent single-echelon problems | Network structure |
| Guaranteed service model (GSM) | A multi-echelon framework fixing a guaranteed replenishment time per stage | Confused with a stochastic service model | Model family |
| Bullwhip | Amplification of order variability moving upstream | Used to describe any variability | Variance ratio |
| Censored demand | Demand truncated by available stock, so unobserved | Forecast from sales and called it demand | Units, with a censoring flag |

Three groups carry most of the confusion, and each deserves a sentence.

**CSL, fill rate, and ready rate are three different promises.** CSL says "we rarely run out". Fill rate
says "we serve most of what is asked". Ready rate says "stock is usually there". A customer who orders
one unit occasionally experiences the first. A customer who orders in bulk experiences the second. A
customer who walks in unannounced experiences the third. Choosing among them is choosing which customer
experience the company is promising, which is a commercial decision and not a statistical one.

**Critical fractile and service target are numerically related and conceptually distinct.** The fractile
derives from costs and is the point that minimises expected cost. A service target is a commitment,
sometimes contractual, sometimes aspirational, sometimes a rule inherited from a template. When they
coincide, the business is lucky. When they differ, the business has chosen to spend money on service
beyond the cost-minimising point, and that choice should be stated with its price.

**Forecast quantile and policy quantile differ by a horizon and a supply process.** A forecast quantile
is a statement about demand at a forecast horizon. A policy quantile is a statement about the quantity
that must be available when an order arrives, given the lead time, the review period, the reliability of
the supplier, and the delay in the information system. The two are related by arithmetic that almost
never appears in a planning deck, which is why a model can be excellent and a policy can still be wrong.

## 5. The mechanics, step by step

Five phases and twenty-two steps, from the promise to the register. The sequence matters: the family is
chosen before the distribution is built, the distribution is built before the level is chosen, and the
level is chosen before the parameter is registered. Most organisations run it in the opposite order,
starting from a number that someone has already committed to and working backwards until the arithmetic
agrees.

### Phase A — Fix the promise

**Step 1. Write the service sentence, with the family named.**

A service sentence has six slots: the family, the level, the population, the horizon, the unit, and the
owner. "The fill rate for ambient grocery SKUs in the northern cluster, measured monthly on shipped
units against ordered units excluding cancellations, will be at least 97 per cent, owned by the category
supply manager." If the sentence cannot be completed, the target is not a target. It is a hope with a
number attached.

The test is mechanical: hand the sentence to a competent analyst in another team and ask them to compute
the number from the warehouse. If two of them produce different answers, the sentence is incomplete.

**Step 2. Check what the customer, or the contract, actually measures.**

The internal target and the contractual measure are frequently different families. A contract may
specify an order-line fill rate, a customer service team may measure OTIF at order level, and the
planning system may optimise a cycle service level. Three measures, three numbers, one relationship.

| Contractual measure | What it rewards | What it ignores | Compatible policy family |
|---|---|---|---|
| Line fill rate | Serving the bulk of each line | Frequent small shortfalls | Fill rate |
| Order fill rate | Complete orders | Large partial shipments | Fill rate at order level |
| OTIF | Complete and on time | Partial credit for near-misses | Joint fill and lead-time measure |
| Cycle availability | Never running out | Size of the shortfall | CSL |
| Downtime or penalty hours | Continuity of supply | Volume shortfalls | Ready rate or high CSL |

> **Note:** A vendor-managed inventory agreement that promises "98 per cent availability" and measures
> it as cycle counts is not the same promise as one that measures unit fill. Both can be honest. The
> error is to compute one and report the other, which is not dishonesty but is indistinguishable from
> it at audit.

**Step 3. Fix the horizon: the protection period, not the forecast horizon.**

The protection period is lead time plus review period. For a weekly review with a three-week lead time,
it is four weeks. For a continuous-review policy with a three-week lead time, it is three weeks. The
distribution must cover that period, because that is the interval during which the policy cannot
respond.

A common and expensive error is to build the distribution at the forecast horizon, which is usually the
planning bucket, and then apply it to the protection period without adjustment. Section 6.7 shows the
size of the resulting error and its direction.

**Step 4. Fix the unit and the time basis.**

A probability per cycle, a proportion of units, and a proportion of time are not interchangeable, and
they are not dimensionless in the same way. A cycle is a defined event with a start and an end; a unit is
a quantity; time is a duration. Write the unit in the field name. `csl_cycle` and `fill_rate_units` are
better names than `service_level`, because the second name will eventually be populated with whichever
one the integration happened to supply.

### Phase B — Build the policy distribution

**Step 5. Start from the predictive distribution at its own horizon.**

Topic 07 gives the object: a distribution over demand at a stated horizon, with calibration evidence and
declared exclusions. Do not rebuild it. The requirement here is that its horizon, its units, and its
conditioning set are known, because the next four steps transform it.

**Step 6. Aggregate to the protection period with the dependence assumption stated.**

If the forecast distribution is weekly and the protection period is four weeks, the four-week
distribution is not four times the weekly distribution. Under independence the variance adds; under
positive autocorrelation it adds faster; under negative autocorrelation it adds more slowly.

```text
   sigma over n periods = sigma_1 * sqrt( n + 2 * sum over k of (n-k) * rho_k )

   n = 4, sigma_1 = 60 units, rho_k = 0 for all k   -> sigma_4 = 120.0
   n = 4, sigma_1 = 60 units, rho_k = 0.3 for all k -> sigma_4 = 165.4
   n = 4, sigma_1 = 60 units, rho_k = 1 for all k   -> sigma_4 = 240.0
```

The dependence assumption is a modelling choice with a large effect and almost no visibility. State it.
Where the history is too short to estimate autocorrelation, run the decision under independence and
under a plausible positive value, and report both order quantities.

**Step 7. Add lead-time variability.**

Lead time is a distribution, not a number. The classical combined formula, in the form used in inventory
textbooks and attributed in the applied literature to Ballou and Srivastava, is:

```text
   sigma_LTD = sqrt( P * sigma_d^2 + d_bar^2 * sigma_L^2 )

   P       = number of periods in the protection period
   sigma_d = standard deviation of demand per period
   d_bar   = mean demand per period
   sigma_L = standard deviation of the lead time, in periods
```

With `P = 4`, `sigma_d = 60`, `d_bar = 120`, and `sigma_L = 0.6` weeks, the first term is `14,400` and
the second is `5,184`, giving `sigma_LTD = 139.9` units. Lead-time variability that looks small in
percentage terms contributes a third of the variance here, because it is multiplied by mean demand
squared.

Where the underlying history is intermittent, this formula is the wrong shape entirely: it assumes a
continuous demand rate multiplied by a continuous lead time, and it will produce a standard deviation
that corresponds to no process. Use the empirical lead-time-demand distribution instead, built by
simulating the arrival process.

**Step 8. Add supply variability.**

A supplier that delivers 95 per cent of what was ordered on average, with variation around that average,
changes both the level and the spread of the quantity required. The first-order correction is to inflate
the target by the reciprocal of the mean yield; the second-order correction is to add the yield variance
to the protection-period variance. The first-order correction alone is usually enough to stop systematic
understocking, and it is routinely omitted.

| Supply process | Effect on the required level | Practical treatment |
|---|---|---|
| Reliable, fixed lead time | None beyond demand | Standard quantile |
| Random lead time, reliable quantity | Widens the distribution | Combined variance formula, or simulation |
| Random quantity, fixed lead time | Raises the mean required | Inflate by `1/mean yield`; add yield variance |
| All-or-nothing availability | Creates a bimodal net-stock distribution | Simulation; the normal formula is invalid |
| Capacity-constrained | Truncates the achievable quantile | Report the achievable level, not the target |

The capacity row is the one most often missed. If the maximum obtainable quantity is `K`, the highest CSL
the policy can deliver is `F(K)`. A target above that is not a target; it is a supply failure recorded in
advance as a demand failure.

**Step 9. Add information delay and review latency.**

The order is placed on a Tuesday using inventory positions extracted on Monday night, which reflect
transactions up to Monday afternoon. The protection period is therefore four weeks plus the extraction
lag plus the confirmation lag. Each of these is small individually and additive in aggregate. Measure
them once, publish them, and add them to the protection period. Topic 09 of Module 01 covers the
diagnostic.

### Phase C — Choose the point

**Step 10. If the family is a cycle service level, read the quantile directly.**

`S = F⁻¹(q)`, read as the smallest `d` with `F(d) ≥ q`. Two conventions exist and they differ at a
support point: the smallest `d` with `F(d) ≥ q`, and the largest with `F(d) ≤ q`. State which one the
implementation uses. On a distribution with wide steps, the difference can be the decision.

**Step 11. If the family is a fill rate, solve the shortfall equation. Do not read a quantile.**

`β(Q) = 1 − E[(D−Q)⁺]/E[D]`. Solve for `Q`. There is no shortcut that survives contact with the data:
the fill rate depends on the order quantity, on the mean, and on the shape of the upper tail. A table that
maps a fill-rate target to a single quantile level is an approximation that holds only for one order
quantity and one distribution shape.

For normal demand the equation has a closed form:

```text
   beta(Q) = 1 - sigma * L(z) / Q ,  z = (Q - mu) / sigma

   L(z) = phi(z) - z * (1 - Phi(z))     the standard normal loss function
```

This is the classical Type 2 service formula. It is exact under the normal assumption, and it is the
right starting point for a fast-moving item. It is wrong for intermittent demand, where the atom at zero
and the discrete support dominate.

**Step 12. If the family is a ready rate, state the time basis and the net-stock definition.**

The ready rate is the long-run fraction of time with positive net stock available, equivalently the
probability that an arbitrary instant finds stock on hand. Under a base-stock policy with a fixed
protection period, it coincides numerically with the probability that protection-period demand does not
exceed the base-stock level, which is the same expression as a CSL over that period. The two diverge
when "cycle" is defined as something other than the protection period, and when demand is lumpy enough
that a period with positive net stock still contains an unfilled order.

The practical consequence: a ready rate reported as a CSL is usually right by accident and wrong in
principle, and the accident fails at the first policy change.

**Step 13. If the family is cost, compute the critical fractile.**

`q* = C_u/(C_u + C_o)`, derived in Module 07 topic 09. What that topic does not cover, and what matters
here, is that the ratio is a property of the commercial context rather than of the item. Section 6.6
computes three ratios for one SKU and obtains order quantities of 400, 560, and 700 units.

**Step 14. When the service target and the cost optimum disagree, decide which governs and price the
difference.**

They will disagree, usually with the service target above the cost optimum. That is not a defect. It is
a purchase of insurance, and it should be recorded as one.

```text
   premium per period = expected cost at the chosen quantile
                      - expected cost at the cost-optimal quantile

   expressed also as  = premium / expected cost at the optimum   (a percentage)
```

Three legitimate positions exist, and each should be written down:

| Position | Rule | When it is defensible | What must be recorded |
|---|---|---|---|
| Cost governs | Order at `q*` | No contractual promise; competitive market; low switching cost | The cost build-up and the price year |
| Promise governs | Order at the contractual level | A penalty or a lost customer is contractually or strategically real | The premium, and who approved it |
| Appetite governs | Order at a level chosen for risk reasons | Board-level risk appetite; regulated continuity; safety-critical item | The premium, the reason, the review trigger |

The failure is not choosing any of the three. It is choosing one by default and reporting it as another.

### Phase D — Multi-period and multi-echelon

**Step 15. Test whether the single-period fractile transfers.**

The newsvendor framing is single-period. In a multi-period base-stock policy with backorders, stationary
demand, and linear costs, the optimal order-up-to level is the critical fractile of the
protection-period demand distribution. The single-period arithmetic transfers, provided the distribution
is the right one. It does not transfer cleanly in four cases:

- **Carryover with high residual value.** If leftover units retain most of their value, the effective
  overage cost is the holding cost per period, not the full unit cost, and the fractile is much higher.
- **Lost sales rather than backorders.** The customer does not wait, so the cost of a shortfall includes
  the lost future relationship, and the effective underage cost is higher.
- **Non-stationary demand.** A fractile of a distribution that is shifting is not a policy parameter.
- **Perishability.** The item's life constrains how much can be carried forward, so the multi-period
  problem collapses toward the single-period one with a shorter effective horizon.

The safe statement: the fractile is the right first approximation, the protection-period distribution is
the right input, and the multi-period correction must be tested rather than assumed.

**Step 16. Recompute at the upstream echelon from the order distribution, not the demand distribution.**

The upstream echelon does not see customer demand. It sees the downstream echelon's orders, which are
batched, rounded, and shaped by the downstream policy. Its required quantile is therefore a quantile of
the order distribution, and the order distribution depends on the downstream service target.

This is the single most misunderstood point in the topic. A downstream target of 95 per cent CSL does not
imply an upstream target of 95 per cent of anything. It implies an order stream whose variability is a
function of the downstream policy, the review period, and the batching rules. Compute that stream, then
set the upstream quantile against it.

**Step 17. Check the safety-stock placement assumption.**

Multi-echelon optimisation places buffers where they buy the most service per unit of holding cost. Two
families dominate. The guaranteed service model fixes a promised replenishment time at each stage and
sizes buffers against the difference between promised and actual lead times. Stochastic service models
propagate distributions stage by stage. Both depend on assumptions about lead-time variability that are
routinely violated.

| Assumption | What the model does | What happens in practice |
|---|---|---|
| Deterministic stage lead times | Sizes buffers from lead-time differences | Variability is pushed upstream, where it is buffered at higher cost |
| Independent stage demand | Treats stages separately | Correlated demand across stages understates joint buffers |
| Reliable supply at every stage | No yield term in the recursion | Upstream stockouts appear as downstream demand |
| Fixed guaranteed times | Promises are met by construction | Expediting becomes the informal buffer |

The applied literature on stochastic lead times in guaranteed-service models makes the first row
concrete: heuristics that ignore lead-time variability, or that substitute the maximum lead time, both
mis-size the buffers, and the direction of the error depends on the network.

**Step 18. Check the joint service level across a family.**

If a customer needs several items together, the probability that all are available is the joint
probability, not the marginal one. Under independence, `0.95^10 = 0.599` and `0.95^50 = 0.077`. A
category reported at 95 per cent service can be at 60 per cent or 8 per cent for a customer who needs
the whole set.

This is a hierarchy fact as much as a service fact, and it is the reason sample-based reconciliation
matters: with coherent draws across the family you can count the joint event directly. With marginal
quantiles you cannot compute it at all.

### Phase E — Govern

**Step 19. Assign ownership at three tiers.**

| Tier | Owns | Cadence | Evidence required to change |
|---|---|---|---|
| Planner | Execution within the policy; flags when the policy does not fit reality | Daily or weekly | None; escalation instead |
| Planning leadership | The policy parameters: family, level, horizon, protection period | Monthly for fast movers, quarterly for slow | Cost build-up plus backtest |
| Finance and executive S&OP | The service target, because it sets working capital and the promise | Quarterly or annual | Premium calculation and approval record |

The separation is the control. A planner who can change the target in response to a stock-out is a
planner who will, and the ratchet follows.

**Step 20. Register the parameter set.**

Every policy parameter gets a row: item or segment, family, level, horizon, protection period,
distribution version, cost parameters and price year, owner, effective date, previous value, approver,
reason. The register is the artefact that makes drift visible, because drift is a difference between the
register and the system.

**Step 21. Fix the review calendar and the evidence required to change.**

A parameter changes on a scheduled review or not at all, except through an incident procedure that
requires an explicit justification and a counterfactual: what would the last twelve months have cost
under the proposed value? This single rule removes most of the ratchet, because most post-stock-out
adjustments do not survive a twelve-month counterfactual.

**Step 22. Audit the drift.**

Twice a year, compare the registered value with the value in the planning system for a sample of items.
Count the differences, the direction, and the age. A register that matches the system is a control that
works. A register that does not is a document nobody reads, and the audit result is the evidence for
asking who changed what.

### 5.7 Closing synthesis of the module

Three topics have run through this module, and they are not three subjects. They are three views of one
object.

**Hierarchy** is the view from above: the same demand observed at item, category, region, and quarter,
with the constraint that the parts add to the whole. **Intermittency** is the view from below: the same
demand observed at the grain where most periods are zero, so that the distribution has an atom and the
mean is a poor summary. **Uncertainty** is the view forward: the same demand observed as a distribution
rather than a number, with a shape, a tail, and a calibration claim.

Each view, pursued to its end, arrives at the same place. Hierarchy ends at a coherent joint
distribution from which someone must read a level. Intermittency ends at a distribution with an atom at
zero from which someone must read a level. Uncertainty ends at a calibrated predictive distribution from
which someone must read a level. The level is a quantile, and the quantile is a decision.

That is why this topic closes the module rather than opening it. The machinery of topics 01 to 09 exists
to produce an object; this topic is about the moment the object becomes an instruction. The instruction
leaves the analytics function at that point and enters the inventory policy, which is Module 10. What
this module owes Module 10 is a quantile that is correctly defined, correctly computed over the right
horizon, correctly calibrated in the tail, and correctly owned. What Module 10 owes in return is a
policy that consumes it without silently changing its meaning.

## 6. Worked example

A distributor of hydraulic components supplies one seal, item HS-2210, to three customer groups from one
warehouse. The forecast horizon is one week, the replenishment lead time is three weeks, and review is
weekly, so the protection period is four weeks. Demand is smooth at the weekly grain but not at the
daily grain. All money is euro at 2026 price levels. The distribution and the costs are constructed to
be internally consistent and to isolate the translation choices; they are not drawn from a published
case.

### 6.1 The protection-period distribution

Four-week demand, with the dependence between weeks already reflected:

| `d` (units) | 200 | 300 | 400 | 480 | 560 | 700 | 900 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p(d)` | 0.05 | 0.10 | 0.20 | 0.30 | 0.20 | 0.10 | 0.05 |
| `F(d)` | 0.05 | 0.15 | 0.35 | 0.65 | 0.85 | 0.95 | 1.00 |

The mean is `10 + 30 + 80 + 144 + 112 + 70 + 45 = 491` units. The median is 480. The 0.95 quantile is
700. The distribution is mildly right-skewed, which is why the mean sits above the median by 11 units.

### 6.2 Four families on one distribution

| Family | Rule | Quantity | Achieved CSL | Achieved fill rate |
|---|---|---:|---:|---:|
| CSL 0.95 | Read `F⁻¹(0.95)` | 700 | 0.950 | 0.9796 |
| Fill rate 0.95 | Solve `E[(D−Q)⁺] = 0.05 × 491 = 24.55` | about 603 | about 0.80 | 0.950 |
| Cost optimum | `C_u/(C_u+C_o) = 0.678` | 560 | 0.850 | 0.9369 |
| Median | `F⁻¹(0.50)` | 480 | 0.650 | 0.8798 |

The arithmetic for the fill rates:

```text
E[(D-400)+] = 80(0.30) + 160(0.20) + 300(0.10) + 500(0.05) = 24 + 32 + 30 + 25 = 111
              fill = 1 - 111/491 = 0.7739
E[(D-480)+] = 80(0.20) + 220(0.10) + 420(0.05) = 16 + 22 + 21 = 59
              fill = 1 - 59/491 = 0.8798
E[(D-560)+] = 140(0.10) + 340(0.05) = 14 + 17 = 31
              fill = 1 - 31/491 = 0.9369
E[(D-700)+] = 200(0.05) = 10
              fill = 1 - 10/491 = 0.9796
```

A 95 per cent fill rate needs an expected shortfall of 24.55 units per cycle, which lies between 31 at
560 and 10 at 700. Linear interpolation gives `560 + (31 − 24.55)/(31 − 10) × 140 = 603` units, an
approximation valid only because the support is coarse and the target is inside the interval.

Read the table by row. **A 95 per cent CSL is a weaker promise than a 95 per cent fill rate in quantity
terms** — it delivers 98 per cent of units — but it is a stronger promise in frequency terms, because it
guarantees stock on 95 per cent of cycles while the fill-rate policy runs out on one cycle in five. The
two statements are not competing descriptions of the same service. They describe different services, and
the choice between them is a choice about which failure the customer will experience.

### 6.3 The cost model

| Underage component | EUR/unit | Overage component | EUR/unit |
|---|---:|---|---:|
| Lost contribution margin | 38 | Unit cost | 40 |
| Expedite premium on recovery | 14 | Recovery on markdown or resale (subtracted) | −12 |
| Contractual shortfall penalty | 22 | Handling and reverse logistics | 3 |
| Goodwill and escalation time | 4 | Carrying cost over residual life | 2 |
| **Total `C_u`** | **78** | Obsolescence allowance | 4 |
| | | **Total `C_o`** | **37** |

```text
q* = C_u / (C_u + C_o) = 78 / (78 + 37) = 78 / 115 = 0.678

F(480) = 0.65 < 0.678 <= F(560) = 0.85   ->   Q* = 560 units
```

The fractile falls inside the 480–560 step, so the cost-minimising order on this discrete support is
560. On a continuous interpolation it would be about 535; on a discrete support the correct reading is
the smallest support point whose cumulative probability reaches the fractile.

### 6.4 Expected cost at each quantity

| `Q` | `E[(D−Q)⁺]` | `E[(Q−D)⁺]` | Expected cost (EUR) | Versus optimum |
|---:|---:|---:|---:|---:|
| 400 | 111 | 20 | `78(111) + 37(20) = 9,398` | +54% |
| 480 | 59 | 48 | `78(59) + 37(48) = 6,378` | +4.2% |
| 560 | 31 | 100 | `78(31) + 37(100) = 6,118` | optimum |
| 603 | 24.55 | 136.6 | `78(24.55) + 37(136.6) = 6,969` | +13.9% |
| 700 | 10 | 219 | `78(10) + 37(219) = 8,883` | +45.2% |
| 900 | 0 | 409 | `78(0) + 37(409) = 15,133` | +147% |

The overage expectations are computed from the same table, for example at `Q = 560`:
`360(0.05) + 260(0.10) + 160(0.20) + 80(0.30) = 18 + 26 + 32 + 24 = 100`.

Two readings matter. **The cost curve is flat near the optimum and steep above it.** Moving from 560 to
603 costs 14 per cent more; moving from 560 to 700 costs 45 per cent more. This asymmetry is the reason a
service target above the cost optimum needs an explicit justification: it is not a small rounding, it is
a purchase.

**The 95 per cent CSL policy is the most expensive of the four service families.** It costs 8,883
against 6,118 at the optimum, a premium of 2,765 euros per cycle, or 45 per cent. Whether that is worth
paying depends on what the promise is worth, which is a commercial question. What is not defensible is
paying it without knowing.

### 6.5 Three contexts, three quantiles

The same seal, sold three ways. The distribution does not change. Only the costs do.

| Context | `C_u` build | `C_o` build | `q*` | Order |
|---|---|---:|---:|---:|
| Contract supply, penalty clause | 38 + 14 + 22 + 4 = 78 | 40 − 12 + 3 + 2 + 4 = 37 | 0.678 | 560 |
| Spot market, no penalty, no expedite | 26 margin only | 41, with higher scrap risk | 0.388 | 400 |
| Line-stopping criticality | 26 + 400 downtime allocation | 41 | 0.912 | 700 |

The range is 300 units, a factor of 1.75, on one item in one warehouse in one week. The spot-market
customer gets 400 units and a 65 per cent cycle service level, because a leftover seal costs more than a
missed spot sale. The line-stopping customer gets 700 units, because an hour of downtime costs more than
a pallet of seals.

This table is the argument against category-level service rules. A rule that says "all hydraulic seals
at 95 per cent" is wrong for the spot business by 45 percentage points of cycle service and wrong for
the critical application by 27. The rule is administratively convenient and commercially indefensible,
and the cost of the indefensibility is computable.

> **Note:** The downtime allocation in the third row is the most contestable number in the table. It is
> an allocation of a plant-level cost to a unit of shortfall, and reasonable people will disagree about
> it by a factor of three. That is precisely why the sensitivity must be shown: the order quantity moves
> from 560 to 700 as the allocation moves from zero to 400 euros, and a reader can locate their own
> belief on that range.

### 6.6 Risk appetite above and below the optimum

The cost optimum is not the only defensible point, in either direction.

**Above the optimum.** A firm may choose 700 units for the contract item, accepting the 45 per cent
premium, because the penalty clause is not the only cost: a missed delivery triggers a qualification
review that costs months. That is a risk-appetite decision, and it is rational if the unmodelled cost is
real. It should be recorded as "we hold at the 0.95 quantile, at a premium of 2,765 euros per cycle,
because the qualification-review risk is not in `C_u`", with a review trigger.

**Below the optimum.** A firm may choose 480 units, accepting the 4 per cent cost increase over 560 and
a CSL of 0.65, because working capital is constrained and the marginal euro of inventory has a higher
return elsewhere. This is also a risk-appetite decision, and it is often made implicitly by a capital
ration rather than explicitly by a policy. Making it explicit converts an accident into a choice.

**The evidence.** The published work on risk-averse inventory finds the premium is usually modest at
moderate service levels and grows sharply at high ones. One simulation study imposing a 90 per cent
quantile constraint on a disservice measure reports a mean cost increase of about 2.5 per cent, and notes
that this may be prohibitive in a low-margin retail setting. A separate 2026 study of coherent risk
measures reports that at cost ratios at or below about 2:1, targeting the 90th percentile instead of the
median raises mean cost while lowering tail risk, and that both improve only at high ratios. The numbers
are context-specific; the shape is not.

### 6.7 The gap to the policy quantile

The forecast distribution is weekly. The policy needs four weeks, plus lead-time variability, plus
supply yield, plus rounding. Each step is arithmetic, and each is usually invisible.

| Step | Object | Level | Quantity | Basis |
|---|---|---:|---:|---|
| Forecast quantile, one week | weekly demand | 0.678 | 148 | `120 + 0.462 × 60` |
| Naive scaling by four | four-week demand | 0.678 | 591 | `4 × 148`, assumes perfect correlation |
| Correct, independent weeks | four-week demand | 0.678 | 535 | `480 + 0.462 × 120` |
| Correct, `rho = 0.3` | four-week demand | 0.678 | 556 | `480 + 0.462 × 165.4` |
| Add lead-time variability | protection-period demand | 0.678 | 563 | `480 + 0.462 × 180.4` |
| Add supply yield of 0.95 | ordered quantity | 0.678 | 593 | `563 / 0.95` |
| Round to a case pack of 24 | purchase order | — | 600 | `ceil(593/24) × 24` |

The chain runs from 148 to 600, a factor of four, and every step is defensible. The first number is what
the planner was shown. The last number is what the buyer must place. An organisation that reviews the
first and executes the last, without the intermediate rows, is reviewing a number that has no
operational meaning.

Three checks before trusting such a chain. Confirm that the dependence assumption is stated, because it
moves the four-week quantity by 21 units on its own. Confirm that lead-time variability is estimated
from receipts, not from the contractual lead time. Confirm that the yield correction uses realised
receipts against ordered quantities, not a supplier's stated on-time percentage, which measures a
different thing.

### 6.8 What the commercial audience sees

| Framing | Number | What the audience hears |
|---|---|---|
| "Service level is 95 per cent" | one number | "We are nearly there" |
| "95 per cent CSL delivers a 98 per cent fill rate" | two numbers | "So we are over-performing" |
| "95 per cent fill rate requires an 80 per cent CSL" | two numbers | "So the target is inconsistent" |
| "The three contexts need 400, 560, and 700 units" | three numbers | "Why is one rule not enough?" |
| "The 95 per cent policy costs 2,765 euros per cycle more than the optimum" | currency | "What does that buy?" |

The last row is the one that produces a decision. The first three produce a discussion about
definitions, which is necessary and does not by itself change anything.

## 7. Code, query, or calculation

The calculation that matters is the one that converts a fill-rate target into a policy level on a
normal-demand item, and the one that converts a cost ratio into a quantile. Both are short enough to do
on paper, and doing them on paper is the point: the numbers are arguable, and the argument is easier
when the arithmetic is visible.

Assumptions for the block below: protection-period demand is normal with mean 480 units and standard
deviation 120 units; the order quantity per cycle is 480 units, equal to the mean; the underage cost is
78 euros per unit and the overage cost 37 euros per unit. These are the discrete example's parameters,
rendered continuous so that the closed forms apply.

```text
NORMAL-DEMAND FILL RATE AND CRITICAL FRACTILE -- hand calculation, units and EUR

given  mu = 480   sigma = 120   Q = 480   beta = 0.95   C_u = 78   C_o = 37

fill-rate route, solve for the safety factor z
  L(z) = (1 - beta) * Q / sigma
       = 0.05 * 480 / 120 = 0.200          standard normal loss function
  L(z) = phi(z) - z * (1 - Phi(z))
  try z = 0.49: phi 0.3538, 1-Phi 0.3121, L = 0.3538 - 0.1529 = 0.2009
  try z = 0.50: phi 0.3521, 1-Phi 0.3085, L = 0.3521 - 0.1543 = 0.1978
  interpolate: z = 0.494
  CSL = Phi(0.494) = 0.689                 <- a 95% fill rate is a 69% CSL
  S   = mu + z * sigma = 480 + 59.3 = 539 units
  check: sigma * L(z) = 24.0 units short per cycle; 1 - 24.0/480 = 0.950

critical-fractile route, solve for the cost ratio
  q*  = C_u / (C_u + C_o) = 78 / 115 = 0.678
  z   = Phi^-1(0.678) = 0.462
  Q*  = mu + z * sigma = 480 + 55.4 = 535 units
  check: F(535) = 0.678; ordering one more unit has zero expected margin

read   the two routes land 4 units apart here, which is an accident of these
       parameter values. Change sigma or Q and they separate immediately.
```

Four checks before trusting any such calculation. Confirm that `Q` in the fill-rate formula is the order
quantity per cycle, not the reorder point; the two are easily swapped and the error is large. Confirm
that `z` is solved from the loss function rather than taken from a cycle-service table, because the two
tables look alike and give different answers. Confirm that `C_o` subtracts realised recovery rather than
adding it. Confirm that `sigma` is the standard deviation over the protection period, not over the
forecast horizon.

> **Note:** A 95 per cent fill rate on this item is a 69 per cent cycle service level. That is not a
> paradox and it is not an error in the formula. It is what a ratio of quantities and a probability of an
> event do when the event can be small and rare. The two numbers are only interchangeable at one special
> case, which is an order quantity of one unit.

## 8. Data, standards, and regulatory context

No international standard defines a service level, a fill rate, or a policy quantile. That absence is
informative: the choice is commercial, and must be documented rather than defaulted. What exists
constrains the vocabulary, the valuation of the two costs, and the control environment around the
parameter.

### 8.1 Vocabulary and measurement

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of probability, distribution, quantile, expectation | Any service measure | B |
| ISO 3534-2 | Applied statistics vocabulary | Terms for process and product measurement | Service-level terminology | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | Uncertainty, precision, trueness; the ban on "accuracy" as a quantity | Forecast or service measures | B |
| ISO 80000-1 | Quantities and units | Unit conventions for money, time, and dimensionless ratios | Which currency and price year | B |
| ISO 8601 | Date and time representation | Week numbering, durations, intervals | Which week a demand observation belongs to | B |
| ASCM SCOR DS | Supply chain reference framework | Metric definitions including perfect order fulfilment and its components | Any binding requirement; it is a reference model | C |

The VIM point transfers directly. "Service level of 95 per cent" is a conformance statement against a
definition and a sample, not a property of a supply chain. The standard does not forbid the phrase; it
forbids pretending the phrase has a unique referent. The ASCM dictionary and the SCOR framework are the
closest thing the field has to an authority on terminology, and they are a reference model rather than a
standard, which is why two organisations can both cite them and measure different things.

### 8.2 Where the costs become accounting numbers

`C_u` and `C_o` are the two numbers that set the quantile, and both are shadows of accounting rules
rather than operational constants.

| Requirement | Where it applies | Effect on the cost pair | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | IFRS-reporting entities | Lower of cost and net realisable value; write-downs raise `C_o` | B |
| ASC 330 *Inventory* | US GAAP entities | Lower-of-cost-or-market logic with industry practice | B |
| IAS 37 *Provisions* | IFRS-reporting entities | A contractual shortfall penalty is a provision; its best estimate belongs in `C_u` | B |
| IFRS 15 / ASC 606 | Revenue | Unsold units generate no revenue, so the full margin is forgone | B |
| Local waste and disposal regulation | Jurisdiction-specific | Disposal fees raise `C_o` directly | B |
| Transfer pricing rules | Multinational groups | The internal margin in `C_u` depends on the transfer price | B |

The practical consequence is that the finance partner and the commercial manager will state different
overage costs, and both will be right at different stages of the same event. Finance describes the
balance-sheet loss before recovery; commercial describes the realised markdown. The quantile depends on
which is used, so the reconciliation has to happen before the policy is set, not during the review.

### 8.3 Contractual instruments

| Instrument | What it fixes | Consequence for the quantile | Grade |
|---|---|---|---|
| Service level agreement (SLA) | The measure, threshold, window, exclusions, and remedy | The family is fixed; the internal target may be a different family | B |
| Vendor-managed inventory agreement | Who sets the level, who owns the stock, how availability is measured | The quantile may be contractually delegated, and the delegation is often incomplete | B |
| Take-or-pay capacity contract | Committed volume and penalty for unused capacity | The reservation is a high quantile of demand, and the penalty is the overage cost | B |
| Penalty and liquidated-damages clauses | A stated sum per unit short or per late order | Directly adds to `C_u` | B |
| Framework agreements with call-off | Volume bands and call-off windows | The protection period is contractual, not operational | B |

The contractual trap is familiar and worth restating in quantile terms: compute an internal fill rate,
compare it with a contractual target, and discover at audit that the contract excludes cancelled lines,
intercompany shipments, or substitutions. The distribution was arithmetically correct and contractually
irrelevant. State the contractual definition next to the modelled one, in the same document.

### 8.4 Internal control over the parameter

Service targets feed inventory valuation, revenue forecasts, and published service commitments. In
listed companies those figures sit within internal control over financial reporting.

| Requirement | Where | What it implies for the quantile | Grade |
|---|---|---|---|
| Sarbanes-Oxley Act 2002, s.404 | US-listed entities | The process that sets and changes the parameter is a control | B |
| Federal Reserve SR 11-7 | US bank holding companies | Model inventory, validation, ongoing monitoring — by analogy only | B |
| UK Corporate Governance Code | UK-listed entities | Internal control and risk management reporting | B |
| Local statutory audit requirements | Most jurisdictions | Parameters material to inventory valuation are auditable | B |

SR 11-7 is a template by analogy and not an obligation for a consumer goods company; citing it as though
it were is a category error. Its useful shape is the triad of model inventory, independent validation,
and ongoing monitoring, which is exactly what a service parameter needs and almost never gets. The
parameter register in Step 20 is the model inventory for this purpose; the drift audit in Step 22 is the
ongoing monitoring.

### 8.5 What is convention rather than requirement

The 95 per cent service level is a policy convention in most companies, not a regulatory requirement;
where it is a requirement, the instrument is a contract or a licence condition, and the instrument names
the family. The `z`-value table is a convention inherited from a normal approximation and a printed
appendix. The 80/15/5 ABC thresholds are a historical convention, not an optimum. The type 1 and type 2
numbering is a textbook convention that is not consistent across sources. Reporting fill rate while
optimising cycle service level is a software convention, not a deliberate design. An analyst who can
separate requirement from convention can challenge a convention; one who cannot treats every inherited
parameter as a physical constant.

## 9. Industry practice and case evidence

### 9.1 The definition dispute, documented

The disagreement is not a pedantic one. It is documented in industry bodies, in consultancy practice,
and in peer-reviewed attempts to choose between the measures.

| Evidence | What it reports | Grade |
|---|---|---|
| APICS industry discussion document on cycle service level and fill rate | Distinguishes Type 1 (event) from Type 2 (quantity); notes that business leaders usually want fill rate because it reflects the customer experience | C |
| Vandeput, "Service Level Definitions: Fill Rate, Cycle, and Period" | One dataset produces a 95.1 per cent fill rate, an 83.3 per cent period service level, and a 50 per cent cycle service level; most software optimises CSL while most practitioners track fill rate | C |
| nVentic, "Supply chain service levels" | Reports 99 per cent OTIF coexisting with 0 per cent fill rate in one example, and 99 per cent raw-material fill achieved at a 30 per cent CSL; notes CSL is not comparable across items with different order cycle lengths | C |
| Ribeiro, "Service Level Is Not One Metric" | Constructs two policies with identical 95 per cent fill rates and cycle service levels of 95 and 50 per cent, and two with identical 95 per cent CSLs and fill rates of 99.95 and 97.5 per cent | C |
| Bricker, University of Iowa lecture notes | A 99 per cent fill-rate target with `Q = 500`, `mu = 150`, `sigma = 50` gives a reorder point of 195, equivalent to an 81.6 per cent cycle service level | C |
| Guijarro Tarradellas et al., conference paper on customer service metric selection | States that the literature offers no guidance on metric selection, and proposes criteria based on customer expectation, demand behaviour, and decision purpose | B |

The fourth row is the cleanest statement of the problem in the material reviewed: the same number, two
policies, a 45-point gap in the measure the other party cares about. The second row is the most useful
for a practitioner, because it shows three legitimate measures computed on one dataset producing numbers
that a room would read as "good", "acceptable", and "failing".

### 9.2 The origin of the formulary, because the inheritance explains the habits

The service-level formulas most organisations use were not designed as a family. They accumulated.

| Contribution | Contribution to the formulary | Grade |
|---|---|---|
| Brown, *Statistical Forecasting for Inventory Control* (1959) | The safety-stock expression as safety factor times standard deviation of lead-time demand; the normal-approximation habit | B |
| Aucamp and Barringer (1987), *Journal of Operations Management* | Documents "Brown's method" for fill-rate service levels: compute a partial expectation, look the factor up in Brown's table, set safety stock | A |
| Silver (1970), *AIIE Transactions* | Shows the commonly used formula errs when forecast error is large relative to the replenishment quantity, which is the origin of the expected-shortage correction | A |
| Johnson and Montgomery, and later textbooks | Standardise the Type 1 and Type 2 distinction and the loss-function table | B |

The consequence of the inheritance is that two habits arrived together and were never separated. The
`z` table made quantile reasoning routine and simultaneously hid the probability inside a multiplier
called a safety factor, so that "1.65" became a configuration value rather than a statement that the
business had chosen a 95 per cent cycle service level. And the fill-rate table arrived as a second,
separate apparatus, so that the two measures were implemented in different modules of the same software
and never reconciled.

That is why a 2026 planning system can compute a fill rate correctly and a reorder point incorrectly,
and report both without noticing that they describe different policies.

### 9.3 The economics of going above the cost optimum

The question "what does a service target above the cost optimum buy" has a small but consistent
evidence base.

| Evidence | What it reports | Grade |
|---|---|---|
| Angün and Kleijnen (2024), Winter Simulation Conference | A 90 per cent quantile constraint on a disservice measure raises mean inventory cost by about 2.5 per cent against the risk-neutral optimum; notes this may be prohibitive at low retail margins | B |
| Ngartera et al. (2026), *Journal of Risk and Financial Management* | At cost ratios at or below about 2:1, moving from the median to the 90th percentile raises mean cost while reducing CVaR at the 95 per cent level; at high ratios both improve | A |
| Jammernegg and Kischka (2013), *International Journal of Production Economics* | A binding service constraint implies higher order quantities and risk-taking behaviour relative to a loss constraint | A |
| MetricGate practitioner documentation on the base-stock quantile policy | The cost-minimising quantile and a fixed target CSL give the same level only when the target equals the fractile | D |

The first two rows say the same thing in different languages: buying tail protection costs mean
performance, and the price is modest at moderate service levels and steep at high ones. The third says
something subtler and more useful for governance: once a service constraint binds, the firm is no longer
minimising expected cost and should stop describing its policy as cost-optimal.

### 9.4 Multi-echelon and lead-time variability in practice

| Evidence | What it reports | Grade |
|---|---|---|
| Humair, Ruark, Tomlin and Willems (2013), *Interfaces* | Extends the guaranteed-service model to stochastic lead times with closed-form safety-stock expressions; heuristics that ignore lead-time variability or use the maximum lead time mis-size buffers | A |
| Hosoda and Disney (2006), *Omega* | In a three-echelon order-up-to chain, variance amplification is determined by cumulative lead time from the customer and local replenishment lead time, not by echelon position | A |
| Urlu and Erkip (2020), *Flexible Services and Manufacturing Journal* | Safety-stock placement in serial systems under supply-process uncertainty; all-or-nothing availability changes the placement and the cost | A |
| Achkar, Brunaud and Grossmann, CMU working paper on multi-echelon inventory optimisation | Notes that stochastic-lead-time treatments can produce significant base-stock levels at upstream nodes, and extends lead-time-variance pushing downstream across divergent networks | B |
| Bryksina (2005), NCSU thesis on strategic safety-stock placement | Service performance degrades sharply as disruption probability or the worst-case-to-normal lead-time ratio rises, so analytic guaranteed-service models under-buffer | B |

The second row is the one to carry into a meeting: the upstream quantile is not a function of the
downstream service level. It is a function of the order stream the downstream policy generates, and of
the lead times along the path. A network that sets every echelon to 95 per cent has not set a coherent
policy. It has set five policies that happen to share a number.

### 9.5 The governance gap

| Evidence | What it reports | Grade |
|---|---|---|
| BEIREK, "Who Owns the Reorder Threshold" | Parameters loaded as provisional defaults at enterprise resource planning (ERP) go-live and never claimed; every stock-out raises a threshold and no surplus lowers one | D |
| Netray, inventory planning parameter governance checklist | Recommends monthly review for A items and quarterly or semi-annual for C items; materiality thresholds for second approval; logging of user, date, old and new value | D |
| Umbrex, inventory management playbook | Splits ownership across planner (values), planning leadership (policy), and finance or executive S&OP (targets) | D |
| Sarbanes-Oxley Act 2002, s.404 | Internal control over financial reporting, which covers parameters material to inventory valuation | B |

The vendor and consultancy sources are graded D because they are marketing or advisory content, not
evidence. Their value here is different: they are reporting a pattern that practitioners recognise, and
the pattern is consistent across all three. That is weak evidence of prevalence and no evidence of
effect size, and it should be cited as such.

### 9.6 Closing synthesis of the module, in practice

The module's three views meet in the governance layer, and the meeting is uncomfortable.

**Hierarchy** says the quantile must be coherent across levels, so the item-level level and the
category-level level cannot be chosen independently, and the joint service level is not the product of
the marginals.

**Intermittency** says the distribution has an atom at zero and a discrete support, so the quantile is
read from a staircase rather than a curve, and the fill-rate formula in its normal form is the wrong
instrument.

**Uncertainty** says the distribution is a calibrated claim with a tested tail, so the level chosen is a
statement about a probability that someone has measured and can be asked to defend.

All three end at the same object: a number in a parameter register, with a family, a horizon, a level,
and a name beside it. The field still argues about which family should be standard, whether the cost
optimum or the promise should govern, and how much tail protection is worth buying. Those arguments are
legitimate and unresolved, and they are resolved in each organisation by a decision rather than by a
literature.

What Module 10 inherits is the object this module produces. Whether it is used well is a question about
policy design, allocation, and multi-echelon coordination, and that is where the next module begins.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Family not named | Two reports show different service numbers for the same week and both are correct | CSL, fill rate, and ready rate are computed from the same data by different teams | Decisions made on incomparable numbers; arguments that never resolve | Publish a metric dictionary with numerator, denominator, and window for each family |
| CSL and fill rate used interchangeably | A target is described as "95 per cent" with no qualifier in any document | The two are numerically related at one order quantity and diverge everywhere else | Order quantities wrong by 20 to 40 per cent | State the family in the field name and in the target sentence |
| Fill-rate target applied as a quantile | The reorder point equals `F⁻¹(0.95)` for a 95 per cent fill-rate goal | Fill rate depends on the order quantity; a quantile does not | Systematic overstocking relative to the target | Solve the shortfall equation; report the implied CSL |
| Protection period wrong | The distribution is weekly and the lead time is three weeks | The quantile is computed on the forecast horizon, not the policy horizon | Under- or over-ordering by a factor set by the dependence | Build the distribution over lead time plus review period |
| Naive scaling of a quantile | The four-week level equals four times the one-week level | Multiplication ignores diversification and correlation | Over-protection under independence; exact only at perfect correlation | Aggregate the distribution, not the quantile |
| Lead-time variability omitted | Lead time is a single number in the parameter register | The supplier's observed spread is not modelled | Safety stock too low in exactly the periods that matter | Estimate the lead-time distribution from receipts |
| Supply yield ignored | The order equals the requirement with no yield term | The supplier delivers less than ordered, on average | Chronic understocking attributed to forecast error | Inflate by `1/mean yield`; add yield variance |
| Cost ratio inherited from a category | Every item in a family shares one `C_u` and `C_o` | Costs are properties of the commercial context, not the product | Wrong quantile for most items in the family | Segment the cost model; publish the segments |
| Service target above optimum, unpriced | Nobody can state what the target costs against the cost optimum | The purchase of insurance is invisible | Money spent without a decision | Compute the premium per period and record who approved it |
| Quantile re-tuned after every stock-out | The parameter history shows rises and no falls | Incident-driven adjustment with no counterfactual | Ratchet; inventory creeps; nobody can defend the level | Scheduled review; twelve-month counterfactual required for change |
| Upstream level copied from downstream | Every echelon in the network is set to the same service level | The upstream echelon faces an order stream, not customer demand | Incoherent policy; buffers in the wrong place | Recompute upstream from the order distribution |
| Joint service ignored | A family is reported at 95 per cent; customers who buy the set are at 60 per cent | Marginal targets do not determine the joint probability | Commercial failure hidden by a correct marginal metric | Compute the joint probability from coherent draws |

Two contrasts are worth stating in a review. **Family not named** is the most common failure because it
does not look like one: two competent analysts compute two correct numbers from the same warehouse and
neither is wrong, so the review becomes a recurring argument about arithmetic that neither party can
lose. **Upstream level copied from downstream** looks like consistency and is the opposite, because the
upstream echelon faces an order stream shaped by the downstream policy, not the demand that policy
serves.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The service family is identifiable and agreed | Every downstream step depends on which functional is read | The wrong summary passes into the policy and the error is invisible | Write the six-slot service sentence; ask two analysts to compute it |
| The protection period is known and stable | The distribution must cover the interval the policy cannot respond to | Quantiles computed on the wrong horizon | Compare modelled protection period with observed order-to-receipt time |
| Demand over the protection period is stationary | A quantile is a property of one distribution | Promotions, trends, and new listings move the target | Compare distributions across sub-periods; flag structural breaks |
| The dependence between periods is known | Aggregation from forecast horizon to protection period needs it | The four-week spread is wrong by tens of per cent | Estimate autocorrelation; run the decision under independence and a plausible positive value |
| Lead time is a distribution, not a number | Lead-time variability enters the variance directly | Safety stock too low where lead times are erratic | Plot lead time by lane over time; test for a trend |
| Supply yield is known and stable | The required order is the required receipt divided by yield | Chronic understocking | Compare receipts with orders by supplier and period |
| The cost pair is known well enough to place the quantile | The fractile is a ratio of two commercial quantities | The order sits at the wrong level | Compute the optimum across a plausible cost range and check whether the decision survives |
| The demand distribution is adequate in the tail used | The decision reads the 0.9 to 0.99 region | The tail is wrong where the cost is | Compare empirical and fitted tail quantiles; report calibration at the decision level |
| The single-period fractile transfers to the policy | Most operational policies are multi-period | Carryover, lost sales, or perishability shift the effective costs | Compare against a dynamic programme on simulated data |
| The hierarchy is stable over the horizon | Reconciliation and joint service need a fixed structure | A new item or a re-organisation invalidates the mapping | Version the hierarchy with the parameter |

The cost pair is the weak part, not the statistics: every quantile depends on two numbers the business
estimates approximately and sometimes politically, which is a reason to show the build-up and its range,
not a reason to hide both inside a default target. A quantile is also not a service outcome, because
execution, substitution, and the customer's willingness to wait intervene between the level and the
realised fill; track that gap rather than assume it away. No standard will settle the definition dispute,
because none of them says which family a supply chain should promise.

## 12. Exercises

### Exercise 1 — Classify four service sentences (analysis, 30 minutes)

For each sentence, identify the family, the level, the population, the horizon, the unit, and the missing
slot. (1) "We target 98 per cent availability on critical spares." (2) "The contract requires 97 per cent
line fill, measured monthly, excluding cancelled orders." (3) "The planner keeps the reorder point at the
95th percentile." (4) "Stock was on hand 96 per cent of the time last quarter."

**Deliverable:** a six-column table for each sentence, with the missing slots marked and a corrected
version of the sentence.
**Acceptance criteria:** every family correctly identified; the unit distinguished as probability,
proportion of units, or proportion of time; at least one sentence has two plausible readings and both are
stated.
**Time:** 30 minutes.

### Exercise 2 — Convert between CSL and fill rate (calculation, 40 minutes)

A normal-demand item has protection-period mean 240 units, standard deviation 60 units, and an order
quantity of 240 units per cycle. Compute the fill rate at a 95 per cent cycle service level, the cycle
service level implied by a 95 per cent fill rate, and the order quantity each requires.

**Deliverable:** the loss-function values, both conversions, both quantities, and a table of the four
combinations.
**Acceptance criteria:** `L(z)` shown at each step; the two quantities differ; a sentence explaining why
the order quantity enters the fill-rate calculation and not the CSL one.
**Time:** 40 minutes.

### Exercise 3 — Build a service table on a discrete distribution (calculation, 45 minutes)

Using the distribution below, compute the CSL, fill rate, and expected cost at quantities of 300, 400,
500, 600, and 700 units, with `C_u = 60` and `C_o = 25`.

| `d` | 200 | 300 | 400 | 500 | 600 | 800 |
|---|---:|---:|---:|---:|---:|---:|
| `p(d)` | 0.10 | 0.20 | 0.25 | 0.25 | 0.15 | 0.05 |

**Deliverable:** the cumulative column, the two tail expectations at each quantity, the cost column, and
the identified optimum.
**Acceptance criteria:** probabilities sum to one; the optimum matches the critical fractile read from
the staircase; the table shows at least one quantity where the CSL ranking and the cost ranking disagree.
**Time:** 45 minutes.

### Exercise 4 — Trace the translation chain (application, 50 minutes)

For an item with a weekly forecast distribution, a four-week lead time, weekly review, lead-time standard
deviation of 0.8 weeks, and a supplier delivering 92 per cent of ordered quantity on average, build the
chain from the one-week 0.9 quantile to the purchase order quantity.

**Deliverable:** a seven-row table in the style of section 6.7, with the basis for each row, plus the
final order rounded to a case pack of 12.
**Acceptance criteria:** the dependence assumption is stated; lead-time variability is included with its
arithmetic; the yield correction is applied to the quantity, not to the level; the rounding is shown.
**Time:** 50 minutes.

### Exercise 5 — Price a service premium (analysis, 35 minutes)

Using the table from Exercise 3, compute the expected cost at the cost optimum, at the 0.30 quantile, and
at the 0.95 quantile. Express each difference from the optimum in currency per cycle and as a percentage.

**Deliverable:** a three-row table with the two premium measures, plus a one-paragraph recommendation on
which level to adopt for a contract with a per-unit penalty.
**Acceptance criteria:** the premium is computed against the optimum and not against the current policy;
the percentage uses the optimum as the denominator; the recommendation names the evidence that would
change it.
**Time:** 35 minutes.

### Exercise 6 — Segment one item into three contexts (design, 45 minutes)

Take one real or constructed item and specify `C_u` and `C_o` for three commercial contexts: a contracted
customer with a penalty, a spot customer, and an application where downtime dominates. Show the three
critical fractiles and the three resulting order quantities on one distribution.

**Deliverable:** three cost build-ups with named components and a price year, three fractiles, and three
quantities.
**Acceptance criteria:** every component is named and signed; recovery is subtracted from `C_o`; the
quantity range is computed rather than asserted; the most contestable component is flagged with a
sensitivity range.
**Time:** 45 minutes.

### Exercise 7 — Audit for drift (critique, 40 minutes)

Obtain or construct twelve months of parameter history for five items: the registered service level, the
value in the planning system, the date of each change, and the reason recorded. Assess the arrangement
against the three-tier ownership model.

**Deliverable:** a table of changes with direction and rationale, a drift measure (final value minus
registered value), and three findings.
**Acceptance criteria:** the direction of every change is classified as rise, fall, or correction; the
counterfactual requirement is checked; at least one finding is about the process rather than the number.
**Time:** 40 minutes.

### Exercise 8 — Multi-echelon coherence (analysis, 55 minutes)

A three-echelon chain sets every echelon to a 95 per cent cycle service level. Explain why this is not a
coherent policy, describe what the upstream echelon actually faces, and specify the computation you
would run to set the upstream level correctly.

**Deliverable:** a one-page explanation, a diagram of the order stream at each echelon, and a numbered
procedure for the upstream computation.
**Acceptance criteria:** the distinction between customer demand and order stream is explicit; the
downstream policy parameters that shape the order stream are listed; the procedure names its inputs and
its dependence assumptions.
**Time:** 55 minutes.

### Exercise 9 — Write the governance note (design, 50 minutes)

Write a one-page governance note for a service parameter on a product family: the family, the level, the
horizon, the owner at each tier, the review calendar, the evidence required to change, and the drift
audit.

**Deliverable:** the note plus a parameter register template with at least ten fields.
**Acceptance criteria:** ownership is separated across three tiers; the review cadence differs for fast
and slow movers; the change evidence includes a counterfactual; the audit has a frequency and a
tolerance.
**Time:** 50 minutes.

### Exercise 10 — Synthesise the module (design, 40 minutes)

Write a 600-word memo arguing that hierarchy, intermittency, and uncertainty are three views of one
problem, using one concrete example to carry the argument. End by stating what Module 10 must do with
the quantile your example produces.

**Deliverable:** the memo with a single worked thread through all three views.
**Acceptance criteria:** each view is defined in one sentence; the example is carried through all three
rather than described three times; the handover to Module 10 is specific about what is passed and what
is not.
**Time:** 40 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

| # | Family | Level | Population | Horizon | Unit | Missing |
|---|---|---:|---|---|---|---|
| 1 | Ambiguous: CSL, fill, or ready | 0.98 | Critical spares | Unstated | Unstated | Horizon, unit, owner, family |
| 2 | Fill rate, line level | 0.97 | Contracted lines | Monthly | Proportion of units | Owner; definition of a cancelled order |
| 3 | CSL | 0.95 | Items held by that planner | Protection period, unstated | Probability per cycle | Population, horizon, owner |
| 4 | Ready rate | 0.96 | All stocked items | Last quarter | Proportion of time | Definition of "on hand", treatment of zero-demand periods |

Sentence 1 has at least three plausible readings and the corrected version must choose one: "The fill
rate for critical spares at the central warehouse, measured monthly on units shipped against units
requested, will be at least 98 per cent, owned by the service-parts manager." Sentence 4 needs the
zero-demand treatment stated, because a period with no demand and no stock is counted as available by
some conventions and unavailable by others, and the difference moves the number by points.

### 13.2 Exercise 2

```text
given  mu = 240   sigma = 60   Q = 240

CSL 0.95 -> z = 1.645
  L(1.645) = 0.10314 - 1.645(0.05) = 0.0209
  expected short = 60 * 0.0209 = 1.25 units per cycle
  fill rate      = 1 - 1.25/240 = 0.9948
  quantity S     = 240 + 1.645(60) = 338.7 -> 339 units

fill rate 0.95 -> L(z) = (1 - 0.95) * 240 / 60 = 0.200
  z = 0.494 ; CSL = Phi(0.494) = 0.689
  quantity S = 240 + 0.494(60) = 269.6 -> 270 units
```

| Policy | Quantity | CSL | Fill rate |
|---|---:|---:|---:|
| CSL 0.95 | 339 | 0.950 | 0.995 |
| Fill rate 0.95 | 270 | 0.689 | 0.950 |

The order quantity enters the fill-rate calculation because a shortfall is spread over the units
ordered: the same expected shortage is a smaller fraction of a larger order. The CSL does not contain
`Q` at all, because it is the probability that the level is not exceeded, and the level is the level.

### 13.3 Exercise 3

Cumulative: `0.10, 0.30, 0.55, 0.80, 0.95, 1.00`. Mean `435` units. Critical fractile
`60/85 = 0.7059`, which falls between `F(400) = 0.55` and `F(500) = 0.80`, so the optimum is 500.

| `Q` | CSL | `E[(D−Q)⁺]` | `E[(Q−D)⁺]` | Fill rate | Cost (EUR) |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.30 | 145 | 10 | 0.667 | `60(145) + 25(10) = 8,950` |
| 400 | 0.55 | 75 | 40 | 0.828 | `60(75) + 25(40) = 5,500` |
| 500 | 0.80 | 30 | 95 | 0.931 | `60(30) + 25(95) = 4,175` |
| 600 | 0.95 | 10 | 175 | 0.977 | `60(10) + 25(175) = 4,975` |
| 700 | 0.95 | 5 | 270 | 0.989 | `60(5) + 25(270) = 7,050` |

The disagreement is at 600: it has the highest CSL of the five and a cost 19 per cent above the
optimum. The 700 row shows the same CSL as 600 and a worse cost, because the extra 100 units have no
effect on the probability of stock-out and a large effect on the expected overage. That is a discrete
support effect and it is not a defect: it says that 600 already satisfies the 0.95 level, and anything
above it is pure cost.

### 13.4 Exercise 4

Worked with `rho = 0.1` at all lags and a case pack of 12.

| Step | Object | Level | Quantity | Basis |
|---|---|---:|---:|---|
| Weekly quantile | one week | 0.90 | 151 | `100 + 1.2816(40)` |
| Naive scaling by five | five weeks | 0.90 | 756 | `5 × 151` |
| Independence | five weeks | 0.90 | 615 | `500 + 1.2816(89.4)` |
| `rho = 0.1` | five weeks | 0.90 | 636 | `500 + 1.2816(105.8)` |
| Lead-time `sd` 0.8 weeks | protection period | 0.90 | 670 | `500 + 1.2816(132.7)` |
| Yield 0.92 | ordered quantity | 0.90 | 728 | `670 / 0.92` |
| Case pack 12 | purchase order | — | 732 | `ceil(728/12) × 12` |

The variance arithmetic behind rows three to five: `89.4 = 40√5`; `105.8 = 40√(5 + 2(0.1)(10))`;
`132.7 = √(11,200 + 10,000(0.64))`. The naive row is the largest because multiplying a quantile by `n`
multiplies the safety margin by `n`, which is an upper bound attained only under perfect correlation.

### 13.5 Exercise 5

| Policy | Quantity | Cost (EUR) | Premium (EUR) | Premium (%) |
|---|---:|---:|---:|---:|
| Cost optimum | 500 | 4,175 | 0 | 0.0 |
| 0.30 quantile | 300 | 8,950 | 4,775 | 114.4 |
| 0.95 quantile | 600 | 4,975 | 800 | 19.2 |

The asymmetry is the point. Under-ordering by two support steps costs 114 per cent more than the
optimum; over-ordering by one step costs 19 per cent. A business that responds to every stock-out by
raising the level is making 19 per cent errors repeatedly and 114 per cent errors never; a business that
responds to every surplus by cutting is doing the reverse. Neither habit is a policy.

A defensible recommendation for a contract with a per-unit penalty: hold at the 0.95 quantile only if
the unmodelled consequence of a shortfall — the escalation, the qualification review, the relationship —
is worth at least 800 euros per cycle. Otherwise hold at the optimum and disclose the CSL of 0.80 in the
customer-facing service report.

### 13.6 Exercise 6 rubric

A strong answer builds each cost from named components with a price year and a sign. The contracted
context adds the penalty and the expedite premium to the margin. The spot context removes both and
raises the overage cost through scrap risk. The downtime context allocates a plant-level cost per hour
to a unit of shortfall and states the allocation basis, because the number is otherwise unarguable. It
flags the downtime allocation as the most contestable component and gives a range. It then shows the
three fractiles on one distribution and the three quantities, and notes which pairs of contexts are
close enough to share a target and which are not. A weak answer produces three numbers without
component build-ups, or applies a single cost ratio and varies only the service target.

### 13.7 Exercise 7 rubric

A strong audit tabulates every change with date, old value, new value, approver, and recorded reason,
then classifies direction. It computes drift as the difference between the registered and live values
and reports it per item and in aggregate. It checks whether any change was accompanied by a
counterfactual, and finds that most were not. Its three findings should include at least one process
finding — for example, that the register is maintained by the same role that adjusts the parameter,
which defeats the separation — and at least one about the direction of drift, typically that rises
outnumber falls by a wide margin. A weak audit lists changes without direction or approval data and
concludes that governance could be improved.

### 13.8 Exercise 8 model answer

A uniform 95 per cent CSL at every echelon is not coherent for three reasons. The upstream echelon does
not face customer demand; it faces the downstream echelon's orders, which are batched, rounded, and
shaped by the downstream policy, so the variance it must cover differs in both level and shape. The
protection period differs by echelon, because lead times and review periods differ, so the same numeric
service level corresponds to different quantities. And the joint service experienced by the end
customer is not the product of the marginals, so the network's realised service is not 95 per cent at
any level of aggregation that matters commercially.

The computation is: simulate or derive the order stream at each echelon given the downstream policy
parameters — review period, order-up-to level, batching rule, minimum order quantity — aggregate that
stream over the upstream protection period, add lead-time and yield variability at that stage, and read
the upstream level from the resulting distribution. The dependence assumptions between stages must be
stated, because they determine how the stage distributions combine.

### 13.9 Exercise 9 rubric

A strong note states the family in the title, not in a footnote, and gives the level to two decimal
places with its unit. It names a role for each of the three tiers and states what each may change. The
review cadence differs by velocity: monthly for A items, quarterly for C items, with an incident
procedure that requires a counterfactual. The register template carries item, family, level, horizon,
protection period, distribution version, cost pair and price year, owner, effective date, previous value,
approver, and reason — eleven fields, one more than the exercise requires. The audit names a frequency,
a sample size, and a tolerance for drift. A weak note names a department rather than a role, or gives a
single annual review for everything.

### 13.10 Exercise 10 rubric

A strong memo defines the three views in one sentence each and then carries one item through all three:
the same item observed at category level is a hierarchy problem, at weekly grain is an intermittency
problem, and as a distribution is an uncertainty problem, and all three routes produce a level that
someone must defend. It should be explicit that the three are not alternative methods but alternative
observations of one object, and that the reconciliation between them is where the module's difficulty
lives. The handover paragraph should state what Module 10 receives — a calibrated, coherent,
correctly-horizoned quantile with a named owner — and what it does not receive, namely the policy design,
the allocation rule, and the multi-echelon placement. A weak memo describes the three topics in
sequence and appends a paragraph about how they are all important.

## 14. Interview questions

**1. "What is a 95 per cent service level?"**

Strong: asks which family, then defines CSL as the probability that a cycle ends without a stock-out,
fill rate as the share of units served, and ready rate as the share of time with stock on hand, and notes
that a 95 per cent CSL and a 95 per cent fill rate are different promises requiring different
quantities. Weak: gives one definition without qualification, which reveals that the candidate has not
had to implement one.

**2. "Why is a 95 per cent CSL a weaker promise than a 95 per cent fill rate?"**

Strong: because the CSL counts events and the fill rate counts units, so a policy can run out rarely and
badly, or often and trivially; gives the arithmetic at a typical coefficient of variation, where 95 per
cent CSL implies roughly 99.5 per cent fill and 95 per cent fill implies roughly 69 per cent CSL. Weak:
says one is stricter without being able to say in which direction or by how much.

**3. "Derive the order quantity for a fill-rate target."**

Strong: writes `β(Q) = 1 − E[(D−Q)⁺]/E[D]`, notes there is no closed-form quantile, uses the normal loss
function where the distribution justifies it, and says the answer depends on the order quantity as well
as on the distribution. Weak: reads the 0.95 quantile and calls it a 95 per cent fill rate, which is the
most common error in the field.

**4. "How does a service target become an order quantity?"**

Strong: describes the chain — distribution, protection period, dependence assumption, lead-time
variability, supply yield, rounding, minimum order quantity — and says each step moves the number and
each is owned by a different function. Weak: says the system does it, which is true and does not answer
the question.

**5. "Why does the same item need different service levels in different contexts?"**

Strong: because the critical fractile is a ratio of underage to overage cost, and those costs are
properties of the commercial relationship; gives two or three contexts and the resulting fractiles.
Weak: says service levels are a commercial judgement, which is true but does not explain why the
quantity changes.

**6. "A business sets every item at 95 per cent. What is wrong with that?"**

Strong: identifies the category-rule failure, notes that costs differ by contract, shelf life, and
criticality, and quantifies the spread on one item, typically 40 to 75 per cent in order quantity. Weak:
says 95 per cent is arbitrary, which is a different and weaker criticism.

**7. "How would you set the service level at an upstream echelon?"**

Strong: says the upstream echelon faces an order stream, not customer demand, so the level must be
computed from the downstream policy's order distribution plus the upstream protection period and supply
variability. Weak: says the same level as downstream, or a higher one to be safe, which is the
incoherence the question is testing for.

**8. "What does a service target above the cost optimum buy, and what does it cost?"**

Strong: it buys tail protection and reduces the frequency of severe shortfalls; it costs the expected
cost premium, which is modest at moderate service levels and steep at high ones; and the premium should
be computed and recorded with the reason. Weak: says it buys customer satisfaction, which is not
quantified and cannot be reviewed.

**9. "How do you stop a service parameter drifting after every stock-out?"**

Strong: separates the roles that execute from the roles that set the parameter, fixes a review calendar,
requires a counterfactual for any change, and audits the register against the live system twice a year.
Weak: says to monitor it closely, which is the behaviour that produced the drift.

**10. "Is a good forecast enough for good service?"**

Strong: no — the forecast is an input, and the policy, the protection period, the supply process, and
the allocation rule all intervene; a well-calibrated distribution consumed through a wrong protection
period produces a wrong quantity. Weak: yes, which confuses model quality with decision quality.

**11. "What would you put in a one-page note setting a service level?"**

Strong: the family, the level, the population, the horizon, the protection period, the distribution
version and its calibration evidence, the cost pair with its price year, the premium over the cost
optimum, the owner at each tier, the review date, and the exclusions. Weak: a number and a target date.

## 15. Key terms

- **Service level** — an umbrella term; always qualify with family, level, population, and horizon.
- **Cycle service level (CSL)** — probability that a replenishment cycle ends without a stock-out.
- **Fill rate (β)** — share of demanded units served from stock; depends on the order quantity.
- **Ready rate** — share of time with stock available; a time-based, not cycle-based, measure.
- **Period service level** — probability that demand in an arbitrary period is fully met.
- **γ service level** — a time-weighted or backorder-weighted service measure.
- **OTIF** — on-time in-full; a joint completeness and timing measure.
- **Critical fractile** — `C_u/(C_u+C_o)`; the cost-minimising quantile in the single-period problem.
- **Underage cost (`C_u`)** — cost of one unit short, including margin, expedite, and penalty.
- **Overage cost (`C_o`)** — cost of one unit left, net of recovery, plus handling and carrying.
- **Cost-optimal quantile** — the fractile read on the demand distribution; not necessarily the policy.
- **Policy quantile** — the level actually encoded in the ordering rule.
- **Forecast quantile** — the level of the predictive distribution at the forecast horizon.
- **Protection period** — lead time plus review period; the interval the policy cannot respond to.
- **Lead-time demand** — demand realised between order placement and receipt.
- **Safety stock** — buffer above expected protection-period demand; a quantile offset in units.
- **Standard normal loss function** — `L(z) = φ(z) − z(1−Φ(z))`; converts a factor to a shortfall.
- **Safety factor** — `z`; a probability disguised as a multiplier.
- **Service-level premium** — expected cost of the chosen level minus that of the cost optimum.
- **Risk appetite** — the organisation's stated willingness to accept shortfall risk, with a reason.
- **Chance constraint** — a requirement that a probability or quantile stay inside a limit.
- **CVaR** — conditional value at risk; expected cost in the worst `α` fraction of outcomes.
- **Parameter register** — the record of every policy parameter, its owner, and its history.
- **Ratchet effect** — asymmetric adjustment upward after failures and never downward after surpluses.
- **Drift** — unlogged movement of a parameter until nobody can defend it.
- **Joint service level** — probability that a set of items is simultaneously available.
- **Supply yield** — fraction of ordered quantity actually received; divides the required order.
- **Censored demand** — demand truncated by available stock, so unobserved in the sales series.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Aucamp, D. C., and Barringer, R. L. (1987). "A Table for the Calculation of Safety Stock." *Journal of
  Operations Management*, 7(1–2), 153–163. DOI: 10.1016/0272-6963(87)90014-3.
- Silver, E. A. (1970). "A Modified Formula for Calculating Customer Service Under Continuous Inventory
  Review." *AIIE Transactions*, 2(3). DOI: 10.1080/05695557008974758.
- Humair, S., Ruark, J. D., Tomlin, B., and Willems, S. P. (2013). "Incorporating Stochastic Lead Times
  Into the Guaranteed Service Model of Safety Stock Optimization." *Interfaces*, 43(5).
  DOI: 10.1287/inte.2013.0699.
- Hosoda, T., and Disney, S. M. (2006). "On Variance Amplification in a Three-Echelon Supply Chain with
  Minimum Mean Square Error Forecasting." *Omega*, 34(4), 344–358.
- Urlu, B., and Erkip, N. K. (2020). "Safety Stock Placement for Serial Systems Under Supply Process
  Uncertainty." *Flexible Services and Manufacturing Journal*. DOI: 10.1007/s10696-019-09374-3.
- Jammernegg, W., and Kischka, P. (2013). "Risk Preferences of a Newsvendor with Service and Loss
  Constraints." *International Journal of Production Economics*, 143(2), 410–415.
- Ngartera, I., Nadarajah, S., Rodoumta, G., and Gningue, Y. (2026). "Stochastic Inventory Optimization
  with Coherent Risk Measures: A Decision-Theoretic Framework for Probabilistic Forecasting and
  Constrained Optimization." *Journal of Risk and Financial Management*. DOI: 10.3390/jrfm19030173.
  Numerical results reported as the published abstract states them; not independently replicated.
- Lee, D., and Rim, S.-C. (2019). "A Mathematical Safety Stock Model for DDMRP Inventory
  Replenishment." *Mathematical Problems in Engineering*. DOI: 10.1155/2019/6496309.

**Working papers, conference proceedings, and theses — Grade B**

- Angün, E., and Kleijnen, J. P. C. (2024). "Black-Box Simulation-Optimization with Quantile Constraints:
  An Inventory Case Study." *Proceedings of the 2024 Winter Simulation Conference*; also CentER
  Discussion Paper 2024-005, Tilburg University. Reports a mean cost increase of about 2.5 per cent; the
  figure is as the paper states it.
- Achkar, V. G., Brunaud, B., and Grossmann, I. E. "Multi-Echelon Inventory Optimization Based on the
  Guaranteed Service Model." Working paper, Carnegie Mellon University.
- Bryksina, O. (2005). *Assessing the Impact of Strategic Safety Stock Placement in a Multi-Echelon
  Supply Chain.* Master's thesis, North Carolina State University.
- Guijarro Tarradellas, E., et al. (2013). "On the Selection of Customer Service Metrics in Inventory
  Control." Conference paper, indexed in Dialnet.
- Gudum, C. K., and de Kok, A. G. (2002). "A Safety Stock Adjustment Procedure to Enable Target Service
  Levels in Simulation of Generic Inventory Systems." Report, Copenhagen Business School.

**Standards, official, and regulatory — Grade B**

- ISO 3534-1, *Statistics — Vocabulary and Symbols — Part 1*. Current edition.
- ISO 3534-2, *Statistics — Vocabulary and Symbols — Part 2: Applied Statistics*.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM), 3rd edition.
- ISO 80000-1, *Quantities and Units — Part 1: General*.
- ISO 8601, *Date and Time — Representations for Information Interchange*.
- IAS 2, *Inventories*; IAS 37, *Provisions, Contingent Liabilities and Contingent Assets*. IFRS
  Foundation.
- ASC 330, *Inventory*; ASC 606, *Revenue from Contracts with Customers*. Financial Accounting Standards
  Board.
- IFRS 15, *Revenue from Contracts with Customers*. IFRS Foundation.
- Sarbanes-Oxley Act of 2002, Section 404, internal control over financial reporting.
- Board of Governors of the Federal Reserve System, SR 11-7, *Guidance on Model Risk Management*. Cited
  by analogy; not an obligation for non-bank entities.

**Books and professional references — Grade B or C**

- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill. The origin of the
  safety-factor-times-standard-deviation formulation. Grade B.
- Silver, E. A., Pyke, D. F., and Thomas, D. J. *Inventory and Production Management in Supply Chains*.
  Current edition. Grade B.
- Zipkin, P. H. *Foundations of Inventory Management*. McGraw-Hill. Grade B.
- APICS/ASCM Dictionary, current edition, entries for cycle service level, fill rate, and ready rate.
  Grade C.
- ASCM SCOR Digital Standard, metric definitions including perfect order fulfilment. Grade C.
- Bricker, D. "Reorder Point" lecture slides, University of Iowa. Grade C; teaching material.
- Rossetti, M. D. *Analysis of Inventory Systems*, chapters 7 and 8. Open textbook. Grade C.

**Practitioner, consultancy, and vendor — Grade C, D, or E**

- APICS industry discussion document, "Cycle Service Level and Fill Rate." Grade C.
- Vandeput, N. "Service Level Definitions: Fill Rate, Cycle, and Period." Practitioner article. Grade C.
  Reports the 95.1 / 83.3 / 50 per cent example as its own worked case.
- nVentic, "Supply chain service levels." Consultancy insight. Grade C.
- Ribeiro, D. "Service Level Is Not One Metric." Practitioner analysis, 2025. Grade C.
- InventoryPath, "Service Level vs Fill Rate." Vendor blog. Grade D.
- Kvantis Atlas, "Service level and fill rate" and "The newsvendor model." Vendor knowledge base.
  Grade D.
- MetricGate, "Base-Stock Quantile (Newsvendor Order-up-to) Policy" and service-level calculator
  documentation. Grade D.
- Netray, "Inventory Planning Parameter Governance Checklist" and service-level and loss-function
  tables. Grade D.
- Umbrex, *Inventory Management Playbook*, roles and governance section. Grade D.
- BEIREK, "Who Owns the Reorder Threshold: The Quiet Governance Gap in Inventory Triggering." Vendor
  blog. Grade D.
- Open practitioner discussions on safety stock with a fill-rate criterion, including the OR
  StackExchange thread on the fill-rate criterion. Grade E; useful for seeing what practitioners ask,
  not for the formulas.

**Note on citations.** Bibliographic details were checked against the sources returned by the searches.
Volume and page numbers are given where confirmed and omitted where not. The figures attributed to
Angün and Kleijnen, Ngartera et al., Vandeput, nVentic, and Bricker are reported as those sources state
them and were not independently replicated. The distribution and cost model in section 6 are
constructed and internally consistent, not drawn from a published case, and every derived number shows
its derivation. Where a source is a vendor or consultancy publication it is graded D or E and used only
as evidence of what practitioners say, never as evidence of what is true.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_scoring-rules.md`](09_scoring-rules.md)
- The distribution this topic consumes: [`07_probabilistic-forecasts.md`](07_probabilistic-forecasts.md)
- Distribution-free intervals: [`08_conformal-prediction.md`](08_conformal-prediction.md)
- Intermittent demand and the atom at zero: [`05_intermittent-classification.md`](05_intermittent-classification.md)
- Hierarchy and aggregation: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- The newsvendor derivation this topic extends: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Inventory policy that consumes the quantile: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Intermittent measures and the zero problem: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Return to the part map: [`../_index.md`](../_index.md)
