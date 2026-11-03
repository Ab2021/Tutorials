# Newsvendor and the Critical Fractile — optimal order quantity under a single selling season

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 03 of 10
> Estimated study time: 3 hours
> Prerequisites: Module 07 topic 09; Module 08 topic 10; Module 10 topics 01–02
> Version 2026.5

---

## 1. Why this topic exists

A merchandiser places an order in March for a range that will sell between September and January.
The factory needs eight weeks, the fabric is dyed to a colour and cannot be re-cut, and there is no
second order. If the range sells out in week three, the season is lost. If it sells through only half,
the remainder leaves through an outlet at a fraction of cost. Every season, somebody commits several
million euros of open-to-buy on a spreadsheets cell that contains a forecast plus a percentage.

This is the newsvendor problem, and it is the smallest complete model of an inventory decision. One
order, one demand distribution, one selling season, a salvage value for what is left, no opportunity
to reorder. From those four facts the optimal quantity follows in a page of algebra, and that quantity
is not the mean, not the mean plus a safety margin, and not the same number as last year.

**Why a fractile and not a mean.** A mean is a property of the demand. A fractile is a property of the
demand *and* of the relative cost of being wrong in each direction. Two items with identical demand
distributions and different margins do not deserve the same order, and no forecast improvement will
fix a rule that ignores the difference. The critical fractile is the point on the distribution at
which the expected cost of ordering one more unit equals the expected cost of not ordering it. It is a
position in the distribution, not a location, and installing that idea is this topic's job.

**Why it sits third.** Topic 01 built the cost–service frontier; topic 02 built the economic order
quantity (EOQ), which assumes certain, steady demand. The newsvendor answers the question the frontier
implies but never computes: given a cost structure, where should the order sit? Everything later is a
generalisation. Safety stock is the fractile applied to lead-time demand, multi-echelon optimisation
applies it at each stage, and pooling applies it to an aggregated distribution whose shape is better
than any of its parts.

**Who is blocked.** The merchandiser who cannot say what risk the order carries, the planner whose
system computes a quantity from a service level nobody chose, the finance partner asked to fund
inventory without seeing what is being bought, the data scientist whose calibrated distribution is
consumed as a mean plus a percentage, and the category manager applying one 95 per cent rule to forty
stock-keeping units (SKUs) whose cost structures differ fivefold.

**Three failures recur.**

The first is the **uplift order**: a forecast plus 15 per cent "for safety". The uplift is a habit,
not a policy, and silently encodes a cost ratio nobody has examined. On a high-margin short-life
product it is far too low; on a low-margin long-life product, far too high. Never written as a ratio,
it is never challenged.

The second is the **unwritten cost pair**: the underage cost taken as gross margin, the overage as
unit cost, and the expedite premium, penalty clause, goodwill effect, and recovery value left out.
The omissions do not cancel; they usually push the same way, and the order lands in the wrong place
behind a defensible-looking calculation.

The third is the **single-period model applied unadjusted to a repeated decision**. The newsvendor's
overage cost is the full value destroyed at season end; in replenishment nothing is destroyed, because
the leftover carries forward. Applying the seasonal pair to a weekly problem overstates the cost of a
leftover and systematically under-orders, which then gets blamed on the forecast.

**What this topic does not repeat.** Module 07 topic 09 introduces the fractile at point-forecast
level; Module 08 topic 10 covers the decision use of quantiles and the forecast-versus-policy
distinction. What is added is the optimisation: the cost pair built component by component, the
discrete and continuous cases, sensitivity, the budget-constrained extension, and the limits.

> **Note:** The critical fractile is also called the critical ratio, the optimal service level, or
> Littlewood's rule depending on the field. It is one object. What differs between fields is which
> costs sit on each side, and that is where the errors live.

## 2. Learning outcomes

By the end of this topic you can:

1. **Test** whether a named decision satisfies the four conditions of a single-period problem — one
   order, one season, a salvage value, no reorder — and state what changes when one fails.
2. **Derive** the critical fractile from the marginal argument without a textbook, and **explain** why
   the answer is a fractile rather than a mean.
3. **Build** an underage and overage cost from signed components with a price year, including lost
   contribution margin, expedite premium, contractual penalty, and goodwill, and **name** what is
   most often omitted and what is most often double-counted.
4. **Compute** the optimal quantity on continuous and discrete demand, and **explain** why the
   discrete answer is the smallest support point reaching the fractile rather than an interpolation.
5. **Quantify** the sensitivity of the quantity to the cost ratio, and **state** when a small cost
   error produces a small quantity error and when it does not.
6. **Translate** the seasonal cost pair into the repeated order-up-to problem, naming the protection
   period and the substitution of holding cost for unit cost.
7. **Solve** a budget-constrained multi-item allocation with a Lagrangian multiplier, and **explain**
   why it does not give every item the same service level.
8. **Audit** a cost ratio imputed from a service target, and **write** the paragraph that breaks the
   circularity honestly.

Each outcome maps to an exercise in section 12 and to an answer or rubric in section 13.

## 3. The core mental model

Hold one picture. A cumulative distribution of demand rises from left to right. A horizontal line
crosses it at a level set by the cost ratio. The order quantity is where the line meets the curve,
read down onto the axis. Move the line, not the curve: the curve is the forecast's job, the line is
the commercial decision.

```text
   CUMULATIVE DEMAND, ONE SEASON, ONE STYLE

   F(d)
   1.00 |                                                  o
        |                                            o
   0.90 |                                     o
        |                              o
   0.724|- - - - - - - - - - - - - - o- - - - - - - -  <- critical fractile
        |                        o                        q* = 70.20/(70.20+26.75)
   0.50 |                 o
        |          o
   0.25 |    o
        |
   0.00 +----|--------|--------|--------|--------|------> d, units
             2,600    3,500    4,800    5,572    6,500
                                          ^
                                          |
                              Q* = F^-1(0.724) = 5,572 units

   The fractile is a horizontal level. The quantity is the vertical read-off.
   Change the cost ratio and the line moves; the curve does not.
```

The level comes from a comparison of two expected marginal costs, and the comparison is the model:

```text
   ONE MORE UNIT ORDERED, BEFORE DEMAND IS KNOWN

                     +---- demand exceeds it, probability 1 - F(Q)
                     |        the unit sells; contribution C_u is earned
                     |        expected gain   (1 - F) * C_u
   unit Q+1 ---------+
                     |
                     +---- demand does not exceed it, probability F(Q)
                              the unit is left; C_o is lost
                              expected loss   F * C_o

   order while   (1 - F) * C_u  >  F * C_o
   rearrange     F  <  C_u / (C_u + C_o)
   at equality   F(Q*) = C_u / (C_u + C_o)

   equal costs     C_u = C_o      ->  F = 0.50   the median
   3 : 1           C_u = 3 C_o    ->  F = 0.75
   9 : 1           C_u = 9 C_o    ->  F = 0.90
   19 : 1          C_u = 19 C_o   ->  F = 0.95
```

Six properties organise what follows.

**Property 1 — The model compares two expected marginal costs; it does not forecast.** The forecast
supplies `F`, the cost pair supplies the level, and the decision is the intersection. A better
forecast does not rescue a wrong cost pair, and a wrong cost pair wastes a good forecast.

**Property 2 — The fractile is dimensionless; the quantity is not.** The fractile is a ratio of two
currencies per unit, so the currency cancels. The quantity inherits every property of the demand
distribution: scale, skew, atoms, support. Changing currency does not move the fractile; changing the
unit of measure moves the quantity and the rounding.

**Property 3 — The answer is a point on the distribution, not on the calendar.** The model knows how
much, not when. Two seasons with the same total distribution and different arrival timing produce the
same order: a strength for one season, a fatal weakness once a replenishment option exists inside it.

**Property 4 — The interesting number is the gap between the fractile and the median.** At equal
costs the answer is the median; the gap is what the cost asymmetry buys and the only part of the
answer the cost analysis contributes. Reporting the quantity without the fractile hides which input
moved.

**Property 5 — Everything absent from the cost pair is assumed to be zero, and the zeros are
decisions.** If the expedite premium is not in `C_u`, the model has decided it is zero. If the penalty
clause is absent, the model has decided the contract has no consequence. If recovery is not
subtracted from `C_o`, the model has decided leftovers are worthless. None of these is neutral and
none is visible in the output.

**Property 6 — The single-period answer is a boundary case of the repeated answer.** With stationary
demand, linear holding cost `h` per unit per period, and linear backorder penalty `b` per unit per
period, the optimal order-up-to level satisfies `F(S*) = b/(b + h)`. The algebra is identical; the
meanings change. `C_o` becomes the cost of carrying a unit for one period rather than destroying it,
`C_u` becomes the penalty per period rather than the lost margin, and the distribution is over the
protection period rather than the season. Substituting `h` for the full unit cost raises the fractile
sharply, which is why a seasonal pair applied to a replenishment problem under-orders.

The discipline that follows is one sentence long:

> **Write the cost pair, name the distribution, read the fractile, then check which kind of problem
> you are actually solving.**

Every failure mode in section 10 violates one clause, and every exercise in section 12 practises one.

## 4. Definitions and vocabulary

The forecasting and quantile vocabulary is defined in Module 08 topic 10. What follows is the
single-period decision vocabulary, with the unit and the coding case that decides whether the number
you compute is the number you meant.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Single-period problem | One order, one demand realisation, no opportunity to reorder | Applied to any item with a forecast and called it a season | Decision type, declared in the model inventory |
| Selling season | The interval in which a unit can be sold at full price | Treated as a calendar quarter rather than a demand window | Start and end dates, ISO 8601 |
| Order quantity (`Q`) | Units committed before demand is observed | Called a forecast, which it is not | Units, saleable eaches |
| Demand realisation (`D`) | The quantity that would have sold had stock been unlimited | Replaced by observed sales, which are censored | Units, season total |
| Critical fractile (`q*`) | `C_u/(C_u+C_o)`; the cost-minimising cumulative probability | Called the optimal service level without naming the family | Proportion, 0 to 1 |
| Underage cost (`C_u`) | Expected loss from being one unit short at the moment of demand | Stated as gross margin only, or as price | Currency per unit, price year stated |
| Overage cost (`C_o`) | Expected loss from one unit left when the window closes | Stated as full unit cost, ignoring recovery | Currency per unit, price year stated |
| Lost contribution margin | Selling price minus variable cost, forgone on a lost sale | Replaced by revenue, which overstates it by the cost | Currency per unit |
| Salvage value (`s`) | Realised net recovery per leftover unit | Taken as the markdown price rather than the net receipt | Currency per unit, net of costs |
| Expedite premium | Extra cost of recovering a shortfall through air freight, overtime, or a spot buy | Omitted, so recovery is assumed impossible | Currency per unit recovered |
| Goodwill cost | Present value of future cash flows lost when a customer defects | Either omitted or inflated without a basis | Currency per unit short |
| Liquidated damages | A contractual sum payable per unit short or per late order | Left out because it sits in a different budget | Currency per unit or per order |
| Carrying cost | Cost of holding one unit for one period: capital, storage, insurance, obsolescence | Quoted as an annual percentage and applied to the wrong base | Currency per unit per period, or percent per year |
| Censored demand | Demand truncated by available stock, so the excess is unobserved | Forecast from sales and called it demand | Units, with a censoring flag |
| Empirical distribution | The distribution formed by observed demand values, each with mass equal to its frequency | Smoothed until the atoms disappear | Probability mass per support point |
| Demand atom | A demand value with strictly positive probability | Treated as an artefact to be interpolated away | Units at one support point |
| Support | The set of demand values with positive probability | Assumed continuous when the item moves in cases | Sorted unit values |
| Cumulative distribution function (`F`) | Probability that demand is at most a given quantity | Read backwards, giving the survivor function | Probability, 0 to 1 |
| Expected shortage | `E[(D−Q)⁺]`, mean unmet demand per season | Omitted when the fill rate is calculated | Units per season |
| Expected leftover | `E[(Q−D)⁺]`, mean unsold quantity per season | Called waste, which it may not be | Units per season |
| Expected mismatch cost | `C_u` times expected shortage plus `C_o` times expected leftover | Reported as total cost, which it is not | Currency per season |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as a fill rate; equals the fractile only in the single-period case | Proportion per cycle |
| Fill rate | Share of demanded units served from stock | Used as though it were a quantile | Proportion of units |
| Order-up-to level (`S`) | Target inventory position in a periodic-review policy | Confused with the order quantity | Units |
| Protection period | Lead time plus review period; the interval the policy cannot respond to | Replaced by lead time alone in periodic review | Days or weeks |
| Backorder | Demand unmet now and satisfied later | Treated as a lost sale, which changes `C_u` completely | Units, with an age |
| Lost sale | Demand unmet and not recovered, so the margin is forgone | Assumed for every shortfall when some customers would wait | Units |
| Case pack | The indivisible multiple in which an item is ordered | Rounds away the decision at low volume | Units per case |
| Minimum order quantity (MOQ) | Supplier-imposed floor on an order quantity | Ignored when converting a fractile into an order | Units |
| Budget constraint | A cap on total purchase value or inventory investment across items | Applied per item, which removes the allocation problem | Currency per season |
| Lagrangian multiplier (`λ`) | Shadow price of a binding constraint, in objective units per constraint unit | Read as a probability or a tax rate | Euros of expected profit per euro of budget |
| Pull-to-centre | Behavioural tendency to order between the optimum and the mean | Described as risk aversion, which the experiments ruled out | Units, or proportion of the gap |

Three groups carry most of the confusion.

**Fractile, ratio, service level, Littlewood.** One object, four names, four communities. The
operations researcher means the cumulative probability; the revenue manager the booking limit at
which the next fare equals the expected displacement cost; the planner whatever the system labels a
service level. When two of them meet they agree on the number and disagree on what it controls.
Write the definition, not the name.

**Underage cost and lost margin.** The margin is one component of `C_u`. A shortfall that can be
expedited costs the premium on the recovered share and the margin on the unrecovered share, not both
on every unit. A chargeback adds to the margin; a qualification clause adds a relationship asset with
no accounting line.

**Salvage value and clearance price.** Salvage is what the company receives per leftover unit, net of
getting it there, not the ticket price and not zero. Moving it from €38 to €20 on a €62 unit shifts
the order by about 400 units, more than most forecast improvements.

## 5. The mechanics, step by step

Seven phases and twenty steps. The order matters: frame the problem before building the costs, build
the costs before fixing the distribution, fix the distribution before reading the fractile, and read
the fractile before rounding. Most organisations run it backwards, starting from a quantity somebody
has already committed to.

### Phase A — Confirm that it is a single-period problem

**Step 1. Write the decision sentence.** "On 15 March we commit to a quantity of style SK-114 for the
autumn–winter season ending 31 January, with no reorder, owned by the category manager." Five slots:
commitment date, quantity, demand window, reorder opportunity, owner. If the reorder slot cannot be
filled with "none", the problem is not a newsvendor.

**Step 2. Test the four conditions.** One order: no committed quantity can change after demand is
seen. One season: unsold stock has no second window at a different price. Salvage value: some market
exists for leftovers, however poor. No reorder: lead time exceeds the useful season. Three of four is
common; say which fails and what it does to the answer. A fast-fashion range with a three-week
replenishment inside a twelve-week season is a two-stage problem, and treating it as a newsvendor
under-orders the first buy and ignores the option value of the second.

**Step 3. Fix the window and the unit.** The window is the season, not the fiscal quarter. The unit is
the saleable each at the finest level at which demand is distinguishable, because a colourway that
cannot be substituted is a different SKU. Use ISO 8601 dates so two systems agree on the weeks.

### Phase B — Build the two costs

**Step 4. Build the underage cost from signed components.**

| Component | How to compute it | Frequency of omission |
|---|---|---|
| Contribution margin on the unrecovered share | `(p − c) × (1 − r)` | Low; usually the only component used |
| Expedite premium on the recovered share | `(emergency cost − c) × r` | High |
| Contractual penalty or chargeback | Contract terms, weighted by covered volume | High |
| Goodwill and future cash flow | Defection probability × lifetime contribution, per unit short | Very high, or grossly overstated |
| Service-desk and escalation cost | Labour hours × loaded rate | Medium |
| Scorecard consequence | Expected future volume effect of a vendor downgrade | High |

The recovery probability `r` is the pivot. A shortfall recovered by air freight with probability `r`
costs the premium on that fraction and the margin on the rest. Adding both on every unit
double-counts the recovered ones and can push `C_u` above the selling price. If `r` is unknown, run
`r = 0` and `r = 1` and report both; they are the bounds.

**Step 5. Build the overage cost from signed components.**

| Component | How to compute it | Frequency of omission |
|---|---|---|
| Unit cost committed | Landed cost, including freight and duty | Low |
| Recovery on clearance | Net receipt after markdown, handling, and outlet costs | Medium; often entered without the costs |
| Handling and reverse logistics | Labour and transport to the clearance channel | High |
| Carrying cost to the clearance date | `c × rate × days / 365` | Medium |
| Obsolescence and disposal | Unsaleable fraction × disposal cost | High for fashion and perishables |

**Step 6. Check the pair for double counting and sign errors.** Three checks take two minutes. Confirm
the margin in `C_u` is a contribution, not a price; using price overstates `C_u` by the unit cost,
which is a factor of two on a 50 per cent margin item. Confirm recovery is subtracted from `C_o`, not
added; a misplaced sign turns valuable salvage into a disposal fee and roughly halves the order.
Confirm the expedite premium and the margin apply to complementary shares.

**Step 7. State the price year and currency.** Costs drift, and a fractile computed on 2023 costs is
not a fractile for 2026. Write "EUR, 2026 price level" beside the pair. If you index, index both sides
by the same factor or the ratio moves for reasons unrelated to the decision.

### Phase C — Fix the demand distribution

**Step 8. Choose between the empirical distribution and a fitted family.** The empirical distribution
preserves atoms but its tail is one observation; a fitted family smooths the tail but imposes a
shape, and a normal fit on right-skewed seasonal demand understates the upper tail where high service
levels read. Use the empirical with twenty comparable seasons, a fitted family with a stated shape
with fewer than ten, and a scaled analogue for a new product, labelled as an assumption.

**Step 9. Correct for censoring.** Sales are not demand: a sold-out season records sales equal to the
order, and everything above is unobserved. Correct with a survival model, or at minimum flag censored
seasons and report with and without them.

**Step 10. Fix the horizon and check the scaling.** The distribution must cover the season total, in
saleable eaches, for the same stores as the order. Weekly demand times 22 is not the season
distribution unless weeks are perfectly correlated; under independence the spread grows with the
square root of the horizon.

### Phase D — Read the fractile

**Step 11. Compute `q* = C_u/(C_u+C_o)` and state it as a probability.** Read it aloud: "the order
should sit at the 72nd percentile of season demand". Outside `(0, 1)`, a cost has the wrong sign: a
negative `C_o` means a leftover is worth more than the unit cost, so order to the top of the support;
a negative `C_u` means the price is below cost, and the model should not be used.

**Step 12. Read the quantity, continuous case.** `Q* = F⁻¹(q*)`: `Q* = μ + σΦ⁻¹(q*)` for normal,
exponentiate a log-scale calculation for lognormal, invert numerically for gamma. State the
convention at a support point, because the smallest value with `F ≥ q*` and the largest with `F ≤ q*`
differ by a unit and occasionally a case.

**Step 13. Read the quantity, discrete case.** `Q* = min{Q in support : F(Q) ≥ C_u/(C_u+C_o)}`. Not an
interpolation and not a rounded continuous answer: every unit between two atoms has the same sale
probability, so the gap adds overage without reachable sales, and interior quantities are dominated
by an endpoint. The answer therefore over-protects by a fraction of a step.

**Step 14. Apply the operational rounding and say what it costs.** Case pack, pallet, MOQ, container
fill, and capacity each move the quantity after the model has finished. Round as the constraint
requires, then recompute the achieved fractile and expected mismatch cost. On a slow mover one case
can be 20 to 40 per cent of the mean, and the rounding is the decision.

### Phase E — Price the answer

**Step 15. Compute the expected mismatch cost at the optimum and at the alternatives.**
`E[cost] = C_u × E[(D−Q)⁺] + C_o × E[(Q−D)⁺]`. Evaluate at the optimum, the current policy, the
contractual service level if one exists, and the median. The curve is flat near the optimum and steep
away from it, and the asymmetry is the argument for or against a target above the optimum.

**Step 16. Report the premium, not just the difference.** The premium is the cost at the chosen
quantity minus the cost at the optimum, in currency per season and as a percentage of the optimum. A
target above the optimum is a purchase of insurance against a tail the cost model does not contain.
That is legitimate and should be recorded as a purchase, with the reason and the approver. What is
not legitimate is presenting it as the cost-optimal answer.

### Phase F — Multi-item under a constraint

**Step 17. Set up the constrained problem, not a per-item rule.** With budget `B` and items indexed by
`i`, maximise `Σ E[profit_i(Q_i)]` subject to `Σ c_i Q_i ≤ B`. The Lagrangian is
`L = Σ E[profit_i(Q_i)] − λ(Σ c_i Q_i − B)`, and the first-order condition is

`F_i(Q_i) = (p_i − c_i − λ c_i) / (p_i − s_i)`, with `λ ≥ 0` and `λ(Σ c_i Q_i − B) = 0`.

At `λ = 0` this reduces to the unconstrained fractile. As `λ` rises, every fractile falls and the
budget goes to the items whose marginal expected profit per euro is highest. The rule is therefore
**equal marginal return per euro of budget, not equal service level**.

**Step 18. Solve for `λ` by monotone search, and handle the corner case.** Spend decreases in `λ`, so
bisection converges: slack budget, `λ = 0`; moderately tight, all quantities positive; very tight,
some fractiles fall below zero, and those items are dropped and the remainder re-solved.

### Phase G — Hand it to the policy

**Step 19. Check which problem you are actually solving.** If the item will be replenished inside the
horizon, replace the seasonal `C_o` with the holding cost per period and the seasonal `C_u` with the
per-period penalty, then rebuild the distribution over the protection period. Section 5.7 gives the
translation. Applying the seasonal pair to a replenishment problem is the most consequential error
here, because it is invisible: the arithmetic is right, the model is right, and the problem is wrong.

**Step 20. Register the inputs.** Record the cost pair, price year, recovery probability,
distribution and version, fractile, quantity, rounding rule, owner, and review date. The register is
what makes the next review a comparison rather than an archaeology exercise.

### 5.7 The repeated setting, stated precisely

| Element | Single period | Repeated, periodic review |
|---|---|---|
| Distribution | Season demand | Demand over lead time plus review period |
| Overage cost | `c − s +` handling `+` carrying | `h`, the cost of holding one unit for one period |
| Underage cost | Lost margin plus penalty plus goodwill | Backorder penalty per period, or lost margin if sales are lost |
| Quantity | Order placed once | Order-up-to level `S`, order `= S −` inventory position |
| Fractile | `C_u/(C_u+C_o)` | `b/(b + h)` |

For normal protection-period demand with mean 1,200 units and standard deviation 260 units, a holding
cost of €0.60 per unit per week and a backorder penalty of €6.00 per unit per week give a fractile of
`6.0/6.6 = 0.909`, a safety factor of 1.335, and an order-up-to level of 1,547 units. The same item
with a seasonal salvage would sit near 0.72. The difference is not a modelling preference. It is the
difference between destroying a unit and carrying it.

> **Note:** `F(S*) = b/(b+h)` holds for stationary demand, linear costs, and backorders filled when
> stock returns. With lost sales the exact optimal policy is not a simple fractile, because the
> stockout removes demand from future periods as well. The fractile remains the right first
> approximation and the right language; it is not an exact solution.

## 6. Worked example

A mid-size skiwear brand commits to one style of insulated jacket, SK-114, for a single autumn–winter
season. The order goes in March with a five-month lead time; the fabric is dyed to a colourway, the
factory has a minimum run, and there is no in-season replenishment. The style sells through a
wholesale account with a service chargeback and through the brand's own e-commerce channel. All money
is in euro at the 2026 price level. The distribution and costs are constructed to be internally
consistent and to isolate the decisions this topic teaches; they are not drawn from a published case.

### 6.1 The decision

| Slot | Value |
|---|---|
| Commitment date | 15 March 2026 |
| Selling window | 12 September 2026 to 31 January 2027, 20 weeks |
| Reorder opportunity | None; fabric is dyed and the lead time exceeds the useful season |
| Owner | Category manager, outerwear |
| Unit | Saleable each, one colourway, one region |

All four structural conditions hold. This is a newsvendor.

### 6.2 The cost pair

The unit lands at €62.00 (free-on-board price, ocean freight, duty, inland delivery) and sells at
€140.00, so the contribution margin is €78.00. The underage cost is built with a recovery probability
of 0.4:

| Component | Arithmetic | EUR per unit short |
|---|---|---:|
| Contribution margin on the unrecovered 60 per cent | `0.60 × 78.00` | 46.80 |
| Expedite premium on the recovered 40 per cent | `0.40 × 26.00` | 10.40 |
| Volume-weighted chargeback on the wholesale account | `0.35 × 22.86` | 8.00 |
| Goodwill, escalation, and service-desk time | Direct estimate | 5.00 |
| **Total `C_u`** | | **70.20** |

The first two rows are complementary, not additive on every unit: a recovered shortfall costs the
premium, an unrecovered one costs the margin, and adding both everywhere overstates `C_u` by about
€26. The chargeback is weighted by wholesale share; the direct channel carries no penalty. The
goodwill figure is the weakest number in the table.

The overage cost:

| Component | Arithmetic | EUR per unit left |
|---|---|---:|
| Unit cost committed | Landed cost | 62.00 |
| Recovery on clearance, net of outlet handling | Realised receipt | −38.00 |
| Carrying cost to the clearance date | `62.00 × 0.18 × 90/365` | 2.75 |
| Obsolescence and disposal allowance | 2 per cent unsaleable at zero | 0.00 |
| **Total `C_o`** | | **26.75** |

The €38.00 is net of transport, ticketing, and handling; the ticket price is not the salvage value.
The zero obsolescence allowance is a judgement, because the style is basic enough to clear, and it
should be recorded as one.

`q* = 70.20 / (70.20 + 26.75) = 0.7241`. The order should sit at the 72nd percentile of season
demand. Not the mean, not the median, and not last year plus 15 per cent.

### 6.3 The distribution and the quantity

Season demand is normal with mean 4,800 units and standard deviation 1,300, estimated from six
comparable seasons of a predecessor style and adjusted for a planned distribution change. Normality
is a convenience: the empirical distribution has six points and its upper tail is one observation.

```text
z* = Phi^-1(0.7241) = 0.594
Q* = 4,800 + 0.594 x 1,300 = 4,800 + 772 = 5,572 units
```

Two checks before the number leaves the room. The standard deviation is 27 per cent of the mean,
plausible for a seasonal style and implausible for a replenished basic; if the planner recognises it
as too high, the distribution is wrong. And the fractile sits in the body of the distribution, where
the normal approximation is least harmful.

### 6.4 Expected mismatch cost, and the premium

```text
L(0.594) = phi(0.594) - 0.594 x (1 - Phi(0.594)) = 0.3345 - 0.1640 = 0.1705
expected shortage = 1,300 x 0.1705 = 221.7 units
expected leftover = 1,300 x (0.594 + 0.1705) = 993.9 units
cost at optimum   = 70.20 x 221.7 + 26.75 x 993.9 = 15,563 + 26,587 = EUR 42,150
```

| Level | `z` | Quantity | Expected shortage | Expected leftover | Mismatch cost | Premium |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.000 | 4,800 | 518.6 | 518.6 | €50,278 | +19.3% |
| 0.724 | 0.594 | 5,572 | 221.7 | 993.9 | €42,150 | optimum |
| 0.90 | 1.282 | 6,466 | 61.4 | 1,727.4 | €50,518 | +19.9% |
| 0.95 | 1.645 | 6,938 | 27.2 | 2,165.6 | €59,839 | +42.0% |
| 0.99 | 2.326 | 7,824 | 4.4 | 3,028.6 | €81,324 | +93.0% |

**The cost curve is flat near the optimum and steep above it.** The 90th percentile costs about 20
per cent more than the optimum and the 95th 42 per cent more: above the optimum the firm is buying
protection against a tail, and the price is computable before it is promised. The 50th percentile
costs almost as much as the 90th, so 70 versus 75 is a rounding error while 70 versus 95 is 42 per
cent of the mismatch budget.

### 6.5 Sensitivity to the cost ratio

| Scenario | `C_u` | `C_o` | Fractile | `z` | Quantity | Change |
|---|---:|---:|---:|---:|---:|---:|
| Base | 70.20 | 26.75 | 0.724 | 0.594 | 5,572 | — |
| `C_u` −30% | 49.14 | 26.75 | 0.648 | 0.379 | 5,293 | −279 |
| `C_u` +30% | 91.26 | 26.75 | 0.773 | 0.752 | 5,778 | +206 |
| `C_o` −30% | 70.20 | 18.73 | 0.789 | 0.804 | 5,845 | +273 |
| `C_o` +30% | 70.20 | 34.78 | 0.669 | 0.437 | 5,368 | −204 |
| Salvage €20 not €38 | 70.20 | 44.75 | 0.611 | 0.281 | 5,165 | −407 |

A 30 per cent error in either cost moves the quantity by 4 to 5 per cent. The reason is arithmetic:
the fractile is a ratio bounded in `(0,1)`, and `dq/dC_u = C_o/(C_u+C_o)²`. Near equal costs, doubling
`C_u` moves the fractile from 0.500 to 0.667, not to 1.000. The cost ratio has to be wrong by a factor
of 3 before a 75 per cent target is implied, and by 19 before a 95 per cent target is.

> **Note:** Boundedness of the fractile does not bound the quantity equally. Near the middle of a
> normal distribution one point of probability buys about `0.025σ`; at the 99th percentile about
> `0.06σ`. A cost error harmless at the median can be material at a 99 per cent target.

The salvage row is the exception. Changing recovery from €38 to €20 is a 47 per cent change in `C_o`,
not 30, and it moves the answer by 407 units. Salvage is the component most often estimated by
assertion and least often audited, and the clearance-pricing literature explains why: when the
clearance price is chosen optimally it depends on the quantity left, so a fixed salvage is an
approximation whose error grows as the season proceeds.

### 6.6 Three items under one budget

Three styles share a €55,000 open-to-buy. With `F_i = (p_i − c_i − λc_i)/(p_i − s_i)` and bisection on
`λ`, the constrained fractiles come out at 0.37, 0.62, and 0.05 against 0.57, 0.75, and 0.41
unconstrained — not a common service level, and not a proportional cut. The constraint equalises
marginal return per euro, and `λ ≈ 0.35` says one more euro of open-to-buy is worth about 35 cents of
expected profit here: the number for the finance partner.

> **Note:** The constraint is on purchase value at the start of the season. For a replenished
> category it should be average inventory or working capital.

### 6.7 What rounding does

The model returns 5,572 units; cartons of 50 make the order 5,600. The achieved fractile is
`Φ((5,600−4,800)/1,300) = Φ(0.615) = 0.731`, against a target of 0.724. The net effect is under €100
of expected profit, smaller than the uncertainty in the goodwill estimate.

That is the benign case. An item with a mean of 40 units and a case pack of 12 has a rounding step of
30 per cent of the mean, and the discrete optimum may sit a whole case from the continuous one. Round
last, recompute the achieved fractile, and report both the modelled and the shipped quantity; report
only the shipped quantity and nobody can tell whether the model was wrong or the carton was.

## 7. Code, query, or calculation

The calculation below is the one to be able to do on paper, because the numbers will be argued over
and the argument is easier when the arithmetic is visible. It is a six-outcome discrete demand table
with a hand marginal comparison.

Assumptions: price €60, unit cost €28, salvage €10, so `C_u = 32` and `C_o = 18`. Demand takes six
values. Quantities must be one of them, which is the discrete case of Step 13.

```text
HAND NEWSVENDOR:  p = 60   c = 28   s = 10 EUR/unit   ->   C_u = 32,  C_o = 18

   d       80    120    160    200    240    300
   p(d)   .10    .20    .25    .25    .15    .05
   F(d)   .10    .30    .55    .80    .95   1.00      mean = 173 units

   Q     F(Q)   P(D>Q)   expected gain   expected loss   net per unit
                        P(D>Q) x 32      F(Q) x 18
    80   .10     .90        28.8             1.8            +27.0   order on
   120   .30     .70        22.4             5.4            +17.0   order on
   160   .55     .45        14.4             9.9             +4.5   order on
   200   .80     .20         6.4            14.4             -8.0   STOP

   fractile = 32 / (32 + 18) = 0.640
   smallest Q with F(Q) >= 0.640   ->   Q* = 200 units

   expected profit at Q*:  60 x 162 + 10 x 38 - 28 x 200 = 4,500 EUR
   check at Q = 160:       60 x 144 + 10 x 16 - 28 x 160 = 4,320 EUR
   check at Q = 240:       60 x 170 + 10 x 70 - 28 x 240 = 4,180 EUR
```

Three checks before trusting such a table. Confirm that each row's marginal applies to the whole
block between one support point and the next, because every unit in the block has the same sale
probability. Confirm that the expected profit at the optimum exceeds its neighbours on both sides.
Confirm that `F(Q*)` is at or above the fractile rather than at it, which is the signature of the
discrete case.

> **Note:** The block from 160 to 200 contains 40 units each worth €4.50 in expectation, so moving
> from 160 to 200 gains about €180. That matches the profit difference exactly: 4,500 − 4,320 = 180.
> When a marginal table and a total-profit table disagree, the marginal table has usually been
> applied to one unit and read as if it applied to the block.

## 8. Data, standards, and regulatory context

No international standard defines the newsvendor model, the critical fractile, or the choice of a
service level. That absence is informative: the decision is commercial, and the documents that
constrain it constrain the valuation of its inputs and the control environment around its use.

### 8.1 The valuation of the two costs

| Requirement | Where it applies | Effect on the cost pair | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | IFRS-reporting entities | Inventory at the lower of cost and net realisable value; a write-down raises `C_o` and lowers `s` | B |
| ASC 330 *Inventory* | US GAAP entities | Lower-of-cost-or-market logic with industry practice; same direction | B |
| IAS 37 *Provisions* | IFRS-reporting entities | A contractual shortfall penalty is a provision; its best estimate belongs in `C_u` | B |
| IFRS 15 / ASC 606 | Revenue | Unsold units generate no revenue; the forgone amount is contribution, not price | B |
| Incoterms 2020 | Global contract convention | Determines which freight, duty, and insurance enter the landed cost `c` | B |
| Waste and take-back rules | Jurisdiction-specific | Disposal fees and producer responsibility raise `C_o` directly | B |

Finance and commercial will state different overage costs, and both will be right at different
moments: finance describes the write-down before recovery, commercial the realised clearance receipt
after it. The fractile depends on which is used, so the reconciliation happens when the policy is
set, in writing, not during the post-season review.

### 8.2 Identity, time, and the definition of a season

| Standard | Governs | What it fixes here | Grade |
|---|---|---|---|
| GS1 GTIN | Product identification | The SKU whose distribution is modelled; a colourway with its own GTIN is its own distribution | B |
| GS1 GLN | Location identification | Which stores or channels are inside the demand population | B |
| ISO 8601 | Date and time | The season's start and end dates, and the week numbering used to aggregate sales | B |
| GS1 EPCIS 2.0 | Event data | Whether sell-through is observable at unit level or only at shipment | B |
| Fiscal calendar | Management reporting | Whether the "season" in the planning system matches the season in the demand data | Convention |

The fiscal calendar causes silent errors. A 4-4-5 retail calendar puts the same week number on
different dates in different years, so a season defined by week number moves by up to six days. On a
twenty-week season that is 4 per cent of the window, enough to make two years of history
incomparable without anyone noticing.

### 8.3 Binding requirements, conventions, and data sources

The 95 per cent service level is a convention, not a requirement. Where it is required the instrument
is a contract or licence condition, and it names the measure — usually a fill rate or an
on-time-in-full (OTIF) target, almost never a fractile. What binds is narrow: accounting standards fix
how `c` and `s` are valued, the contract fixes any penalty in `C_u`, and internal control fixes who may
change the parameter.

| Data source | What it records | What it hides |
|---|---|---|
| Point-of-sale transactions | Units sold at item and store level | Demand above the stock level; returns; substitution |
| Shipments to the retailer | Units transferred | Sell-through; channel inventory; everything after the transfer |
| Warehouse management system issues | Units released to fulfilment | Cancellations; failed deliveries |
| Order book | What customers asked for | What they would have asked for had stock been available |

Censoring is not a data-quality defect that a better extract will fix. It is a structural property of
selling from finite stock. The correction belongs in the model, and the minimum honest treatment is
to flag censored periods, fit with and without them, and report both answers.

## 9. Industry practice and case evidence

### 9.1 The problem has four names and a longer history than most of its users know

Edgeworth's 1888 paper on banking reserves solved the same problem — how much cash to hold against
random withdrawals — with the same structure: a stock, a random demand, and an asymmetric cost of
being wrong in each direction. The modern formulation is Arrow, Harris, and Marschak's 1951
*Econometrica* paper, which came out of a 1950 Office of Naval Research working group at RAND; Arrow's
retrospective describes how the static problem and the dynamic `(s, S)` policy were separated there.

The naming has its own history. "Newsboy" appears in Morse and Kimball's 1951 *Methods of Operations
Research*, a formerly classified wartime text; Magee called it "the classical newsboy problem" in
1953; Hadley and Whitin called it "the Christmas tree problem or newsboy problem" in 1963. The
gender-neutral "newsvendor" is generally credited to Matt Sobel, with "newsperson" appearing in some
textbooks in the 1980s. Work in the 2000s argued the problem was never really about newsboys: it was
a military logistics problem in a civilian metaphor, and the no-returns contract that defines it
became common in practice only in the late nineteenth century.

Section 16 grades these sources. The history matters for one practical reason: the four names carry four cost conventions. The revenue
manager's overage is the empty seat, the retailer's the markdown, the spare-parts planner's the
downtime. When a meeting contains two of those worlds, "cost" does different work for each, and the
fractiles differ even with an identical distribution.

### 9.2 Fashion and apparel

| Evidence | What it reports | Grade |
|---|---|---|
| Cachon and Kök (2007), *M&SOM* | With clearance pricing chosen rationally the fixed-salvage model is mis-specified; average, marginal, and weighted-average salvage heuristics systematically over-order; the weighted-average heuristic is near-optimal when demand is positively correlated across periods. Uses Sport Obermeyer as the motivating case | A |
| Caro and Gallien (2012), *Operations Research* | A clearance pricing and forecasting system at Zara; a field experiment in stores raised clearance revenue by about 6 per cent | A |
| Mitra (2018), *EJOR* | Treats the salvage price as a decision variable rather than a constant | A |
| Namin, Ratchford, and Soysal (2017), *JRCS* | Empirical markdown study with a specialty apparel retailer; early but small markdowns maximise revenue | A |
| Kajeeratwatthana (2022), *Thai Journal of Operations Research* | One street-fashion firm; reports roughly a 26 per cent order reduction and an 85 per cent profit increase against company policy | B |

Cachon and Kök attack the assumption rather than the arithmetic. If the clearance price falls when
more stock is left, `C_o` rises with `Q` and is not a constant. Section 6 assumes a fixed €38
recovery; the direction of that error is known (over-ordering) and the size is not, which is why the
salvage sensitivity in 6.5 matters more than the point estimate. The Zara result is a reminder that
the model is one component of a system: the reported six per cent came from pricing and forecasting
together, not from the order quantity alone.

### 9.3 Grocery, e-commerce, and the implied cost of a service promise

| Evidence | What it reports | Grade |
|---|---|---|
| Ulrich, Jahnke, Langrock, Pesch, and Senge (2020), *EJOR* | Overage cost is estimable from margins; underage cost reflects long-run strategic objectives and "cannot be stated"; the authors invert a stated target instead, `b = h α/(1−α)`. At `h = €1` and `α = 0.97` the implied `b` is about €32.30. The paper cites a grocery retailer's 98.8 per cent target | A |
| Stitch Fix engineering (2019) | Practitioner account of implementing the model in subscription apparel, including the difficulty of estimating the costs | C |
| OPLab / de Treville teaching material | An emergency-replenishment option lowers the effective underage cost from the margin to the premium, dropping a worked fractile from 0.70 to 0.40 | C |

The Ulrich calculation is the honest version of a calculation most firms do dishonestly. `b = 1 ×
0.97/0.03 = €32.33` says a unit short is treated as 32 times as costly as a unit left over. For a
grocery item with a 30 per cent margin and a unit cost of a few euros, no plausible lost-margin
figure produces that ratio. The 97 per cent target is not a cost calculation at all; it is a service
promise, and the ratio is its shadow. The de Treville case carries the complement: a second-order
supplier does not change the distribution, it changes `C_u`, and the fractile falls from 0.70 to 0.40
with no change to the forecast.

### 9.4 What the experiments say about the humans in the loop

| Evidence | What it reports | Grade |
|---|---|---|
| Schweitzer and Cachon (2000), *Management Science* | Known uniform demand, fractiles 0.75 and 0.25, optima 225 and 75; average orders 176.68 and 134.06, pulled toward the mean in both directions. Risk, waste, and stockout aversion were tested and rejected; supported accounts are ex-post inventory-error minimisation and anchoring on the mean | A |
| Bostian, Holt, and Smith (2008), *M&SOM* | Names the "pull-to-centre" effect and models it as adaptive learning with recency and reinforcement | A |
| MS Replication Project (2021) | Independent replications reproduce the direction and approximate magnitude: 196.29/131.31 in one laboratory, 177.51/134.50 in another | B |

A planner who anchors on the mean under-orders a high-fractile item and over-orders a low-fractile
one, largest where the asymmetry is largest. The effect appears in both student and manager samples,
so the remedy is not exhortation; it is a system-computed quantity and a review of overrides.

### 9.5 Estimating the underage cost, which is the hard part

| Evidence | What it reports | Grade |
|---|---|---|
| Schwartz (1966), *Management Science* | A fixed per-unit stockout penalty misrepresents goodwill; introduces "perturbed demand" models in which a stockout reduces future demand rather than imposing an immediate charge | A |
| Oral and Subba Rao (1972), Case Western working paper | Separates immediate shortage cost from long-term goodwill; models the latter with a semi-Markov process and an eroded demand rate | B |
| Xu (2016), *Journal of Applied Statistics* | Practitioners assume a fixed penalty or target a service level because goodwill is hard to measure; proposes statistical estimation of expected shortage cost | A |
| Liberopoulos et al. (2009) | Surveys what the backorder penalty `b` can represent and how it might be estimated | A |

`C_u` is therefore a belief, not a measurement, in any organisation that has not run an experiment.
The belief can be disciplined three ways: estimate the measurable components and label the residual
as goodwill; bound the residual by asking what defection rate would justify the observed target; or
reveal it from behaviour, since a planner who expedites at a known premium has revealed that the
shortfall cost exceeds that premium, which gives a free lower bound.

### 9.6 What companies do, against what the model says

| Model recommendation | Observed practice | Why the gap persists |
|---|---|---|
| Compute the cost pair and the fractile | Apply a percentage uplift to a forecast | The uplift is inherited and nobody owns the ratio |
| Estimate salvage net of handling | Use the markdown price, or zero | Clearance accounting lives in another team |
| Include expedite and penalty in `C_u` | Use the gross margin | The other costs sit in other budgets |
| Use the protection-period distribution for replenishment | Use the season distribution everywhere | The planning system has one parameter field |
| Allocate a constrained budget by marginal return | Cut every item by the same percentage | Proportional cuts are explainable and wrong |
| Report the premium over the optimum | Report only the chosen service level | The premium invites a question nobody wants |

The fifth row is the most expensive. A proportional cut is the default response to an open-to-buy
reduction because it is fair, explainable, and needs no analysis. It is also systematically wrong: it
takes the same share from the thinnest-margin item, which the constraint cuts by 30 per cent in the
Exercise 6 answer, as from the best, which it cuts by 11.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Order equals forecast plus uplift | The quantity is a round percentage above the mean and nobody can name the percentage's origin | A habit encodes a cost ratio nobody computed | Wrong in a direction set by the item's margin | Replace the uplift with a fractile and a written cost pair |
| `C_u` is the gross margin only | The cost table has one row | Expedite, penalty, and goodwill are omitted | Under-order, worst on high-margin or penalty-bearing items | Build the six-component table and sign each row |
| `C_o` is the full unit cost | The salvage line is missing or zero | Recovery is treated as impossible | Under-order, worst where a clearance channel exists | Compute net salvage after handling and carry |
| Margin and expedite double-counted | `C_u` exceeds the selling price | Both applied to every unit short | Over-order, with an implausibly high fractile | Apply the costs to complementary recovery shares |
| Seasonal pair applied to a replenished item | `C_o` equals the unit cost in a weekly policy | Destruction cost used where only carrying applies | Systematic under-stocking, blamed on the forecast | Substitute `h` and rebuild over the protection period |
| Distribution fitted to sales | The mean rises every time service improves | Censoring truncates the upper tail | The fractile reads on a distribution with no tail | Flag censored periods; fit with and without |
| Continuous answer rounded without a check | The order is round and the achieved probability is unstated | Case pack or MOQ applied last and unexamined | On slow movers the rounding is larger than the model | Report modelled and shipped quantity, and the achieved fractile |
| Discrete item treated as continuous | An order of 137 units for an item shipping in 24s | Interpolation between atoms | Up to a case of error, and an unreproducible fractile | Use the smallest-support-point rule and round outward |
| Cost ratio imputed from the target | The cost model reproduces the target exactly | The same number is used on both sides | A circular calculation presented as evidence | State the implied `C_u` and ask whether it is plausible |
| Constrained budget cut proportionally | Every item loses the same percentage | Proportionality is not optimality | High-margin items starved, low-margin items overstocked | Solve the Lagrangian, or rank by margin per euro |
| Fractile reported without the quantity | A slide says "72nd percentile" and no number of units | The fractile is an input, not an instruction | The warehouse cannot act on it | Report level, quantity, and rounding together |
| The number is re-derived from memory each season | The same style gets 0.72, 0.80, 0.68 in three years | No record of the inputs | Drift, and no learning | Register the cost pair, distribution version, and owner |

**The imputed ratio** is the most common honest error, because it does not feel like one. A manager
sets 95 per cent because the contract says so. The analyst builds a cost model, finds it gives 68 per
cent, and adjusts the goodwill term until it gives 95. The document is internally consistent, cites
two real cost components, and contains no independent information about the service level. The fix is
not to forbid the target; it is to stop pretending it was derived, and to print the implied ratio
beside it.

**The seasonal pair in a replenishment system** happens when a model built for a seasonal pilot is
extended to the replenished range without changing the cost definitions. The arithmetic keeps
working, the service level keeps being reported, and the inventory is chronically short. The
diagnostic is simple: if `C_o` in the parameter register equals the unit cost and the item is
replenished weekly, the parameter is wrong by construction.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Demand is one random variable over one window | One decision, one uncertainty | Multi-stage structure is collapsed and recourse value is lost | Ask whether any quantity can change after demand is seen |
| The distribution is stationary within the season | The fractile reads from a fixed `F` | Trend, promotion, or a competitor exit moves it after the order | Compare distributions across sub-periods; condition on known events |
| The distribution is adequate where the fractile reads | The answer is one point on `F` | A tail fit that is wrong at the decision | Compare empirical and fitted quantiles at the decision level; report sample size |
| Observed demand is uncensored | Sales are used as demand | Upper tail truncated, mean understated | Flag sold-out periods; fit with and without |
| Costs are constant per unit | `C_u` and `C_o` are scalars | Volume discounts, learning, and FX move them with `Q` | Plot unit cost against quantity; index the price year |
| The salvage value is independent of quantity | `C_o` does not depend on `Q` | Clearance price falls as leftovers rise, so `C_o` rises with `Q` | Test against historical markdown depth by leftover volume |
| A shortfall is a lost sale, not a backorder | `C_u` is the lost margin | Customers wait, so the true cost is delay and the model over-orders | Measure repeat purchase after a stockout |
| The budget is the only coupling between items | The Lagrangian is separable | Shared shelf space, shared demand, or substitution couples them | Test for cross-item demand effects |
| `λ` is constant across the allocation | One shadow price for the budget | Segment budgets, supplier minimums, or tiered financing | Check whether the budget is fungible |
| The decision-maker maximises expected profit | The fractile is an expectation | Risk aversion, bonus timing, or a career cost of a visible stockout | Compare historical orders with the expected-profit optimum |

**The model is silent on timing, and on what a stockout does to next season.** A season that sells
out in week three and one that sells out in week nineteen have the same demand and the same answer,
and different consequences; any in-season supply option needs a two-stage model. And a stockout today
changes future demand — some substitute, some defer, some leave — so being short this season makes
next season's distribution smaller. That cost is not in `C_u` unless somebody puts it there with a
number and a basis.

## 12. Exercises

### Exercise 1 — Build the cost pair (application, 45 minutes)

Choose a real or realistic item with a finite selling window and at least one contractual or service
consequence beyond the lost margin. Build `C_u` and `C_o` from named components with signs, a price
year, and a currency.

**Deliverable:** two signed component tables, a paragraph on the recovery probability and why it
takes that value, and an explicit statement of the three components you are least confident in.
**Acceptance criteria:** every component has a sign and a unit; margin is contribution, not price;
recovery is subtracted from `C_o`; margin and expedite apply to complementary shares; price year and
currency stated.
**Time:** 45 minutes.

### Exercise 2 — Fractile arithmetic and its inverse (calculation, 30 minutes)

An item has price €50, unit cost €22, and salvage €6. Season demand is normal with mean 900 units and
standard deviation 240.

(a) Compute `C_u`, `C_o`, the fractile, and the optimal quantity. (b) Suppose the business orders at
the 95th percentile: compute the implied `C_u` and state whether it is plausible. (c) Compute the
quantity at a 3:1 and at a 9:1 cost ratio.

**Deliverable:** the arithmetic, a table of ratio, fractile, `z`, and quantity, and a two-sentence
interpretation of part (b).
**Acceptance criteria:** the fractile is stated as a probability; the implied `C_u` is in euros per
unit; the interpretation names the circularity rather than the arithmetic.
**Time:** 30 minutes.

### Exercise 3 — Discrete demand and the rounding rule (calculation, 45 minutes)

Demand takes values 50, 100, 150, 200, 250, and 350 with probabilities 0.15, 0.20, 0.25, 0.20, 0.15,
and 0.05. Price is €88, unit cost €40, salvage €11.

(a) Build the cumulative distribution and its mean. (b) Compute the fractile and the optimal quantity
with the marginal comparison, row by row. (c) Compute expected profit at the optimum and at the
support points either side. (d) Explain why the answer is a support point rather than an
interpolation.

**Deliverable:** the table, the marginal comparison, the profit check, and a paragraph on (d).
**Acceptance criteria:** the comparison stops at the first negative net margin; the profit check
confirms the optimum; the explanation refers to the dominance of interior quantities by the
endpoints.
**Time:** 45 minutes.

### Exercise 4 — The sensitivity band (analysis, 40 minutes)

Using the section 6 base pair (`C_u = 70.20`, `C_o = 26.75`, mean 4,800, standard deviation 1,300),
compute the quantity when each cost is varied by ±50 per cent, one at a time. State whether the
decision changes at a carton of 50.

**Deliverable:** a five-row table of scenario, costs, fractile, `z`, quantity, and change; a paragraph
on whether the cost estimation effort is worth it; a sentence on where the conclusion fails.
**Acceptance criteria:** fractile and quantity distinguished; the conclusion names the region where
the reassurance fails; the carton check is performed.
**Time:** 40 minutes.

### Exercise 5 — Audit a circular calculation (critique, 40 minutes)

A category plan targets 96 per cent CSL. Its cost model gives `C_o = €26.75` and `C_u = €642.00`, of
which €78.00 is contribution margin, €13.40 expedite premium, and €550.60 "goodwill and strategic
value".

(a) Confirm the model reproduces the target. (b) Compute the goodwill term as a multiple of the
measurable underage components. (c) Write the three-sentence paragraph that makes the calculation
honest.

**Deliverable:** the two calculations and the paragraph.
**Acceptance criteria:** the paragraph neither endorses nor rejects the target; it states the implied
ratio, compares it to the measurable components, and asks what the target is buying.
**Time:** 40 minutes.

### Exercise 6 — Three items under one budget (calculation, 60 minutes)

Three items share a €55,000 open-to-buy; demand is normal for each.

| Item | `p` | `c` | `s` | Mean | SD |
|---|---:|---:|---:|---:|---:|
| X | 60 | 25 | 5 | 800 | 200 |
| Y | 120 | 45 | 15 | 300 | 90 |
| Z | 30 | 20 | 12 | 1,500 | 350 |

(a) Compute the unconstrained quantity and spend per item and the total. (b) Solve the constrained
problem by bisection on `λ`, reporting `λ`, the three fractiles, quantities, and total spend. (c)
Compare the constrained and unconstrained fractiles and explain the ranking. (d) State what `λ` means
to a finance partner.

**Deliverable:** both tables, a bisection trace at three or four values of `λ`, and a paragraph for
(c) and (d).
**Acceptance criteria:** the budget is met within €200; the fractiles differ; the ranking is explained
with margin per euro of purchase, not margin alone.
**Time:** 60 minutes.

### Exercise 7 — Translate to the repeated setting (analysis, 45 minutes)

Take the section 6 item and suppose it is replenished weekly. Weekly demand has mean 240 and standard
deviation 290.7, consistent with the season figures under independence. The protection period is four
weeks, holding cost €0.90 per unit per week, backorder penalty €14.00 per unit per week.

(a) Compute the order-up-to level under the repeated pair. (b) Compute the level that results from
applying the seasonal pair to the same four-week distribution. (c) Explain the difference in one
paragraph, naming the three substitutions. (d) State one reason the repeated answer might still be
wrong.

**Deliverable:** both calculations and the two paragraphs.
**Acceptance criteria:** the protection-period distribution is built correctly, including the
independence assumption; the three substitutions are named; part (d) names a structural limitation,
not an estimation error.
**Time:** 45 minutes.

### Exercise 8 — Write the decision memo (design, 45 minutes)

Write a one-page memo to a merchandising director recommending an order quantity. The director is
numerate, has no operations research background, and has already said "just add 15 per cent like we
always do".

**Deliverable:** the decision; the cost pair with its two largest components; the fractile; the
quantity; the expected mismatch cost; the premium at 95 per cent in currency and percentage; the two
assumptions that would change the answer; the review date.
**Acceptance criteria:** no undefined acronyms; every number has a unit and a currency; the memo
states what would make it wrong.
**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

Rubric, not an answer. A strong submission has two signed tables: recovery is negative in the overage
table, and expedite applies only to the recovered share. A weak one has one "margin" and one "cost"
and calls them `C_u` and `C_o`, which is right only without recovery, penalty, or goodwill. Watch
salvage, goodwill, and recovery: each is a decision masquerading as an estimate.

### 13.2 Exercise 2

```text
(a)  C_u = 50 - 22 = 28    C_o = 22 - 6 = 16    q* = 28/44 = 0.6364
     z* = 0.348            Q* = 900 + 0.348 x 240 = 984 units
(b)  q = 0.95 -> C_u/C_o = 19 -> implied C_u = 19 x 16 = 304 EUR, 10.9x the margin
(c)  3:1 -> z 0.674 -> 1,062        9:1 -> z 1.282 -> 1,208
```

The mapping is section 3's: 1:1 gives 900, 3:1 gives 1,062, 9:1 gives 1,208, 19:1 gives 1,295. A
ninefold change in relative costs moves the quantity 34 per cent; the next step adds 7.

Part (b) is the circularity: a 95 per cent target with a €16 overage demands a €304 underage, and no
component here plausibly reaches it. The honest statement names the implied 19:1, the measurable
1.75:1, and the residual as a strategic choice.

### 13.3 Exercise 3

| `d` | 50 | 100 | 150 | 200 | 250 | 350 |
|---|---:|---:|---:|---:|---:|---:|
| `p(d)` | 0.15 | 0.20 | 0.25 | 0.20 | 0.15 | 0.05 |
| `F(d)` | 0.15 | 0.35 | 0.60 | 0.80 | 0.95 | 1.00 |

Mean `= 160` units; `C_u = 48`, `C_o = 29`, `q* = 48/77 = 0.6234`.

```text
 Q    F(Q)   P(D>Q)   0.48 x P(D>Q)   0.29 x F(Q)   net per unit
 50   .15     .85         40.80            4.35         +36.45   order on
100   .35     .65         31.20           10.15         +21.05   order on
150   .60     .40         19.20           17.40          +1.80   order on
200   .80     .20          9.60           23.20         -13.60   STOP
```

`Q* = 200`, the smallest support point with `F(Q) ≥ 0.6234`. Expected profit at the three candidates:

```text
Q = 150:  125.0 x 88 + 25.0 x 11 - 150 x 40 = 11,000 + 275 - 6,000 = 5,275 EUR
Q = 200:  145.0 x 88 + 55.0 x 11 - 200 x 40 = 12,760 + 605 - 8,000 = 5,365 EUR
Q = 250:  155.0 x 88 + 95.0 x 11 - 250 x 40 = 13,640 + 1,045 - 10,000 = 4,685 EUR
```

The marginal check agrees: the step from 200 to 250 is `−13.60` per unit over 50 units, or `−680`, and
`5,365 − 680 = 4,685`. Part (d): any quantity inside `(150, 200)` is dominated by an endpoint, because
every unit in the gap has the same sale probability, `P(D > 150) = 0.40`, so the marginal value is
constant and drives the order to one end.

### 13.4 Exercise 4

| Scenario | `C_u` | `C_o` | Fractile | `z` | Quantity | Change |
|---|---:|---:|---:|---:|---:|---:|
| `C_u` +50% | 105.30 | 26.75 | 0.797 | 0.833 | 5,883 | +311 |
| `C_u` −50% | 35.10 | 26.75 | 0.568 | 0.171 | 5,022 | −550 |
| `C_o` +50% | 70.20 | 40.13 | 0.636 | 0.349 | 5,254 | −318 |
| `C_o` −50% | 70.20 | 13.38 | 0.840 | 0.994 | 6,092 | +520 |
| Base | 70.20 | 26.75 | 0.724 | 0.594 | 5,572 | — |

The band is 5,022 to 6,092, about ±9 per cent around 5,572: roughly 22 cartons across the whole range.
A 50 per cent cost error produces a 10 per cent quantity error, a factor-of-five attenuation, which
is the boundedness of the fractile at work.

The conclusion fails near the extremes. At a 99 per cent target, `z` moves from 2.054 to 2.326 for a
25 per cent change in the ratio, 0.27σ or 350 units, because the normal density falls in the tail. It
also fails on a coarse discrete support, where the answer jumps between atoms.

### 13.5 Exercise 5

```text
(a)  implied fractile = 642.00/(642.00 + 26.75) = 642.00/668.75 = 0.9600
     the model reproduces the 96 per cent target exactly, by construction

(b)  measurable components = 78.00 + 13.40 = 91.40
     goodwill and strategic value = 550.60
     multiple = 550.60/91.40 = 6.0 times the measurable underage cost
```

The paragraph:

"The target for this category is 96 per cent CSL. Taken at face value it implies an underage cost of
€642 per unit against an overage cost of €26.75, a ratio of 24:1. The measurable underage components
— contribution margin and expedite premium — total €91.40, which supports about 3.4:1 and a service
level near 77 per cent. The remaining €550.60 is a strategic judgement about availability; it is
larger than the margin on the item, and we are recording it as a judgement rather than presenting it
as a cost estimate."

It is honest because it computes the implied ratio, compares it with the measurable part, and labels
the residual; it does not say 96 per cent is wrong, because it may not be.

### 13.6 Exercise 6

**(a) Unconstrained. (b) Constrained**, at `λ ≈ 0.395` from
`F_i = (p_i − c_i − λc_i)/(p_i − s_i)`; spend falls from €68,525 at `λ = 0` to €54,823 at `λ = 0.40`,
so the root sits just below 0.40.

| Item | `C_u` | `C_o` | Uncon. fractile | `Q` | Spend | Con. fractile | `Q` | Spend | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X | 35 | 20 | 0.636 | 870 | €21,750 | 0.457 | 778 | €19,450 | −11% |
| Y | 75 | 30 | 0.714 | 351 | €15,795 | 0.545 | 310 | €13,950 | −12% |
| Z | 10 | 8 | 0.556 | 1,549 | €30,980 | 0.117 | 1,083 | €21,660 | −30% |
| **Total** | | | | | **€68,525** | | | **€55,060** | |

**(c) The ranking.** The cuts are not proportional: Z loses 30 per cent and Y 12, with the budget down
20. The ranking follows margin per euro of purchase at the unconstrained optimum — Y 1.67, X 1.40,
Z 0.50 — so the item that converts capital into margin fastest keeps its stock. A proportional cut
would leave Z near 1,240 and Y near 280, the wrong side of both errors.

**(d) What `λ` means.** `λ ≈ 0.40` means one more euro of open-to-buy is worth about 40 cents of
expected profit at this allocation. It is the marginal value of budget, not an interest rate. Freeing
€5,000 of cash is worth about €2,000 of expected profit before financing cost.

### 13.7 Exercise 7

```text
(a) repeated:  h = 0.90, b = 14.00, q = 14.00/14.90 = 0.9396, z = 1.553
               four-week mean = 4 x 240 = 960
               four-week SD   = 290.7 x sqrt(4) = 581.4
               S = 960 + 1.553 x 581.4 = 960 + 903 = 1,863 units

(b) seasonal pair:  q = 70.20/96.95 = 0.7241, z = 0.594
               S = 960 + 0.594 x 581.4 = 960 + 345 = 1,305 units
```

The repeated answer is 558 units higher, 43 per cent above the seasonal answer on the same
distribution.

**(c)** Three substitutions produce the difference: the overage moves from €26.75 per unit destroyed to
€0.90 per unit per week carried; the underage from a lost margin of €70.20 to a backorder penalty of
€14.00 per week; the distribution from season to protection-period demand. The first dominates.

**(d)** The answer may still be wrong because demand is not stationary: skiwear demand over a four-week
period in November is not drawn from the same distribution as in September.

### 13.8 Exercise 8

A strong memo has eight elements: the decision and its date; the cost pair with its two largest
components and their sources; the fractile; the quantity and its rounding; the expected mismatch
cost; the premium at 95 per cent in currency and percentage; the two assumptions that would change
the answer; the review date. A weak memo leads with the forecast and says "the model recommends"
rather than naming the owner, which turns a decision into a suggestion.

Sample opening: "I recommend 5,600 units of SK-114, the 73rd percentile of season demand. The
cost-optimal point is the 72nd percentile at 5,572 units; the carton size of 50 accounts for the
difference and costs under €100 in expected profit. At 95 per cent — last season's level — the same
distribution implies 6,938 units and about €17,700 more expected mismatch cost, or 42 per cent. The
two inputs I would most want to test are the €38 net salvage and the €5 goodwill allowance."

## 14. Interview questions

These test whether you can defend a quantity, not whether you can recite a formula. Each answer line
names what a strong response contains and what a weak one reveals.

**1. Why not order the mean forecast?**

Strong: the costs are unequal, so the minimum sits where marginal costs balance, which is a quantile,
not the mean; they coincide only at equal costs. Adds the formula and a ratio. Weak: "the mean is just
an average" — no mechanism, no cost pair, no statement of when they coincide.

**2. Derive the critical fractile in four lines.**

Strong: the last unit sells with probability `1 − F(Q)` for `C_u` and is left with `F(Q)` for `C_o`;
order while `(1 − F)C_u > F C_o`; hence `F(Q*) = C_u/(C_u + C_o)`. Weak: recites the formula and
cannot say why the inequality flips.

**3. Demand is discrete. Where exactly does the order go?**

Strong: the smallest support point at or above the fractile; no interpolation, because every unit in
a gap has the same marginal value and interior quantities are dominated by the endpoints; then round
to the case pack and report the achieved fractile. Weak: takes the continuous inverse, rounds to the
nearest integer, and calls the difference immaterial.

**4. What does a 3:1 cost ratio imply, and why not 75 per cent of the maximum?**

Strong: `3/(3+1) = 0.75`; the fractile is bounded and saturates, so quantity is far less sensitive
than the ratio at moderate service levels, though sensitivity returns in the tail. Weak: says "higher
service level" with no number, or claims quantity scales linearly with the ratio.

**5. What belongs in the underage cost?**

Strong: lost contribution on the unrecovered share, expedite premium on the recovered share,
penalties and chargebacks, measurable goodwill and service-desk cost, each signed, with a unit and a
price year. Weak: lists "lost profit" only, or adds margin and expedite to every unit.

**6. A colleague adds the full margin and the full expedite premium. What breaks?**

Strong: the two are alternatives for the same unit — recovered or not — so adding them everywhere
overstates `C_u` and can push it above the selling price; the correct expression is a
probability-weighted pair. Weak: accepts it as "conservative" without asking which direction that is.

**7. How is this different from setting an order-up-to level?**

Strong: the overage is holding cost per period, not unit cost; the underage is a backorder or lost
sale per period; the distribution is demand over the protection period; the same algebra with three
substitutions, often a very different level. Weak: says the models are "basically the same" and
reuses the seasonal pair.

**8. You have a budget and ten items. Do you set the same service level for each?**

Strong: no; the Lagrangian sets `F_i = (p_i − c_i − λc_i)/(p_i − s_i)`, equalising marginal profit per
euro, so high-margin-per-euro items keep stock and low-margin items are cut, sometimes to zero. Weak:
proposes a proportional cut or a common service level and defends it as fair.

**9. Your plan says 95 per cent. Where did the number come from?**

Strong: asks whether it is an output or a target input; if the latter, computes the implied ratio
`19:1`, compares it with the measurable components, and labels the residual a strategic judgement.
Weak: recomputes the quantity at 95 per cent and calls the job done.

**10. Why is your demand history biased?**

Strong: sales are censored at stock; every sold-out period is a lower bound on demand, and the bias
is largest where the order decision matters; flag censored periods, fit with and without them, report
both. Weak: treats sales as demand and attributes the residual to forecast error.

**11. A planner over-orders low-fractile items and under-orders high-fractile ones. What is that?**

Strong: pull-to-centre — orders anchor on the mean; it is not risk aversion, training did not remove
it, and the control is a system-computed quantity plus a review of overrides. Weak: calls it caution,
or recommends more training.

## 15. Key terms

- **Newsvendor problem.** Single-period stocking: one order, random demand, no replenishment.
- **Critical fractile.** `C_u/(C_u + C_o)`, the cumulative probability at which the order sits.
- **Underage cost (`C_u`).** Expected cost of being one unit short, currency per unit.
- **Overage cost (`C_o`).** Expected cost of being one unit long, currency per unit.
- **Contribution margin.** Price minus variable cost; the base of `C_u`, not the selling price.
- **Net salvage (`s`).** Receipt per leftover unit after handling and markdown, not the ticket price.
- **Recovery probability (`r`).** Chance a shortfall is covered by expedite; splits margin and premium.
- **Expedite premium.** Incremental cost of covering a shortfall faster than the normal channel.
- **Critical ratio.** The older name for the critical fractile, common in revenue management.
- **Cycle service level (CSL).** Probability of no stockout in a period; what the fractile defines.
- **Fill rate.** Fraction of demand met from stock; a different measure, not the fractile.
- **Order-up-to level (`S`).** Target inventory position in a periodic-review policy.
- **Protection period.** Time the inventory position must cover: review period plus lead time.
- **Newsvendor quantity (`Q*`).** The optimal order; a support point when demand is discrete.
- **Demand distribution (`F`).** Cumulative distribution of demand over the relevant period.
- **Lost-sales model.** Demand above stock is forgone; the standard single-period assumption.
- **Backorder model.** Demand above stock is deferred; the standard repeated-setting variant.
- **Censored demand.** Sales truncated by stock availability; a lower bound on demand.
- **Expected mismatch cost.** `C_u × E[shortage] + C_o × E[leftover]`, the objective minimised.
- **Standard normal loss function.** `L(z) = φ(z) − z(1 − Φ(z))`, shortage in units of σ.
- **Shadow price (`λ`).** Marginal expected profit per extra euro of a binding budget.
- **Open-to-buy.** Purchase budget available for a period; the constraint in the multi-item problem.
- **Lagrangian allocation.** Equalising marginal return per euro across items under a budget.
- **Pull-to-centre.** Tendency to order toward the mean regardless of the fractile.
- **Implied cost ratio.** The ratio recovered by inverting a stated service target.
- **Littlewood's rule.** Booking limit set at the fare ratio; the newsvendor in revenue management.
- **Markdown.** Price reduction to clear leftovers; makes salvage endogenous.

## 16. Sources

Grouped by type. Grades: A peer-reviewed or audited; B regulatory, official, or scholarly monograph;
C company case or practitioner account; D vendor marketing; E press or anecdote. No D or E source is
used for a quantitative claim in this file.

### Theory and history

- Edgeworth (1888), *Journal of the Royal Statistical Society* — the reserve problem with asymmetric
  costs. A
- Arrow, Harris, and Marschak (1951), *Econometrica* — the modern single-period formulation and
  `(s, S)` separation. A
- Arrow (2002), *Operations Research* — first-hand account of the 1950 RAND working group. A
- Morse and Kimball (1951), *Methods of Operations Research* — the "newsboy" label. B
- Hadley and Whitin (1963), *Analysis of Inventory Systems* — "Christmas tree problem". B
- Kraiselburd and North-Shea (2007) — military origin and contract history of the model. B
- Porteus (2008), "The Newsvendor Problem," in *Building Intuition*, Springer — attributes the term
  "newsvendor" to Sobel. B

### Multi-item and constrained allocation

| Source | Contribution here | Grade |
|---|---|---|
| Turken et al. (2012), *Handbook of Newsvendor Problems* | The budget-constrained Lagrangian rule | B |
| Zhang, Xu, and Hua (2009), *IJPE* | Binary search on marginal benefit; linear-time optimum | A |

### Salvage, clearance, and fashion

| Source | Contribution here | Grade |
|---|---|---|
| Cachon and Kök (2007), *M&SOM* | Endogenous clearance price; salvage heuristics over-order | A |
| Caro and Gallien (2012), *Operations Research* | Zara clearance system; field experiment | A |
| Mitra (2018), *EJOR* | Salvage price as a decision variable | A |
| Namin, Ratchford, and Soysal (2017), *JRCS* | Empirical markdown depth and timing | A |
| Kajeeratwatthana (2022), *Thai Journal of Operations Research* | Single-firm order and profit comparison | B |

### Underage cost, goodwill, and service targets

| Source | Contribution here | Grade |
|---|---|---|
| Schwartz (1966), *Management Science* | Goodwill as perturbed demand rather than a fixed penalty | A |
| Oral and Subba Rao (1972), Case Western working paper | Immediate versus long-term shortage cost | B |
| Xu (2016), *Journal of Applied Statistics* | Statistical estimation of expected shortage cost | A |
| Liberopoulos et al. (2009) | Survey of what the backorder penalty represents | A |
| Ulrich, Jahnke, Langrock, Pesch, and Senge (2020), *EJOR* | Inversion of a stated target to an implied penalty | A |

### Behavioural evidence

| Source | Contribution here | Grade |
|---|---|---|
| Schweitzer and Cachon (2000), *Management Science* | Pull-to-centre; risk and waste aversion rejected | A |
| Bostian, Holt, and Smith (2008), *M&SOM* | Adaptive-learning account of the effect | A |
| MS Replication Project (2021) | Independent replication of direction and magnitude | B |

### Standards and official material

| Source | Contribution here | Grade |
|---|---|---|
| IAS 2 *Inventories*, IFRS Foundation | Lower of cost and net realisable value; write-downs | B |
| IAS 37 *Provisions*, IFRS Foundation | Contractual shortfall penalties as provisions | B |
| IFRS 15 / ASC 606, IFRS Foundation and FASB | Revenue recognition; forgone revenue is contribution | B |
| ASC 330 *Inventory*, FASB | US GAAP inventory measurement | B |
| Incoterms 2020, International Chamber of Commerce | Which freight and duty enter the landed cost | B |
| GS1 GTIN, GLN, and EPCIS 2.0 | Product, location, and event identity | B |
| ISO 8601 | Date and time representation for season boundaries | B |

### Practitioner accounts

| Source | Contribution here | Grade |
|---|---|---|
| Stitch Fix engineering (2019) | Implementation account, including cost estimation | C |
| OPLab / de Treville teaching material | Emergency replenishment lowering `C_u` | C |

## 17. Links

**This module**

- [Module 10 index](_index.md) — topic table, prerequisites, status.
- [01. The Cost-Service Frontier](01_cost-service-frontier.md) — the trade-off curve this sits on.
- [02. EOQ and Its Limits](02_eoq-limits.md) — the repeated model this one borrows `C_o` from.

**Prerequisites**

- [08.10 Quantiles and Decisions](../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md)
  — the quantile vocabulary this file extends.
- [07.09 Error Metrics and Cost](../07_time_series_forecasting/09_error-metrics-cost.md) — asymmetric
  loss and why mean absolute error is the wrong target for an asymmetric decision.

**Wider map**

- [Part 2 index](../_index.md) — the methods modules in sequence.
- [Part 1 index](../../01_foundations/_index.md) — the foundations the cost pair rests on.
- [Module 10 survey](../../../course/10_inventory_optimization_multi_echelon.md) — the topic map.

**Not yet written**

These siblings are planned. They are plain text because a link to a missing file is worse than none.

```text
04_safety-stock.md                     (planned)
05_service-metrics.md                 (planned)
06_policy-families.md                 (planned)
07_lead-time-demand.md                (planned)
08_multi-echelon.md                   (planned)
09_pooling-postponement.md            (planned)
10_segmentation-shelf-life.md         (planned)
```

When 04 is written, read it next: safety stock is the repeated-setting version of this fractile,
with lead-time demand in place of season demand.
