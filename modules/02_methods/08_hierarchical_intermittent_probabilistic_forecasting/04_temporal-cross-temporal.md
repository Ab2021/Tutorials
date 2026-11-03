# Temporal and Cross-Temporal Hierarchies

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01A topic 04 (calendars, weeks, and time zones); Module 08 topics 01–03 (hierarchy structures, aggregation consistency, minimum-trace reconciliation)
> Version 2026.5

---

## 1. Why this topic exists

The monthly sales and operations planning (S&OP) pack says 42,000 eaches will ship in April. The weekly build schedule for the same April sums to 44,100. The daily replenishment plan for the first fortnight sums to 15,900, against 15,300 for the same fortnight in the weekly plan. Three numbers, one product, one month, three meetings. Nobody is lying, and every number was computed correctly by somebody competent.

This is the second dimension of reconciliation. Topics 01 to 03 treated the entity hierarchy — product, location, customer — and made the numbers add up *at one time bucket*. They said nothing about whether the monthly number equals the sum of its weeks, or whether the weeks equal the sum of their days. In most planning suites those two reconciliations are performed by different code, owned by different teams, on different cadences, and neither knows the other exists. The result is a plan that is coherent left-to-right and incoherent top-to-bottom.

**Time is a hierarchy.** A day sums into a week, a week into a month, a month into a quarter, a quarter into a year. That is the same algebraic object as a product rolling into a category. It behaves differently in three ways that make it harder, and those three differences are the spine of this file.

**The constraint matrix scales with the length of the history, not with the width of the taxonomy.** An entity hierarchy with 500 stock-keeping units (SKUs) and 20 category nodes has 20 constraints per time bucket. A temporal hierarchy over three years of daily data has roughly 200 constraints per series, and the cross-temporal system multiplies the two.

**The calendar can make the constraint system infeasible.** In the entity hierarchy a node always has a parent: a SKU belongs to one category, or the taxonomy is broken and you fix the taxonomy. In the temporal hierarchy a March does not contain a whole number of International Organization for Standardization (ISO) weeks. Thirty-one days is four weeks and three days. There is no set of whole seven-day weeks that covers March, so "the month equals the sum of its weeks" and "the week is an ISO week" cannot both be true. Module 01A topic 04 taught the calendar. This topic is where that calendar defect arrives inside the forecast layer and stops being a formatting problem.

**Aggregation can create information as well as destroy it.** In the entity hierarchy, aggregation usually destroys signal: a regional total hides which SKU is moving. In the temporal hierarchy, aggregation often creates signal: a slow-moving part that is zero in 70% of months is dense at the quarterly and annual levels, and a seasonal shape invisible at weekly resolution is obvious at annual resolution. This is the strongest practical argument for temporal hierarchies, and it has nothing to do with the accuracy claim that usually gets quoted.

**What is at stake.** Not "should we reconcile across time" — the answer to that is always yes in principle and rarely affordable in full. The stake is the question in section 5, step 19: *which time bucket does the planning process actually need coherent, and what is the documented bridge to the others?* A company that answers that question has a plan. A company that reconciles everything uniformly has a plan nobody has read.

**What this topic is not.** It is not an introduction to time series forecasting, which is Module 07, and it does not re-teach the calendar, which is Module 01A topic 04. It also does not cover the probabilistic version of any of this: reconciling predictive distributions across a temporal hierarchy is a live research area, and this file stays with point forecasts so the algebra stays checkable by hand. Topic 07 of this module picks up distributions; the temporal extension there is a separate problem.

> **Note:** Nothing in this topic claims that temporal reconciliation improves accuracy in general. Section 9 shows the evidence running against the popular claim: gains are usually largest at the coarsest levels and smallest at the finest, and are sometimes negative at the shortest horizon.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a temporal hierarchy for a named series from day to year, label each summing relation with its factor, and mark where the Gregorian calendar breaks the tree.
2. **Construct** the temporal summing matrix `S_te` for a series with four aggregation levels and state its dimensions, its independent constraint count, and the dimension of the coherent subspace.
3. **Demonstrate** that a month cannot equal the sum of its ISO weeks, using the arithmetic of a named month, and state the three resolutions available.
4. **Compute** by hand a three-node temporal reconciliation and show how the weights move the period total.
5. **Compute** the dimensions of a cross-temporal constraint system from an entity count, a temporal ladder, and a history length.
6. **Distinguish** temporal-only reconciliation, sequential cross-temporal reconciliation, and optimal cross-temporal combination, and state what each guarantees.
7. **Explain** why a separable (Kronecker) error covariance is used in practice and what it assumes.
8. **Choose** the bucket that must be coherent for a named planning process and write the bridge rule to adjacent buckets.
9. **Critique** an accuracy claim for temporal reconciliation on the grounds of horizon, aggregation level, calendar, and evaluation design.

Outcomes 2, 3, 4, 5, and 8 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures: a ladder, a month that does not close, and a grid. The ladder is the hierarchy. The month is where the calendar breaks it. The grid is what happens when the entity hierarchy and the temporal hierarchy are laid across each other.

### 3.1 Panel A — the temporal ladder

```text
   Panel A — the temporal ladder for one series, year 2026

   level          summing relation
   ------------   ---------------------------------------------
   YEAR 2026      the root; 12 months or 4 quarters sum to it
      ^
      |  12 months -> 1 year ; 4 quarters -> 1 year
      |
   QUARTER        Q1 Q2 Q3 Q4 ; 3 months -> 1 quarter
      ^
      |  3 months -> 1 quarter
      |
   MONTH          Jan Feb ... Nov Dec
      ^
      |  4 or 5 whole weeks -> 1 month    <-- THIS IS THE BREAK
      |  (true only in a 4-4-5 style calendar; in the Gregorian
      |   calendar a month holds 4.43 weeks on average)
      |
   WEEK           W01 W02 ... W52 (W53) ; 7 days -> 1 week
      ^
      |  7 days -> 1 week
      |
   DAY            1 Jan ... 31 Dec

   1 year + 4 quarters + 12 months + 53 weeks + 365 days = 435 nodes,
   of which 365 are bottom-level, so a strict tree has 70 constraints.
```

Read the ladder from the bottom. Every arrow is a summing relation and every summing relation is a row of the constraint matrix. It is a **strict tree** only if every bucket at a level lies inside exactly one bucket above. Quarters in years hold. Months in quarters hold. Days in months hold. Weeks in months do not, and that single failure is what the rest of this file is about.

### 3.2 Panel B — March 2026, where the ladder breaks

```text
   Panel B — March 2026 in ISO weeks

   ISO week    W09      W10    W11    W12    W13    W14
   range       23Feb-1  2-8    9-15   16-22  23-29  30Mar-5Apr
   March days  1        7      7      7      7      2

   days of March ..................  1 + 7 + 7 + 7 + 7 + 2 = 31   correct
   whole ISO weeks overlapping ....  6 x 7                 = 42   not 31
   any cover by whole 7-day weeks .  28 or 35              never 31
                                     because 31 = 4 x 7 + 3

   So "March = sum of its weeks" has no solution in whole ISO weeks.
   Writing March = W09/7 + W10 + W11 + W12 + W13 + 2*W14/7 is not a
   constraint. It is an allocation assumption wearing a constraint's
   clothes.
```

The panels are worth memorising because the numbers are small and the conclusion is general. Thirty-one modulo seven is three, so no month whose length is not a multiple of seven can be partitioned by whole seven-day weeks. February in a common year is the only month that can, and only when it starts on the right weekday.

### 3.3 Panel C — the cross-temporal lattice

```text
   Panel C — the cross-temporal hierarchy drawn as a grid

                       Jan   Feb   Mar   Apr   May   Jun  ...   <- time
                    +-----+-----+-----+-----+-----+-----+-----+
   TOTAL            |     |     |     |     |     |     |     |
                    +-----+-----+-----+-----+-----+-----+-----+
   DIVISION A       |     |     |     |     |     |     |     |   ^
                    +-----+-----+-----+-----+-----+-----+-----+   |
   DIVISION B       |     |     |     |     |     |     |     |   | entity
                    +-----+-----+-----+-----+-----+-----+-----+   | sums
   CATEGORY 1       |     |     |     |     |     |     |     |   | run
   CATEGORY 2       |     |     |     |     |     |     |     |   | upward
                    +-----+-----+-----+-----+-----+-----+-----+   |
   SKU 1            |     |     |     |     |     |     |     |   |
   SKU 2            |     |     |     |     |     |     |     |   |
   ...              |     |     |     |     |     |     |     |   v
                    +-----+-----+-----+-----+-----+-----+-----+

   temporal sums run right: months into quarters into years,
   inside every row.

   The coherent subspace has dimension = (number of bottom cells)
                                         = SKUs x months.
```

A temporal-only reconciliation projects each row onto its own time constraints and leaves the columns incoherent: monthly totals will not sum across SKUs to the category monthly total. A cross-sectional-only reconciliation projects each column and leaves the rows incoherent. Doing one then the other, in either order, does not in general produce a cross-temporally coherent result, because the second projection breaks the first. Only a projection onto the intersection satisfies both, and the intersection is spanned by the bottom cells.

### 3.4 Three properties to carry forward

**The leaves are the same series, cut finely.** In the entity hierarchy two leaves are two different things and their errors may be nearly independent. In the temporal hierarchy the twelve monthly leaves of a year are one series observed twelve times, so a parent's error *is* the sum of its children's errors plus whatever the parent model added. That structure is why the simple weightings (section 5, step 10) work unusually well in the temporal case.

**The tree exists only if the buckets nest.** Topics 01 to 03 assumed a summing matrix that could be written down. A temporal summing matrix can be written down only after a calendar is chosen in which the levels nest. Choose the calendar first, or the matrix does not exist.

**The coherent subspace is spanned by the bottom cells.** For a cross-temporal system with `N_b` bottom entities and `T_b` bottom buckets, the coherent subspace has dimension `N_b x T_b`. Every aggregate, in either direction, is a fixed linear combination of those cells. That sentence generates every dimension in this file: `N x T` total nodes, `N_b x T_b` degrees of freedom, `N x T - N_b x T_b` independent constraints.

### 3.5 Reading the three panels together

The three panels answer three different questions, and a meeting usually asks them in the wrong order.

Panel A answers *what should add up to what*. It is a design question, decided before any model is fitted, and it is decided twice: once when you choose the calendar and once when you choose the ladder. If those two choices disagree, Panel A is a picture of an intention rather than a structure.

Panel B answers *where the design fails*. It is a diagnostic, and it is local: it is about one month, not about the structure in general. The value of running it on a named month rather than reasoning abstractly is that the failure has a size — 42 against 31, or 63 eaches against 1,000 — and a size can be argued about in a meeting where an abstraction cannot.

Panel C answers *what has to be true everywhere at once*. It is the operational question, and it is the one that costs money. A company can survive an incoherent month. It cannot survive an incoherent quarter in which the monthly plans, the SKU plans, and the financial forecast all point different ways, because then every meeting becomes a negotiation over which number is real.

Read in that order — design, then diagnose, then integrate — and the arithmetic in section 6 reads as a worked answer rather than as a set of unrelated calculations.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Temporal hierarchy | The non-overlapping aggregations of one series to progressively coarser time buckets | Called "the time dimension" of an entity hierarchy | Ordered list of aggregation factors |
| Temporal aggregation | Summing a series over consecutive, disjoint intervals | Applied to ratios and prices, which do not aggregate additively | Factor, e.g. 7 for day to week |
| Non-overlapping aggregation | Aggregation in which each bottom bucket belongs to exactly one aggregate at each level | Confused with a rolling or trailing window | Boolean per level |
| Overlapping aggregation | Aggregation over a moving window, so consecutive aggregates share buckets | Used to build a "temporal hierarchy", which it cannot | Window length in buckets |
| Aggregation factor | The number of bottom buckets that sum to one aggregate at a level, denoted `k` | Assumed to be any integer | Integer, usually a divisor of the seasonal period |
| Seasonal period | The number of observations in one seasonal cycle at a level, denoted `m` | Assumed to stay the same across levels | Observations per cycle; 12 monthly, 52 weekly |
| Divisor rule | Only factors of `m` keep the seasonal period integral at the aggregate level | Ignored, producing a seasonal period of 4.33 | Set of admissible factors |
| Temporal summing matrix `S_te` | The `n_te x T_b` 0/1 matrix mapping bottom buckets to every temporal node | Assumed identical to the entity summing matrix | Matrix, entries 0 or 1 |
| Temporal constraint matrix `C_te` | The `(n_te - T_b) x n_te` matrix of constraints, with `C_te y = 0` | Omitted because "the summing matrix is enough" | Matrix, entries 0, 1, and -1 |
| Coherent subspace (temporal) | The set of vectors in which every aggregate equals the sum of its children in time | Treated as a single target vector | Subspace of dimension `T_b` per series |
| Coherence gap (temporal) | The difference between an aggregate base forecast and the sum of its temporal children | Reported as forecast error | Units, and ratio to the aggregate |
| Cross-temporal hierarchy | The product of an entity hierarchy and a temporal hierarchy: every entity at every bucket | Called "hierarchical forecasting" without qualification | Lattice, `N x T` nodes |
| Cross-temporal coherence | Every entity constraint holds at every bucket and every temporal constraint holds for every entity | Confused with sequential coherence | Boolean over the lattice |
| Sequential reconciliation | Reconciling cross-sectionally then temporally, or the reverse | Assumed to produce cross-temporal coherence | Ordered pair of projections |
| Iterative reconciliation | Alternating the two projections to convergence | Assumed to converge in one or two passes | Iteration count plus tolerance |
| Optimal cross-temporal combination | A single projection onto the intersection of both constraint families | Called "MinT" without saying which covariance | Estimator plus covariance structure |
| Kronecker product | The block matrix `A ⊗ B` whose `(i,j)` block is `a_ij B` | Treated as an implementation detail | Matrix operation |
| Separable covariance | A covariance written `Σ_cs ⊗ Σ_te`, so entity and time errors factor | Assumed without testing | Two matrices plus an assumption |
| Bottom level | The finest bucket the hierarchy is built on | Confused with the level a model forecasts at | One bucket, e.g. one day |
| Top level | The coarsest bucket, usually the year | Assumed to be the most reliable | One bucket, e.g. one year |
| Temporal disaggregation | Distributing a known aggregate to finer buckets under a stated rule | Presented as estimation when it is assumption | Units per bucket |
| Benchmarking | Adjusting a high-frequency series so its sums match a known low-frequency total | Confused with forecasting | Constraint plus adjustment method |
| Denton method | Least-squares benchmarking minimising deviation from a proxy under the aggregation constraint | Used without stating the differencing order | Method plus difference order |
| Chow–Lin method | Regression-based disaggregation with an indicator series and a residual model | Assumed to be a forecasting method | Method plus residual model |
| Proportional first difference | A Denton variant preserving the period-to-period growth of the proxy | Called "Denton" without the variant | Method variant |
| Allocation key | The rule that splits an aggregate across buckets | Left implicit in a spreadsheet | Rule plus evidence |
| Day-weighted bridge | Allocating a month across weeks in proportion to overlapping days | Treated as reconciliation | Fraction per bucket, summing to 1 |
| 4-4-5 calendar | A retail fiscal calendar whose 13-week quarters split into months of 4, 4, and 5 weeks | Used as the generic name for the National Retail Federation (NRF) 4-5-4 calendar | Week pattern per quarter |
| 52/53-week year | A fiscal year of 364 or 371 days ending on a fixed weekday | Reported without saying which | Weeks per year |
| ISO week date | A date expressed as week-year, week number, and weekday, Monday-start | Joined to a fiscal week number | `YYYY-Www-D` |
| Weekday composition | The count of each weekday in a period | Ignored when comparing months | Counts per weekday |
| Trading day | A day on which the business books volume | Counted as a calendar day | Count per period |
| Leap-year regressor | A calendar regressor absorbing the extra day of 29 February | Replaced by dividing by 365 | Typically `0.75` and `-0.25` |
| Calendar regressor | A regressor encoding trading days, moving holidays, or leap years | Omitted from a model that needs it | Regressor plus coding |
| Seasonal adjustment | Removal of the seasonal and calendar components from a series | Confused with deseasonalising a forecast | Adjustment method plus regressors |
| regARIMA | The regression-with-ARIMA-errors model used to estimate calendar effects before seasonal adjustment | Treated as a forecasting model | Model specification |
| X-13ARIMA-SEATS | The seasonal adjustment program maintained by the U.S. Census Bureau | Assumed to be a forecasting package | Software plus version |
| Frequency aliasing | A cycle that appears at a different period, or vanishes, when a series is aggregated | Read as a genuine change in dynamics | Period ratio |
| Horizon-specific gain | An accuracy improvement measured at a stated forecast horizon | Reported as a single average across horizons | Percentage at horizon `h` |
| Mean absolute scaled error (MASE) | Mean absolute error scaled by the in-sample naive error | Compared across different bucket lengths | Dimensionless |
| Forecast bucket | The bucket a forecast is expressed in | Confused with the bucket the plan executes in | Bucket length in days |
| Consumption bucket | The bucket in which a forecast is consumed by actual orders | Left unconfigured | Bucket length plus consumption direction |
| Planning bucket | The interval a plan is expressed in, chosen from the decision cadence | Chosen to match a report | Days or weeks |
| Frozen horizon | The portion of the plan that cannot be changed | Ignored when choosing the temporal bottom level | Days |
| Multiple aggregation prediction (MAPA) | Averaging forecasts produced at several non-overlapping aggregation levels | Confused with temporal-hierarchy reconciliation | Method label |
| Temporal hierarchy forecasting | The reconciliation approach implemented in the `thief` R package | Used as a synonym for MAPA | Method plus weight scheme |

Four rows repay expansion.

**Temporal aggregation versus a moving average.** A temporal hierarchy uses non-overlapping aggregation: January, February, and March are disjoint and the quarter is their sum. A twelve-month trailing total is also an aggregate, but consecutive trailing totals share eleven months, so they cannot be nodes in a tree. Confusing the two produces a "hierarchy" whose constraints are all satisfied and whose bottom level does not exist.

**The divisor rule.** For monthly data `m = 12`, so the admissible factors are 1, 2, 3, 4, 6, and 12. A factor of 5 gives an aggregate whose seasonal period is `12/5 = 2.4`, which is not a period. This is why the ladder in industry is month, quarter, half-year, year rather than month, two-month, five-month. For daily data the natural period is 7 and the only non-trivial factor is 7, so any ladder reaching a month has left the divisor rule and entered the calendar problem.

**Benchmarking versus reconciliation.** Benchmarking takes a known aggregate as true and adjusts the detail. Reconciliation treats every level as fallible and finds a compromise. Statistical offices benchmark gross domestic product because the annual figure is audited and the quarterly one is estimated. A company forecasting demand has no audited level, so reconciliation is the right frame. The two are often implemented in the same software and described with the same words.

**Separability.** A separable covariance says the correlation between two cells factorises into a part depending only on which entities they are and a part depending only on which buckets they are: the correlation between SKU A in January and SKU B in February is an entity correlation times a time correlation, with no interaction. It is a strong assumption and it is the only one that makes the covariance estimable at realistic sizes.

## 5. The mechanics, step by step

Five phases. Phase A fixes the calendar and the ladder. Phase B builds the temporal constraint system. Phase C reconciles in time alone. Phase D reconciles across time and entities together. Phase E decides which bucket must be coherent. Each step names its output and the branch at which the analysis can legitimately stop.

### Phase A — Fix the calendar and the ladder

**Step 1. Choose the bottom bucket from the decision cadence and the data granularity, not from the report.** The bottom level is the finest bucket you will ever reconcile to. If nobody acts on a daily number, a daily bottom level multiplies the constraint count by roughly thirty for no operational benefit.

- *Output:* one bottom bucket, stated in days, with the decision it serves.
- *Branch:* if the finest decision is weekly, build the ladder from week upward and treat daily data as an allocation problem downstream.
- *Failure if skipped:* the hierarchy is built on the granularity of the data warehouse, which is daily because that is how events arrive, and the reconciliation cost is set by a schema nobody chose for this purpose.

**Step 2. Choose the ladder, respecting the divisor rule.** Take the seasonal period at the bottom level, list its divisors, build the ladder from those. Monthly with `m = 12` gives 1, 3, 6, 12. Weekly with `m = 52` gives 1, 2, 4, 13, 26, 52 — note that 4 is admissible and 4.33 is not, which is why a four-week period is a natural level and a calendar month is not.

- *Branch:* if the business needs a level whose factor is not a divisor, that level must be reached through the calendar (step 3), not through the hierarchy.

**Step 3. Choose the calendar in which the ladder nests.** Three options, and the choice is not cosmetic.

| Calendar | Levels that nest exactly | What does not nest | Cost |
|---|---|---|---|
| Gregorian | day into month, month into quarter, quarter into year | week into month | The weekly plan cannot equal the monthly plan |
| NRF 4-5-4 retail | day into week, week into period, period into quarter, quarter into year | fiscal period into Gregorian month | Periods are not calendar months; a second calendar is needed for finance |
| 52/53-week fiscal | day into week, week into fiscal year | week into any sub-annual month | Sub-annual buckets have to be defined inside the fiscal year |

- *Branch:* if the company reports to a parent on Gregorian months and plans on 4-5-4, it needs two calendars and an explicit bridge. No single calendar satisfies both.

**Step 4. Verify nesting on a date spine before building any matrix.** Expand every date in the planning window, attach every level's label, and group. Three assertions: every date appears exactly once; the day counts within each node sum to the node's length; every node at level `k` has exactly one parent at level `k+1`.

- *Branch:* if an assertion fails, stop. A summing matrix built on a non-nesting calendar encodes a falsehood, and every number downstream inherits it.

### Phase B — Build the temporal constraint system

**Step 5. Build `S_te`.** One row per temporal node, one column per bottom bucket. A row holds a 1 in every column whose bucket lies inside that node's interval, 0 elsewhere. The root row is all ones; a bottom row is a unit vector. Every column sums to the number of levels; every row sums to the number of bottom buckets inside that node.

**Step 6. Build `C_te` and verify `C_te S_te = 0`.** One row per aggregate node: the node itself with coefficient +1 and each of its children with coefficient -1. If step 4 passed, the rows are independent and there are exactly `n_te - T_b` of them. If the rows are not independent, the structure is not a tree: a redundant row is the algebraic signature of an overlapping aggregation smuggled in.

**Step 7. Measure the coherence gap at every level pair, before reconciling.** Report three numbers per pair: the gap in units, the gap as a ratio to the parent, and the gap per bottom bucket.

- *Branch:* if the month-against-weeks gap is not reproducible from the month-against-days gap, the week level is not nested and no reconciliation can fix it. The structure is wrong, not the forecast.

**Step 8. Decide whether the gap is a disagreement or a calendar artefact.** A gap that appears in one month and vanishes in the next is usually a calendar artefact — a five-Sunday month, a 53rd week, a leap day. A gap that persists across every month with the same sign is a disagreement between models, which reconciliation averages away rather than resolves.

- *Branch:* if the cause is a calendar artefact, fix the calendar or add the regressor. Do not reconcile a modelling defect.

### Phase C — Reconcile in time alone

**Step 9. Produce base forecasts independently at every temporal level.** A monthly model, a quarterly model, and an annual model see different history lengths, different signal-to-noise ratios, and different seasonal structure. That difference is the resource reconciliation exploits.

- *Failure if skipped:* using the monthly forecast divided by the number of weeks as the weekly base produces a hierarchy that is coherent by construction and carries no weekly information at all.

**Step 10. Choose the weights `W_te`.** Three schemes dominate, and their names come from the temporal hierarchy literature.

| Scheme | Weight for a level | What it encodes | When it is defensible |
|---|---|---|---|
| Hierarchy variance (WLS-H) | Proportional to that level's estimated error variance | Each level's own estimated reliability | Enough history at every level |
| Series variance (WLS-V) | Proportional to the in-sample variance at that level | Noisier series get less weight | Levels differ greatly in volume |
| Structural scaling (WLS-S) | Proportional to the number of bottom buckets in the aggregate | Longer aggregates average more, hence carry relatively less noise | No reliable error estimates, or judgemental base forecasts |

**Step 11. Apply the projection.** `y_tilde = S_te (S_te' W_te^-1 S_te)^-1 S_te' W_te^-1 y_hat_te`. On a strict temporal tree with a diagonal `W` this has a closed form in terms of the node weights, which is why the three-node case in section 6.5 can be done by hand.

- *Check:* `C_te y_tilde = 0` to tolerance; a coherent input returns unchanged.

**Step 12. Report the residual, not just the result.** The residual `(I - S_te P_te) y_hat_te` is the part of the base forecasts the projection discarded. Its size at each level tells you which level was overruled, which is the number a planner will ask about in the meeting.

### Phase D — Reconcile across time and entities

**Step 13. Stack the two structures.** The cross-temporal summing matrix is the Kronecker product `S_te ⊗ S_cs`, mapping the `N_b x T_b` bottom cells to all `N x T` nodes. If the vector is ordered entity-major the product is `S_cs ⊗ S_te` instead. The ordering is arbitrary; mixing the two conventions inside one implementation is not. Work the dimensions out longhand once, for one small case, and write the answer into the register.

**Step 14. Choose the covariance structure.** The full covariance has `(N x T)^2` entries, which for 256 entities and 51 buckets is over 170 million.

| Structure | Parameter count | Assumption | When it is defensible |
|---|---|---|---|
| Scaled identity | 1 | All cells equally reliable | Never, but a useful baseline |
| Separable, `Σ_cs ⊗ Σ_te` | `N^2 + T^2` | Entity and time errors factor | The default; test it |
| Full | `(N x T)^2` | Nothing | Almost never estimable |

- *Branch:* if a separability test rejects, the honest response is a richer separable form — entity groups by time structures — not a full covariance.

**Step 15. Solve, and budget for the solve.** The optimal estimator requires one inversion per horizon, or a single large system. The recent work on sequential and iterative approaches exists because that solve is expensive on large lattices, and because a well-chosen iteration can reach the same answer.

**Step 16. Compare against the two sequential heuristics.** Reconcile cross-sectionally then temporally (`cs -> te`), and the reverse (`te -> cs`). Neither is cross-temporally coherent in general, because the second projection breaks the first. Iterating either to convergence can recover the optimal solution under conditions stated in the literature, which is a useful fact and not a licence to skip the comparison.

- *Output:* four results — optimal, `cs -> te`, `te -> cs`, and iterated — plus the base.
- *Branch:* if the four agree to within the noise of the backtest, use the cheapest. Agreement is information.

### Phase E — Judge, and choose the bucket

**Step 17. Backtest with common origins, at every level and every horizon.** Hold the base forecasts fixed, hold the origins fixed, and evaluate at each temporal level separately. The horizon dimension is the one almost always collapsed and almost always material.

**Step 18. Report horizon-specific gains, never a single average.** A table with horizons as rows and levels as columns is the minimum honest presentation. A single number across a 52-week horizon is dominated by the long horizons, where errors are large and percentage gains are easy.

**Step 19. Decide which bucket must be coherent, write the bridge, and publish both.**

| Planning process | Bucket that must be coherent | Why | Bridge to the rest |
|---|---|---|---|
| Daily replenishment | Day, within the current week | Orders are placed per day and consumed against a weekly forecast | The weekly total is a consumption cap, not a constraint |
| Weekly build scheduling | Week, within the frozen horizon | The schedule is a commitment to the plant | The monthly plan is the capacity envelope; restate weekly |
| Monthly S&OP | Month, within the quarter | The plan is signed at month level by function | Weekly detail is allocated inside the month |
| Financial forecast | Quarter, within the fiscal year | Revenue and cost are reported quarterly | Month is a management view; the audited constraint is the quarter |

- *Output:* one row per planning stream, naming the coherent bucket, the bridge rule to each adjacent bucket, and the owner of the bridge.
- *Failure if skipped:* every bucket is reconciled to every other, the results conflict, and the organisation resolves the conflict by meeting, which is the most expensive reconciliation method available.

The output of Phase E is a single page, and it should be treated as the deliverable of the whole exercise. Everything in phases A to D exists to make that page true. A hierarchy that is arithmetically perfect and has no consumer is a research result; a hierarchy that is approximate and named in the planning calendar is a control. Most companies need the control.

## 6. Worked example

Meridian Components is a fictional distributor of one industrial part. The figures are constructed to be internally consistent and the arithmetic is shown so you can check it. The unit is eaches and the planning window is March 2026.

### 6.1 The calendar facts

March 2026 has 31 days, begins on Sunday 1 March, and ends on Tuesday 31 March. Because 31 days starting on a Sunday gives five Sundays, five Mondays, and five Tuesdays, the month has an unusual weekday composition.

| Weekday | Occurrences | Dates |
|---|---:|---|
| Sunday | 5 | 1, 8, 15, 22, 29 |
| Monday | 5 | 2, 9, 16, 23, 30 |
| Tuesday | 5 | 3, 10, 17, 24, 31 |
| Wednesday | 4 | 4, 11, 18, 25 |
| Thursday | 4 | 5, 12, 19, 26 |
| Friday | 4 | 6, 13, 20, 27 |
| Saturday | 4 | 7, 14, 21, 28 |

If Sunday, Monday, and Tuesday are the high-volume days — plausible for a distributor whose customers order early in the week and whose consumer channel peaks at the weekend — then March 2026 carries three extra high-volume days relative to a 30-day month with the same start. That is a calendar effect of roughly the size of a small promotion, invisible in any model that works in months.

The ISO weeks overlapping March are six, and only four of them lie wholly inside it.

| ISO week | Date range | March days |
|---|---|---:|
| 2026-W09 | Mon 23 Feb – Sun 1 Mar | 1 |
| 2026-W10 | Mon 2 Mar – Sun 8 Mar | 7 |
| 2026-W11 | Mon 9 Mar – Sun 15 Mar | 7 |
| 2026-W12 | Mon 16 Mar – Sun 22 Mar | 7 |
| 2026-W13 | Mon 23 Mar – Sun 29 Mar | 7 |
| 2026-W14 | Mon 30 Mar – Sun 5 Apr | 2 |
| **Total** | | **31** |

### 6.2 Three base forecasts at three levels

Three models ran independently, as they do in a real planning stack, and each produced a March number.

| Level | March total implied (eaches) | Method |
|---|---:|---|
| Month | 1,000 | Seasonal exponential smoothing on 48 months of history |
| Week | 1,063 | Six weekly forecasts, allocated to March by overlapping days |
| Day | 1,047 | 31 daily forecasts from a day-of-week profile model |

The weekly figure needs one line of explanation. The six full-week base forecasts are 232, 238, 231, 245, 252, and 224 eaches. Allocating them to March by the number of overlapping days:

| Week | Full-week base | Fraction in March | Allocated |
|---|---:|---:|---:|
| W09 | 232 | 1/7 | 33.143 |
| W10 | 238 | 7/7 | 238.000 |
| W11 | 231 | 7/7 | 231.000 |
| W12 | 245 | 7/7 | 245.000 |
| W13 | 252 | 7/7 | 252.000 |
| W14 | 224 | 2/7 | 64.000 |
| **Total** | | | **1,063.143** |

The daily figure comes from a weekday profile of 38 on Sunday, 42 on Monday, 39 on Tuesday, 30 on Wednesday, 31 on Thursday, 32 on Friday, and 20 on Saturday, applied to the weekday counts of section 6.1.

| Weekday | Profile (eaches/day) | Occurrences | Contribution |
|---|---:|---:|---:|
| Sunday | 38 | 5 | 190 |
| Monday | 42 | 5 | 210 |
| Tuesday | 39 | 5 | 195 |
| Wednesday | 30 | 4 | 120 |
| Thursday | 31 | 4 | 124 |
| Friday | 32 | 4 | 128 |
| Saturday | 20 | 4 | 80 |
| **Total** | | **31** | **1,047** |

Three numbers for one month: 1,000, 1,063, and 1,047. The spread between highest and lowest is 63 eaches, 6.3% of the monthly base. At a unit cost of 30 euro and a 22% annual holding cost, a 63-unit error held for a month costs about 34 euro; across 400 active parts it is 13,700 euro a month of avoidable inventory or shortage, before any service consequence.

### 6.3 Why these three numbers cannot be reconciled as a tree

The coherence gap depends on which pair you compare, and the pairs are not independent.

| Comparison | Parent base | Sum of children | Gap | Gap as % of parent |
|---|---:|---:|---:|---:|
| Month against weeks | 1,000 | 1,063.143 | -63.143 | -6.31% |
| Month against days | 1,000 | 1,047.000 | -47.000 | -4.70% |
| Weeks against days (March window) | 1,063.143 | 1,047.000 | +16.143 | +1.52% |

Now add the requirement that the week level be a genuine parent of the day level and a genuine child of the month level. It cannot be both, because W09's one February day belongs to February and W14's five April days belong to April. The constraint system would need the row `March - W09_march - W10 - W11 - W12 - W13 - W14_march = 0`, and `W09_march` is not a series. It is a fraction of a series, and a fraction of a series cannot appear in a summing matrix.

The same defect from the other side: the week-to-day constraint `W09 - sum(days in W09) = 0` is valid over a week spanning two months, and the month-to-day constraint `March - sum(days in March) = 0` is valid. Both are true. But there is no month-to-week row that is true, so the three-level tree does not exist and only the two-level ones do.

### 6.4 The month-and-days reconciliation, done properly

The honest structures are month and days, or week and days. Take month and days. The month base is 1,000 and the 31 daily base forecasts sum to 1,047, so the gap is -47. With structural scaling the month node has weight 31 and each day has weight 1, so the total weight is 62.

For a tree with a single constraint, the projection moves each node by its weight share of the gap, with the parent moving opposite the children:

| Node | Calculation | Base | Reconciled |
|---|---|---:|---:|
| March | `1,000 - 31 x (-47)/62` | 1,000 | 1,023.500 |
| Each day | `base - 1 x (-47)/62` | as modelled | base - 0.758 |
| **Sum of days** | `1,047 - 31 x 0.758` | | **1,023.500** |

The month moves up by 23.5 and every day is scaled down by 0.758. Coherent, and note what the weighting did: the month node has weight 31 against a total of 62, so it carries half the adjustment despite being one series out of thirty-two. That is the structural-scaling claim in one number — a node representing 31 buckets is treated as carrying 31 times the information of one bucket, so it moves 31 times as far.

### 6.5 A feasible temporal tree, reconciled by hand

Now suppose the company plans on a 4-4-5 style calendar in which the planning month is four whole weeks. Take a period `P` of four weeks, split into two halves `H1` (weeks 1 and 2) and `H2` (weeks 3 and 4). This tree nests exactly. Base forecasts: `P = 1,000`, `H1 = 420`, `H2 = 560`. Children sum to 980, so the gap is `d = +20`.

With `W = diag(400, 900, 1600)`, read as error variances in units squared, each node moves by `d x w_i / sum(w)`:

| Node | Calculation | Base | Reconciled |
|---|---|---:|---:|
| P | `1,000 - 20 x 400/2,900` | 1,000 | 997.241 |
| H1 | `420 + 20 x 900/2,900` | 420 | 426.207 |
| H2 | `560 + 20 x 1,600/2,900` | 560 | 571.034 |

The check: `426.207 + 571.034 = 997.241`. Coherent to rounding. H2 moves most because its variance is largest, so the method trusts it least.

Change the weights to `W = diag(1600, 400, 900)` and the period total becomes `1,000 - 20 x 1,600/2,900 = 988.966`, with `H1 = 422.759` and `H2 = 566.207`. The period total moves by 8.276 eaches — 0.83% of the base — purely from a statement about which level is more reliable. The weights are the model, and the data does not supply them.

### 6.6 The bridge assumption, measured

Section 6.2 allocated the boundary weeks by overlapping days, which assumes volume is uniform within a week. The daily model says it is not: Sunday carries 38 eaches against a week total of 232, which is 16.4% of the week, not 14.3%. Allocate the boundary weeks by the profile share instead.

| Week | Full-week base | Profile share in March | Allocated |
|---|---:|---:|---:|
| W09 | 232 | 38/232 = 0.16379 | 38.000 |
| W10 to W13 | 238, 231, 245, 252 | 1.0 | 966.000 |
| W14 | 224 | 81/232 = 0.34914 | 78.207 |
| **Total** | | | **1,082.207** |

Both boundary weeks contain the same seven weekdays, so both profiles sum to 232 and the shares are comparable. The two keys give 1,063.143 and 1,082.207. The difference, 19.06 eaches, is 1.9% of the monthly base and is entirely an artefact of the allocation rule. No forecast improved. The bridge choice alone moved the number by about a third of the total coherence gap.

That is the measured cost of a non-nesting calendar, and it is why section 5 step 19 asks which bucket must be coherent rather than assuming all of them must.

### 6.7 The cross-temporal system for the same business

Meridian's planning structure has 240 SKUs, 12 categories, 3 divisions, and a total: 256 entity nodes, of which 240 are bottom. The temporal ladder is 36 months, 12 quarters, and 3 years: 51 temporal nodes, of which 36 are bottom.

| Quantity | Value | Derivation |
|---|---:|---|
| Cross-temporal nodes `N x T` | 13,056 | 256 x 51 |
| Bottom cells `N_b x T_b` | 8,640 | 240 x 36 |
| Independent constraints | 4,416 | 13,056 - 8,640 |
| Naive stacked constraint rows | 4,656 | 256 x 15 temporal + 16 x 51 entity |
| Redundant rows in the stack | 240 | 4,656 - 4,416 |
| Entity-only constraints per bucket | 16 | 256 - 240 |
| Ratio, cross-temporal to entity-only rows | 291 | 4,656 / 16 |

The last row is what makes this topic operationally different from topic 02. The entity constraint matrix has sixteen rows. The cross-temporal one has four and a half thousand, and the redundancy in the naive stack is a warning: an implementation that stacks both families and inverts will produce a singular matrix unless it reduces to an independent set or uses a pseudo-inverse.

The covariance tells the same story. A full covariance over 13,056 cells has 170,459,136 entries. A separable covariance `Σ_cs ⊗ Σ_te` has 65,536 plus 2,601, about 68,137 parameters. The separable form is not a convenience; it is the difference between a model that can be estimated and one that cannot.

### 6.8 What this shows and does not show

It does not show that temporal reconciliation improves accuracy. There is no realised March here, only base forecasts, and section 9 is clear that the accuracy evidence is conditional and often runs against the popular claim.

It does show four things. The three levels disagree by 6.3% on one month of one part. The disagreement cannot be closed by a tree, because the tree does not exist on an ISO calendar. The bridge assumption moves the answer by 1.9%. And the cross-temporal system is roughly three hundred times the size of the entity system it sits underneath, which is why companies reconcile in time selectively and across entities comprehensively.

## 7. Code, query, or calculation

One hand calculation. The point is the arithmetic of the calendar and of the projection, both easier to check on paper than in a library.

```text
MARCH 2026 — day, week, month (eaches)

  March: 31 days, begins Sunday 1 March.
  ISO weeks overlapping March, with March days inside each:
    W09 23Feb-1Mar 1   W10 2-8  7   W11 9-15  7
    W12 16-22      7   W13 23-29 7  W14 30Mar-5Apr 2
    check: 1+7+7+7+7+2 = 31

  Weekly base allocated by overlapping days:
    232*(1/7) + 238 + 231 + 245 + 252 + 224*(2/7)
    = 33.143 + 238 + 231 + 245 + 252 + 64.000 = 1,063.143

  Daily base, profile 38,42,39,30,31,32,20 over 5,5,5,4,4,4,4 days:
    5*38 + 5*42 + 5*39 + 4*30 + 4*31 + 4*32 + 4*20 = 1,047

  Monthly base 1,000
  gap month - weeks = 1,000 - 1,063.143 = -63.143   (-6.31%)
  gap month - days  = 1,000 - 1,047.000 = -47.000   (-4.70%)

MONTH AGAINST DAYS, STRUCTURAL SCALING (month weight 31, each day 1)
  sum of weights = 31 + 31*1 = 62 ; gap d = 1,000 - 1,047 = -47
  March    -> 1,000 - 31*(-47)/62 = 1,000 + 23.500 = 1,023.500
  each day -> day - 1*(-47)/62    = day - 0.758
  check: 1,047 - 31*0.758 = 1,047 - 23.500 = 1,023.500
```

The closing check is the one to run every time. It costs one line and catches every normalisation error, which is the failure this projection is most prone to.

## 8. Data, standards, and regulatory context

There is no standard for temporal forecast reconciliation. There are standards for labelling time, for seasonal adjustment with calendar effects, and for interim financial reporting, and each constrains what a temporal hierarchy can be built on.

### 8.1 Time representation

| Standard | Governs | What it fixes | Grade |
|---|---|---|---|
| ISO 8601-1:2019 | Date and time representation | Calendar dates, week dates, Monday as first weekday, week 01 as the week containing the first Thursday | B |
| ISO 8601-2:2019 | Extensions | Reduced precision, uncertain dates, intervals and durations | B |
| RFC 3339 | Internet timestamps | A mandatory offset on every timestamp | B |
| IANA time zone database | Civil time | Zone identifiers and transition rules, with daylight saving time (DST) transitions | B |

The relevant consequence is narrow and decisive. ISO 8601 gives a week that is seven days and a month that is 28 to 31 days. It does not give a calendar in which the two nest, because no such calendar exists within it. A temporal hierarchy built on ISO weeks and Gregorian months is built on a standard that guarantees the two will not line up. DST matters at the daily level for the same reason: a local day can be 23, 24, or 25 hours long, so a daily series in local time does not have a constant bucket length, and any temporal ladder reaching the hour inherits that.

### 8.2 Fiscal and retail calendars

| Instrument | Jurisdiction | What it establishes | Grade |
|---|---|---|---|
| NRF 4-5-4 calendar | Retail, international by adoption | A 52/53-week merchandising calendar with 4-5-4 months and 13-week quarters | C |
| 26 CFR §1.441-2 | United States | Election of a tax year of 52 or 53 weeks ending on a fixed weekday | B |
| IRS Publication 538 | United States | The 52-53-week year: same weekday each year, last or nearest occurrence to a month end | B |
| IAS 34 | IFRS reporters | Interim reporting and the comparability of interim periods | B |
| IFRS 8 | IFRS reporters | Reconciliation of segment totals to the financial statements | B |

IAS 34 is the temporal one: it requires interim periods to be comparable, which in a 52/53-week year means stating which normalisation was used. Neither it nor IFRS 8 tells a company which bucket to plan in.

### 8.3 Calendar adjustment in seasonal adjustment

| Item | What it does | Source and grade |
|---|---|---|
| Trading-day regressors | Seven day-of-week dummies, or six contrasts if an intercept is present | X-13ARIMA-SEATS documentation, U.S. Census Bureau. Grade B |
| Leap-year regressor | Codes 29 February, typically `0.75` in a leap February and `-0.25` in a common February, so the coefficients sum to zero over the four-year cycle | X-13ARIMA-SEATS documentation. Grade B |
| Moving-holiday regressors | Shift Easter and similar holidays by their date each year | X-13ARIMA-SEATS documentation. Grade B |
| Length-of-month adjustment | Divide by days in month, or by 30.4375, before modelling | Cleveland and Devlin (1982). Grade A |
| Leap-year pre-adjustment | Adjust February to an average length of 28.25 days before seasonal adjustment | Office for National Statistics methodological note. Grade B |

The Office for National Statistics note makes the arithmetic concrete: in a flow series an extra 29 February can move February output by roughly 3.6% with no change in the underlying business, and whether it does depends on which weekday 29 February falls on. That is the same mechanism as section 6.1's five Sundays, one level down.

> **Note:** these adjustments remove a calendar effect from a *series*. They do not make a temporal hierarchy nest. A trading-day-adjusted monthly series is still not the sum of four or five weekly series. Adjustment and reconciliation solve different problems and are often confused because both are described as "calendar correction".

### 8.4 What is convention and what is requirement

- No jurisdiction requires that a company's weekly plan sum to its monthly plan.
- The 4-5-4 calendar is a voluntary retail convention. The tax rules permit a 52/53-week year but do not require one and do not prescribe a week pattern.
- The choice of aggregation factors is unconstrained by any standard. The divisor rule is a mathematical convenience, not a regulation.
- IAS 34's comparability requirement binds the *disclosure*, not the planning bucket. A company can plan weekly, report quarterly, and comply.
- IFRS 8's reconciliation requirement is the closest a regulator comes to the temporal question: a segment total must be reconciled to the financial statements. That is the entity analogue of the bridge in section 5 step 19, and it is the strongest argument that a documented bridge is a governance artifact rather than a modelling nicety.

## 9. Industry practice and case evidence

The temporal hierarchy literature is substantial and its findings are more conditional than the practitioner summaries suggest. This section separates what the peer-reviewed record establishes from what vendors claim and what companies appear to do.

### 9.1 The line of work

The modern temporal hierarchy is usually dated to Athanasopoulos, Hyndman, Kourentzes, and Petropoulos (2017), "Forecasting with temporal hierarchies", *European Journal of Operational Research*. The construction is the one in section 5: non-overlapping aggregation to levels whose factors divide the seasonal period, independent base forecasts at each level, and an optimal combination with a summing matrix, reusing the hierarchical reconciliation algebra. Three weight schemes were proposed — hierarchy variance, series variance, and structural scaling — and the last needs no error estimates, which is why it survives where some levels are judgemental.

It has a direct predecessor. Multiple aggregation prediction (MAPA), by Kourentzes, Petropoulos, and Trapero (2014) in the *International Journal of Forecasting*, averaged forecasts produced at several aggregation levels. The difference is the one that matters theoretically: MAPA is a *combination*, with weights that need not respect any constraint, whereas the temporal hierarchy is a *projection*, which guarantees the levels add up. A combination can be more accurate; a projection is coherent.

The cross-temporal extension is later and is associated with Di Fonzo and Girolimetto. Their 2023 paper in the *International Journal of Forecasting* gives the closed-form optimal combination satisfying both contemporaneous and temporal constraints, plus an iterative heuristic that alternates the two one-dimensional reconciliations. The 2025 follow-up in *Statistical Methods & Applications* analyses when the sequential and iterative routes coincide with the optimal one: roughly, when the error covariances are constant across levels and the entity and temporal covariances are compatible, iteration converges to the optimal solution regardless of order.

### 9.2 The accuracy evidence, stated with its gradient

This is where the popular claim and the literature diverge, and the divergence is the most useful thing in this topic.

| Study | Setting | Reported result | Grade |
|---|---|---|---|
| Athanasopoulos et al. (2017), *EJOR* | M3 monthly and quarterly; UK accident and emergency (A&E) weekly demand | Monthly best symmetric mean absolute percentage error (sMAPE) 13.61% against Theta's 13.85%; quarterly best 9.70% against Theta's 8.96%; A&E mean absolute scaled error (MASE) improvements of roughly 17–19% at short horizons and about 43% at the annual level | A |
| Spiliotis, Petropoulos, Assimakopoulos (2019), *PLOS ONE* | 2,045 monthly M and M3 series | Accuracy improvements of 1.8% to 5.4%, higher at the coarsest levels and declining as the frequency rises; bias improvements 4.3% to 7.9% | A |
| Nystrup et al. (2019), *EJOR* | Electricity load and realised volatility | Improvements of 12–21% at the daily level and 0–8% at the hourly level | A |
| Petropoulos, Hollyman, Spiliotis (2025), *JORS* | M5 Walmart daily data, 28-day horizon, sub-hierarchies | Slight deterioration at horizons 1–7 for some weight schemes; gains of roughly 7.5–8.5% beyond the first two weeks; about 4% overall | A |
| Lipiecki, Bilińska, Kourentzes, Weron (2026), *IJF* | Day-ahead electricity prices, Germany and Spain | Up to 5.5% for hourly and 13.4% for baseload prices, with gains increasing at higher aggregation levels | A |

Read the table as a gradient rather than a verdict. In four of the five studies the improvement is **largest at the coarsest level and smallest at the finest**, and in one the finest level is slightly worse in the first week. The M3 result is a split decision: temporal hierarchies beat the best benchmark monthly and lost to it quarterly.

**So what of the claim that temporal reconciliation helps most at the shortest horizon?** It is true in the A&E case, where the weekly series is short and noisy and the annual level carries a well-estimated seasonal shape and a stable level the weekly model cannot see. It is not the general finding. The mechanism is real — a coarse forecast can inform a fine one — but it only pays when the coarse forecast knows something the fine forecast does not. When the annual forecast is a smoothed version of the same three years of data, the projection adds an average and subtracts a signal, and the short horizon is where the signal was.

> **Note:** this is the single most important honesty point in the topic. Vendor material and conference talks routinely claim "improves short-term accuracy" without stating the horizon profile or the aggregation level, and the claim does not survive contact with the studies that report both. Ask for the horizon-by-level table.

### 9.3 The intermittent-demand argument, which is different

There is a second argument for temporal hierarchies that is stronger than the accuracy one and much less quoted. A slow-moving part that is zero in 70% of months is dense when aggregated to quarters and denser still at the year. Temporal aggregation makes an intermittent series forecastable, and the reconciliation distributes that forecastability back down.

That is a different mechanism from "long-bucket information informs short-bucket forecasts". It is "aggregation reduces the zero share and the reconciliation preserves the aggregate". Topics 05 and 06 of this module treat the intermittent methods themselves. The point here is that the strongest case for a temporal hierarchy in a spare-parts business is density, not accuracy, and a business case built on density is easier to defend than one built on a percentage.

### 9.4 What practitioners do

The public record on corporate practice is thin and mostly inferential.

| Practice | Evidence status | Reasoning |
|---|---|---|
| Monthly planning, weekly execution, daily replenishment, unreconciled | Inference from planning-system design and job descriptions | Each cadence has an owner and a system; nobody owns the joins between them |
| Calendar effects handled by a "weeks" helper variable in the forecast model | Documented in vendor tooling | Forecast Pro describes de-weighting history to sales per week, forecasting, then re-weighting; one published example reports adjusted R-squared rising from 0.50 to 0.91 |
| 4-4-5 chosen so forecasts break down to whole weeks | Documented in vendor tooling and retail practice | A 4-5-4 period has no partial weeks, so the weekly plan is a genuine decomposition |
| Cross-temporal reconciliation rare; temporal-only reconciliation occasional | Inference | The estimator is expensive and the organisational change is larger than the method |
| Cross-sectional reconciliation common in retail and consumer goods | Inference from job descriptions and vendor material | It is built into planning suites and is a reporting requirement in practice |

The Forecast Pro example is worth noting for what it measures. The adjusted R-squared improvement from 0.50 to 0.91 is a *fit* improvement from removing a calendar artefact, not a forecast accuracy improvement from reconciliation, and the two are routinely conflated in vendor material. Grade D: vendor documentation, single example, no replication.

Two further data points on the operational side. Oracle's supply chain planning documentation distinguishes the forecast bucket from the consumption bucket and requires the consumption direction (forward, backward, or both) to be configured, which is an explicit acknowledgement that the monthly forecast and the daily order do not live in the same bucket. Anaplan community discussion of converting 4-4-5 weekly data to Gregorian months describes the fringe-day delta method and the alternative of moving the difference between months so that the annual total is preserved. Both are Grade D to E: practitioner material, useful as evidence of the problem, not as evidence of a solution.

### 9.5 What practitioners argue about

Four arguments recur, and knowing them is more useful than knowing the method.

**"Should the annual level be in the hierarchy at all?"** Including it means a five-year judgement enters a four-week forecast through the projection. Excluding it means the top of the ladder is a quarter, whose forecast is a smoothed month. There is no right answer; there is an answer that matches the horizon of the decision.

**"Is 4-4-5 worth the second calendar?"** It buys exact nesting and comparable months. It costs a fiscal calendar that finance, payroll, and statutory filing do not use, and a permanent translation layer. Retailers accept the cost; most manufacturers do not need it.

**"Optimal or sequential?"** The optimal solve is expensive and the sequential heuristics are not guaranteed coherent. The 2025 result that iteration can reach the optimum under stated conditions narrows the gap, and the practical answer is usually to run both and compare.

**"Does any of this survive intermittent demand?"** Yes, and the mechanism is density rather than accuracy. A hierarchy built on a series that is mostly zeros at the bottom and dense at the top is doing something different from one built on a dense series, and the reconciliation weights should reflect that difference.

### 9.6 What the evidence does not settle

Three questions are open, and a good practitioner says so rather than closing them with an opinion.

**Whether the cross-temporal gain is worth its cost on realistic lattices.** The studies that report cross-temporal results use moderate dimensions. The published work on scalable reconciliation through sub-hierarchies exists because full lattices are expensive, and the cost-benefit question at a thousand entities by fifty buckets is not settled empirically.

**Whether the temporal gain survives intermittent demand.** The competition datasets are largely dense at the level studied. A spare-parts business has a bottom level that is mostly zero, and the aggregation argument there is about density, which is a different mechanism with a different (and much thinner) empirical base.

**Whether the reconciliation should propagate uncertainty as well as point forecasts.** Reconciling predictive distributions across a temporal hierarchy is an active area, and the naive approach — reconcile the mean, then apply an interval derived from the reconciled mean — does not preserve coverage. Treat any software that does this without saying so as unverified.

## 10. Failure modes and red flags

| # | Failure mode | Red flag in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Month built from weeks by dividing by 4.33 | The weekly plan is exactly the monthly plan divided by a constant | A non-integer factor applied uniformly, ignoring weekday composition | Weekly plan is flat and cannot be executed | Build weeks from days; use a profile key, not a constant |
| 2 | ISO weeks joined to Gregorian months on week number | Annual totals reconcile, but the weekly series has a spike or a gap at month ends | A week straddles two months and is assigned wholly to one | Permanent unexplained variance read as forecast error | Join on a date spine; allocate, do not assign |
| 3 | A temporal "hierarchy" built on overlapping aggregates | Twelve-month rolling totals used as a level | Constraints are satisfied but the bottom level does not exist | The summing matrix is not a summing matrix | Use non-overlapping aggregation only |
| 4 | Coherence claimed where none is possible | "The month equals the sum of the weeks" written in a data dictionary | The calendar does not permit it | An audit that can never pass, so it is switched off | State the bridge explicitly and stop calling it reconciliation |
| 5 | Leap day treated as an ordinary day | 29 February shows a demand spike or a productivity dip | Daily averages computed over a fixed 365 | Annual comparisons biased in leap years | Count days per period; add a leap-year regressor |
| 6 | Weekday composition ignored | Two months of equal length compared with no adjustment | A month with five Mondays is not a month with four | Month-over-month change mixes real movement with calendar | Report weekday counts; use trading-day adjustment |
| 7 | 53-week year inside a temporal hierarchy | The annual node has 371 days and nobody notices | One level's bucket length differs from the others' | A 1.9% level shift attributed to demand | Flag the 53-week year in the calendar table; restate or exclude |
| 8 | Daily forecast produced by dividing the month | Every day in the month carries the same number | False granularity: monthly information at daily resolution | Narrow intervals, poor execution, no weekly pattern | Disaggregate with an explicit day-of-week profile |
| 9 | Structural scaling applied without the closing check | The children do not sum to the reconciled parent | Weights normalised over the wrong total | The output is incoherent in the way the method promised to fix | Compute the sum check before publishing |
| 10 | Separable covariance assumed and never tested | The implementation has a Kronecker product and no diagnostic | Entity and time errors assumed to factor | Weights wrong in a structured, invisible way | Fit a richer separable form; compare out of sample |
| 11 | Whole-horizon average reported for a horizon-specific claim | "Improves accuracy by 8%" with no horizon stated | Long horizons dominate the average | The claim is true and uninformative; the short horizon may be worse | Report a horizon-by-level table |
| 12 | Reconciling to a bucket nobody acts on | The daily reconciled plan is published and the weekly plan is used | The coherent bucket was chosen for completeness, not for a decision | Effort spent on a number with no consumer | Name the decision per bucket; reconcile where a commitment exists |
| 13 | Annual level contaminating a four-week decision | A four-week forecast moves when the five-year outlook is revised | The projection uses every level's base forecast | Short-horizon plan destabilised by long-horizon judgement | Truncate the ladder to the decision horizon, or down-weight the top |
| 14 | Forecast bucket and consumption bucket mismatched | Orders consume a monthly forecast in a daily bucket and the residual is nonsense | Consumption direction and bucket not configured | Phantom forecast error and double counting | Configure both buckets and the consumption direction explicitly |

**Failure 2 is the one with the highest consequence**, because it passes the check organisations actually run. An annual reconciliation compares totals, and a total is precisely what a boundary misallocation preserves. Failure 4 is the most corrosive, because an audit that cannot pass is an audit that gets switched off, and then nobody knows whether the numbers add up.

The failures also have a diagnostic order. Run the calendar checks (5, 6, 7) and the structure checks (2, 3, 4) before the weighting checks (9, 10), because a calendar or structure defect produces symptoms that look exactly like a weighting defect and no weighting change will fix them. A team that starts by tuning weights has usually started in the wrong place.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The chosen calendar nests at every level of the ladder | The summing matrix exists only if buckets nest | ISO weeks against Gregorian months; 4-5-4 periods against calendar months | Run step 4's three assertions on a two-year date spine |
| 2 | Aggregation is additive | Makes the constraint linear | Averages, ratios, prices, service levels, rates per day | Ask whether the parent is a sum or a mean of its children |
| 3 | The aggregation factors divide the seasonal period | Keeps the aggregate seasonal period integral | A factor of 5 on monthly data; daily to month | Compute `m/k` at every level; a non-integer invalidates the level |
| 4 | Base forecasts are independent across levels | The projection treats them as separate estimates | The weekly forecast is the monthly divided by four | Trace a base forecast to its source; look for a division |
| 5 | The error covariance is estimable | Weighted least squares needs it | `N x T` larger than the history; a separable assumption that fails | Compare `N x T` with the number of observations; test separability |
| 6 | The temporal structure is stable across the horizon | One summing matrix serves the whole window | A calendar change, a 53-week year, a fiscal migration mid-window | Diff the calendar table across the horizon |
| 7 | The bottom bucket matches the decision cadence | Otherwise the hierarchy reconciles detail nobody uses | A daily bottom level for a weekly decision | List the decisions and their cadences; compare with the ladder |
| 8 | Reconciliation is worth its cost | Justifies the exercise | The reconciled forecast loses to its base at the decision bucket | Backtest reconciled against base at that bucket |
| 9 | A separable covariance is adequate | Makes the system estimable | Strong entity-by-time interaction, such as a promotion calendar | Fit a structured alternative and compare out of sample |
| 10 | Calendar effects are removed or modelled | Otherwise a calendar artefact is reconciled as a disagreement | Trading days, leap years, moving holidays ignored | Residual seasonality and calendar tests on the base forecasts |
| 11 | The top level carries information the bottom does not | The whole accuracy argument rests on it | The annual forecast is a smoothed version of the same data | Compare the top-level base forecast with the bottom-level sum over history |
| 12 | The bridge between calendars is documented and owned | Otherwise the bridge is an undocumented model | A day-weighted split nobody has written down | Ask three people to reproduce the monthly number from the weekly one |

Two limits of the framing deserve stating plainly.

**Temporal coherence is not the same as a good plan.** A coherent hierarchy distributes whatever the projection decided into every bucket, and if the projection decided wrongly, coherence makes the wrongness harder to challenge. This is the warning from topic 02, and it is sharper here because the temporal projection touches every planning horizon at once.

**The literature's gains are measured on competition data and a handful of case studies.** The M3 and M5 series are not a company's plan, and the reported percentages do not transfer. Treat the section 9.2 table as evidence about the *direction and gradient* of the effect, not as a forecast of what a specific business will obtain.

## 12. Exercises

Ten exercises. Exercises 1 to 5 are calculation and application, 6 and 7 are critique, and 8 to 10 are design. Each states a deliverable, acceptance criteria, and an estimated time.

### Exercise 1 — Build the temporal summing and constraint matrices

A series is observed monthly for 24 months, January 2025 to December 2026. Build a temporal hierarchy with four levels: month, quarter, half-year, and year.

**Deliverable:** the dimensions of `S_te` and `C_te`, the number of independent constraints, the dimension of the coherent subspace, the row sums of `S_te` for one month row, one quarter row, one half-year row, and the year row, and the product `C_te S_te`.

**Acceptance criteria:** `S_te` is 38 x 24; `C_te` is 14 x 38; the node count is derived and not asserted; the row sums are 1, 3, 6, and 24; the product is the zero matrix exactly; the fact that 14 equals 38 minus 24 is explained as the rank-nullity relation for a tree.

**Estimated time:** 30 minutes.

### Exercise 2 — March 2026 by hand

March 2026 has 31 days and begins on a Sunday. The six ISO weeks overlapping March have full-week base forecasts of 232, 238, 231, 245, 252, and 224 eaches. The monthly base forecast is 1,000.

**Deliverable:** the number of March days in each ISO week, the day-weighted March total, the coherence gap against the monthly base in units and as a percentage, and one sentence on why the total is not equal to the calendar month.

**Acceptance criteria:** the day counts are 1, 7, 7, 7, 7, and 2, summing to 31; the allocated total is 1,063.143 to three decimals; the gap is -63.143 eaches, or -6.31%; the sentence identifies non-nesting rather than forecast error as the cause.

**Estimated time:** 25 minutes.

### Exercise 3 — Prove the infeasibility

Using March 2026 as the example, show that no partition of the month into whole ISO weeks exists.

**Deliverable:** a short argument of no more than 200 words, with the arithmetic, plus a statement of the three resolutions available to a planner.

**Acceptance criteria:** the argument uses `31 mod 7 = 3` and the fact that any union of whole seven-day weeks has a length that is a multiple of seven; the three resolutions are (a) allocate across the boundary with a stated key, (b) drop the week level and reconcile month against day, and (c) adopt a calendar in which weeks nest; no resolution is presented as costless.

**Estimated time:** 20 minutes.

### Exercise 4 — Three-node temporal projection under two weightings

A four-week period `P` has base forecast 1,000; its two halves have 420 and 560. Reconcile with `W = diag(400, 900, 1600)` and then with `W = diag(1600, 400, 900)`.

**Deliverable:** both reconciled vectors, the movement of each node in units, and the difference in the period total between the two weightings.

**Acceptance criteria:** the first vector is 997.241, 426.207, 571.034; the second is 988.966, 422.759, 566.207; each sums correctly to two decimals; the period total differs by 8.276 eaches, stated as 0.83% of the base; the write-up says which weight expresses which belief about reliability.

**Estimated time:** 25 minutes.

### Exercise 5 — Size the cross-temporal system

A company has 120 SKUs, 8 families, 2 divisions, and one total. Its temporal ladder is 36 months, 12 quarters, and 3 years.

**Deliverable:** the number of entity nodes, temporal nodes, bottom cells, cross-temporal nodes, independent constraints, and the ratios against the entity-only system.

**Acceptance criteria:** 131 entity nodes, 51 temporal nodes, 6,681 cross-temporal nodes, 4,320 bottom cells, 2,361 independent constraints; the temporal and entity constraint rows are derived separately before being combined; the ratio against the entity-only 11 rows is stated and explained.

**Estimated time:** 30 minutes.

### Exercise 6 — Critique the vendor claim

A vendor slide reads: "Temporal reconciliation improves forecast accuracy by 12% across all horizons and all planning levels."

**Deliverable:** a critique of no more than 350 words naming at least four defects, and a rewritten claim that would be defensible.

**Acceptance criteria:** the critique identifies the collapse of the horizon dimension, the collapse of the level dimension, the absence of a base-forecast comparison, and the absence of a stated calendar; the rewrite names a horizon, a level, a base, and a calendar; no sentence asserts that temporal reconciliation never helps.

**Estimated time:** 30 minutes.

### Exercise 7 — Separate a calendar artefact from a disagreement

A monthly series shows gaps between the monthly base forecast and the sum of its weekly base forecasts of -6.3%, -1.2%, +0.4%, -5.8%, +0.1%, and -6.0% over six consecutive months. Each month begins on a different weekday.

**Deliverable:** a diagnosis with a mechanism, the additional data you would need, and the test that would distinguish the two explanations.

**Acceptance criteria:** the diagnosis notes that the large gaps cluster on months with five occurrences of a high-volume weekday; the additional data is weekday counts per month, not more history; the test compares the gap against weekday composition across at least twelve months; the write-up does not conclude that the weekly model is biased without ruling out the calendar.

**Estimated time:** 30 minutes.

### Exercise 8 — Choose the bucket that must be coherent

A food manufacturer runs daily replenishment, a weekly production schedule frozen for two weeks, a monthly S&OP cycle, and a quarterly financial forecast. It currently reconciles all four to each other and the numbers conflict.

**Deliverable:** a one-page recommendation naming the coherent bucket for each of the four processes, the bridge rule between adjacent buckets, the owner of each bridge, and the failure the recommendation risks.

**Acceptance criteria:** each bucket is justified by a commitment, not by a report; at least one process is explicitly left incoherent with another and the bridge is stated; the bridge rules are executable arithmetic rather than "align with"; the risk statement names the process most likely to override the rule.

**Estimated time:** 40 minutes.

### Exercise 9 — Write the calendar specification

For a company planning on a 4-5-4 retail calendar but reporting to a parent on Gregorian months, write the calendar specification for the temporal hierarchy.

**Deliverable:** one page stating the bottom bucket, the ladder and its factors, the calendar in which the ladder nests, the bridge to the Gregorian month, the leap-year and 53-week handling, and five assertions a data pipeline could run.

**Acceptance criteria:** the ladder respects the divisor rule; the bridge states an allocation key and its measured effect in percentage terms; the 53-week handling chooses restatement, exclusion, or weekly averaging and says which; every assertion is executable against a date spine.

**Estimated time:** 40 minutes.

### Exercise 10 — Audit a real dataset for nesting

Take any time-stamped dataset you can inspect that has at least two levels of time labelling, such as a weekly and a monthly extract of the same measure.

**Deliverable:** a one-page audit stating whether the two levels nest, the evidence, the size of the discrepancy, and the smallest change that would make them nest.

**Acceptance criteria:** nesting is tested by summing the finer level within each coarse bucket and comparing, not by inspecting labels; the discrepancy is reported in units and as a percentage; at least one period is examined in daily detail; the recommendation distinguishes a calendar fix from an allocation fix.

**Estimated time:** 45 minutes.

## 13. Self-check answers

Answers to all ten exercises, with the reasoning. Design exercises get a rubric rather than a single answer.

### Answer 1

Twenty-four months, eight quarters, four half-years, and two years give 38 nodes. The bottom level is the 24 months, so `S_te` is 38 x 24 and `C_te` is 14 x 38. The fourteen constraints are eight quarter rows, four half-year rows, and two year rows, all independent because each corresponds to a distinct parent in a strict tree. The coherent subspace has dimension 24, one per month.

Row sums of `S_te`: a month row has 1, because a month contains only itself; a quarter row has 3; a half-year row has 6; the year row has 24. The product `C_te S_te` is 14 x 24 and every entry is zero, because each constraint row subtracts exactly the columns its parent row sums.

The identity 14 = 38 - 24 is rank-nullity for a tree: the number of edges equals the number of non-root nodes, and each edge is one constraint. A team that computes 14 by counting rows and 14 by subtracting has a check on its own structure.

The common error is to write both `year = sum of 2 half-years` and `year = sum of 4 quarters`. Those two rows are not independent given the half-year rows, and including both makes `C_te C_te'` singular. One constraint per parent, no more.

### Answer 2

March 2026 begins on Sunday 1 March and has 31 days. The ISO weeks overlapping it contain 1, 7, 7, 7, 7, and 2 March days respectively; 1 + 7 + 7 + 7 + 7 + 2 = 31.

Day-weighted allocation: 232 x (1/7) = 33.143; 238, 231, 245, and 252 each at 7/7; 224 x (2/7) = 64.000. The total is 1,063.143.

The gap against the monthly base of 1,000 is -63.143 eaches, or -6.31% of 1,000. The total is not equal to the calendar month because six ISO weeks overlap March and two of them extend beyond it; the allocation assigns fractions of two weekly forecasts to March, and those fractions are an assumption rather than an observation.

### Answer 3

Any union of whole seven-day weeks has a length that is a multiple of seven. March 2026 has 31 days, and 31 = 4 x 7 + 3, so 31 mod 7 = 3. Therefore no set of whole ISO weeks has total length 31: the nearest possibilities are 28 days and 35 days. Since a partition must cover all 31 days exactly once, no partition into whole ISO weeks exists. The same argument applies to every month whose length is not a multiple of seven, which is every month except a 28-day February.

The three resolutions. First, **allocate across the boundary**: assign a fraction of each straddling week to the month using a stated key, so the month is coherent with a *transformed* week series rather than with the week series itself. Second, **drop the week level**: reconcile month against day, which nests, and treat the week as an operational aggregation of the day that is not part of the hierarchy. Third, **adopt a nesting calendar**: use a 4-4-5 or 52/53-week fiscal calendar in which a period is a whole number of weeks, at the price of a fiscal period that is not a calendar month.

None is costless. Resolution 1 embeds an assumption whose size section 6.6 measured at 1.9% of the monthly base. Resolution 2 loses the weekly level from the reconciliation. Resolution 3 requires a second calendar for finance.

### Answer 4

First weighting, `W = diag(400, 900, 1600)`, gap d = 1,000 - 420 - 560 = +20, sum of weights = 2,900:

- P = 1,000 - 20 x 400/2,900 = 1,000 - 2.759 = 997.241
- H1 = 420 + 20 x 900/2,900 = 420 + 6.207 = 426.207
- H2 = 560 + 20 x 1,600/2,900 = 560 + 11.034 = 571.034
- check: 426.207 + 571.034 = 997.241

Second weighting, `W = diag(1600, 400, 900)`:

- P = 1,000 - 20 x 1,600/2,900 = 1,000 - 11.034 = 988.966
- H1 = 420 + 20 x 400/2,900 = 420 + 2.759 = 422.759
- H2 = 560 + 20 x 900/2,900 = 560 + 6.207 = 566.207
- check: 422.759 + 566.207 = 988.966

The period total moves from 997.241 to 988.966, a difference of 8.276 eaches, or 0.83% of the 1,000 base.

The first weighting says the period forecast is the most reliable of the three, because its variance is the smallest. The second says it is the least reliable. In a temporal hierarchy the first claim is the more usual: an aggregate error variance grows with the aggregation factor in absolute units, but the aggregate also averages more, so whether its *relative* reliability is higher depends on the autocorrelation of the underlying process. This is why structural scaling weights by the number of bottom buckets rather than by raw variance, and why the choice must be stated.

### Answer 5

Entity structure: 120 + 8 + 2 + 1 = 131 nodes, of which 120 are bottom, so there are 11 independent entity constraints per bucket. Temporal structure: 36 + 12 + 3 = 51 nodes, of which 36 are bottom, so there are 15 independent temporal constraints per series.

Cross-temporal: nodes = 131 x 51 = 6,681; bottom cells = 120 x 36 = 4,320; independent constraints = 6,681 - 4,320 = 2,361. The stacked count is 120 x 15 = 1,800 temporal rows at the leaves plus 11 x 51 = 561 entity rows, which totals 2,361 and matches the independent count exactly. That agreement is a check, and it holds because with a strict tree the temporal constraints at aggregate entity nodes are implied by the leaf temporal constraints together with the entity constraints at each bucket.

The ratio against the entity-only 11 rows is 2,361 / 11 = 215. A naive full-stack count using 131 temporal rows would be 1,965 + 561 = 2,526, which overstates by 165 and would produce a singular matrix if inverted without reduction.

### Answer 6

Four defects at minimum. **The horizon dimension is collapsed:** a 12% figure across "all horizons" is an average over a range in which the sign may change, and the studies in section 9.2 report gains near zero or negative in the first week and much larger at the annual level. **The level dimension is collapsed:** a single number across all planning levels hides the gradient the literature consistently reports, with the largest gains at the coarsest level. **There is no base comparison:** "improves accuracy" against what? If the comparison is against a different set of base forecasts, the number measures the base models. **No calendar is stated:** on an ISO calendar the weekly level does not nest in the month, so the hierarchy the claim depends on may not exist.

A defensible rewrite: "In a backtest on 36 months of history with a Gregorian-calendar hierarchy of month, quarter, and year, temporal reconciliation with structural scaling reduced mean absolute scaled error at the quarter level by 8.1% and at the month level by 1.4%, relative to the same base forecasts unreconciled, with common origins and a 12-month horizon. The week level was excluded because it does not nest in the calendar month."

That rewrite states the calendar, the levels, the base, the metric, and the horizon, and it does not claim a result the evidence would not support. The percentages are placeholders for whatever the candidate's own backtest produced.

### Answer 7

The pattern — large negative gaps at -6.3%, -5.8%, and -6.0%, and small gaps at -1.2%, +0.4%, and +0.1% — alternates with the calendar, not with the business. The mechanism is weekday composition: a month with five occurrences of a high-volume weekday carries more volume than weekly models expect if they were fitted on months with four, and the monthly model, which has seen the calendar in its own seasonal terms, does not reproduce the same effect.

The additional data needed is the weekday count per month, not more history. With counts for each of the seven weekdays across at least twelve months, regress the gap on the count of each weekday and check whether the coefficients on the high-volume days are large and stable.

The distinguishing test is whether the gap is explained by weekday composition across twelve or more months. If the fit is good and the residuals are small, the gap is a calendar artefact and the fix is a trading-day adjustment, not a reconciliation. If the gap persists after weekday adjustment with a consistent sign, it is a disagreement between models and reconciliation is the right response.

The error to avoid is concluding that the weekly model is biased. It may be. But a gap that moves with the calendar is evidence about the calendar, and a bias claim requires the calendar to be ruled out first.

### Answer 8 — rubric

A strong answer names a commitment for each process rather than a report. Daily replenishment is coherent at the day, because orders are placed per day, but the week is the consumption cap. Weekly scheduling is coherent at the week within the frozen horizon, because the schedule is a promise to the plant. Monthly S&OP is coherent at the month within the quarter, because that is where the functions sign. The quarterly financial forecast is coherent at the quarter, because that is what is reported and audited.

The bridges: day to week is a consumption rule, not a sum, and the weekly forecast acts as a cap on daily consumption. Week to month is an allocation with a stated key inside the month, restated when the frozen horizon rolls. Month to quarter is a sum, and the quarterly figure is the authoritative one for finance.

The process most likely to override the rule is monthly S&OP, because it has the most senior participants and the strongest incentive to make the quarter add up by adjusting the month. A strong answer names that risk and proposes that the monthly adjustment be logged as a bridge variance rather than absorbed silently.

A weak answer reconciles all four to each other, or declares one bucket authoritative for everything, which is the current failure restated as a recommendation.

### Answer 9 — rubric

A strong specification reads like a data contract. The bottom bucket is the day, because the replenishment decision is daily. The ladder is day, week, 4-5-4 period, quarter, year, with factors 7, 4 or 5, 13, and 4. The nesting calendar is 4-5-4, in which every level nests exactly. The bridge to the Gregorian month is a day-weighted allocation of the fiscal period across calendar months, with the measured effect stated as a percentage of the monthly total — section 6.6's 1.9% is a usable calibration. The 53-week year is handled by restating the following year, or by exclusion, stated explicitly. The five assertions are executable: every date appears once; each fiscal period contains a whole number of weeks; the days in each period sum to 28 or 35; every period has one parent quarter; the annual total equals the sum of the periods.

A weak specification states the ladder and the calendar and stops, leaving the bridge to whoever builds the pipeline.

### Answer 10 — rubric

A strong audit tests nesting by arithmetic: sum the finer level within each coarse bucket and compare with the coarse value, reporting the discrepancy in units and as a percentage, and naming the buckets where it is largest. It examines at least one period in daily detail, because the aggregation rule is only visible at the finest grain. It distinguishes a calendar fix — changing the labels so the levels nest — from an allocation fix — keeping the labels and adding a stated key.

A weak audit inspects the label columns, observes that the weekly labels look consistent with the monthly ones, and concludes that they nest. Labels are consistent by construction in most systems. The arithmetic is the only test.

## 14. Interview questions

Twelve questions, each with what a strong answer contains and what a weak answer reveals.

### Q1. "What is a temporal hierarchy?"

**Strong answer:** the set of non-overlapping aggregations of one series to progressively coarser buckets, treated as a tree, with each aggregate equal to the sum of its children, and reconciled by the same projection algebra as an entity hierarchy. It notes that the bottom level is a choice and that the ladder must respect the divisor rule for the seasonal period.
**Weak answer:** "the time dimension", which suggests the candidate has only seen time as a column in a star schema.

### Q2. "Why is temporal reconciliation harder than reconciling products and locations?"

**Strong answer:** three reasons. The constraint matrix scales with the length of the history rather than the width of the taxonomy, so it is hundreds of rows per series rather than tens. The calendar may make the constraint system infeasible, because a month does not contain a whole number of ISO weeks. And the leaves are the same series cut finely, so the error covariance is structured rather than free, which changes how the weights behave.
**Weak answer:** "it is the same thing with time instead of products", which is the answer of someone who has not tried it.

### Q3. "Does the month equal the sum of its weeks?"

**Strong answer:** not in a Gregorian calendar, and the candidate shows why — 31 is not a multiple of seven, so no union of whole seven-day weeks covers a 31-day month. It then names the three resolutions: allocate across the boundary with a stated key, drop the week level, or adopt a 4-4-5 style calendar in which weeks nest.
**Weak answer:** "yes, if you are careful with the boundary", which is the answer that produces the undocumented bridge.

### Q4. "What does cross-temporal coherence mean?"

**Strong answer:** every entity constraint holds at every time bucket and every temporal constraint holds for every entity, simultaneously. The coherent subspace has dimension `N_b x T_b`, one per bottom cell. Reconciling in one direction and then the other does not in general produce it, because the second projection breaks the first.
**Weak answer:** "both hierarchies are reconciled", which does not say whether the result satisfies both.

### Q5. "How large is a cross-temporal constraint matrix?"

**Strong answer:** the candidate computes it. For 240 bottom entities and 36 bottom buckets with 16 entity constraints and 15 temporal constraints, the lattice has 13,056 nodes, the bottom cells number 8,640, and there are 4,416 independent constraints — roughly three hundred times the 16 entity constraints alone. It also mentions that naive stacking produces redundant rows.
**Weak answer:** "large", with no structure.

### Q6. "Why does everyone use a Kronecker covariance?"

**Strong answer:** because the full covariance over `N x T` cells has `(N x T)^2` entries, over 170 million for a modest hierarchy, while a separable form `Σ_cs ⊗ Σ_te` has `N^2 + T^2`, about 68,000. It states the assumption the separable form makes — that entity and time errors factor with no interaction — and notes that the assumption should be tested rather than assumed.
**Weak answer:** "for speed", with no statement of what is being assumed away.

### Q7. "Does temporal reconciliation improve short-horizon forecasts?"

**Strong answer:** sometimes, and the general finding is the opposite of the popular claim. In the published studies the gains are usually largest at the coarsest level and smallest at the finest, and one large study reports slight deterioration in the first week. The short-horizon gain is real where the coarse forecast carries information the fine one does not, as in the accident and emergency case; it is not a general law.
**Weak answer:** "yes, because long-horizon information informs short-horizon forecasts", recited without a horizon-by-level result.

### Q8. "What is the difference between benchmarking and reconciliation?"

**Strong answer:** benchmarking treats one level as true and adjusts the others to match it; reconciliation treats every level as fallible and finds a weighted compromise. Statistical offices benchmark because the annual figure is audited. A demand plan has no audited level, so reconciliation is the right frame and benchmarking is a special case with infinite weight on one node.
**Weak answer:** both are "making the numbers add up", which misses the entire weighting question.

### Q9. "A planner says the weekly plan must add up to the monthly plan. What do you ask?"

**Strong answer:** which month, which weeks, and which calendar. Then whether the weekly plan is a decomposition of the monthly commitment or an independent forecast, because those are two different objects and only the first is a constraint. The candidate may propose a 4-4-5 calendar or a documented allocation key, and will say what the key costs.
**Weak answer:** "we will reconcile them", which promises a projection onto a subspace that may not exist.

### Q10. "How would you decide whether to reconcile in time at all?"

**Strong answer:** start from the decision. If one bucket carries the commitment and the others are views, reconcile that bucket against the entity hierarchy and bridge the rest. Reconcile in time only where two buckets both carry commitments, and then measure whether the reconciled forecast beats its own base at the bucket that matters.
**Weak answer:** "always reconcile, coherence is good", which is a preference rather than a decision.

### Q11. "What is a 4-4-5 calendar and why would a forecaster care?"

**Strong answer:** a retail fiscal calendar in which each 13-week quarter is split into months of 4, 4, and 5 weeks, so every month contains a whole number of weeks and begins on the same weekday. The forecaster cares because it makes the week-to-month relation exact, which makes a temporal hierarchy feasible, and because it removes the weekday-composition artefact from month-on-month comparisons. The cost is a fiscal month that is not a calendar month, requiring a second calendar for finance.
**Weak answer:** "it makes all months equal", which is false: the months are 28 or 35 days.

### Q12. "What would make you abandon a temporal hierarchy?"

**Strong answer:** a backtest showing the reconciled forecast losing to its base at the decision bucket; a calendar in which the levels cannot be made to nest without a bridge whose measured effect exceeds the coherence gap; a bottom level so sparse that the aggregate carries all the information and the projection adds nothing; or a computational cost that exceeds the value of the coherence at the scale the business operates.
**Weak answer:** no condition is named, which means the method is a habit.

## 15. Key terms

- **Temporal hierarchy** — the non-overlapping aggregations of one series to coarser buckets, treated as a tree with summing constraints.
- **Temporal summing matrix `S_te`** — the `n_te x T_b` 0/1 matrix mapping bottom buckets to every temporal node.
- **Temporal constraint matrix `C_te`** — the `(n_te - T_b) x n_te` matrix of aggregation constraints, with `C_te y = 0` for a coherent vector.
- **Aggregation factor** — the number of bottom buckets that sum to one aggregate at a level.
- **Divisor rule** — only factors of the seasonal period keep the aggregate seasonal period integral.
- **Coherence gap (temporal)** — the difference between an aggregate base forecast and the sum of its children in time.
- **Cross-temporal hierarchy** — the lattice of every entity at every time bucket.
- **Cross-temporal coherence** — every entity constraint at every bucket and every temporal constraint for every entity, simultaneously.
- **Sequential reconciliation** — reconciling cross-sectionally then temporally, or the reverse, which does not in general give cross-temporal coherence.
- **Iterative reconciliation** — alternating the two projections to convergence.
- **Optimal cross-temporal combination** — a single projection onto the intersection of both constraint families.
- **Kronecker product** — the block matrix `A ⊗ B`, the structure of the cross-temporal summing matrix.
- **Separable covariance** — `Σ_cs ⊗ Σ_te`, which makes the cross-temporal system estimable.
- **Temporal disaggregation** — distributing a known aggregate to finer buckets under a stated rule.
- **Benchmarking** — adjusting a fine series so its sums match a known coarse total.
- **Denton method** — least-squares benchmarking minimising deviation from a proxy under the aggregation constraint.
- **Chow–Lin method** — regression-based temporal disaggregation with an indicator series.
- **Allocation key** — the rule that splits an aggregate across buckets.
- **Day-weighted bridge** — allocating a month across weeks in proportion to overlapping days.
- **4-4-5 calendar** — a 52/53-week retail calendar whose months are 4, 4, and 5 weeks within each 13-week quarter; the NRF calendar is 4-5-4.
- **52/53-week year** — a fiscal year of 364 or 371 days ending on a fixed weekday.
- **ISO week date** — `YYYY-Www-D`, Monday-start, whose weeks do not nest in Gregorian months.
- **Weekday composition** — the count of each weekday in a period, the mechanism behind month-length artefacts.
- **Trading-day adjustment** — correction for the number of active days in a period.
- **Leap-year regressor** — a calendar regressor absorbing 29 February, typically `0.75` and `-0.25` in a four-year cycle.
- **Seasonal adjustment** — removal of seasonal and calendar components from a series, which does not make a hierarchy nest.
- **Horizon-specific gain** — an accuracy improvement at a stated horizon, the only honest unit for an accuracy claim.
- **Structural scaling (WLS-S)** — weights proportional to the number of bottom buckets in each aggregate, needing no error estimates.
- **MAPA** — multiple aggregation prediction algorithm, which averages forecasts across aggregation levels rather than projecting them.
- **Forecast bucket versus consumption bucket** — the bucket a forecast is expressed in against the bucket in which orders consume it.

## 16. Sources

Grouped by type. Every entry carries a grade: A for peer-reviewed or audited, B for regulatory or official standard, C for company case or investor claim, D for vendor marketing or vendor documentation, E for press or anecdote. Digital object identifiers are given only where the search returned them and they matched the cited title.

### Peer-reviewed literature (Grade A)

| Source | What it supports | Grade |
|---|---|---|
| Athanasopoulos, Hyndman, Kourentzes, Petropoulos, "Forecasting with temporal hierarchies", *European Journal of Operational Research*, 262(1), 60–74, 2017. DOI: 10.1016/j.ejor.2017.02.046 | The temporal hierarchy construction, the three weight schemes, the M3 results, and the UK A&E case | A |
| Kourentzes, Petropoulos, Trapero, work on multiple temporal aggregation published in the *International Journal of Forecasting*, 2014 | MAPA as the averaging predecessor of temporal reconciliation | A |
| Di Fonzo, Girolimetto, "Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives", *International Journal of Forecasting*, 39(1), 39–57, 2023. DOI: 10.1016/j.ijforecast.2021.08.004 | The closed-form optimal cross-temporal estimator and the iterative heuristic | A |
| Girolimetto, Di Fonzo, "Cross-temporal forecast reconciliation: insights on sequential, iterative, and optimal approaches", *Statistical Methods & Applications*, 2025. DOI: 10.1007/s10260-025-00822-z | Conditions under which iteration reaches the optimal solution, and the separable Kronecker covariance | A |
| Athanasopoulos, Hyndman, Kourentzes, Panagiotelis, "Forecast reconciliation: A review", *International Journal of Forecasting*, 40, 430–456, 2024. DOI: 10.1016/j.ijforecast.2023.10.010 | The survey framing of temporal and cross-temporal reconciliation | A |
| Spiliotis, Petropoulos, Assimakopoulos, work on improving the forecasting performance of temporal hierarchies, *PLOS ONE*, 2019. DOI: 10.1371/journal.pone.0223422 | The 1.8–5.4% range of improvement and its decline as frequency rises | A |
| Nystrup and co-authors, work on temporal hierarchies with autocorrelated errors in load forecasting, *European Journal of Operational Research*, 2019 | The 12–21% daily against 0–8% hourly gradient | A |
| Petropoulos, Hollyman, Spiliotis, "Scalable forecast reconciliation through (un)guided sub-hierarchies", *Journal of the Operational Research Society*, 2025 | The M5 result that gains are slight or negative at horizons 1–7 and larger beyond two weeks | A |
| Lipiecki, Bilińska, Kourentzes, Weron, work on temporal hierarchy forecasting for day-ahead electricity prices, *International Journal of Forecasting*, 2026 | The 5.5% hourly and 13.4% baseload results, with gains increasing at higher aggregation | A |
| Chow, Lin, "Best linear unbiased interpolation, distribution, and extrapolation of time series by related series", *Review of Economics and Statistics*, 1971 | The regression-based temporal disaggregation method | A |
| Denton, "Adjustment of monthly or quarterly series to annual totals: an approach based on quadratic minimization", *Journal of the American Statistical Association*, 1971 | Least-squares benchmarking under an aggregation constraint | A |
| Cleveland, Devlin, "Calendar effects in monthly time series: modeling and adjustment", *Journal of the American Statistical Association*, 1982. DOI: 10.1080/01621459.1982.10477841 | Length-of-month and trading-day adjustment | A |
| Chen, work on comparing methods for temporal distribution and interpolation at the national accounts, U.S. Bureau of Economic Analysis working paper, 2007 | The comparative performance of Denton variants and Chow–Lin in official statistics | A |

### Standards and official bodies (Grade B)

| Source | What it supports | Grade |
|---|---|---|
| ISO 8601-1:2019, *Date and time — Representations for information interchange — Part 1: Basic rules* | Week dates, Monday as first weekday, the first-Thursday rule | B |
| ISO 8601-2:2019, *Part 2: Extensions* | Reduced precision, intervals, durations | B |
| RFC 3339, *Date and Time on the Internet: Timestamps* | Mandatory offset on every timestamp | B |
| IANA time zone database and its theory documentation | Zone identifiers, daylight saving transitions | B |
| 26 CFR §1.441-2, *Election of taxable year consisting of 52-53 weeks* | The 52/53-week year and its fixed weekday | B |
| IRS Publication 538, *Accounting Periods and Methods* | The 52-53-week tax year in practice | B |
| IAS 34, *Interim Financial Reporting* | Comparability of interim periods | B |
| IFRS 8, *Operating Segments* | The reconciliation of segment totals, the precedent for a documented bridge | B |
| U.S. Census Bureau, X-13ARIMA-SEATS documentation | Trading-day, moving-holiday, and leap-year regressors | B |
| Eurostat, *ESS Guidelines on Seasonal Adjustment* | The European practice standard for calendar adjustment | B |
| Office for National Statistics, methodological note on leap year adjustments | The 3.6% February effect and the 28.25-day pre-adjustment | B |

### Industry conventions (Grade C)

| Source | What it supports | Grade |
|---|---|---|
| National Retail Federation, 4-5-4 calendar | The retail calendar convention, the rule for a 53rd week, and the list of 53-week years | C |

### Vendor and practitioner material (Grade D and E)

| Source | What it supports | Grade |
|---|---|---|
| Forecast Pro documentation, "Adjusting for 4 vs 5 Week Periods" | The weeks-helper variable method and the reported adjusted R-squared improvement from 0.50 to 0.91; used as evidence of what tools offer, not of what works | D |
| Oracle Supply Chain and Manufacturing Cloud documentation, forecast consumption | The distinction between forecast bucket and consumption bucket, and the consumption direction setting | D |
| SQLBI, "Week-Based Time Intelligence in DAX" | The reporting-layer failure of standard time intelligence on 4-4-5 calendars | D |
| Anaplan community discussion of grouping 4-4-5 sales into calendar months | Practitioner methods for the fringe-day delta and the annual-total-preserving transfer | E |
| R packages `thief` and `FoReco` on the Comprehensive R Archive Network (CRAN) | Reference implementations of temporal hierarchy forecasting and cross-temporal reconciliation | D |

**Note on citations and grades.** Several figures in section 9.2 are reported as the papers state them and were taken from published abstracts and summaries rather than from a full reading of each paper; treat them as the authors' reported results, not as independently verified measurements. Four entries give the topic and venue rather than an exact title, because only the topic and venue were verified. No Grade E material is used to support a quantitative claim.

## 17. Links

Within this module:

- Module map and topic status: [`_index.md`](_index.md)
- Prerequisite, hierarchy structures: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- Prerequisite, aggregation consistency and the four classical methods: [`02_aggregation-consistency.md`](02_aggregation-consistency.md)
- Next topic, minimum-trace reconciliation in detail: `03_mint-reconciliation.md` (planned)
- Later topic, intermittent demand classification and methods: `05_intermittent-classification.md` (planned)

Within the part:

- Part index: [`../_index.md`](../_index.md)
- Prerequisite module, hierarchy and aggregation: [`../07_time_series_forecasting/05_hierarchy-aggregation.md`](../07_time_series_forecasting/05_hierarchy-aggregation.md)
- Backtesting design, which the horizon-by-level table depends on: [`../07_time_series_forecasting/08_rolling-origin-backtesting.md`](../07_time_series_forecasting/08_rolling-origin-backtesting.md)
- Regression with lags and Fourier terms, the base models used at each temporal level: [`../07_time_series_forecasting/04_regression-lags-fourier.md`](../07_time_series_forecasting/04_regression-lags-fourier.md)

Elsewhere in the course:

- Calendars, weeks, and time zones: [`../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Lead-time definitions, the other time quantity a planner needs: [`../../01_foundations/01A_foundations_units_time_assumptions/05_lead-time-definitions.md`](../../01_foundations/01A_foundations_units_time_assumptions/05_lead-time-definitions.md)
- Planning processes and cadences: [`../../01_foundations/01_supply_chain_foundations/04_planning-processes.md`](../../01_foundations/01_supply_chain_foundations/04_planning-processes.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module 01A survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Module survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)

> **Note:** Topic 03 of this module is listed in the index as `03_mint-reconciliation.md` and has not been written at the time of this file. This file uses the minimum-trace material it will contain only at the level of the projection formula in section 5, which topic 02 already establishes.
