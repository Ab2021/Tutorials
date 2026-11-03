# Probabilistic Forecasts: What a Distribution Is For

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topics 10–11; Module 04 topic 02; Module 07 topics 01 and 09; Module 08 topics 05–06
> Version 2026.5

---

## 1. Why this topic exists

A planner asks a demand scientist how many units of a part will be needed over the next three weeks.
The scientist answers "about 60". The planner orders 60. The part stocks out in eleven of the next
twenty cycles, and everybody concludes that the forecast was wrong.

The forecast was not wrong. It was a mean. The planner needed an order quantity that would leave stock
on the shelf in nineteen cycles out of twenty, and nobody had named which property of the future they
were asking for. One number was produced, a different one was consumed, and the gap between them stayed
invisible because neither party had a word for it.

This is a **representation failure**, not a modelling failure. A point forecast compresses a distribution
into a single number, and the compression is free only if the decision that consumes it reads exactly
the functional that was retained. Most decisions do not. A cycle service level reads a quantile. A
fill-rate target reads an expectation over the shortfall. A cost-based order reads a critical fractile.
A capacity reservation reads a high quantile, because capacity is lumpy and slow to change. A budget
reads a quantile of a cost distribution, or a mean plus a policy buffer. Five decisions, five
functionals, one distribution underneath.

**Who is blocked.** The demand planner who cannot answer "how confident are you?" with anything but a
number. The inventory manager who sets a reorder point at the forecast plus a fixed percentage and cannot
say what service level that produces. The finance partner who must accrue for forecast risk and has no
distribution to accrue against. The data scientist whose planning system accepts one number per item per week.

**Why a distribution, and what it costs.** A distribution is the smallest object that answers all of
those questions at once, because each is a functional of it. The cost is that a distribution is a much
stronger claim than a number. A point forecast asserts "this is roughly the centre". A distribution
asserts a probability for every event, including the events nobody checked: the 99th percentile, the
probability of zero, the probability of more than twice the mean. Those assertions are testable, and
they will be tested. Quoting a number is cheap and unfalsifiable; quoting a distribution is expensive
and falsifiable. An organisation that has not decided to be falsifiable will keep producing numbers and
keep being surprised.

**The historical inheritance.** Safety-stock formulas built on a normal quantile have been standard
practice since the mid-twentieth century, and the z-value table is probably the oldest interface between
a probability and an order quantity. That made quantile reasoning routine, and it hid the quantile inside
a multiplier called a "safety factor", which detached the number from the probability it encoded. A z of
1.65 became a habit rather than a statement that the business had chosen a 95% cycle service level.
Modern quantile forecasting is partly a recovery of what the field already knew and partly an addition:
estimating the quantile without assuming a family, and testing whether the stated probability is true.

**What this topic is not.** It is not a catalogue of distributions; topic 06 of this module gives one
generative route and Module 01A topic 11 gives the families supply chains use. It is not a treatment of
distribution-free intervals (topic 08), of scoring (topic 09), or of how to choose a service level
(topic 10). It asks a prior question: what is a distribution for, what does it commit you to, and how do
you know whether the one you have is any good?

> **Note:** Nothing here claims that a predictive distribution is knowable. It is a model output, built
> from a sample and conditioned on assumptions. The discipline is not to produce the true distribution.
> It is to produce one whose claims are stated, whose calibration has been tested, and whose exclusions
> are written down.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** four supply chain decisions to the functional of the predictive distribution each reads, and state the horizon and probability involved.
2. **Distinguish** a point forecast, a mean, a median, and a quantile, and show with arithmetic where they diverge on skewed or intermittent demand.
3. **Explain** what is lost at each step of the compression ladder, and which decisions each compression is safe for.
4. **Decompose** predictive uncertainty into observation noise, parameter error, structural error, and exogenous uncertainty, and say which a model's interval contains.
5. **Test** calibration rather than assume it, using coverage counts, coverage by horizon and segment, and a probability integral transform (PIT) histogram.
6. **Distinguish** a forecast distribution from a scenario set, and explain what is lost and gained by refusing to probabilise a scenario set.
7. **Diagnose** the failure of naive quantile reconciliation, demonstrate why quantiles do not add, and choose a method for a stated case.
8. **Write** a release note meeting the reporting standard: central prediction interval, horizon, calibration evidence, decision functional, exclusions.

Each outcome maps to an exercise in section 12 and to an answer or rubric in section 13.

## 3. The core mental model

Hold one picture: a distribution, and three arrows leaving it at three different places. The
distribution is the object. The arrows are decisions. Each touches the object at a different point, and
none touches it where the point forecast sits.

```text
        THE PREDICTIVE DISTRIBUTION AND THE FUNCTIONALS THAT READ IT

  density f(d)
     ^
     |                ####
     |               ######
     |              ########
     |             ##########
     |            ############
     |           ##############
     |          ################
     |         ##################
     +-----|-------|-------|-------|------|-----> d   (units, lead-time demand)
            0      40      80     100    140
                            |       |      |
    cost optimum -----------+       |      +---- Q_0.95 = 140    (CSL 0.95)
    critical fractile 0.75         |
    Q_0.75 = 80                    +----------- Q_0.90 = 100

  fill rate 0.95  ->  an EXPECTATION over the shortfall, solved at Q ~ 123 units

  three readers, three objects, one distribution:
     CSL 0.95 -> 140 units      fill 0.95 -> ~123 units      cost -> 80 units
     a 75% spread in the order quantity, from the same distribution
```

Six properties follow, and they organise the rest of the topic.

**Property 1 — A point forecast is a functional, and therefore a claim.** A mean minimises squared
error, a median minimises absolute error, a mode maximises density. Every point forecast answers a loss
function, stated or not, and the loss it answers is usually not the loss the decision incurs. Module 07
topic 09 develops the metric consequences; here the point is representational.

**Property 2 — Compression is free only when the decision reads the retained functional.** The ladder
runs from full density to a quantile grid to a single quantile to an interval to a point. Each step
discards information, and the discard is harmless if and only if the downstream decision never needed
it. A cycle service level of 0.95 needs one quantile, so a grid is more than enough and a point forecast
is not. A fill-rate target needs the tail integral, so a single quantile is not enough at any level. A
capacity plan evaluated against expected cost needs the whole density, because the cost function is
nonlinear. The safe compression is set by the decision, not by the reporting habit.

```text
  THE COMPRESSION LADDER

  full density --(quantile grid)--> quantile --(two points)--> interval --(one point)--> point
  retains         everything         one F^-1(q)   two quantiles    a mean or median
  safe when       the cost is        the decision  the decision is  the decision is
                  nonlinear          reads a level  a stated interval  linear and symmetric
```

**Property 3 — A distribution commits you to everything, including what you never checked.** Saying "the
90% interval is 20 to 100" is not weaker than a full density; it is a claim about two quantiles that
implies something about all the others, and it will be read as though the whole shape were known. If the
true distribution is bimodal, or has an atom at zero, the endpoints will be correct and the reader will
infer the wrong shape. Hence calibration testing, and hence exclusions.

**Property 4 — Predictive uncertainty has four sources, and a model's interval usually contains two.**
Observation noise, parameter error, structural error, and future exogenous uncertainty. A parametric
interval from residual variance contains observation noise and, if Bayesian or bootstrapped, some
parameter error. It almost never contains structural error, because it conditions on the model being
right, and it never contains exogenous uncertainty, because that is outside its likelihood.

**Property 5 — Calibration is a property you test, not a property you assume.** A forecast is calibrated
if, over a long run of comparable occasions, events stated at probability p happen about p of the time.
That is an empirical statement about a sample, not a theorem about a model. It is checked with a coverage
count, a PIT histogram, or a reliability diagram, on a window that was not used to fit. A model can be
well calibrated and useless: an interval of zero to infinity is perfectly calibrated. Hence sharpness,
and hence topic 09.

**Property 6 — Hierarchical coherence is a property of the joint distribution, not of the marginals.** A
set of quantiles is coherent in the everyday sense if the children sum to the parent at every level.
That is weaker than probabilistic coherence, which requires the whole joint predictive distribution to
live on the coherent subspace. Quantiles do not add: the sum of two 0.95 quantiles is not the 0.95
quantile of the sum except under perfect positive dependence. A hierarchy reconciled quantile by column
can therefore look coherent and correspond to no distribution at all.

> **Note:** The ladder is not a hierarchy of quality. A one-number forecast is the correct output for
> most production scheduling and transport planning. The failure is not using a point forecast. It is
> using one for a decision that reads a different functional, and not knowing that you have.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Predictive distribution | The probability distribution of a future quantity, conditional on information available now | Called a "confidence interval" or a "range" | Distribution over units at a stated horizon |
| Predictive density | The density function of a continuous predictive distribution | Read as the probability of a specific value | Density per unit; integrates to 1 |
| Cumulative distribution function (CDF) | `F(d) = P(D ≤ d)`, the probability that demand does not exceed `d` | Confused with the density | Dimensionless, monotone from 0 to 1 |
| Quantile | The value `F⁻¹(q)` below which the variable falls with probability `q` | Quoted without `q`, which makes it meaningless | Units, indexed by `q ∈ (0,1)` |
| Quantile forecast | A forecast of the `q`-quantile of the future variable | Treated as a point forecast plus a margin | Units, at a stated `q` and horizon |
| Prediction interval (PI) | A pair of quantiles `[F⁻¹(α/2), F⁻¹(1−α/2)]` intended to contain the outcome with probability `1−α` | Called a confidence interval, which is a statement about a parameter | Two values in units, plus nominal coverage |
| Central prediction interval | A PI with equal tail probabilities, usually 80%, 90%, or 95% nominal | Assumed to be the narrowest interval at that coverage | Units, nominal coverage, horizon |
| Highest density interval (HDI) | The shortest interval containing a stated probability mass | Assumed equal to the central interval under skew | Units; narrower than central for skewed `F` |
| Nominal coverage | The probability an interval claims, for example 0.90 | Reported as though it were achieved coverage | Proportion |
| Empirical coverage | The fraction of realised outcomes that fell inside the interval | Computed on the fitting window and called out-of-sample | Proportion, with sample size |
| Calibration | Agreement between stated probabilities and observed frequencies over repeated occasions | Asserted from the model family rather than measured | Diagnostic, reported as a table or plot |
| Probabilistic calibration | `P(D ≤ F⁻¹(q)) = q` on average across forecast origins | Confused with conditional calibration at every `q` | Diagnostic over a forecast sample |
| Marginal calibration | Agreement between the forecast distribution averaged over origins and the observed marginal | Assumed to imply probabilistic calibration | Diagnostic over a forecast sample |
| Sharpness | The concentration of the predictive distribution; narrower is sharper | Rewarded alone, producing confident nonsense | Width in units, or a spread statistic |
| Probability integral transform (PIT) | The value `F(y)` of the realised outcome under the forecast CDF | Plotted without stating the forecast origins | Dimensionless; uniform if calibrated |
| Reliability diagram | Observed frequency against forecast probability for a binary event derived from the distribution | Read without the bin counts | Plot; needs counts per bin |
| Proper scoring rule | A score whose expectation is optimised only by reporting the true predictive distribution | Applied to point forecasts without qualification | Function of forecast and outcome |
| Continuous ranked probability score (CRPS) | The integral of pinball loss over all quantile levels; a proper score in units of the variable | Compared across variables in different units | Units of the forecast |
| Pinball loss | The asymmetric absolute loss minimised by a stated quantile | Unknown outside specialist teams | Units, at level `q` |
| Critical fractile | `C_u/(C_u+C_o)`, the cost-minimising quantile | Confused with a service-level target | Dimensionless, 0 to 1 |
| Cycle service level (CSL) | The probability that a replenishment cycle ends without a stock-out | Reported as fill rate | Proportion |
| Fill rate | The fraction of demand units served from stock | Treated as a quantile | Proportion |
| Expected shortage per cycle | `E[(D−Q)⁺]`, the mean unmet demand at order-up-to level `Q` | Omitted when fill rate is computed | Units per cycle |
| Safety stock | The buffer above expected lead-time demand | Treated as a policy constant rather than a quantile offset | Units |
| Lead-time demand (LTD) | Total demand realised between placing an order and receiving it | Computed on the planning bucket instead of the lead time | Units over `L` periods |
| Observation noise | Irreducible period-to-period variation in the quantity forecast | Called "forecast error" | Variance in units squared |
| Parameter uncertainty | Error from estimating model parameters on a finite sample | Ignored in plug-in intervals | Variance in units squared |
| Structural (model) uncertainty | Error from the model family being wrong | Almost always omitted from reported intervals | Not directly estimable; bounded by backtests |
| Exogenous uncertainty | Uncertainty about future drivers outside the model: price, promotion, competitor, policy | Mixed into residual variance with no separation | Scenario-dependent |
| Scenario | An internally consistent narrative of a possible future | Called a forecast | Narrative plus quantified drivers |
| Scenario set | A collection of scenarios intended to span a range of futures | Called a probability distribution | Set, typically without weights |
| Probabilistic coherence | The joint predictive distribution lies on the aggregation subspace | Confused with marginal summing | Property of a joint distribution |
| Marginal coherence | Each quantile vector sums across the hierarchy | Assumed to imply probabilistic coherence | Property of a set of quantiles |
| Quantile crossing | Two estimated quantiles out of order, `Q̂_0.8 < Q̂_0.7` | Silently sorted by the planning system | Data-quality defect |
| Reconciliation | Adjusting forecasts so that levels of a hierarchy agree | Applied to quantiles by direct summation | Method, with a covariance assumption |
| Forecast origin | The date at which a forecast was made and the information set it used | Overwritten by later vintages | Timestamp |

Three pairs carry most of the confusion. **Prediction interval versus confidence interval:** the first is
a statement about a future random variable, the second about an unknown fixed parameter defined by
coverage over repeated samples. In supply chain work the object is almost always the first, and calling
it the second invites the reply "so there is a 95% chance the parameter is in there". **Calibration
versus accuracy:** accuracy asks how close the forecast was, calibration asks whether the probabilities
meant what they said, and a forecast can be accurate on average and badly calibrated in the tails, which
is the case that produces stock-outs. **Scenario versus distribution:** the first says "here are futures
we consider plausible", the second says "here is the probability of each". Using the first where the
second is needed produces an order quantity chosen by the loudest person in the room.

## 5. The mechanics, step by step

Six phases and twenty-four steps, from the decision to the report. The procedure is long because the
failure mode is not ignorance of a single step. It is skipping a step and not noticing, usually the step
that names the probability.

### Phase A — Fix the decision and its functional

**Step 1. Write the decision sentence.** One sentence, with an owner and a cadence: "The category manager
sets a three-week order-up-to level for each item in the ambient range every Tuesday." If you cannot
write it, stop. A forecast without a decision has no functional, and a distribution without a functional
cannot be tested for fitness.

**Step 2. Identify which functional the decision reads.** Five families cover most supply chain
decisions, and they are not interchangeable.

| Decision | Functional read | What must be stated | Typical range |
|---|---|---|---|
| Service commitment: CSL target | A single quantile | `q`, and the lead-time horizon | 0.90–0.99 |
| Service commitment: fill-rate target | Expectation of the shortfall, `E[(D−Q)⁺]/E[D]` | Target, order quantity, and the distribution | 0.90–0.99 |
| Cost-based ordering | Critical fractile `C_u/(C_u+C_o)` | Both unit costs, and the price year | 0.25–0.95 |
| Capacity reservation | A high quantile over the reservation horizon | Probability, horizon, and the cost of unused capacity | 0.80–0.99 |
| Budget or provision | A quantile of a cost distribution, or a mean plus policy buffer | Probability, and whether the buffer is a quantile at all | Policy-dependent |

The distinction between the first two rows is the most consequential in the table and the least
understood. A cycle service level is exactly a quantile of lead-time demand at the reorder point. A fill
rate is not a quantile at any level: it depends on the order quantity, on the shape of the distribution
above the reorder point, and on the mean. Section 6 shows one distribution producing a 95% CSL at 140
units and a 95% fill rate at about 123.

**Step 3. State the horizon of the distribution, not of the report.** The functional is over lead-time
demand, not over the planning bucket. A weekly forecast with a three-week lead time needs a distribution
of three-week demand. Computing the weekly distribution and multiplying the quantile by three is wrong
in both directions: it ignores the diversification of independent weekly variation, which makes it too
wide, and it ignores serial correlation, which makes it too narrow in the regime that matters. Module
01A topic 11 treats lead-time demand construction; the requirement here is only that the horizon be
stated and match the decision.

**Step 4. Attach a probability to every quantile, every time.** "The reorder point is 90" is a number.
"The reorder point is the 0.95 quantile of three-week lead-time demand, which is 90 units" is a claim
that can be checked and can be wrong. Write the probability in the same sentence as the number, in every
deck, ticket, and API field. `p95_leadtime_demand` is a better field name than `safety_stock`.

### Phase B — Build the distribution and account for its uncertainty

**Step 5. Enumerate the four sources of predictive uncertainty.** This table is the diagnostic instrument
for the whole topic. When a vendor says "our model produces a 95% interval", the correct question is
which row is inside it. The honest answer is usually the first, sometimes the second, and rarely the
third or fourth.

| Source | What it is | In a typical model interval? | How to estimate it |
|---|---|---|---|
| Observation noise | Period-to-period variation not explained by the model | Yes, usually | Residual variance |
| Parameter error | Error from estimating parameters on a finite sample | Sometimes, if Bayesian or bootstrapped | Posterior or bootstrap spread |
| Structural error | Error from the model family being wrong | Almost never | Backtest bias, ensemble disagreement |
| Exogenous uncertainty | Future drivers outside the model | No | Scenario analysis, judgement, market data |

**Step 6. Decide what your interval actually contains, and say so.** A parametric interval from residual
variance contains observation noise, under the assumption that the model is correctly specified and the
residuals are the right shape. A bootstrap over residuals adds parameter uncertainty if the parameters
are refitted in each replicate. A Bayesian posterior predictive distribution adds it by construction.
None contains structural error, because each conditions on a model family being true, and none contains
exogenous uncertainty, because the likelihood has never seen a competitor's price change.

**Step 7. Widen for the missing sources, deliberately and visibly.** Three routes, in increasing cost.
**Backtest residual quantiles:** take the empirical error distribution at the decision horizon from a
rolling-origin backtest and add its quantiles to the point forecast, which captures observation noise
plus whatever structural error was present in that window. **Ensemble spread:** run several model
families and treat the spread as a lower bound on structural uncertainty; it is usually too narrow,
because the models share data and assumptions, but better than zero. **Scenario overlay:** add an
explicit exogenous component as a mixture over named scenarios with stated weights.

**Step 8. Represent exogenous futures as a mixture, not as a wider residual.** A wider residual inflates
the whole distribution symmetrically and leaves the shape alone. A mixture changes the shape: a
promotion with probability 0.3 produces a bimodal predictive distribution, and the 0.9 quantile of the
mixture can be far above either component's. If you know a promotion is planned, model it. Do not widen
the interval and call it uncertainty.

**Step 9. Check the support before the shape.** Demand is non-negative, often discrete over a short
horizon, and intermittent demand has an atom at zero, sometimes with probability above 0.5. A normal
distribution fitted to such a series produces negative low quantiles, and the planning system will
either reject them or silently floor them. Flooring is worse than rejecting: it moves probability mass
without renormalising, so the distribution no longer integrates to one and the upper quantiles are too
low by exactly the mass that was moved. Check that `F⁻¹(q) ≥ 0` for every published `q`, and check
whether the atom at zero is represented.

### Phase C — Compress for the interface

**Step 10. Choose the representation the decision can consume.** Most planning systems accept one number
per item per period. That constraint is real, and it is not an argument for point forecasts. It is an
argument for computing the functional the system needs and passing that: a reorder point is a quantile,
so pass the quantile. The system does not need to know it is a quantile. The organisation does.

**Step 11. Check monotonicity and quantile crossing.** Quantile regression estimates each level
separately and can produce crossing, a fitted 0.8 quantile below a fitted 0.7. This is not a numerical
nuisance. It means the estimated object is not a distribution, and any functional read from it is
undefined. Sort the quantiles and report how often crossing occurred; if it happens in more than a small
fraction of series, the model needs a constraint or a different parameterisation.

**Step 12. Check that the compressed object still answers the functional.** Run the decision twice, once
from the full distribution and once from the compressed object, and compare. If the order quantity moves
by more than the rounding to case pack, the compression is not safe for that decision. This test catches
most representation failures, including the commonest: compressing to a mean and reading a quantile from it.

**Step 13. Never publish a quantile without its probability, or an interval without its nominal coverage
and horizon.** This is a formatting rule with a substantive purpose. The probability is the part that
makes the number falsifiable. An organisation that publishes `1,240` cannot learn. One that publishes
`1,240 (0.90 quantile, 4-week horizon, calibrated on 26 rolling origins)` can.

### Phase D — Test calibration

**Step 14. Count coverage on data the model never saw.** For each forecast origin in a rolling-origin
backtest, record the interval and the realised outcome. Empirical coverage is the share inside. Compare
with nominal, and report the count with the rate, because a 90% interval evaluated on 20 origins has a
standard error near 7 percentage points, so 17 of 20 is not evidence of anything.

**Step 15. Break coverage down by horizon and by segment.** Aggregate coverage hides the two failures
that matter. Coverage degrading with horizon means the uncertainty is not growing fast enough, the
signature of a model that ignores parameter or structural error. Coverage that is acceptable overall but
poor on fast movers, promotions, or new items means the calibration is an average over a mixture and the
important segment is the wrong one.

**Step 16. Plot the PIT histogram and read its shape.** For each outcome compute `u = F(y)`, the forecast
CDF at the realised value. If the forecasts are calibrated, `u` is uniform on (0,1).

| Shape | Reading | Typical cause |
|---|---|---|
| Flat | Consistent with calibration | None, or a well-specified model |
| U-shaped | Too many outcomes in the tails; intervals too narrow | Underestimated uncertainty, missing structural error |
| Humped | Too few in the tails; intervals too wide | Overdispersion, or an unnecessarily wide policy band |
| Rising | Outcomes systematically above the forecast | Positive bias, or a distribution shifted low |
| Falling | Outcomes systematically below the forecast | Negative bias |
| Spike at 0 or 1 | Outcomes outside the support, or an atom not modelled | Support error, zero-inflation ignored |

The histogram is a diagnostic, not a test. A flat histogram is necessary but not sufficient, and a formal
uniformity test on dependent forecast origins over-rejects because consecutive origins share
information. Treat the shape as evidence, not as a verdict.

**Step 17. Report calibration with sharpness.** An interval of zero to infinity is calibrated and
useless. Report the mean interval width, or the interquartile spread, alongside the coverage. A model
with 90% coverage and a width of 400 units is not better than one with 88% coverage and a width of 120;
it is a different trade, and the decision determines which trade is right. Topic 09 formalises the trade;
the reporting standard in Step 23 requires both numbers.

### Phase E — Make it coherent across the hierarchy

**Step 18. Distinguish marginal from probabilistic coherence.** For point forecasts, reconciliation is a
projection onto the aggregation constraint, and Module 08 topics 02 and 03 treat the methods. For
probabilistic forecasts there are two requirements. **Marginal coherence** means each quantile's vector
sums across the levels. **Probabilistic coherence** means the joint predictive distribution lives on the
coherent subspace, so every draw from it sums correctly. Only the second is a distribution; the first can
be satisfied by a set of numbers that no distribution could produce.

**Step 19. Understand why quantiles do not add.** For two random variables, `Q_τ(X+Y) ≠ Q_τ(X) + Q_τ(Y)`
in general. Equality holds only when `X` and `Y` are comonotonic, moving together perfectly; it fails
when they are independent and fails most when they are negatively dependent, because the `τ` quantile of
a sum depends on the dependence between components, and marginal quantiles carry no information about
dependence. Summing item-level 0.95 quantiles therefore produces a category value that is almost always
too high: it assumes perfect positive dependence, the most pessimistic case. Summing 0.5 quantiles works
only because the median of a sum of symmetric variables is the sum of medians, a special case not a rule.

**Step 20. Prefer sample-based reconciliation, which preserves coherence by construction.** Produce a
sample of the joint predictive distribution at the bottom level, reconcile the sample so every draw sums
across the hierarchy, and compute quantiles from the reconciled draws. Because each draw is coherent,
every functional computed from the sample is coherent: quantiles, means, tail expectations, expected
costs. The projection is the same linear operator used for point forecasts, applied draw by draw, so it
costs a sample and a dependence structure and needs no new theory.

**Step 21. If you must reconcile quantiles, constrain the problem and name the assumption.** One line of
work reconciles the median with a regularised quantile loss and then imposes coherence through a
variance-additivity identity, an assumption about shape rather than a consequence of it. Another
reconciles samples and projects them, and its authors argue the quantile-constrained approach is not
coherent in general. Prefer the sample route unless the system cannot carry samples.

**Step 22. Check the hierarchy's calibration level by level.** A reconciled hierarchy can be well
calibrated at the total and badly at the leaf, because the weights are fitted to minimise error where the
data is. Report coverage at each level. The same caution applies across time: a quarterly 0.9 quantile is
not three monthly 0.9 quantiles.

### Phase F — Report

**Step 23. Meet the reporting standard.** Five fields, printed with the distribution, every time.

| Field | Content | Why it is required |
|---|---|---|
| Central prediction interval | Two numbers, the nominal coverage, and the horizon | The reader sees the spread and knows what it claims |
| Stated horizon | The exact future period the distribution covers | A distribution without a horizon is not a distribution |
| Calibration evidence | Empirical coverage, sample size, window, segmentation | Makes the claim falsifiable and shows where it fails |
| Decision functional | Which quantile or expectation the number is for, and the probability | Prevents the number being read as a different functional |
| Exclusions | What the distribution does not contain | Prevents the reader inferring a shape the model never claimed |

A worked note for a spare part:

```text
  item         BR-4471, bearing, warehouse Rotterdam
  horizon      3 weeks (lead time), origin 2026-W44
  central PI   80% interval [20, 100] units; 90% interval [0, 140] units
  calibration  80% nominal: 33/40 = 82.5% empirical, rolling 40 origins
               90% nominal: 28/40 = 70.0% empirical   <-- undercovers
  functional   0.95 quantile of lead-time demand = 140 units (CSL policy)
  excludes     planned maintenance shutdown in W46; price change announced for Q1
```

The 90% row is the most important line in the note, because it is the line that says the interval is not
doing what it claims. A report that omits it is not a report; it is an advertisement.

**Step 24. Version the distribution and pre-register the evaluation.** Store the predictive distribution
with its forecast origin, model version, and parameter set, and score it later against the same origin.
Any change to the family, the calibration window, or the reconciliation weights is a new version with a
restatement note. This prevents the commonest institutional failure: a distribution quietly widened
after every stock-out until it covers everything and informs nothing.

### A short history, because the inheritance explains the habits

Quantile reasoning entered inventory practice before statistics had a theory of quantile estimation. The
newsvendor gave the order quantity as a fractile, and the normal-approximation safety-stock formula gave
it as `z` times a standard deviation. Both were taught as formulas with a multiplier, and by the time
planning software arrived, "safety stock factor 1.65" was a configuration parameter whose meaning was
folklore. Probabilistic forecasting then arrived from weather and energy rather than retail: the energy
forecasting competitions scored submissions on pinball loss at many quantile levels, and M4 in 2018 was the
first of the M-series to elicit prediction intervals across its large benchmark, producing the cleanest
public evidence that forecasters' intervals are too narrow. The gap persists because a point forecast is easier to put in a
spreadsheet.

## 6. Worked example

A distributor of industrial spare parts plans a bearing, item BR-4471, at one warehouse. The lead time is
three weeks, review is weekly, and the order-up-to level is set per replenishment cycle. Demand is
intermittent. Quantities are units and money is euro at 2026 prices. The numbers are constructed to be
internally consistent and to isolate the representation choice; they are not a published case.

### 6.1 Why the mean, the median, and the point forecast diverge

One week of demand for the item, assuming the weekly distribution is stable:

| Weekly demand (units) | 0 | 1 | 2 | 4 | 8 | 12 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probability | 0.52 | 0.13 | 0.13 | 0.09 | 0.07 | 0.04 | 0.02 |

Three summaries of this one distribution:

```text
mean   = 0(0.52) + 1(0.13) + 2(0.13) + 4(0.09) + 8(0.07) + 12(0.04) + 20(0.02)
       = 0 + 0.13 + 0.26 + 0.36 + 0.56 + 0.48 + 0.40 = 2.19 units per week

median        = 0 units,   because F(0) = 0.52 >= 0.50
0.90 quantile = 8 units,   because F(4) = 0.87 < 0.90 and F(8) = 0.94 >= 0.90
0.95 quantile = 12 units,  because F(8) = 0.94 < 0.95 and F(12) = 0.98 >= 0.95
```

A model trained on mean absolute error returns 0. One trained on squared error returns 2.19. The reorder
point at a 0.95 cycle service level is 12, and expected demand is 2.19. The three differ by a factor of
more than five, and the median is not merely different but qualitatively different: it says "no demand"
about a process that has demand 48% of the time. For smooth demand the mean and median coincide and the
divergence looks like a curiosity. For intermittent demand it is the whole problem.

### 6.2 The lead-time distribution and three functionals

Three weeks of lead-time demand, with the dependence between weeks already reflected. It is given rather
than derived; topic 06 shows one generative route and Module 01A topic 11 shows others.

| `d` (units) | 0 | 20 | 40 | 60 | 80 | 100 | 140 | 180 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p(d)` | 0.10 | 0.15 | 0.20 | 0.20 | 0.15 | 0.10 | 0.07 | 0.03 |
| `F(d)` | 0.10 | 0.25 | 0.45 | 0.65 | 0.80 | 0.90 | 0.97 | 1.00 |

The mean is 60.2 units, the median 60, the 0.90 quantile 100, the 0.95 quantile 140. Now read three
functionals from the same table.

**Cycle service level.** `CSL(Q) = F(Q)`, read straight from the cumulative column: 140 units at 0.95,
100 units at 0.90.

**Fill rate.** `FR(Q) = 1 − E[(D−Q)⁺]/E[D]`, computed at each candidate.

```text
E[D] = 60.2 units

Q = 60:  E[(D-60)+] = 20(0.15) + 40(0.10) + 80(0.07) + 120(0.03) = 16.2
         FR = 1 - 16.2/60.2 = 0.731
Q = 80:  E[(D-80)+] = 20(0.10) + 60(0.07) + 100(0.03) = 9.2
         FR = 1 - 9.2/60.2 = 0.847
Q = 100: E[(D-100)+] = 40(0.07) + 80(0.03) = 5.2
         FR = 1 - 5.2/60.2 = 0.914
Q = 140: E[(D-140)+] = 40(0.03) = 1.2
         FR = 1 - 1.2/60.2 = 0.980
```

A 0.95 fill rate needs `E[(D−Q)⁺] = 0.05 × 60.2 = 3.01`, which sits between `Q = 100` (5.2) and
`Q = 140` (1.2), near 123 units. That is a quantile of about 0.93, not 0.95, and the exact level depends
on the order quantity and the shape of the tail. The fill rate is not a quantile, and any table that
converts one to the other with a single multiplier is approximating a relationship that depends on the
order size.

**Cost-based ordering.** With an underage cost of 30 euros per unit short and an overage cost of 10 euros
per unit left, the critical fractile is `30/(30+10) = 0.75`, which reads 80 units. The expected cost at
each candidate confirms it:

| `Q` | `E[(D−Q)⁺]` | `E[(Q−D)⁺]` | Expected cost (EUR) |
|---:|---:|---:|---:|
| 40 | 27.2 | 7.0 | `30(27.2) + 10(7.0) = 886` |
| 60 | 16.2 | 16.0 | `30(16.2) + 10(16.0) = 646` |
| 80 | 9.2 | 29.0 | `30(9.2) + 10(29.0) = 566` |
| 100 | 5.2 | 45.0 | `30(5.2) + 10(45.0) = 606` |
| 140 | 1.2 | 81.0 | `30(1.2) + 10(81.0) = 846` |

| Policy | Rule | Order | CSL achieved | Fill rate achieved |
|---|---|---:|---:|---:|
| Mean | Order the mean | 60 | 0.65 | 0.731 |
| Cost-optimal | Critical fractile 0.75 | 80 | 0.80 | 0.847 |
| Fill-rate target | Solve FR = 0.95 | about 123 | about 0.94 | 0.950 |
| CSL target | 0.95 quantile | 140 | 0.95 | 0.980 |

The cost minimum is at 80, the critical fractile, and the 140-unit service policy costs 846 against 566,
a 49% increase. That increase is not waste: it is the price of a service promise the business may have
made contractually. Somebody should decide to pay it, and the distribution lets them see the price. Four
policies, one distribution, order quantities from 60 to 140.

### 6.3 What compression loses

Suppose the planning system accepts only an interval. The 90% central interval is `[0, 140]`, since
`F(0) = 0.10 ≥ 0.05` and `F(140) = 0.97 ≥ 0.95`. The interval is correct and it destroys three things:
the location of the mass inside it, the atom at low demand, and the right skew with a long upper tail. It
says nothing about the 0.75 quantile, so it cannot support a cost-based order, which depends on the ratio
of the two tail expectations and hence on the shape between and beyond the endpoints. It can support a
capacity reservation at 0.95, because that is exactly what it states. That is Property 2 in action: name
the decision, name the functional, then ask whether the compressed object contains enough to evaluate it.

### 6.4 Calibration evidence, and what it says

The model behind the table claims an 80% interval of `[20, 100]` and a 90% interval of `[0, 140]`. It has
been run on 40 rolling origins with three-week horizons. Outcomes fell inside the 80% interval 33 times
and inside the 90% interval 28 times.

| Interval | Nominal | Empirical | Count | Reading |
|---|---:|---:|---:|---|
| 80% central | 0.80 | 0.825 | 33/40 | Consistent with calibration |
| 90% central | 0.90 | 0.700 | 28/40 | Undercovers by 20 points |

The 90% upper bound of 140 was exceeded 12 times in 40 origins, three times the nominal rate. The 80%
bound of 100 was exceeded 7 times, close to the nominal 8. The pattern is diagnostic: the central mass is
roughly right and the tail is too thin, the signature of a model whose uncertainty is mostly observation
noise with structural and exogenous error absent. That is Property 4 made concrete.

Two responses are available. Widen the 90% interval to the empirical 0.95 quantile of the backtest
errors, near 180, and accept a loss of sharpness. Or decompose the misses: how many fell outside because
of a promotion, a price change, or a supplier shutdown, and how many because the model understated
ordinary variation. The first produces a defensible number today; the second produces a better model.

> **Note:** Forty origins is a small sample. The standard error of a coverage estimate at 0.90 with
> `n = 40` is `sqrt(0.9 × 0.1 / 40) = 0.047`, about 4.7 percentage points. The observed 70% is about four
> standard errors below nominal and worth acting on. A single origin outside the interval is not.

### 6.5 The hierarchy trap, with arithmetic

Fifty bearings in the same family, each held at an independent 0.95 cycle service level. The category
manager reports that the family is at 95% service. It is not.

```text
P(all 50 in stock) = 0.95^50 = 0.077      about  7.7%
P(all 10 in stock) = 0.95^10 = 0.599      about 59.9%
P(all  5 in stock) = 0.95^5  = 0.774      about 77.4%
```

If the items are independent, the probability that a customer can get every item in the family on one
visit is 7.7%, although every item meets its own 95% target. Positive correlation, from items bought
together or sharing a supplier, raises the joint probability; negative correlation lowers it. The joint
service level is a property of the joint distribution, and marginal targets do not determine it.

This is also the practical argument for sample-based reconciliation. If you hold draws that are coherent
across the family, you can compute the joint probability directly, by counting the draws in which every
item is in stock. If you hold only marginal quantiles, you cannot compute it at all, and you will report
7.7% or 95% depending on which question you thought you were answering.

### 6.6 What the example establishes

Four findings survive the arithmetic. A mean, a median, and a quantile of one distribution can differ by
a factor of five on intermittent demand, so the choice of summary is a decision, not a formatting choice.
A cycle service level and a fill rate at the same nominal level require different order quantities, so
they are not two names for one target. An interval supports a quantile decision and not a cost decision,
so compression must be checked against the functional. And marginal service targets do not determine the
joint service level, so coherence must be a property of the joint distribution.

## 7. Code, query, or calculation

The calculation that matters is reading quantiles and tail expectations off a small distribution by hand.
It is short enough to do on paper, and doing it on paper is the point: the numbers are arguable, and the
argument is easier when the arithmetic is visible. The block below is section 6.2 written as a table walk.

```text
LEAD-TIME DEMAND QUANTILES BY HAND -- 3-week lead time, units

  d_i    p_i     F_i = cum p      q read
  ---------------------------------------------------------------
    0   0.10       0.10
   20   0.15       0.25
   40   0.20       0.45
   60   0.20       0.65           median (0.50) = 60
   80   0.15       0.80           Q_0.75 = 80    Q_0.80 = 80
  100   0.10       0.90           Q_0.90 = 100
  140   0.07       0.97           Q_0.95 = 140
  180   0.03       1.00

  check   sum p_i = 1.00
  mean  = 0(0.10) + 20(0.15) + 40(0.20) + 60(0.20) + 80(0.15)
        + 100(0.10) + 140(0.07) + 180(0.03) = 60.2

  tail  E[(D-60)+] = 20(0.15) + 40(0.10) + 80(0.07) + 120(0.03) = 16.2
        fill(60)   = 1 - 16.2/60.2 = 0.731
        E[(D-80)+] = 20(0.10) + 60(0.07) + 100(0.03) = 9.2
        fill(80)   = 1 - 9.2/60.2 = 0.847

  read  a quantile needs the CDF column; a fill rate needs the tail integral.
```

Three checks before trusting such a table. Confirm the probabilities sum to one after any truncation or
flooring, because a distribution that lost mass at zero produces upper quantiles that are too low.
Confirm the quantile is read as the smallest `d` with `F(d) ≥ q`, not the largest with `F(d) ≤ q`,
because the two conventions differ by one support point and that point is often the decision. Confirm the
tail expectation uses the same units and horizon as the quantile.

## 8. Data, standards, and regulatory context

No international standard defines a predictive distribution, a prediction interval, or a calibrated
forecast. That absence is informative: the choice is statistical and commercial, and must be documented
rather than defaulted. What exists constrains the vocabulary, the inputs, and the reporting.

### 8.1 Statistical vocabulary and measurement

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of probability, distribution, quantile, interval | Any forecasting procedure | B |
| ISO 5725-1 | Trueness and precision of measurement methods | Accuracy as a combination of bias and precision | Forecast evaluation, which has no repeated reference | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | Precision, trueness, uncertainty, and the ban on "accuracy" as a quantity | Anything specific to forecasts | B |
| ISO 80000-1 | Quantities and units | Unit conventions for money, time, ratios | Which currency or price year to use | B |
| ISO 8601 | Date and time representation | Week numbering and ordering, hence horizon definition | Which week a demand observation belongs to | B |

The VIM point transfers directly. "A 90% prediction interval" is a conformance claim relative to a stated
coverage, exactly as "98% inventory accuracy" is a conformance rate relative to a tolerance. The interval
has no meaning without the coverage, the horizon, and the sample. The standard does not forbid the
phrase; it forbids pretending the phrase has a unique referent.

### 8.2 Where the distribution becomes an accounting number

| Requirement | Where it applies | Effect on the forecast distribution | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | IFRS-reporting entities | Lower of cost and net realisable value; the upper tail drives write-down risk | B |
| IAS 37 *Provisions* | IFRS-reporting entities | A provision is the best estimate of expenditure; a distribution, not a point, is the natural input | B |
| ASC 330 *Inventory* | US GAAP entities | Lower-of-cost-or-market logic with industry practice | B |
| IFRS 15 / ASC 606 | Revenue | Unsold units generate no revenue, so the lower tail is a revenue risk | B |
| Sarbanes-Oxley Act 2002, s.404 | US-listed entities | Internal control over financial reporting, including forecast inputs to valuation | B |
| Federal Reserve SR 11-7 | US bank holding companies | Model risk management: inventory, validation, ongoing monitoring | B |
| ASCM SCOR DS | Global reference framework | Defines service metrics such as perfect order fulfilment | C |

The SR 11-7 entry is a template by analogy, not an obligation for a consumer goods company. Its useful
shape is the triad of model inventory, independent validation, and ongoing monitoring, which is exactly
what a predictive distribution needs and almost never gets.

### 8.3 Contractual service definitions

| Instrument | What it fixes | Consequence for the distribution | Grade |
|---|---|---|---|
| Service level agreement (SLA) | Fill rate, on-time in full, order completeness, measurement window, exclusions | The functional is fixed by contract; the internal target may be a different functional | B |
| Supplier scorecards | Delivery reliability and quality thresholds | A supplier's measured service is not your demand distribution | B |
| Internal planning targets | Target CSL or fill rate by item class | A target is a policy, not a measurement | C |
| Capacity contracts | Reserved volume, take-or-pay thresholds | The reservation is a high quantile, and the penalty is the cost of the unused portion | B |

The contractual trap is to compute an internal fill rate, compare it with a contractual target, and
discover at audit that the contract excludes cancelled lines, intercompany shipments, or substitutions.
The distribution is arithmetically correct and contractually wrong. State the contractual definition
next to the modelled one.

### 8.4 The fields a distribution artefact needs

| Field | Why it is needed | Common absence |
|---|---|---|
| Forecast origin | Identifies the information set and prevents leakage | Overwritten each planning cycle |
| Horizon | The future period the distribution covers | Implied rather than stated |
| Model version and parameters | Reproducibility and restatement | Only the latest version retained |
| Quantile grid or sample | The object itself, not a summary of it | Only a point and a "confidence" |
| Calibration window | The sample on which coverage was measured | The fitting window used instead |
| Segmentation | Coverage by item class, horizon, and demand regime | Aggregate only |
| Reconciliation method | Which dependence assumption the hierarchy encodes | "Reconciled" with no method named |
| Exclusions | Promotions, shutdowns, price changes, new listings | Not recorded anywhere |

The first two rows are the most consequential and least often fixed. If the forecast is overwritten every
week, no calibration test can run, because the forecast that was made is gone. A distribution without an
origin cannot be scored, and an unscored distribution cannot be improved.

### 8.5 What is convention rather than requirement

The 80% and 95% interval levels are conventions inherited from statistics, not requirements; a business may
prefer 50% and 90%. The choice of a normal, gamma, or empirical family is a modelling decision, not a
compliance one, and the 0.95 cycle service level is a policy in most companies. The "three standard
deviations" heuristic for a wide interval is folklore, and it is wrong for skewed demand. An analyst who
can distinguish a requirement from a convention can challenge a convention; one who cannot will treat
every inherited interval level as a physical constant, including the ones that decide how much inventory
the company holds.

## 9. Industry practice and case evidence

### 9.1 The evidence that forecasters' intervals are too narrow

The single most robust empirical finding in probabilistic forecasting is that stated intervals
undercover. It has been observed in competitions, in operational weather forecasting, and in demand
planning.

| Evidence | What it shows | Grade |
|---|---|---|
| Grushka-Cockayne and Jose (2019), M4 interval analysis | Across the twenty valid interval submissions in M4, the submissions "fail to estimate the uncertainty properly"; combining intervals improves calibration and accuracy | A |
| Makridakis, Hyndman and Petropoulos (2020) | Intervals are "often narrower than they should be", with nominal 95% intervals from conventional methods achieving coverage as low as 76.8% on earlier competition data | A |
| Gneiting, Balabdaoui and Raftery (2007) | Formalises calibration and sharpness, defines probabilistic and marginal calibration, and shows a uniform PIT histogram is necessary but not sufficient | A |
| Magnus and Vasnev (2023) | Ignoring positive correlation between combined forecasts produces intervals that are far too narrow; central-bank forecasts described as misleadingly precise | B |
| Wang, Kang, Petropoulos and Li (2021) | Feature-based combination improves interval scores and coverage on M4 data | B |

The 76.8% figure is reported as the source states it and was not independently replicated here. What
matters is the direction and the consistency: the undercoverage is not a bug in one implementation. It is
what happens when a model reports the uncertainty it can see and omits the uncertainty it cannot. The
three mechanisms in the second row map onto the four sources in Step 5: ignored model uncertainty is
structural error, unaccounted parameter uncertainty is estimation error, and Gaussian assumptions failing
in fat tails is a support or shape error.

### 9.2 Quantile forecasting as a commercial practice

| Evidence | What it shows | Grade |
|---|---|---|
| Bruzda (2018), *Central European Journal of Operations Research* | Compares bootstrap, GARCH, and quantile regression for multi-step quantile forecasts; simulation-based approaches reduced average supply chain cost per unit substantially at 12-month horizons and high service levels on 474 M3 monthly series | A |
| Ngartera et al. (2026), *Journal of Risk and Financial Management* | Gradient-boosted quantile regression with a constrained newsvendor; moving from the median to the 90th percentile cut CVaR95 by 26.7%, raised CSL from 44.4% to 89.5%, and cut mean cost by 48.7% at a 10:1 stockout-to-holding ratio | A |
| Wang, Kang and Petropoulos (working paper) | Probabilistic combinations for intermittent demand, evaluated on calibration, sharpness, and inventory utility; simple averaging is robust, and accuracy and inventory performance trade off | B |
| Kvantis Atlas practitioner note | States the quantile-to-newsvendor correspondence and warns against confusing a distribution with a confidence interval | D |
| TimesFM quantile documentation | A P10–P90 forecast grid mapped to service levels, with a caveat that P90 should contain about 90% of actuals | D |

Two things are worth taking from the peer-reviewed rows. The benefit is concentrated at high service
levels and long horizons, exactly where the tail dominates. And the reported gains depend on the cost
ratio: the 2026 study measured its improvements at a 10:1 stockout-to-holding ratio, and at ratios near
2:1 the higher quantile raises mean cost while lowering tail risk. Quantile targeting buys tail
protection and pays for it in average cost, which is a commercial decision and should be made explicitly.

### 9.3 Calibration practice and its instruments

| Instrument | What it diagnoses | Limitation | Grade |
|---|---|---|---|
| PIT histogram | Whether the forecast CDF is uniform at the realised outcomes | Needs many origins; shape is diagnostic, not a test | A |
| Reliability diagram | Whether stated probabilities match observed frequencies for a derived binary event | Bin counts matter; sparse bins mislead | A |
| Coverage by horizon and segment | Where calibration fails | Requires a retained forecast archive | A |
| Uniformity test (chi-square or similar) | Formal rejection of calibration | Over-rejects under temporal dependence, the normal case | B |

The warning is Gneiting, Balabdaoui and Raftery's: calibration alone is not enough, because a forecast
can be calibrated and uninformative. Their criterion is to maximise sharpness subject to calibration,
which is why the reporting standard in Step 23 pairs the interval with its empirical coverage and its
width.

### 9.4 The scenario-versus-distribution argument

| Position | Source | Claim | Grade |
|---|---|---|---|
| Scenarios should not be probabilised | Ha Duong (2003) | A scenario set is conventionally presented without quantifying confidence, while a forecast is attached to a probability distribution | B |
| Scenarios carry no relative likelihood in Intuitive Logics practice | Goodwin and Wright (2025) | Scenario sets are internally plausible narratives with no relative likelihood; the space between them represents the uncertainty | A |
| Oxford scenario planning eschews probabilities | Ramírez, Churchhouse, Palermo and Hoffmann (2017) | Distinguishes plausible from probable futures; best-case and worst-case scenarios are disguised forecasts | C |
| Scenarios are "just-about plausible, not necessarily likely" | Behar and Hlatshwayo (2021), IMF guidance | Probability is not the relevant metric; scenarios are not quantified sensitivity analyses | B |
| Forecasters and scenario planners should be combined | Scoblic and Tetlock (2020) | Frames the two camps and proposes bridging devices | E |

Keep the two objects separate and know which one the decision needs. A capacity investment with a
fifteen-year life and no meaningful probability model needs a scenario set, because the value is in
widening the range of futures and testing robustness across them. A weekly order needs a distribution,
because the decision is a quantile and a quantile requires a probability. Using a scenario set for the
order means the order is set by whichever scenario the loudest person prefers. The bridge, when one is
needed, is to treat the scenario set as a mixture with stated weights and publish the weights as an
assumption, which makes it falsifiable and destroys the property that made the set useful.

### 9.5 What practitioners argue about

**Quantile grid or distribution family.** One camp fits a parametric family and reads quantiles from it,
which is smooth, compact, and wrong in the tail if the family is wrong. The other fits each quantile
directly, which is flexible and can produce crossing. The evidence does not settle it, and the practical
answer is usually to fit a family for the central mass and calibrate the tail empirically.

**Reconcile samples or reconcile quantiles.** The sample route guarantees coherence and costs a
dependence structure. The quantile route is cheaper and can produce a set that is not a distribution. The
published work is split, partly over which is easier to implement in a system that stores one number per
cell.

**Calibration or sharpness.** A model can be tuned to cover everything, and a model can be tuned to be
narrow. The proper-score literature says to optimise a score that balances both, and the reporting
literature says to show both. In practice the argument is about which failure is more expensive: an
interval that missed a stock-out, or an interval so wide that nobody acts on it.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Quantile quoted without its probability | A slide says "the upper bound is 1,400" with no percentage | The number is uninterpretable and unfalsifiable | Nobody can tell whether it is a 0.8 or a 0.99 | Require the probability in the same sentence and in the field name |
| Point forecast used as a quantile | The reorder point equals the forecast plus a fixed 10% | A mean or median is read as a tail value | Systematic understocking on skewed demand | Compute the quantile from the distribution; state the level |
| Model interval treated as total uncertainty | The validation slide shows nominal coverage and no empirical coverage | Residual variance contains only observation noise | Intervals too narrow; stock-outs attributed to bad luck | Add backtest residual quantiles; report empirical coverage |
| Nominal coverage assumed | The documentation says "95% confidence" and nothing was measured | The model family is taken as evidence | Undetected undercoverage | Count coverage on held-out origins |
| Calibration tested on the fitting window | Coverage is 94% in-sample and 71% out-of-sample | The interval was fitted to the same data | A false assurance that survives until the first bad quarter | Rolling-origin evaluation with retained vintages |
| Interval width identical across items | Every SKU has the same percentage band | The model ignores item-level variance | Fast movers too wide, slow movers too narrow | Segment the width; test coverage by class |
| Negative quantiles passed to the planning system | The 0.05 quantile is −3 units | An unbounded family was fitted to non-negative demand | The system floors at zero without renormalising | Check support; use a family with non-negative support |
| Quantile crossing | The 0.8 quantile is below the 0.7 for some series | Independent estimation per level | The object is not a distribution; functionals undefined | Sort and report the crossing rate; constrain the fit |
| Quantiles summed up a hierarchy | Category 0.95 equals the sum of item 0.95 values | Quantiles are not additive | The category quantile is too high; capacity over-reserved | Reconcile draws, then take quantiles |
| Reconciliation applied to quantile columns | The reconciled 0.9 column sums and the 0.5 column does not | Each level adjusted independently | Coherent-looking numbers with no joint distribution | Enforce coherence sample by sample |
| A scenario set presented as a distribution | Three scenarios with "30/40/30" weights invented in the meeting | Weights were chosen to make the story work | The order quantity is a narrative artefact | Either state weights as an assumption or do not probabilise |
| Fill rate computed as a quantile | A 95% fill-rate target is set at the 0.95 quantile | The two are different functionals | Overstocking relative to the target, or understocking | Solve the shortfall equation; state the order quantity |
| Horizon mismatch | The distribution is weekly and the lead time is three weeks | The functional is computed on the wrong object | The quantile is too low when correlation is positive | Build the distribution over lead-time demand |
| Truncation without renormalisation | The probabilities sum to 0.93 after flooring | Mass was removed at zero | Upper quantiles biased low | Renormalise and record the removed mass |
| Mean and median used interchangeably | Two reports give 60 and 62 for the same item | Different loss functions were minimised | Small on smooth demand, large on intermittent | State which summary and which loss |

Two rows deserve expansion. **Quantiles summed up a hierarchy** most often survives review, because the
arithmetic looks right and the result is plausible; it over-reserves whenever the components are less than
perfectly correlated, which is almost always, and the over-reservation grows with the number of
components. **A scenario set presented as a distribution** most often survives a steering committee,
because the weights sound like judgement and nobody wants to challenge judgement. Both are caught by one
question: what dependence or probability assumption did this number require, and who chose it?

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The decision's functional is identifiable | The whole chain from distribution to number depends on it | The wrong summary passes downstream and the error is invisible | Write the decision sentence and name the functional; check the system field |
| The distribution's horizon matches the decision's horizon | A quantile of weekly demand is not a quantile of lead-time demand | Order quantities wrong, in a direction set by the correlation | Compare horizons in model metadata and policy |
| Forecast origins are comparable enough for calibration to transfer | Coverage measured on a backtest is used to justify the live interval | Regime change invalidates the calibration | Plot coverage by year; test for a break |
| The distribution family is adequate in the tail that matters | The decision reads a quantile or a tail expectation | The tail is wrong where the cost is | Compare empirical and fitted tail quantiles |
| The calibration sample is large enough | Coverage is an estimate | Random variation is read as miscalibration | Report the count and the standard error |
| The cost parameters are known well enough to pick a fractile | The optimum is a ratio of two costs | The order sits at the wrong quantile | Compute the optimum across a plausible cost range |
| Demand is not censored by stock-outs | The distribution is fitted to observed demand | The model learns the stock-out and reproduces it | Flag stock-out periods; bound or reconstruct demand |
| The hierarchy structure is stable over the horizon | Reconciliation requires a fixed aggregation matrix | A re-organisation or a new item invalidates the mapping | Version the hierarchy with the forecast |
| Quantile estimates are monotone in level | A distribution must have non-crossing quantiles | Functionals are undefined | Count crossings; constrain if frequent |
| Exogenous drivers are modelled or excluded explicitly | The distribution is conditional on something | A promotion or price change arrives outside the conditioning set | List the exclusions and check them against the calendar |

Three limits deserve statement. A predictive distribution is not a forecast of the world: it is a forecast
of a variable under a model, and the model's exclusions are part of the forecast. A calibrated
distribution is not necessarily a useful one: an interval that covers everything informs nothing, which
is why sharpness must be reported. And probabilistic coherence cannot be recovered from marginals: no care
with a column of quantiles reveals the joint distribution, and joint service requires a joint object.

## 12. Exercises

### Exercise 1 — Map four decisions to four functionals (analysis, 35 minutes)

For each decision, name the functional of the predictive distribution it reads, the horizon, and the
parameter that must be stated. (1) A hospital sets a reorder point for a critical consumable at a 0.98
cycle service level. (2) A retailer commits to a 97% fill rate on a private-label line. (3) A contract
manufacturer reserves oven capacity for the next quarter. (4) A finance team sets a provision for
slow-moving stock.

**Deliverable:** a four-row table with decision, functional, horizon, parameter, and the object to store.
**Acceptance criteria:** each functional classified as quantile, expectation, or full distribution;
horizon in time units; the parameter named with its unit or probability. **Time:** 35 minutes.

### Exercise 2 — Read a distribution by hand (calculation, 30 minutes)

Using the nine-point distribution below, compute the mean, the median, the 0.80 quantile, the 0.95
quantile, and the fill rate at the median and at the 0.80 quantile.

| `d` | 0 | 10 | 20 | 30 | 40 | 60 | 80 | 120 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p(d)` | 0.15 | 0.15 | 0.20 | 0.20 | 0.12 | 0.09 | 0.05 | 0.03 | 0.01 |

**Deliverable:** the five numbers, the cumulative column, and the two tail expectations. **Acceptance
criteria:** probabilities sum to one; quantiles read with the `F(d) ≥ q` convention; fill rate computed as
a ratio of expectations. **Time:** 30 minutes.

### Exercise 3 — Find the quantile implied by a fill-rate target (calculation, 40 minutes)

Using the distribution from Exercise 2, find the order quantity that achieves a 0.95 fill rate, by
interpolation between support points if necessary. State the implied quantile level and explain why it
differs from 0.95.

**Deliverable:** the shortfall equation, the candidates tested, the interpolation, and the implied
quantile. **Acceptance criteria:** the equation is written before it is solved; the answer states that
fill rate depends on the order quantity; the implied level is compared with 0.95. **Time:** 40 minutes.

### Exercise 4 — Audit a compressed report (critique, 35 minutes)

A report says: "Demand for the next quarter is forecast at 12,400 units, with a confidence range of
±15%." Identify every ambiguity and every piece of information lost, and state which decisions the
statement can and cannot support.

**Deliverable:** at least eight defects, each with the decision it would mislead, plus a rewritten
statement meeting the reporting standard. **Acceptance criteria:** the interval's nominal coverage,
horizon, and dependence assumptions are named as missing; the point forecast's functional is identified
as unstated; the rewrite includes calibration evidence and exclusions. **Time:** 35 minutes.

### Exercise 5 — Decompose uncertainty (analysis, 40 minutes)

A model reports a 90% interval of ±8% of the point forecast. A rolling-origin backtest shows residuals
with a standard deviation twice that implied by the interval. An ensemble of three model families
disagrees by ±5% around the point. A promotion is planned with probability 0.4.

**Deliverable:** a four-source decomposition table with an estimated contribution from each source, and a
widened interval that states which sources it includes. **Acceptance criteria:** all four sources
addressed; the widened interval is wider than the model's and the reason is stated; the promotion is
handled as a mixture. **Time:** 40 minutes.

### Exercise 6 — Calibration by hand (calculation, 45 minutes)

A 90% interval was evaluated on 36 rolling origins and contained the outcome 26 times. Compute the
empirical coverage, the standard error of the estimate, and the number of standard errors between observed
and nominal. Sketch the PIT histogram you would expect if the misses are concentrated above the upper
bound.

**Deliverable:** the arithmetic, the standard error, the distance, and a labelled sketch. **Acceptance
criteria:** coverage stated with its count; standard error formula shown; the sketch shows a U-shape with
the right tail heavier than the left. **Time:** 45 minutes.

### Exercise 7 — The hierarchy multiplication (calculation and critique, 45 minutes)

A family has 30 items, each held at a 0.97 cycle service level, independently. Compute the probability
that all are in stock, then for 8 items and 3 items. Explain what changes if the items are positively
correlated, and state how you would compute the joint probability from a coherent sample.

**Deliverable:** three probabilities, a sentence on the correlation effect, and a procedure for the
sample-based computation. **Acceptance criteria:** exponentiation shown; independence stated as an
assumption; the sample procedure counts draws, not quantiles. **Time:** 45 minutes.

### Exercise 8 — Scenario set or distribution (critique, 30 minutes)

A team presents three scenarios for a new distribution centre: base, e-commerce acceleration, and supply
disruption. They propose weighting them 50/30/20 "based on judgement" and using the weighted result as the
demand distribution for a fifteen-year capacity decision.

**Deliverable:** a one-page critique separating what is legitimate from what is not, and a recommendation
for how the capacity decision should consume the scenario set. **Acceptance criteria:** the weights are
identified as an assumption rather than a measurement; plausible and probable are used correctly; the
recommendation is specific and does not merely reject the analysis. **Time:** 30 minutes.

### Exercise 9 — Write the release note (design, 45 minutes)

Write a one-page forecast release note for a weekly replenishment decision on a slow-moving item with a
four-week lead time, meeting the five-field reporting standard in Step 23.

**Deliverable:** the note, plus a short paragraph on which field is most likely to be omitted by habit and
why. **Acceptance criteria:** all five fields present; the central interval has a nominal coverage and a
horizon; calibration evidence includes a count and a window; the functional and probability are named;
at least three exclusions are listed. **Time:** 45 minutes.

### Exercise 10 — Design the exclusions list (design, 40 minutes)

For a promotional forecast for a consumer good, list the exclusions the predictive distribution should
declare. For each, state whether it can be modelled, needs a scenario, or must simply be declared.

**Deliverable:** a table of at least ten exclusions with a treatment and a consequence if ignored.
**Acceptance criteria:** at least three exclusions are exogenous and cannot be modelled from history; the
treatment column distinguishes modelled, scenario, and declared; the consequence column is signed.
**Time:** 40 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

(1) **Hospital reorder point, 0.98 CSL.** A single quantile, the 0.98 quantile of lead-time demand over
the replenishment lead time in days; store the quantile plus the calibration evidence. (2) **97% fill
rate.** An expectation, the solution of `1 − E[(D−Q)⁺]/E[D] = 0.97`, needing both the target and the order
quantity because fill rate depends on both; store the distribution or the tail expectation as a function
of `Q`. (3) **Capacity reservation.** A high quantile over the reservation period, typically 0.90 to
0.99, chosen from the cost of unused capacity against unmet demand; store the distribution over that
horizon plus the two costs. (4) **Provision.** A quantile of the cost distribution, or an expectation over
the lower tail of demand, over the remaining product life; store the distribution over the residual life
plus the valuation rule.

A submission that gives "the forecast" for all four has not done the exercise. The distinguishing feature
is that row 2 is not a quantile at any level.

### 13.2 Exercise 2

Cumulative probabilities: `0.15, 0.30, 0.50, 0.70, 0.82, 0.91, 0.96, 0.99, 1.00`.

```text
mean = 0(0.15) + 10(0.15) + 20(0.20) + 30(0.20) + 40(0.12)
     + 60(0.09) + 80(0.05) + 120(0.03) + 160(0.01)
     = 0 + 1.5 + 4.0 + 6.0 + 4.8 + 5.4 + 4.0 + 3.6 + 1.6 = 30.9 units

median        = 20, because F(20) = 0.50 >= 0.50
0.80 quantile = 40, because F(30) = 0.70 < 0.80 and F(40) = 0.82 >= 0.80
0.95 quantile = 80, because F(60) = 0.91 < 0.95 and F(80) = 0.96 >= 0.95

E[(D-20)+] = 10(0.20) + 20(0.12) + 40(0.09) + 60(0.05) + 100(0.03) + 140(0.01)
           = 2 + 2.4 + 3.6 + 3.0 + 3.0 + 1.4 = 15.4
fill(20)   = 1 - 15.4/30.9 = 0.502

E[(D-40)+] = 20(0.09) + 40(0.05) + 80(0.03) + 120(0.01) = 1.8 + 2.0 + 2.4 + 1.2 = 7.4
fill(40)   = 1 - 7.4/30.9 = 0.761
```

The mean of 30.9 sits above the median of 20 because the distribution is right-skewed. The fill rate at
the median is 50.2%, close to the CSL of 50% by coincidence of this shape, while the fill rate at the
0.80 quantile is 76.1%, well below 80%. That gap is the subject of Exercise 3.

### 13.3 Exercise 3

Target: `E[(D−Q)⁺] = 0.05 × 30.9 = 1.545`.

```text
Q =  60: E[(D-60)+]  = 20(0.05) + 60(0.03) + 100(0.01) = 1.0 + 1.8 + 1.0 = 3.8
Q =  80: E[(D-80)+]  = 40(0.03) + 80(0.01) = 1.2 + 0.8 = 2.0
Q = 120: E[(D-120)+] = 40(0.01) = 0.4
```

Interpolating between 80 (2.0) and 120 (0.4) for a target of 1.545: the fraction along is
`(2.0 − 1.545)/1.6 = 0.284`, so `Q ≈ 80 + 0.284 × 40 ≈ 91.4` units.

The implied quantile level lies between `F(80) = 0.96` and `F(120) = 0.99`, so a 95% fill rate corresponds
to roughly the 0.97 quantile, not the 0.95 quantile. The fill rate is a ratio of expectations: it weights
the shortfall by its size, so rare large shortfalls above the upper support points matter. The mapping
between a fill-rate target and a quantile therefore depends on the order quantity, the mean, and the shape
of the upper tail.

### 13.4 Exercise 4

Defects: the functional of "12,400" is unstated; "±15%" has no nominal coverage; the interval may be
symmetric by construction, which is wrong for skewed demand; the horizon is "next quarter" but the
decision cadence and lead time are not stated; the dependence between periods is unstated; no calibration
evidence and no sample size are given; the exclusions are unstated; and "confidence range" invites a
confidence-interval reading, which is a statement about a parameter rather than a future outcome.

It cannot support a cost-based order or a fill-rate target, because the shape and tail expectations are
unknown, nor a capacity reservation above the stated upper bound. It can weakly support a budget at rough
precision, if the mean is the intended functional and the interval is read as a rough range.

A rewrite: "Three-month demand to the end of 2026-Q1 is forecast at 12,400 units (mean). The 80% central
prediction interval is 10,900 to 14,100 units, calibrated on 24 rolling origins with 79% empirical
coverage. The 0.95 quantile is 15,300 units. Excludes the February promotion and the March supplier
shutdown. Owner: category demand planning." The numbers are illustrative; the structure is the answer.

### 13.5 Exercise 5

The model's 90% interval of ±8% implies a standard deviation of about `8/1.645 = 4.9%` of the point if
normal. The backtest residual standard deviation is twice that, about 9.8%, so the residual-based interval
should be roughly twice as wide before any other correction. The ensemble spread of ±5% is not a standard
deviation and should not be added arithmetically; it is a lower bound on structural uncertainty.

A defensible decomposition: observation noise contributes about 4.9% and is already in the model;
parameter error contributes between 1% and 3% depending on the sample; structural error is at least the
ensemble spread of 5%; exogenous uncertainty is dominated by the promotion, which cannot be represented as
a variance. A widened 90% interval of roughly ±14% before the promotion, with the promotion handled as a
two-component mixture, is the shape of a good answer. The acceptance point is not the number: it is that
the answer names which sources are inside the interval and which are outside.

### 13.6 Exercise 6

```text
coverage       = 26/36 = 0.722
standard error = sqrt(0.90 * 0.10 / 36) = sqrt(0.0025) = 0.050
distance       = (0.722 - 0.900) / 0.050 = -3.56 standard errors
```

A shortfall of 3.6 standard errors is large enough to act on, though dependence between overlapping
origins means the effective sample is smaller than 36 and the true distance is larger. If the misses are
concentrated above the upper bound, the PIT histogram is U-shaped but asymmetric, with more mass near 1
than near 0. A symmetric U-shape indicates intervals uniformly too narrow; an asymmetric one indicates a
thin upper tail specifically, the signature of a model that captures ordinary variation and misses the
events that produce stock-outs.

### 13.7 Exercise 7

```text
30 items: 0.97^30 = exp(30 * ln 0.97) = exp(-0.9138) = 0.401
 8 items: 0.97^8  = exp(-0.2437) = 0.784
 3 items: 0.97^3  = 0.9127
```

Positive correlation raises the joint probability toward the marginal value, because the items fail
together rather than independently; perfect correlation gives 0.97 for any number of items. The sample-based
computation is to hold coherent draws across the family, count the draws in which every item's demand is at
or below its order-up-to level, and divide by the number of draws. That count is the joint service level,
and it is available only because the draws are jointly coherent.

### 13.8 Exercise 8 rubric

A strong critique concedes that the scenario set is a legitimate tool for a fifteen-year capacity
decision, because the value is in widening the range of futures and testing robustness. It then identifies
three problems. The weights are judgement, not measurement, and publishing them as a distribution converts
an assumption into a number that will be treated as evidence. The set is almost certainly not exhaustive,
so the weighted result is a mixture over three points, not a distribution over the future. And a long-lived
capacity decision needs a robustness criterion, such as minimax regret across scenarios, rather than an
expected value from weights nobody can defend.

A strong recommendation: use the scenarios to test the plan's performance under each, report the cost or
regret under each, and choose a plan acceptable across the set. Treat the 50/30/20 weights as a sensitivity,
not as a probability. A weak critique says only that scenarios are not probabilities and stops, which is
true and does not help the decision.

### 13.9 Exercise 9 rubric

A strong release note has all five fields in a fixed order. The central interval carries a nominal coverage
and a horizon. The calibration evidence carries a count, a window, and a segmentation, not just a
percentage. The functional is named with its probability. The exclusions are specific and dated. The
paragraph on habit should identify the exclusions field as the most often omitted, because exclusions
require knowing what changed in the business, and that knowledge usually sits outside the analytics team.
A weak note leads with a point forecast and adds a range as a courtesy.

### 13.10 Exercise 10 rubric

A strong table distinguishes three treatments. Modelled: own-price elasticity, seasonality, the base lift
from a known promotion mechanic. Scenario: competitor response, a new entrant, a regulatory change.
Declared: a supply disruption with no history, a packaging change, a listing decision. At least three
exogenous items are required, and the signed consequence column is the discriminator: a promotional
forecast that excludes competitor response is biased high, while one that excludes a supply constraint is
biased low on realised sales. A weak table lists generic risks with no treatment and no sign.

## 14. Interview questions

**1. "What is the difference between a point forecast and a predictive distribution?"** Strong: a point
forecast is a functional of the distribution, chosen by an implicit loss function, while the distribution
is the object from which every functional can be read; a mean is optimal for squared error and a quantile
for a service target, and compression is safe only when the decision reads the retained functional. Weak:
the distribution "includes uncertainty", which does not explain why the choice of summary changes the order
quantity.

**2. "A planner asks for the 95% service level. What do you give them?"** Strong: ask which service level,
because a cycle service level is the 0.95 quantile of lead-time demand while a fill rate is an expectation
over the shortfall that depends on the order quantity, and the two produce different quantities from the
same distribution. Weak: return the 0.95 quantile without asking.

**3. "Why do model prediction intervals usually undercover?"** Strong: give the four sources and note that
a residual-variance interval contains observation noise and sometimes parameter error but almost never
structural or exogenous uncertainty, cite the competition evidence that nominal 95% intervals can cover
far less, and name the remedies. Weak: the model is badly fitted, which is one cause among several and not
the most common.

**4. "What is calibration, and how would you test it?"** Strong: agreement between stated probabilities and
observed frequencies over repeated occasions, tested with coverage counts and sample sizes, coverage by
horizon and segment, and a PIT histogram whose shape diagnoses the direction of failure, paired with
sharpness because a calibrated interval can be uselessly wide. Weak: "check that the actuals fall inside
the interval", which is the first instrument and not the test.

**5. "A colleague sums item-level 0.95 quantiles to get the category 0.95 quantile. What is wrong?"**
Strong: quantiles are not additive, equality holds only under comonotonicity, and the sum of marginal
quantiles is generally too high because it assumes perfect positive dependence; the result is no quantile
of the category distribution, and the fix is to reconcile draws and then take quantiles. Weak: "it is an
approximation", which understates a bias that grows with the number of items.

**6. "What is the difference between a scenario set and a probability distribution?"** Strong: a scenario
set is a collection of internally plausible narratives usually without relative likelihoods, a distribution
assigns a probability to every event, scenario planning withholds probabilities deliberately, and a
distribution is required when the decision reads a quantile; the bridge is to assign weights as a stated
assumption, at the cost of turning judgement into apparent evidence. Weak: treat them as two formats for
the same content.

**7. "How would you decide between a quantile grid and a full distribution?"** Strong: ask what the decision
needs, because a service target needs one quantile and a grid is more than enough, a cost decision needs
tail expectations and hence the shape, and a hierarchy decision needs a joint object, usually samples; note
that a grid is easier to store and that crossing must be checked. Weak: the full distribution is always
better, which ignores storage and governance cost.

**8. "Your 90% interval covers 72% of outcomes. What do you do?"** Strong: compute the standard error, check
whether the shortfall exceeds sampling variation, break coverage down by horizon and segment to locate the
failure, inspect the PIT histogram for asymmetry, then choose between widening the interval empirically and
decomposing the misses, recording the choice. Weak: widen the interval by a guessed factor.

**9. "What does it mean for a hierarchical probabilistic forecast to be coherent?"** Strong: distinguish
marginal coherence, where each quantile vector sums, from probabilistic coherence, where the joint
distribution lives on the coherent subspace; only the second guarantees that every functional computed from
the forecasts is coherent, and sample-based reconciliation achieves it by construction. Weak: "the children
add up to the parent", which is the point-forecast definition.

**10. "Why can a well-calibrated forecast still be a bad forecast?"** Strong: the infinite interval covers
everything with the stated probability and carries no information; sharpness is the second criterion, the
proper scoring rules trade the two, and the reporting standard requires both numbers. Weak: calibration is
not everything, without being able to say what the other thing is.

**11. "How do you handle a forecast conditional on no promotion when a promotion is planned?"** Strong: the
distribution must be of the variable the decision faces, so model the promotion as a mixture component with
a stated probability and lift distribution rather than widening the interval, because widening changes the
spread and not the shape; a mixture can be bimodal and its upper quantile can be far above either
component's. Weak: add a percentage to the forecast.

**12. "What would you put in a one-page release note for a probabilistic forecast?"** Strong: the five
fields, namely the central prediction interval with nominal coverage, the horizon, calibration evidence with
counts and window and segmentation, the decision functional with its probability, and the exclusions, which
are the field most often omitted and the one that most often explains a surprise because they record what
the model was never asked to see. Weak: a point forecast, an accuracy number, and a chart.

## 15. Key terms

- **Predictive distribution** — the probability distribution of a future quantity given present information.
- **CDF** — cumulative distribution function, `F(d) = P(D ≤ d)`.
- **Quantile** — `F⁻¹(q)`, the value below which the variable falls with probability `q`.
- **Prediction interval** — a pair of quantiles intended to contain the outcome at a stated coverage.
- **Central prediction interval** — a prediction interval with equal tail probabilities.
- **HDI** — highest density interval, the shortest interval at a stated coverage.
- **Nominal coverage** — the probability an interval claims.
- **Empirical coverage** — the fraction of outcomes that actually fell inside.
- **Calibration** — agreement between stated probabilities and observed frequencies.
- **Sharpness** — the concentration of the predictive distribution; narrow is sharp.
- **PIT** — probability integral transform, `F(y)` at the realised outcome; uniform if calibrated.
- **Reliability diagram** — observed frequency against forecast probability for a binary event.
- **Proper scoring rule** — a score optimised only by reporting the true predictive distribution.
- **CRPS** — continuous ranked probability score, the integral of pinball loss over all levels.
- **Pinball loss** — the asymmetric absolute loss minimised by a stated quantile.
- **Critical fractile** — `C_u/(C_u+C_o)`, the cost-minimising quantile.
- **Cycle service level** — the probability of no stock-out in a replenishment cycle; a quantile.
- **Fill rate** — the fraction of demand units served from stock; an expectation, not a quantile.
- **Expected shortage per cycle** — `E[(D−Q)⁺]`, the mean unmet demand at level `Q`.
- **Lead-time demand** — demand realised between order placement and receipt.
- **Observation noise** — irreducible period-to-period variation.
- **Parameter error** — error from estimating parameters on a finite sample.
- **Structural error** — error from the model family being wrong.
- **Exogenous uncertainty** — uncertainty about future drivers outside the model.
- **Scenario set** — a collection of plausible futures, usually without probabilities.
- **Probabilistic coherence** — the joint predictive distribution lies on the aggregation subspace.
- **Marginal coherence** — each quantile vector sums across the hierarchy; weaker than the above.
- **Quantile crossing** — estimated quantiles out of order, so the object is not a distribution.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). "Probabilistic Forecasts, Calibration and Sharpness." *Journal of the Royal Statistical Society: Series B*, 69(2), 243–268. Grade A.
- Grushka-Cockayne, Y., and Jose, V. R. R. (2019). "Combining Prediction Intervals in the M4 Competition." *International Journal of Forecasting*; SSRN 3329413. Grade A.
- Makridakis, S., Hyndman, R. J., and Petropoulos, F. (2020). "Forecasting in Social Settings: The State of the Art." *International Journal of Forecasting*, 36(1), 15–28. DOI: 10.1016/j.ijforecast.2019.05.011. Grade A.
- Makridakis, S., and Petropoulos, F. (2020). "The M4 Competition: Conclusions." *International Journal of Forecasting*. Grade A.
- Bruzda, J. (2018). "Multistep Quantile Forecasts for Supply Chain and Logistics Operations." *Central European Journal of Operations Research*. DOI: 10.1007/s10100-018-0591-2. Grade A.
- Ngartera et al. (2026). "Stochastic Inventory Optimization with Coherent Risk Measures." *Journal of Risk and Financial Management*. DOI: 10.3390/jrfm19030173. Grade A; numerical results reported as the published abstract states them, not independently replicated.
- Goodwin, P., and Wright, G. (2025). "Can Narrative-Based Scenarios Support Quantitative Judgmental Forecasting?" *Futures and Foresight Science*. Grade A.
- Han, Y., Dasgupta, S., and Ghosh, J. (2021). "Simultaneously Reconciled Quantile Forecasting of Hierarchically Related Time Series." *AISTATS 2021*, PMLR 130. Grade A.
- Rangapuram, S. S., et al. (2021). "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series." *ICML 2021*, PMLR 139. Grade A.
- Rangapuram, S. S., et al. (2023). "Coherent Probabilistic Forecasting of Temporal Hierarchies." *AISTATS 2023*, PMLR 206. Grade A.
- Jeon, J., Panagiotelis, A., and Petropoulos, F. (2019). "Probabilistic Forecast Reconciliation with Applications to Wind Power and Electric Load." *European Journal of Operational Research*. Grade A.

**Working papers, preprints, and theses — Grade B**

- Wang, X., Kang, Y., and Petropoulos, F. "Combining Probabilistic Forecasts of Intermittent Demand." arXiv:2304.03092. Grade B; preprint, consult the published version before citing.
- Wang, X., Kang, Y., Petropoulos, F., and Li, F. (2021). "The Uncertainty Estimation of Feature-Based Forecast Combinations." arXiv:1908.02891. Grade B.
- Magnus, J. R., and Vasnev, A. L. (2023). "On the Uncertainty of a Combined Forecast: The Critical Role of Correlation." Grade B; venue not confirmed from the search material.
- Hyndman, R. J., et al. "Probabilistic Forecasts in Hierarchical Time Series." Working paper, robjhyndman.com. Grade B.
- Ha Duong, M. (2003). "Imprecise Probability Bridges Scenario-Forecast Gap." Working paper, CIRED/CNRS and Carnegie Mellon. Grade B.
- Thorarinsdottir, T. L., and Schuhen, N. (2017). "Verification: Assessment of Calibration and Accuracy." NR report SAMBA/17/17. Grade B.
- Gudum, C. K. (2002). "A New Compound Lead Time Demand Distribution Approach and a Comparison Study." PhD thesis, Copenhagen Business School. Grade B.

**Standards and official — Grade B**

- ISO 3534-1, *Statistics — Vocabulary and Symbols — Part 1*. Current edition. Grade B.
- ISO 5725-1, *Accuracy (Trueness and Precision) of Measurement Methods and Results — Part 1*. Grade B.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM), 3rd edition. Grade B.
- ISO 80000-1, *Quantities and Units — Part 1: General*. Grade B.
- ISO 8601, *Date and Time — Representations for Information Interchange*. Grade B.
- IAS 2, *Inventories*; IAS 37, *Provisions*; IFRS 15/ASC 606, *Revenue from Contracts with Customers*. IFRS Foundation and FASB. Grade B.
- Sarbanes-Oxley Act of 2002, Section 404. Grade B.
- Board of Governors of the Federal Reserve System, SR 11-7, *Guidance on Model Risk Management*. Grade B; cited by analogy, not as an obligation for non-bank entities.
- Behar, A., and Hlatshwayo, S. (2021). *Guidance Note: How to Implement Strategic Foresight (and Why)*. IMF Analytical Note. Grade B.

**Software documentation and reference material — Grade B or C**

- Bureau of Meteorology, `scores` library, PIT tutorial and API documentation. Grade B for the implementation; not evidence of general best practice.
- ECMWF Forecast User Guide, Section 12.B, "Statistical Concepts — Probabilistic Data." Grade B.
- Tibshirani, R. (2023). StatLearn lecture notes on forecast scoring and calibration, UC Berkeley. Grade C; teaching material.

**Practitioner, industry, and press — Grade C, D, or E**

- APICS/ASCM, "Cycle Service Level and Fill Rate" industry discussion document. Grade C.
- Ramírez, R., Churchhouse, S., Palermo, T., and Hoffmann, J. (2017). "Using Scenario Planning to Reshape Strategy." *MIT Sloan Management Review*. Grade C.
- Kvantis Atlas, "Probabilistic Forecasting" and "Service Level and Fill Rate." Vendor knowledge base, accessed 2026. Grade D.
- TimesFM repository, `quantile.md` quantile-head documentation. Grade D.
- InventoryPath, "Service Level vs Fill Rate: What Your 98% Actually Promises." Grade D.
- Scoblic, J. P., and Tetlock, P. E. (2020). "A Better Crystal Ball." *Foreign Affairs*. Grade E; press essay.
- Open practitioner discussion threads on safety stock with a fill-rate criterion. Grade E; useful for seeing what practitioners ask, not for the formulas.

**Note on citations.** Details were checked against the sources returned by the searches; volume and page
numbers appear where confirmed and are omitted where not. The M4 and coverage figures attributed to
Grushka-Cockayne and Jose, Makridakis et al., Bruzda, and Ngartera et al. are reported as those sources
state them and were not independently replicated. The distribution in section 6 is constructed and
internally consistent. Where the search material did not confirm a venue, the entry says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_compound-poisson-hurdle.md`](06_compound-poisson-hurdle.md)
- Next topic: [`08_conformal-prediction.md`](08_conformal-prediction.md)
- Later in this module: [`09_scoring-rules.md`](09_scoring-rules.md), [`10_quantiles-decisions.md`](10_quantiles-decisions.md)
- Hierarchy structures this topic assumes: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- Reconciliation mechanics: [`02_aggregation-consistency.md`](02_aggregation-consistency.md)
- Intermittent demand and method families: [`05_intermittent-classification.md`](05_intermittent-classification.md)
- Point-error metrics and the critical fractile: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Benchmarks and forecastability: [`../07_time_series_forecasting/01_forecastability-benchmarks.md`](../07_time_series_forecasting/01_forecastability-benchmarks.md)
- Aggregation before reconciliation: [`../07_time_series_forecasting/05_hierarchy-aggregation.md`](../07_time_series_forecasting/05_hierarchy-aggregation.md)
- Intermittent metrics and the zero problem: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Demand and lead-time distributions: [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Probability and the bootstrap: [`../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md`](../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md)
- Survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- 01A module map: [`../../01_foundations/01A_foundations_units_time_assumptions/_index.md`](../../01_foundations/01A_foundations_units_time_assumptions/_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Return to the part map: [`../_index.md`](../_index.md)
