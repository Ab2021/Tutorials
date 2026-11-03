# Baseline and Incrementality

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 05; Module 09 topics 02–03
> Version 2026.5

---

## 1. Why this topic exists

Every promotion review contains the same sentence, and it is almost always wrong: "We sold
1,860 units a week during the event, against 980 before, so the promotion delivered a 90% lift."

The 1,860 is a fact. The 980 is a fact. The subtraction is arithmetic. The claim is a causal
claim, and it rests entirely on a proposition nobody in the room has stated: that 980 is what
would have sold had the promotion not run. That number does not exist in any system, and it was
never recorded, because the promotion did run. Everything in this topic is an attempt to
construct it honestly, and to say out loud how much the construction costs.

The size of the prize explains why the argument is worth having. Trade spend is routinely
described as the second-largest line on a consumer goods profit and loss statement after cost of
goods sold (COGS), at roughly 15–25% of gross revenue in the vendor and consultancy sources that
report it. A company turning over a billion in gross sales is therefore spending somewhere
between 150 and 250 million on promotions, and deciding next year's allocation from a measurement
that may be wrong by a factor of two.

Three failures recur, and they motivate the rest of this file.

**Failure one: the pre-period average is treated as a fact about a world that did not happen.**
It is a fact about the twelve weeks before the event, which were themselves shaped by the event.
Shoppers who knew the promotion was coming waited for it. The retailer may have run a competing
mechanic two weeks earlier. The preceding month may have been the coldest on record. The
pre-period average measures the past; it does not model an alternative present, and the gap
between those two things is the whole subject.

**Failure two: last year's same week is treated as seasonally clean.** It is clean only if
distribution, assortment, shelf price, pack architecture, competitive set, and store count all
held. In a category where a new pack size launched in March, a year-over-year comparison is
measuring the pack launch and calling it a promotion.

**Failure three: promoted weeks are compared to unpromoted weeks as if the retailer chose those
weeks at random.** The retailer chose them because it expected demand to be high — a holiday, a
payday, a back-to-school window. The manufacturer often chose the deal date to land in a quarter
that needed shipments. Both choices are correlated with the outcome the analysis is trying to
explain.

A fourth reason is institutional rather than statistical. **The number that gets reported is the
number that survives a deduction dispute, not the number that is closest to the truth.** A
manufacturer and a retailer arguing over a 40,000-unit claim need a baseline both sides can
recompute in an afternoon. That requirement selects for simple, deterministic, explainable
baselines and against the estimators an econometrician would prefer. Section 9 develops the
tension; it is the most useful thing to understand about why industry and academic practice
diverge on the same question.

### 1.1 What this topic is, and what it is not

This topic owns **the counterfactual**: how to define it, which construction to use, what each
construction assumes, and how to report a lift so that the reader knows which one was used.

Topic 02 decomposed a sales series into baseline, seasonality, promotion, distribution, and
price. Topic 03 built the event table that records what was promoted, where, when, and at what
shelf price. This topic takes those two inputs and answers the question the decomposition was
always setting up: given that a promotion happened, what did it cause?

Topic 05 takes the number this topic produces and adjusts it — for cannibalisation, halo, and
pull-forward — to turn lift into profit. Topic 07 replaces the single average effect with a
distribution of effects across stores and shoppers. Module 05 owns the design statistics:
randomisation units, cluster variance, power, and switchback inference are taught there and are
cited here, not re-derived.

> **Note:** The boundary between this topic and topic 05 is deliberate but porous. Pull-forward
> is named here because it changes the evaluation window, which is a baseline decision. It is
> owned there because it changes the profit calculation, which is a different decision. If you
> are reading only one of the two, read this one for the window and that one for the money.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** lift as a difference against a named counterfactual, and **state** the counterfactual
   explicitly in any lift figure you report, including the units, the window, and the unit of
   analysis.
2. **Distinguish** the five constructions that are routinely called "the baseline" — pre-period
   average, same period last year, model-based no-promotion forecast, matched control store, and
   randomised holdout — and **state** what each assumes and which direction each biases.
3. **Demonstrate**, with arithmetic, why the pre-period average overstates lift in the presence of
   anticipation, stockpiling, and retailer week selection, and **quantify** the overstatement on a
   small dataset.
4. **Order** the causal designs from before-after through difference-in-differences, synthetic
   control, matched markets, and randomised or switchback holdouts by credibility, and **justify**
   the ordering by naming the assumption each rung removes.
5. **Diagnose** promoted-week selection from evidence in the data — pre-period level differences,
   calendar clustering, deal-date clustering — and **explain** why it inflates naive lift.
6. **Compute** a post-promotion trough, **distinguish** it from a separate anomaly, and **choose**
   an evaluation horizon that includes it with a stated justification.
7. **Separate** the baseline used to measure lift from the baseline used to forecast demand, and
   **explain** why the two have different error structures and different acceptance tests.
8. **Write** a lift report to the reporting standard in section 5.8: point estimate with an
   interval, the counterfactual method named, and the assumptions listed.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one question in your head, and refuse to let it be replaced by a number:

> **What would this store, this product, this week, have sold if the promotion had not run?**

Everything else is machinery for answering it. The machinery is ordered by how much of the answer
it removes from the realm of assumption. Draw the ladder and you can place any real analysis on
it within thirty seconds of opening the file.

```text
   THE COUNTERFACTUAL LADDER
   climb only as far as the data lets you, and always say which rung you are on

   RUNG 5  RANDOMISED HOLDOUT             assumes: assignment is independent of what
           store x week, or switchback            would have happened anyway, and one
           (Module 05, topics 02 and 05)          unit's outcome does not depend on
                                                  another's treatment
                                                  buys: selection and confounding removed
                                                  by design rather than by model

   RUNG 4  MATCHED-MARKET COMPARISON      assumes: the matched markets share the treated
           untreated market set                   market's trend, and shoppers do not
                                                  cross the boundary between them
                                                  buys: common shocks removed; local
                                                  shocks remain

   RUNG 3  SYNTHETIC CONTROL              assumes: a weighted donor pool can reproduce
           donor pool, weights fitted             the treated unit's pre-period path and
           on the pre-period only                 would have continued to track it
                                                  buys: unit-specific trend removed, if
                                                  and only if the pre-period fit is good

   RUNG 2  DIFFERENCE-IN-DIFFERENCES      assumes: parallel trends — absent the promotion,
           treated stores vs control              the gap between the two groups stays
           stores, before and after               constant through the event
                                                  buys: calendar and market shocks shared
                                                  by both groups

   RUNG 1  BEFORE-AFTER                   assumes: nothing else changed but the promotion
           promoted stores vs their own
           pre-period average                     buys: nothing. It removes the promotion's
                                                  own history and no more.
```

The ladder is not a claim that rung 5 is always better. It is a claim that **the credibility of a
lift number is the credibility of the assumption you had to make to get it**, and that different
rungs buy you different assumptions at different costs. Rung 5 costs you the ability to promote
the stores you most want to promote. Rung 4 costs you a market and assumes no spillover. Rung 3
costs you a donor pool and a fit you can defend. Rung 2 costs you a control set and an untestable
parallel-trends claim. Rung 1 costs you almost nothing and buys you almost nothing, which is
precisely why it is the most common rung in industry.

### 3.1 Lift is a difference, and the difference has a decomposition

Write the naive estimate and expand it:

```text
  naive lift   =  actual sales        -  pre-period average
  true lift    =  actual sales        -  true counterfactual

  pre-period average  =  true counterfactual
                       +  anticipation bias        (pre-period depressed -> baseline too low)
                       -  stockpiling carry-in     (pre-period inflated -> baseline too high)
                       +  calendar mismatch        (holiday or weather not present in the event)
                       +  trend drift              (category growing or declining)
                       +  distribution change      (stores, facings, or pack sizes moved)
                       +  price and assortment     (regular price or range changed)

  therefore:  naive lift = true lift + (counterfactual - pre-period average)
```

The last line is the whole topic in one equation. **The error in a naive lift is exactly the
error in the baseline, with the sign flipped.** Every method in section 5 is an attempt to make
that parenthetical term small, and every diagnostic in section 10 is an attempt to see how large
it still is.

Three consequences follow. **The sign of the bias is not fixed.** Anticipation depresses the
pre-period and inflates naive lift. Stockpiling carry-in inflates the pre-period and deflates it.
When both are present they partly cancel, and the cancellation is not a reason to relax: your
error is small by luck, not by method, and it will not be small on the next event.

**A baseline is a claim about a counterfactual, not a summary of history.** Two analysts can
compute the same pre-period mean and be making different claims with it, because one is implicitly
asserting "nothing would have changed" and the other "the calendar would have repeated."

**The window is part of the estimand.** Lift over four promotion weeks and lift over eight weeks
including the trough are different quantities, not the same quantity measured more precisely.
Section 6 shows one event at +61% on the event weeks and +17% over eight weeks. Neither is wrong;
reporting one while calling it "the lift" is.

### 3.2 Why promoted weeks are not a random sample of weeks

This is the part of the mental model that most analytics teams skip, and the part that matters most.

A retailer's promotion calendar is a plan, not a draw. Category managers place events where they
expect traffic — holidays, paydays, back-to-school, weather-driven category peaks — and where they
can execute, which means weeks when the distribution centre is not already overloaded.
Manufacturers push deal dates toward the end of a fiscal quarter when shipment targets are at
risk. Neither party is dishonest; both are doing their job. The joint result is that **the
treatment is assigned on the basis of the potential outcome**, which is the definition of
selection on unobservables.

The literature names this in several ways: *timing selection* in retail analytics, *treatment
endogeneity* in econometrics, and in recent marketing-science work on promotion cycles, the
observation that cyclic promotions endogenously sort consumers into waiting and purchasing
states, so estimated elasticity varies over the cycle even when preferences are static, and
experiments run out of phase with the natural promotion rhythm can bias the estimate predictably.

The implication is blunt: **comparing promoted weeks to unpromoted weeks estimates the difference
between two kinds of week, not the effect of the promotion.** Difference-in-differences against
control stores removes the part of that difference common to both groups in the same calendar
weeks — the holiday. The naive comparison does not. That is why the two numbers in section 6
differ by 29 points, and why those points are almost entirely the holiday the retailer chose.

### 3.3 Two baselines, two jobs

The word "baseline" is used for two objects that share a name and nothing else.

| | Measurement baseline | Forecast baseline |
|---|---|---|
| Question | What did the promotion cause? | What will sell next week? |
| Time direction | Retrospective, one event | Prospective, rolling |
| Unit of analysis | The promoted store-week, or the event aggregate | The stock-keeping unit (SKU)-location-week you will replenish |
| Includes promotion? | Deliberately excludes it | Must include expected future promotions |
| Error that matters | Bias — a systematically wrong counterfactual | Variance and bias, weighted by inventory cost |
| Acceptance test | Pre-trend fit, placebo tests, design credibility | Out-of-sample error on a rolling backtest |
| Who consumes it | Finance, sales, the retailer's claim team | Supply planning, replenishment, sales and operations planning (S&OP) |
| Consequence of error | Wrong allocation decision, disputed deduction | Stock-out or excess inventory |

The measurement baseline must be a world without the promotion; the forecast baseline must be a
world with every promotion you expect to run. They are opposite objects, and confusing them
produces the two most common reporting errors in the field: a "baseline forecast" that quietly
excludes known future promotions and under-forecasts every event week, and a "lift" computed
against a forecast that already contained the promotion and reports near zero.

> **Note:** Module 07 covers the forecast baseline as a modelling problem, including lag structure
> and Fourier seasonality. This topic covers it only to keep the two objects from being confused.

## 4. Definitions and vocabulary

The table below is the working vocabulary for the rest of the module. The "common misuse" column
is not pedantry: each entry names a substitution that changes the answer, and in most cases
changes it in a predictable direction.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Counterfactual | The sales that would have occurred in the promoted unit and window had the promotion not run | "Baseline" used to mean a measured period rather than a constructed alternative | Units per store per week |
| Lift | Actual sales minus the counterfactual, over a stated window | Used interchangeably with "growth", "uplift", and "return" | Units, or per cent of counterfactual |
| Incrementality | The share of observed sales caused by the promotion | Treated as a synonym for lift when lift still contains seasonal error | Per cent |
| Measurement baseline | The counterfactual used to size an event that has already run | Conflated with the forecast baseline | Units per store per week |
| Forecast baseline | The expected level a forecast model predicts absent promotion, used for planning | Conflated with the measurement baseline | Units per week, carrying a forecast error |
| Promoted sales | Everything scanned while the promotion was live | Read as if all of it were incremental | Units |
| Baseline sales | Expected sales absent the promotion over the same window | Assumed without argument to be the pre-period mean | Units |
| Pre-period average | Mean of the n weeks before the event | Called "the baseline" with no qualifier | Units per store per week |
| Same period last year | The equivalent calendar weeks in the prior year | Called a "seasonally adjusted baseline" | Units per store per week |
| Model-based baseline | A no-promotion forecast from a model fitted on non-promoted periods | Credibility assumed from in-sample R-squared | Units per store per week |
| Control store | An unpromoted store used as a comparator | Any unpromoted store, with no matching diagnostics | Store identifier |
| Matched market | An untreated geography matched on size, mix, and trend | Called a "control region" with no evidence of matching | Market identifier |
| Donor pool | The set of candidate untreated units combined into a synthetic control | Treated as a peer group rather than a weighted fit | List of unit identifiers |
| Synthetic control | A weighted combination of donors whose pre-period path matches the treated unit | Described as "a control store we built" | Weights, non-negative, summing to one |
| Difference-in-differences (DiD) | Change in the treated group minus change in the control group | Called "lift versus control" without stating the parallel-trends claim | Percentage points |
| Parallel trends | Absent treatment, treated and control groups would have moved in parallel | Asserted after eyeballing two lines on one chart | None; it is an assumption, not a statistic |
| Randomised holdout | A randomly selected subset of stores or weeks denied the promotion | Called a "control group" when assignment was not random | Share of units, commonly 10–20% |
| Switchback | Treatment alternates over time within the same unit | Mistaken for a before-after comparison | Period length, e.g. two weeks |
| Treatment | The promotion as the shopper experienced it, ideally the shelf price and display | The funded discount substituted for the shelf price | Currency per unit, or per cent off shelf |
| Selection bias | Promoted units or weeks differ systematically from unpromoted ones | Named "endogeneity" and then left unaddressed | None; expressed as a bias in units |
| Timing selection | The retailer's choice of which weeks to promote is correlated with expected demand | Invisible in the analysis because the calendar is taken as given | Week index |
| Stockpiling | Shoppers buy ahead and hold inventory | Referred to as "lift that did not really happen" | Units per household |
| Pull-forward | Demand moved from a later period into the promoted period | Counted as incremental | Units |
| Post-promotion dip | Sales below baseline in the weeks after the event | Reported as a separate curiosity | Units per store per week |
| Pantry loading | The household-level form of stockpiling, concentrated in storable categories | Applied to categories where storage is not plausible | Units per household |
| Forward buying | A retailer buying on deal and holding the inventory | Conflated with shopper stockpiling | Cases, or units |
| Anticipation effect | Sales fall before an announced promotion as shoppers wait | Mistaken for a weak pre-period or a data error | Units |
| Cannibalisation | The promoted SKU's gain taken from the same manufacturer's other SKUs | Sign reversed, or folded into "halo" | Units |
| Halo | Gain in complementary SKUs caused by the promotion | Assumed rather than measured | Units |
| Evaluation window | The weeks included in the lift calculation | Chosen after seeing the result | Weeks |
| Evaluation horizon | The period over which the promotion should be judged on commercial grounds | Used interchangeably with the window | Weeks |
| Incremental return on ad spend (iROAS) | Incremental gross profit or revenue per unit of promotional spend | Called return on investment (ROI) with no cost basis stated | Currency per currency |
| Temporary price reduction (TPR) | A time-limited cut to the shelf price | Used to mean the funded deal | Per cent off shelf |
| Stable unit treatment value assumption (SUTVA) | One unit's outcome does not depend on another unit's treatment | Assumed silently, especially in geo tests | None |
| Pre-trend test | A check that treated and control tracked before the event | Run on a window contaminated by anticipation | A plot, or a joint test statistic |
| Placebo test | Re-running the method on a period with no promotion | Rarely performed | An estimate that should be near zero |
| Counterfactual forecast error | The baseline model's error on genuinely unpromoted periods | Assumed to be zero | Units, or per cent |

Two of these deserve a sentence more than a table cell.

**Counterfactual is not a synonym for baseline.** A baseline is a number. A counterfactual is a
statement about a world. When you write "baseline = 980", you have written a number and hidden
the statement. When you write "counterfactual: the twelve promoted stores would have sold 1,154
units per week, estimated by difference-in-differences against twelve matched control stores over
an eight-week pre-period", you have written the statement, and a reader can disagree with it.

**Incrementality is not lift.** Lift is measured against a counterfactual and can be computed for
a single store-week. Incrementality is the causal share and is only meaningful once you have
decided what counts as the population — one SKU, one category, one manufacturer, one retailer.
A promotion can have positive lift and negative incrementality at the manufacturer level, because
the lift came from the manufacturer's own other SKUs.

## 5. The mechanics, step by step

The procedure below is the order in which the decisions actually bind. Most failed lift analyses
failed at step 1 or step 2, not at step 6.

### 5.1 Step 1 — Name the decision the number will feed

Before touching data, write one sentence: *"This number will decide X, and a wrong number of
magnitude Y would cause Z."*

The decision determines the unit of analysis, the window, and the acceptable error. Three
decisions that look similar and are not:

| Decision | Unit of analysis | Window | Error that matters |
|---|---|---|---|
| Renew the promotion in next year's joint business plan | Store-week aggregate across the account | Event plus trough | Bias in the mean |
| Settle a retailer deduction claim | Store-SKU-week, per the claim | Contract-defined | Agreement with the counterparty's arithmetic |
| Set the promoted volume for supply planning | SKU-DC-week | Shipment lead time | Upper tail, not the mean |

A supply planner needs the 90th percentile of the event week, not the mean lift. A claim team
needs a number the retailer will reproduce. A category manager needs an unbiased mean to compare
mechanisms. The same promotion produces three different correct answers.

### 5.2 Step 2 — Define the event precisely

An event is not "week 14". It is a tuple: SKU, store (or store set), start and end date, mechanic,
shelf price, display type, feature type, and funding. Topic 03 supplies this record. Four
precision failures propagate into the baseline. **Funded versus shelf price:** if the treatment
variable is the deal sheet discount rather than the observed shelf price, the estimated effect
belongs to an intention; use the observed shelf price where it exists, and where it does not,
state the pass-through assumption. **Mechanic boundaries:** a display running a week longer than
the price cut creates a week treated for display and untreated for price, so decide whether that
week is in the window and record the decision. **Overlapping events:** two mechanics in the same
store-week cannot be separated by any baseline method, so combine them into one treatment or
exclude the week, and say which. **Store-count changes:** a store that opened, closed, or was
re-modelled inside the window breaks the panel, so freeze the store set at the pre-period.

### 5.3 Step 3 — Choose the window before you see the result

Define four windows explicitly and write them down:

- **Anticipation window** — the weeks before the event in which shoppers may have deferred
  purchase because the event was announced. Typically one to two weeks for a planned feature,
  longer for a known seasonal event.
- **Event window** — the weeks the promotion was live.
- **Post-event window** — the weeks in which the trough is expected. Typically two to four weeks
  for a storable category, shorter for perishables.
- **Evaluation horizon** — the union of the three, which is the period over which you claim the
  promotion should be judged.

The horizon is a commercial judgement, and it must be argued rather than assumed. A promotion
that generates 60% lift on the event weeks and a full reversal within four weeks has created
almost no incremental volume; it has moved volume and paid a discount for the privilege. A
promotion that generates 20% on the event weeks and holds half of it for a quarter has created
real volume. The event-week number cannot distinguish these. Only the horizon can. Topic 05 owns
the profit consequence of the trough; this topic owns the window arithmetic, and the rule is that
a trough caused by the promotion belongs inside the effect being measured.

### 5.4 Step 4 — Establish what design you actually have

Work down this list and stop at the first line that is true. Do not claim a higher rung than the
data supports.

```text
  randomised holdout or switchback    -> rung 5, report holdout share and assignment unit
  matched untreated market set        -> rung 4, report match diagnostics
  donor pool, 8+ weeks, 20+ donors    -> rung 3, report pre-period fit and weight spread
  untreated control stores, same weeks -> rung 2, report the pre-trend plot
  none of the above                   -> rung 1, say so and report the 5.7 band
```

The order matters because each rung buys an assumption the rung below cannot. A synthetic
control on a badly matched donor pool is worse than a well-matched DiD, so rung 3 is not
automatically preferable to rung 2 — the ladder orders *by assumption removed when executed
correctly*, not by the sophistication of the estimator.

### 5.5 Step 5 — Build the pre-period and test it

The pre-period is the raw material for every rung above 1. Three tests, in this order.

**Length.** Eight weeks is a floor, not a target. Twelve to twenty-six weeks is common for
weekly data; shorter pre-periods cannot separate trend from noise. If a category is highly
seasonal and the event is annual, you may need a full year, which pushes you toward a
model-based or year-over-year approach because a 52-week pre-period with one event inside it is
itself contaminated.

**Contamination.** Remove from the pre-period any week that was promoted, any week inside an
anticipation window for a prior event, and any week with a distribution change. Then state how
many weeks remain. A twelve-week pre-period that loses four weeks to contamination is an
eight-week pre-period, and you should say eight.

**Pre-trend.** Plot the treated and control series for the pre-period and look at them. This is
not a formality. A pre-trend divergence means DiD will not identify anything, and it is visible
to the eye in most cases where it exists. Statistical pre-trend tests exist but have low power in
short panels, so a non-significant test is weak evidence and a visible divergence is strong
evidence.

> **Note:** The pre-trend test is itself biased when the anticipation window sits inside the
> pre-period, because the treated stores are already responding to the coming promotion. Exclude
> the anticipation window from the pre-trend check, or the test will reject good designs and pass
> bad ones.

### 5.6 Step 6 — Construct the baseline

The arithmetic for each rung, stated so you can reproduce it by hand.

**Rung 1 — pre-period average.**

```text
  baseline      = mean(units, weeks -n .. -1)
  lift          = actual - baseline
```

Assumes the counterfactual equals the recent past. Biased upward whenever anticipation or a
rising category trend is present, biased downward whenever the pre-period contained stockpiling
carry-in, a prior promotion, or unusually favourable weather.

**Rung 1b — same period last year.**

```text
  baseline      = mean(units, same weeks, year -1)
  adjusted      = baseline x (category index this year / category index last year)
  lift          = actual - adjusted
```

The adjustment is not optional. Without it you are measuring category growth. With it, you are
still assuming the store's share of the category held, which fails when distribution or
assortment moved.

**Rung 2 — difference-in-differences, ratio form.**

```text
  growth_control = mean(control, event) / mean(control, pre)
  baseline       = mean(treated, pre) x growth_control
  lift           = mean(treated, event) - baseline
```

The ratio form is preferred to the additive form when the treated and control groups differ in
level, because it lets a proportional calendar effect scale with store size. The additive form
`baseline = treated_pre + (control_event - control_pre)` is equivalent when levels are equal and
diverges when they are not. State which you used.

**Rung 2b — regression form.** The same estimate with store and week fixed effects, which is what
most software will run:

```text
  units_it = a_i + g_t + b * promoted_it + e_it
```

`b` is the DiD estimate when the panel is balanced and the treatment is binary. State the
clustering level. Module 05 topic 03 covers why store-level treatment with store-level clustering
needs enough clusters to be credible.

**Rung 3 — synthetic control.** Choose donor weights `w_j >= 0`, summing to one, that minimise
the distance between the weighted donor pre-period path and the treated pre-period path. Then:

```text
  baseline_t = sum_j w_j * units_jt
  lift_t     = units_treated_t - baseline_t
```

Two diagnostics decide whether you believe it: the pre-period fit (root mean squared error in the
pre-period, compared with the post-period divergence) and the weight dispersion (a single donor
carrying 80% of the weight means you have a matched pair with extra steps, and should say so).

**Rung 4 — matched market.** Same arithmetic as rung 3 or rung 2, applied to aggregated markets.
The additional assumption is no cross-market spillover: a shopper in the control market cannot
drive to the treated market. Test it by checking control-market sales during the event for a
dip that would indicate diversion.

**Rung 5 — randomised holdout.** Randomise assignment of stores (or weeks, in a switchback) and
compare means. The arithmetic is trivial; the design is the work. Module 05 topics 02 and 05
cover assignment units and switchback inference. From this topic, the only requirements are that
the holdout share is stated, the assignment mechanism is described, and the analysis is at the
assignment level rather than the shopper level.

### 5.7 Step 7 — Report a band, not a point

Compute the lift under at least two constructions and report both. This is not indecision; it is
the honest representation of a quantity that is defined only relative to an assumption.

The practical form is a **baseline fan**:

| Construction | What it removes | Typical direction of bias |
|---|---|---|
| Pre-period average, full | Nothing | Upward if anticipation, downward if carry-in |
| Pre-period average, recent four weeks | Some trend | Upward, more so |
| Same period last year, category-adjusted | Seasonality | Depends on distribution and price stability |
| Model-based no-promotion forecast | Trend, seasonality, price, distribution | Downward if the model saw promotions in training |
| Difference-in-differences, matched stores | Calendar and market shocks | Near zero if parallel trends holds |
| Randomised holdout | Selection and confounding | Unbiased by design |

If the fan is narrow, the conclusion is robust and you can say so. If the fan spans a factor of
two, the correct output is not the midpoint — it is a statement that the data cannot distinguish
the hypotheses, plus a recommendation about what design would.

### 5.8 Step 8 — Report to the standard

Four elements, every time. A lift number missing any of them is not a finding.

1. **Point estimate with an interval.** Units, unit of analysis, and window. "1,860 versus
   1,154 units per store per week over the four event weeks; lift 706 units per store per week,
   95% interval 540 to 872" is a finding. "Sixty-one per cent lift" is a headline.
2. **The counterfactual method, named.** Not "modelled baseline" but "difference-in-differences
   against twelve matched control stores, ratio form, eight-week pre-period excluding the two
   anticipation weeks".
3. **The assumptions, listed.** Parallel trends for DiD; no spillover for matched markets;
   pre-period fit for synthetic control; independence of assignment for a holdout.
4. **The sensitivity.** The lift under the pre-period average, and the reason you prefer the
   method you chose. If the two disagree by more than the decision threshold, say so.

## 6. Worked example

A worked case, small enough to check by hand and realistic enough that the numbers behave the way
real numbers do.

### 6.1 Setup

Brand K, a 1.5-litre liquid laundry detergent, 12-load pack. The manufacturer funds a 20% off
invoice temporary price reduction (TPR) with feature support for four weeks in six stores of a
regional grocery chain. The retailer schedules the event to span the two weeks before and the two
weeks containing a national holiday — a decision the retailer makes because category traffic is
high, and one that the analysis must absorb rather than ignore.

Twelve matched control stores in the same region did not promote the item and did not promote any
competing laundry detergent in the window. They were matched on store format, prior 52-week
category volume, and shopper demographics. The match is good but not perfect: control stores are
on average 1% smaller in this category. Weekly units per store are measured on point-of-sale (POS)
scan data, not shipments.

### 6.2 The data

| Period | Weeks | Promoted stores | Control stores |
|---|---|---|---|
| Early pre-period | −12 to −5 (8 weeks) | 1,020 | 980 |
| Late pre-period (anticipation) | −4 to −1 (4 weeks) | 900 | 1,010 |
| Event | 1 to 4 (4 weeks) | 1,860 | 1,109 |
| Post-event | 5 to 8 (4 weeks) | 772.5 mean | 1,060 |

The late pre-period in the promoted stores shows the anticipation effect: 900 against an early
pre-period of 1,020, a 12% decline in the four weeks before an event that shoppers could see
coming. The control stores show no such decline; their rise from 980 to 1,010 is ordinary
category drift.

> **Note:** If you compute the pre-period average over all twelve weeks you get 980 for the
> promoted stores and 990 for the controls. That 980 is the number that will appear in the
> promotion review slide, and it is the wrong number for two independent reasons: it contains the
> anticipation dip, and it is not adjusted for the calendar.

### 6.3 Five baselines, five answers

**Baseline A — full twelve-week pre-period average.**

```text
  baseline = (8 x 1,020 + 4 x 900) / 12 = 11,760 / 12 = 980.0
  lift     = 1,860 - 980.0 = 880.0 units per store per week
  lift %   = 880.0 / 980.0 = +89.8%
```

**Baseline B — last four pre-period weeks.** The "recent baseline" that category managers often
prefer because it is more responsive:

```text
  baseline = 900.0
  lift     = 1,860 - 900.0 = 960.0
  lift %   = 960.0 / 900.0 = +106.7%
```

The recent baseline is worse, not better, precisely because it is the anticipation window. This
is the single most useful arithmetic in the file: **the baseline choice that feels most responsive
is the one most contaminated by the event.**

**Baseline C — same period last year.** Last year, the same six stores averaged 1,390 units in
the equivalent four weeks, and the category index has grown 4% since.

```text
  adjusted baseline = 1,390 x 1.04 = 1,445.6
  lift              = 1,860 - 1,445.6 = 414.4
  lift %            = 414.4 / 1,445.6 = +28.7%
```

But last year the item had 14 facings in these stores and this year it has 18, a distribution
change of roughly 29%. The year-over-year baseline is measuring the facings, not the promotion.
Without the category adjustment the number would be +33.8%; with it, +28.7%. Neither is a
promotion effect.

**Baseline D — difference-in-differences, ratio form, eight-week pre-period.**

```text
  growth_control = 1,109 / 980      = 1.13163
  baseline       = 1,020 x 1.13163  = 1,154.27
  lift           = 1,860 - 1,154.27 = 705.73
  lift %         = 705.73 / 1,154.27 = +61.1%
```

This is the estimate to defend. It excludes the anticipation window from the pre-period, which is
correct because the anticipation is part of the treatment's effect; it scales the control's
holiday uplift onto the promoted stores' base level; and it uses stores that did not promote.

Note what the 29-point gap between A and D actually is. The control stores rose 13.2% in the event
weeks. That rise is the holiday. Baseline A attributes the holiday to the promotion. Baseline D
does not. **The 29 points are the price of the retailer's week selection, and nothing else.**

**Baseline E — model-based no-promotion forecast.** A regression on non-promoted weeks with
store fixed effects, a linear trend, holiday dummies, and observed shelf price, fitted on 78 weeks
of history excluding all promoted weeks:

```text
  baseline = 1,180.4
  lift     = 1,860 - 1,180.4 = 679.6
  lift %   = 679.6 / 1,180.4 = +57.6%
```

Close to D, and slightly lower, because the model also absorbs a small positive price effect from
the category's regular-price increase in the spring.

### 6.4 The fan

| Construction | Baseline | Lift (units/store/week) | Lift % |
|---|---:|---:|---:|
| A. Twelve-week pre-period average | 980.0 | 880.0 | +89.8% |
| B. Last four pre-period weeks | 900.0 | 960.0 | +106.7% |
| C. Same period last year, category-adjusted | 1,445.6 | 414.4 | +28.7% |
| D. DiD, matched control stores | 1,154.3 | 705.7 | +61.1% |
| E. Model-based no-promotion forecast | 1,180.4 | 679.6 | +57.6% |

The fan spans 28.7% to 106.7%, a factor of 3.7. If the decision threshold is "renew if lift
exceeds 40%", four of five constructions say renew and one says do not — and the one that says do
not is the one nobody will defend, because it is the crudest. Reporting a single point estimate is
therefore not a presentational choice; it is a choice about which decision the organisation makes.

### 6.5 The window

Now extend to the post-event weeks. The promoted stores fall to 640, 720, 830, and 900 units in
weeks 5 to 8, a mean of 772.5. The control stores settle at 1,060.

```text
  growth_control_post = 1,060 / 980        = 1.08163
  baseline_post       = 1,020 x 1.08163    = 1,103.27
  post lift per week  = 772.5 - 1,103.27   = -330.77
  post lift %         = -330.77 / 1,103.27 = -30.0%
```

The trough is 30% below the counterfactual, and it lasts at least four weeks. Combining:

```text
  event lift, 4 weeks      = 4 x 705.73          = +2,822.9 units per store
  post lift, 4 weeks       = 4 x (-330.77)       = -1,323.1 units per store
  net over 8 weeks         =                       +1,499.8 units per store
  8-week baseline          = 4 x 1,154.27 + 4 x 1,103.27 = 9,030.2
  net lift % over horizon  = 1,499.8 / 9,030.2   = +16.6%
```

Across six stores, the event generated roughly 16,938 incremental units and the trough returned
7,938 of them, for a net of about 8,999 units. **The same promotion is +61.1% on the event weeks
and +16.6% on an eight-week horizon.** Both are correct answers to different questions. A
promotion review reporting +61% and a finance team booking +16.6% are not disagreeing about the
data; they are disagreeing about the window, without either side saying so.

### 6.6 What the trough is not, and whether the stores were selected

The trough is not automatically all promotion-caused. Three components sit inside those 30 points:
stockpiling reversal, which is promotion-caused and belongs in the effect; competitive response,
which is a market event the matched controls should absorb; and seasonal decline, which the ratio
form already removes. Only the first belongs to the promotion. The measurement assigns all three
to it because the controls are matched, but a competitor response local to the promoted stores
would survive the matching, which is a stated limit rather than a solved problem.

Selection deserves the same scepticism. Three quick tests:

| Test | What you compute | What this example shows |
|---|---|---|
| Level match | Pre-period mean, treated versus control | 1,020 versus 980; a 4% gap, absorbed by the ratio form |
| Calendar clustering | Share of promoted weeks in top-decile category weeks | 4 of 4; the event sits entirely in the holiday build |
| Deal-date clustering | Share of events in the last three weeks of a fiscal quarter | 1 of 1; this was a quarter-end shipment push |

Two of three flag selection. That is normal, and it is not a reason to abandon the analysis; it is
a reason to require rung 2 or above, and to say in the report that the promoted weeks were chosen
for expected demand. A randomised holdout would remove the concern entirely, which is what the
recommendation should ask for.

## 7. Code, query, or calculation

The hand calculation below reproduces the two headline numbers from section 6. Run it with a
calculator, or paste it into a spreadsheet as a check on whatever tool produced your lift.

```text
INPUTS (units per store per week)
  promoted, weeks -12..-5   1,020      control, weeks -12..-5   980
  promoted, weeks  -4..-1     900      control, weeks  -4..-1 1,010
  promoted, event weeks     1,860      control, event weeks   1,109

A. PRE-PERIOD AVERAGE (12-week mean, promoted stores only)
   baseline   = (8 x 1,020 + 4 x 900) / 12   =    980.0
   lift units = 1,860 - 980.0                =    880.0
   lift %     = 880.0 / 980.0                =   +89.8 %

B. DIFFERENCE-IN-DIFFERENCES (ratio form, 8-week pre-period)
   control growth = 1,109 / 980              = 1.13163
   baseline       = 1,020 x 1.13163          = 1,154.27
   lift units     = 1,860 - 1,154.27         =   705.73
   lift %         = 705.73 / 1,154.27        =   +61.1 %

SPREAD
   percentage points        = 89.8 - 61.1    =    28.7
   ratio                    = 89.8 / 61.1    =   1.47x
   units per store per week = 880.0 - 705.7  =   174.3
   across 6 stores, 4 weeks = 174.3 x 6 x 4 = 4,183 units
```

Expected output is the four bold values: 980.0, 880.0, +89.8%, and 705.73, +61.1%. If your tool
disagrees, the usual cause is a different pre-period window, a control group that includes
promoted stores, or an average taken across stores of unequal weight.

## 8. Data, standards, and regulatory context

### 8.1 The data you need, and where it lives

| Input | System of record | Granularity | Failure that breaks the baseline |
|---|---|---|---|
| POS scan units and price | Retailer POS feed, or syndicated scanner data | Store-SKU-week, sometimes day | Store set changes; price is the shelf price, not the funded price |
| Shipments | Manufacturer ERP or distributor | DC-customer-week | Pipeline fill counted as shopper demand |
| Promotion calendar | Trade promotion management (TPM) system | Deal, account, date range | Deal dates differ from execution dates |
| Causal flags (feature, display) | Syndicated audit, or retailer | Store-week, often sampled | Sampled, so some promoted weeks look unpromoted |
| Distribution and facings | Syndicated audit, planogram system | Store-SKU-period | Changes mid-window, breaking the year-over-year comparison |
| Retailer inventory | Retailer DC data, rarely shared | DC-SKU-week | Forward buying invisible in POS, visible in shipments |
| Household panel | Syndicated panel | Household-purchase | Small samples at store level; used for stockpiling, not for lift |

Three of these matter more than the rest. **POS rather than shipments** is non-negotiable for
measuring shopper response; shipments measure the retailer's order, a policy variable that
includes forward buying. **Observed shelf price rather than the deal sheet** defines the
treatment. **Execution dates rather than planned dates** define the window — a display that went
up three days late shifts the event boundary and, if ignored, attributes three days of baseline to
the promotion.

### 8.2 Identifiers and standards

Lift analysis joins data across four organisations, so identity is a measurement problem before it
is an analytics one.

- **GS1 GTIN** (Global Trade Item Number) identifies the trade item including pack size. Two pack
  sizes of the same brand are different GTINs and must not be aggregated without a conversion to
  equivalent units. **GS1 GLN** (Global Location Number) identifies the store or ship-to location;
  retailer-internal store numbers are not GLNs and do not survive a banner change.
- **EDI 852** (Product Activity Data) is the common retailer-to-supplier weekly sales and inventory
  feed. It carries units by item and location but not shelf price, and its week definition is the
  retailer's. **EDI 867** (Product Transfer and Resale Report) carries resale data closer to POS,
  but coverage varies by retailer.
- **NRF 4-5-4 calendar** is the retail fiscal calendar most United States retailers use, with
  Sunday week starts and an occasional 53rd week. It does not align with the 4-4-5 manufacturer
  calendar, and the misalignment is a documented source of spurious lead-lag.

> **Note:** These are identifiers and message formats, not measurement standards. Nothing in GS1
> or EDI defines a baseline, and no message format carries a counterfactual. The standards make
> the join possible; the method is yours.

### 8.3 What is regulated, and what is not

No regulator defines promotional lift, and no accounting standard requires a counterfactual. That
absence is itself a finding: the number that allocates 15–25% of revenue is governed by contract
and convention rather than by rule.

Four regulatory and standard-setting areas do constrain the inputs:

1. **Revenue recognition and consideration payable to a customer.** Under ASC 606 and IFRS 15, a
   promotional allowance paid to a retailer is generally treated as a reduction of revenue rather
   than a marketing expense, unless it is in exchange for a distinct good or service. This changes
   what "promotion cost" means in an iROAS denominator, and it changes which number finance will
   defend.
2. **Accrual and deduction accuracy.** Trade accruals are estimates until claims are settled.
   Sarbanes-Oxley controls in United States listed companies require the accrual process to be
   documented and tested, which is one reason the industry baseline is deterministic: it has to
   sit inside a controlled process.
3. **Consumer price indication.** In the European Union, the price-indication rules amended by the
   Omnibus Directive require an announced price reduction to reference the prior price, generally
   the lowest price applied in the preceding 30 days. This constrains what a retailer can call a
   regular price, and therefore what belongs in a pre-period price baseline. Confirm the in-force
   national text for the market you are working in; transposition differs.
4. **Panel and loyalty data.** Household-level stockpiling evidence comes from panel or loyalty
   data, which is personal data. Purpose limitation and minimisation apply; a stockpiling model
   does not need household identifiers.

The honest summary: regulation governs **how you record the money and the price**, not **how you
estimate the counterfactual**. The methodological standard is professional, not legal, and section
5.8 is the closest thing to a normative reference in this topic.

## 9. Industry practice and case evidence

### 9.1 Why the industry baseline and the academic baseline differ

The divergence is not a competence gap. It is a difference in the object being produced.

An industry baseline is a **product**. It must be computed for every SKU, every account, every
week, by a system, by people who are not econometricians, and it must be reproducible by the
counterparty in a deduction dispute. Those requirements select for determinism, transparency,
and speed. A pre-period average satisfies all three. A synthetic control satisfies none of them:
it depends on a donor pool that the retailer does not have, produces a number that changes if the
pool changes, and cannot be explained in a claims meeting.

An academic counterfactual is a **research object**. It is produced once, for one event, by
someone whose job is to be right rather than fast, and it is judged by internal validity. The
published literature on promotion evaluation consequently spends its effort on identification —
selection, endogeneity, dynamic consumer inventory — that a TPM system cannot address.

The syndicated vendors occupy a middle position that matters because their definition is the one
most companies actually use. NielsenIQ's consumer packaged goods (CPG) dictionary defines
incremental lift per week of support as sales above baseline, noting the figure can be negative
when extra volume does not cover the discount. That definition is correct and useful. What it
cannot do, because a syndicated product cannot, is defend a counterfactual for a given event.

### 9.2 The standards body position

In November 2025, IAB and IAB Europe published guidelines for incremental measurement in commerce
media. The document is useful here for one reason: it ranks method types by causal strength, in
an order close to the ladder in section 3 — experiment-based methods (randomised controlled
trials, holdouts, matched markets) at the top, model-based counterfactual methods (synthetic
control, propensity models) next, econometric methods (marketing mix modelling, time-series
regression) below that, and hybrid proxies last. It also names three requirements for a causal
claim: a credible counterfactual, bias control, and separation of signal from noise.

That ranking, published by an industry body rather than a journal, is evidence that the ladder in
section 3 is converging practice rather than an academic idiosyncrasy.

> **Note:** IAB is an industry standards body, not a regulator. Its guidelines are normative for
> members who adopt them and carry no legal force. Grade them accordingly — section 16 does.

### 9.3 Documented cases

**Synthetic control at store level in a supermarket promotion test.** A company-affiliated
research team working with a United States grocery data business evaluated a supermarket
promotion using 16 treated stores against 43 control stores, and extended the synthetic control
approach to account for cross-store correlation in the error structure. The methodological point
is the one that matters for practice: treating stores as independent understates uncertainty,
because stores in the same market move together. Any store-level lift interval that assumes
independence is too narrow. This is company-affiliated research, not an independent audit, and
the specific estimate should be treated as a case rather than a benchmark.

**Two-step synthetic control on scanner data.** A peer-reviewed treatment of the same problem adds
a formal test of the pre-period fit before estimation, and then chooses an estimator that trades
bias against efficiency. The application reported in the abstract is a tax change rather than a
promotion, but the estimator is designed for marketing events and store-category scanner data,
which is the setting here. The transferable lesson is procedural: **test the pre-period fit
before you trust the counterfactual, and report the test.**

**Post-promotion dips in store-level scanner data.** The foundational empirical work estimated
pre- and post-promotion dips on store-level scanner data for storable categories and found the
dip to be real but modest relative to the promotion bump — on the order of 4% to 25% of the
current sales effect, depending on category and model. Later work reconciled this with household
panel evidence, which shows larger purchase acceleration, by observing that feature and display
activity in the post-period has an opposing dynamic effect that partly masks the dip in aggregate
data. A large subsequent study across 83 stores and 10 categories estimated tens of thousands of
dip elasticities and found dips stronger for high-priced, frequently promoted, mature, high-share
items, and stronger where promotion timing is less predictable.

The practical reading is important and slightly uncomfortable: **the trough you can see in
aggregate data is a lower bound on the trough that exists**, because the measurement is
contaminated by whatever else the retailer did in those weeks. Section 6's 30% trough is
plausible for a storable category with a deep discount; a 4% trough would also be plausible for a
different category. The number is category-specific, and the only defensible approach is to
measure it rather than assume it.

**Retailer forward buying.** A peer-reviewed analysis of forward buying by retailers shows that
it can benefit both manufacturer and retailer under some conditions, can arise from a prisoner's
dilemma under competition, and can lead retailers to order beyond the most optimistic demand
scenario when demand is uncertain. For lift measurement the implication is direct: shipment-based
lift is not shopper lift, and the gap is a deliberate inventory decision by the counterparty.

**A consultancy case.** A three-month engagement with a French retail chain built a counterfactual
forecast trained only on non-promotion dates, predicted across all dates, and compared prediction
to actual during promotions. The published account reports roughly 90% forecast accuracy and names
cannibalisation, halo, and anticipation/stockpiling as limitations. Consultancy grade: a workable
workflow, not evidence that the workflow is unbiased.

### 9.4 The uncomfortable numbers

Vendor and consultancy sources converge on trade spend at 15–25% of gross revenue and on a large
majority of promotions failing to break even. The most commonly repeated figure is that around
72% of United States trade promotions do not break even, attributed variously to a management
consultancy and to a syndicated data provider. Section 16 grades it as vendor-sourced: it is
repeated across vendor content and traces to a small number of original studies that are not
reproduced in the retrievable sources. Treat it as a prior about the base rate, not a measurement.
If most promotions do not pay back, a lift estimate that makes every promotion look successful is
probably measuring the baseline wrongly — a weak but real sanity check.

## 10. Failure modes and red flags

Each row names a failure, a red flag visible in a meeting, and one visible in the data. The
meeting flag is usually the faster one.

| # | Failure | Red flag in a meeting | Red flag in the data | Mechanism | Consequence |
|---:|---|---|---|---|---|
| 1 | Pre-period contaminated by anticipation | "We used the four weeks immediately before, because that is most relevant" | The last pre-period weeks sit below the earlier weeks by more than trend explains | Shoppers deferred purchase; baseline is depressed | Lift overstated, often by more than the true effect |
| 2 | Same period last year used without adjustment | "It is seasonally clean" | Distribution, facings, or pack count changed year on year | The comparison is measuring assortment, not promotion | Lift attributed to the wrong cause, direction unpredictable |
| 3 | Promoted weeks selected by the retailer | Nobody mentions why the event was scheduled then | All promoted weeks fall in top-decile category weeks | Treatment assigned on expected outcome | Naive lift includes the calendar |
| 4 | Control stores contaminated | "We used the stores that were not in the promotion" | Control group shows a spike in the event weeks | A different mechanic, a competitor, or a spillover hit the controls | Lift biased toward zero |
| 5 | Control stores unmatched | No matching diagnostics shown | Pre-period levels differ by more than 10% and trends diverge | Control is not comparable | DiD identifies nothing; the interval is meaningless |
| 6 | Window excludes the trough | The chart stops at the last promotion week | Post-event weeks are absent from the dataset extract | The trough is outside the measurement | Lift overstated by the entire reversal |
| 7 | Shipments used as the outcome | "Volume" with no source stated | Event-week shipments exceed POS by a large margin that does not reverse | Pipeline fill and forward buying | Lift measures ordering, not shopping |
| 8 | Funded discount used as the treatment | The deal sheet is on the slide and the shelf price is not | Treatment variable equals the invoice discount exactly | Pass-through is not one | Effect estimated for an intention |
| 9 | Model baseline validated in sample | "The model fits at 0.94" | R-squared reported, out-of-sample error absent | The model saw promoted weeks during fitting | Baseline too low, lift overstated |
| 10 | Single point estimate reported | One number, no interval, no method note | No control group, no pre-trend plot | Assumption hidden | Decision taken on an artefact |
| 11 | Spillover between treated and control | Control stores are in the same trade area as treated | Control sales dip during the event | Shoppers cross the boundary | Control under-reacts; lift understated |
| 12 | Event boundary taken from the plan, not execution | "The promotion ran weeks 14 to 17" | Display flag starts mid-week | Execution lag | Two or three days of baseline attributed to the promotion |

Three of these are systematically under-detected. **Row 8 is invisible unless you look for it.**
Every promotion dataset has a discount field, and in most of them it is the funded discount,
because that is what the manufacturer's system records; the shelf price lives elsewhere, or in an
audit sample, or nowhere. **Row 9 is common in teams that have recently adopted machine learning.**
A gradient-boosted model trained on all weeks will fit the promotion bump and then predict a high
baseline during the next event, producing a lift near zero — visible immediately if you ask what
the training set excluded. **Row 12 is the quietest.** Planned and executed dates differ by days,
usually a partial week; taking the window from the TPM system misclassifies the boundary weeks,
and the error is small enough per week to survive review while large enough to move the estimate
by several points.

## 11. Assumptions and limits

An explicit ledger: each row states the assumption, why the method needs it, what breaks if it
fails, and how to test it. If you cannot fill the test column, you cannot use the method.

| # | Assumption | Why the method needs it | What breaks if it fails | How to test |
|---:|---|---|---|---|
| 1 | The counterfactual is stable over the window | Every method extrapolates a pre-period relationship into the event | The baseline drifts; lift absorbs the drift | Hold out part of the pre-period and predict the rest |
| 2 | Parallel trends between treated and control | DiD differences out common shocks | Treated group was already diverging; the estimate is the divergence | Plot the pre-period; run a placebo on an unpromoted period |
| 3 | No interference between units | Control stores must be untreated in fact | Control under-reacts; lift is understated | Check control sales during the event for a dip; check trade areas |
| 4 | The event window is correctly dated | Everything is conditioned on the window | Boundary weeks are misclassified | Compare planned dates to execution dates and to the price series |
| 5 | The treatment is the shelf price and display | The effect belongs to what the shopper saw | The estimate belongs to the funded deal | Reconcile observed shelf price to the deal sheet for a sample |
| 6 | No other promotion in the window | Confounded events cannot be separated | Lift attributed to the wrong mechanic | Build the calendar matrix from topic 03; flag overlaps |
| 7 | The store set is constant | Panel balance | Entrants and exits move the mean | Freeze the store set at the pre-period and drop changes |
| 8 | The post-period is long enough to capture the trough | The horizon claim depends on it | Trough truncated; net lift overstated | Plot the post-period until sales return to baseline |
| 9 | Pre-period is free of prior promotion carry-in | The baseline must be an unpromoted world | Baseline inflated; lift understated | Identify the prior event and its trough; extend or trim the pre-period |
| 10 | The category is comparable across the year-over-year gap | The last-year baseline must be like-for-like | Assortment or price movement is measured as promotion | Reconcile facings, pack count, and regular price between years |

Three limits cannot be tested away. **You cannot randomise the past**: for an event already run
the best you can do is rung 2 to 4, and the holdout is a recommendation for the next event, not a
fix for this one. **The counterfactual is partly a definition**: two competent analysts can
disagree about whether the anticipation weeks belong in the effect, and the honest response is to
report both. **The trough is a lower bound**: as section 9.3 sets out, aggregate data hides part of
the dip because other activity in those weeks pushes the other way.

## 12. Exercises

Nine exercises, ordered from recall to design. Each states a deliverable, acceptance criteria, and
an estimated time. Exercises 1 to 4 have numeric answers in section 13; 5 to 9 have rubrics.

**Exercise 1 — Recompute the fan.** Using the data in section 6, compute lift under all five
constructions and reproduce the table in section 6.4 to one decimal place.
*Deliverable:* a five-row table with baseline, lift in units, and lift in per cent.
*Acceptance:* every value matches section 6.4; baseline D is shown line by line. *Time:* 25 minutes.

**Exercise 2 — Move the pre-period.** Recompute baseline D using a twelve-week pre-period
including the anticipation weeks, then a six-week pre-period. Report both lifts and explain which
is more defensible and why. *Deliverable:* two lift figures and a five-sentence justification.
*Acceptance:* the twelve-week figure reproduces the 980/1,010 arithmetic; the justification names
anticipation as the reason a longer window is worse. *Time:* 35 minutes.

**Exercise 3 — Window arithmetic.** Using the section 6 data, compute lift over a twelve-week
horizon by adding four further post weeks at 980 units per store. State the net lift in units, in
per cent, and as a share of the event-week lift. *Deliverable:* three numbers and a one-line
interpretation. *Acceptance:* arithmetic correct to one decimal; the interpretation states that
the promotion retained only a minority of its event-week gain. *Time:* 30 minutes.

**Exercise 4 — Diagnose a mis-specified baseline.** A colleague reports a 140% lift computed as
event-week sales divided by the four weeks immediately before the event, in a category where the
promotion was announced two weeks in advance. Identify two distinct biases, state their
directions, and estimate which dominates. *Deliverable:* a short memo with a direction for each
bias and a ranked conclusion. *Acceptance:* both biases named correctly (anticipation; calendar
selection or stockpiling); directions stated; the ranking argued rather than asserted.
*Time:* 30 minutes.

**Exercise 5 — Build a baseline fan for your own data.** Take one real promotion and compute lift
under at least three constructions. Present the fan, state which you would defend in a review, and
state what you would need to move up one rung. *Deliverable:* a one-page fan table plus a paragraph
on the next rung. *Acceptance:* at least three constructions, at least one control-based, each
method named; the next-rung requirement is specific (e.g., "a 20-store holdout for the autumn
event"). *Time:* 3 hours.

**Exercise 6 — Draft the holdout proposal.** Design a randomised holdout for the next cycle:
assignment unit, holdout share, duration, primary metric, analysis level, and the decision rule
that will use the result. Do not compute power; state what power analysis you would commission
and from whom. *Deliverable:* a two-page design memo. *Acceptance:* the assignment unit is
justified against the decision; the holdout share is within 10–20% or the deviation is argued; the
analysis level matches the assignment level; the power analysis is commissioned rather than
invented. *Time:* 2 hours.

**Exercise 7 — Critique a vendor claim.** Find a public vendor or consultancy claim about
promotion lift or incrementality. Assess the counterfactual, the comparator, the window, and the
conflict of interest, then grade the evidence A to E. *Deliverable:* a 400-word appraisal with a
grade and a one-line justification. *Acceptance:* all four dimensions addressed; the grade is
defensible against the section 16 scheme; no claim is accepted without a stated comparator.
*Time:* 1.5 hours.

**Exercise 8 — Separate the two baselines.** For one SKU, write down the measurement baseline and
the forecast baseline side by side for the same future week. Explain in five sentences why they
differ and what would go wrong if planning used the measurement one. *Deliverable:* a two-column
table and five sentences. *Acceptance:* the forecast baseline includes expected future promotions;
the explanation names at least one inventory consequence. *Time:* 45 minutes.

**Exercise 9 — Write the reporting standard.** Produce a one-page template that any lift number
must satisfy before it reaches a commercial committee, including the four elements from section
5.8 and a signature line for the method owner. *Deliverable:* a one-page template.
*Acceptance:* all four elements present; short enough to be used; the assumptions line is
mandatory. *Time:* 1 hour.

## 13. Self-check answers

**Answer 1.** Baseline A: (8 × 1,020 + 4 × 900) / 12 = 11,760 / 12 = 980.0; lift 880.0; +89.8%.
Baseline B: 900.0; lift 960.0; +106.7%. Baseline C: 1,390 × 1.04 = 1,445.6; lift 414.4; +28.7%.
Baseline D: 1,109 / 980 = 1.13163; 1,020 × 1.13163 = 1,154.27; lift 705.73; +61.1%. Baseline E:
1,180.4; lift 679.6; +57.6%. All five rows should match section 6.4. The most common error is
using the twelve-week mean for the control in baseline D (990 rather than 980), which gives a
growth ratio of 1.1202, a baseline of 1,142.6, and +62.8% — close, but not the same number, and
the difference is the anticipation weeks quietly entering the control base.

**Answer 2.** Twelve-week pre-period including anticipation: treated pre-mean 980, control
pre-mean 990, growth ratio 1,109 / 990 = 1.1202, baseline 1,097.8, lift 762.2, +69.4%. Six-week
pre-period (weeks −6 to −1, which is 2 weeks at 1,020 and 4 at 900, mean 940): control over the
same six weeks, 2 weeks at 995 and 4 at 1,010, mean 1,005, growth ratio 1,109 / 1,005 = 1.1035,
baseline 940 × 1.1035 = 1,037.3, lift 822.7, +79.3%. The twelve-week window is more defensible:
it has more weeks, it excludes the anticipation period from the treated base, and its control
series is uncontaminated. The six-week window is worse despite being more "recent" because four
of its six weeks are anticipation. The lesson is that **length is not the criterion; freedom from
treatment contamination is.**

**Answer 3.** Four further post weeks at 980 per store, against a control baseline of 1,060
(ratio 1.08163, baseline 1,103.27): post lift per week = 980 − 1,103.27 = −123.27; over four weeks
= −493.1. Twelve-week event-plus-post lift = 2,822.9 − 1,323.1 − 493.1 = +1,006.7 units per store.
Twelve-week baseline = 4 × 1,154.27 + 8 × 1,103.27 = 4,617.1 + 8,826.2 = 13,443.3. Net lift =
1,006.7 / 13,443.3 = +7.5%. As a share of the event-week lift: 1,006.7 / 2,822.9 = 35.7%. The
interpretation: by twelve weeks the promotion has retained about a third of its event-week gain,
and the remaining two thirds were movement of existing demand rather than creation of new demand.
This is the arithmetic behind the common claim that most promotions move volume rather than
create it — the claim is usually true, and the window is why.

**Answer 4.** Two biases. First, anticipation: the promotion was announced two weeks ahead, so
the four pre-period weeks are depressed by shoppers waiting, which lowers the baseline and
inflates the lift. Direction: upward. Second, calendar selection: the retailer chose the weeks,
probably for expected high demand, and the pre-period does not contain the corresponding calendar
effect, so the seasonal uplift is attributed to the promotion. Direction: upward. Both push the
same way, so the 140% figure is not a case of offsetting errors; it is roughly the sum of two
overstatements. Which dominates depends on the announcement's reach and the size of the calendar
effect, but in a category with a strong holiday or weather cycle the calendar selection usually
dominates, because it operates on every unit in the event window while anticipation operates only
on the shoppers who were both aware and able to defer. A defensible answer states a hypothesis
and names the test — a control-store comparison would separate them, since the control stores
share the calendar but not the announcement.

**Answer 5 (rubric).** Full marks: at least three constructions, one of them control-based; a named
method for each; an identical window across constructions so the comparison is clean; an explicit
statement of which is defended and why; and a specific next-rung requirement. Failures: windows
that differ between constructions; a "model-based" baseline with no statement of what training
excluded; a next-rung requirement that is a wish ("better data") rather than a design ("a 20-store
holdout, store-level assignment, two weeks, units per store per week analysed at store level").

**Answer 6 (rubric).** Full marks: an assignment unit justified against the decision (store for
account renewal; week within store, a switchback, if no store can be denied — Module 05 topic 05);
a holdout share between 10% and 20%, or a reasoned deviation; a duration covering event and
trough; a primary metric in units with a stated unit of analysis; an analysis level matching the
assignment level; and a decision rule written before the result. Randomising shoppers when the
treatment is executed at store level is a fail, because shopper assignment cannot vary it.

**Answer 7 (rubric).** Full marks: all four dimensions addressed explicitly — counterfactual
constructed or assumed; comparator a control group, a pre-period, or another campaign; window
including or excluding the trough; conflict of interest, is the publisher selling the method or the
measurement. A grade of D for vendor marketing is not a criticism of the arithmetic but a statement
about what the claim can bear. Strong answers note that a vendor's definitional content can be
sound while its performance content is unverifiable.

**Answer 8 (rubric).** Full marks: the two baselines differ specifically — the measurement baseline
excludes the promotion under evaluation, the forecast baseline includes every expected promotion
plus the expected response — and the five sentences name at least one inventory consequence, such
as under-forecasting the event week into a stock-out, or over-ordering through a deep trough. The
common failure is describing them as "the same number computed differently".

**Answer 9 (rubric).** Full marks: all four elements from section 5.8 — point estimate with
interval, method named, assumptions listed, sensitivity to an alternative construction — plus a
named owner. A template longer than a page will not be used and should lose marks for that alone.
A template with no assumptions line fails regardless of its other content, because that line is
the mechanism by which the standard has any effect.

## 14. Interview questions

**Q1. What is lift?**
*Strong:* Actual sales minus a counterfactual, over a stated window and unit of analysis, with
the counterfactual method named. It is causal and only as credible as its identifying assumption.
*Weak:* "The increase in sales during a promotion, as a percentage over baseline." Never says
what the baseline is or that it is constructed.

**Q2. Why is the pre-period average a poor baseline?**
*Strong:* Because the pre-period is shaped by the promotion: announcement causes anticipation and
depresses the weeks before; a prior event leaves carry-in that inflates them; the retailer chose
the promoted weeks for expected demand, so the pre-period lacks the calendar effect the event
contains. The bias has no fixed sign, which is worse than a known one.
*Weak:* "Because sales change over time, so use a longer period." Confuses precision with
identification, and would recommend a longer, more contaminated window.

**Q3. When is last year's same week acceptable?**
*Strong:* When distribution, assortment, regular price, pack architecture, and competitive set are
unchanged, and the category index is used to adjust for growth. In practice, rarely — and the
analyst should state which of those held.
*Weak:* "Always, because it controls for seasonality." Seasonality is one of several things that
change between years.

**Q4. What does difference-in-differences buy you?**
*Strong:* It removes shocks common to treated and control stores in the same calendar weeks —
holidays, weather, category-wide price moves. It does not remove selection, local shocks, or
spillover, and parallel trends is untestable in the post-period and only weakly testable before it.
*Weak:* "A control group, so the result is causal." Treats the existence of a control as
sufficient and never names the identifying assumption.

**Q5. Why are promoted weeks different from unpromoted weeks?**
*Strong:* Because the calendar is a plan. Retailers place events where traffic is expected and
manufacturers push deal dates toward quarter end. Treatment is assigned on expected outcome, which
is selection — visible as calendar clustering and above-average promoted-store volume.
*Weak:* "Promotions run when sales are low and the brand needs a boost." States the opposite
mechanism, which holds for clearance but not generally, and would imply downward bias.

**Q6. How do you decide the evaluation window?**
*Strong:* By the decision and the category's storage behaviour. The window must include
anticipation, the event, and the trough; the horizon must be argued commercially, not chosen after
seeing the result. Two to four post weeks is typical for a storable category, less for perishables.
*Weak:* "The promotion weeks plus one week, to be safe." Truncates the trough and systematically
overstates net lift.

**Q7. What is the post-promotion dip, and is it part of the effect?**
*Strong:* The period after the event in which sales fall below the counterfactual, driven mainly
by stockpiling reversal and pull-forward. It is part of the effect if the promotion caused it,
which it usually did, so it belongs inside the window. Aggregate dips are typically smaller than
panel evidence implies, because other post-period activity masks part of them.
*Weak:* "A separate phenomenon from consumers running down pantry stock; report it alongside the
lift." Separates the consequence from the cause and overstates the net effect.

**Q8. You have no control stores. What do you do?**
*Strong:* Go to rung 3 if a defensible donor pool exists, or rung 1b with a category-adjusted
year-over-year comparison if not; report the fan across constructions and state that the design
cannot separate the promotion from the calendar; recommend a holdout next time.
*Weak:* "Build a model to predict the baseline." Without saying what training excluded, the model
learns the promotion and reports a lift near zero.

**Q9. Measurement baseline versus forecast baseline?**
*Strong:* The measurement baseline is a counterfactual excluding the promotion under evaluation;
the forecast baseline includes every promotion expected to run. Different errors, different
acceptance tests, different consumers — finance and claims versus supply planning.
*Weak:* "The same number used for different purposes." That is the confusion itself.

**Q10. How would you report a lift figure to a commercial committee?**
*Strong:* Point estimate with an interval, in stated units over a stated window; the method named;
the assumptions listed; and a sensitivity to one alternative construction. If the fan spans the
decision threshold, say so rather than picking the midpoint.
*Weak:* "A percentage with the promotion dates, versus baseline." Supplies a headline and withholds
everything a reader would need to disagree with it.

**Q11. Why do industry and academic baselines differ so much?**
*Strong:* Because they are different products. The industry baseline must be computable every week
for every SKU by a system and reproducible by a counterparty in a deduction dispute, which selects
for determinism. The academic counterfactual is produced once for one event and judged on internal
validity. Both are rational; the analyst's job is to know which the decision needs.
*Weak:* "Industry analysts are not trained in causal inference." Sometimes true of individuals, and
not an explanation of a systematic difference.

## 15. Key terms

- **Anticipation** — the fall in sales before an announced promotion as shoppers defer purchase.
- **Baseline** — a constructed counterfactual, not a measured period; always state which one.
- **Baseline fan** — the set of lift estimates produced by several constructions of the same event.
- **Control store** — an unpromoted store used as a comparator, ideally matched on pre-period behaviour.
- **Counterfactual** — what would have sold had the promotion not run; the object every method here estimates.
- **Difference-in-differences (DiD)** — change in treated minus change in control, identifying parallel trends.
- **Donor pool** — the untreated units from which a synthetic control's weights are drawn.
- **Evaluation horizon** — the period over which a promotion should be judged commercially.
- **Incrementality** — the share of observed sales caused by the promotion, at a stated population level.
- **Incremental return on ad spend (iROAS)** — incremental gross profit or revenue per unit of promotional spend.
- **Lift** — actual minus counterfactual, over a stated window and unit of analysis.
- **Matched market** — an untreated geography matched on size, mix, and trend.
- **Measurement baseline** — the counterfactual used to size an event already run.
- **Model-based baseline** — a no-promotion forecast fitted on genuinely unpromoted periods.
- **Parallel trends** — the identifying assumption of DiD: absent treatment, the groups would have tracked.
- **Post-promotion dip** — sales below the counterfactual after the event; part of the effect.
- **Pre-period average** — mean of the weeks before the event; a description, not a counterfactual.
- **Pull-forward** — demand moved from a later period into the promoted period.
- **Randomised holdout** — a randomly selected subset of units denied the promotion.
- **Selection bias** — systematic difference between promoted and unpromoted units or weeks.
- **Stable unit treatment value assumption (SUTVA)** — one unit's outcome does not depend on another's treatment.
- **Stockpiling** — shoppers buying ahead and holding inventory; the household form of pull-forward.
- **Switchback** — treatment alternating over time within the same unit; a Module 05 design.
- **Synthetic control** — a weighted combination of donors fitted to the treated unit's pre-period.
- **Timing selection** — the retailer's choice of promoted weeks correlated with expected demand.

## 16. Sources

Grades follow the module scheme: A peer-reviewed or audited, B regulatory or official, C company
case or investor claim, D vendor marketing, E press or anecdote. Working papers are graded A with
that caveat noted.

### Peer-reviewed articles, textbooks, and working papers (A)

- Blattberg, R. C., & Neslin, S. A. (1990). *Sales Promotion: Concepts, Methods, and Strategies.*
  Prentice Hall. Canonical textbook; source of much of the section 4 vocabulary. Grade A.
- Van Heerde, H. J., Leeflang, P. S. H., & Wittink, D. R. (2000). The estimation of pre- and
  postpromotion dips with store-level scanner data. *Journal of Marketing Research*, 37(3),
  383–395. DOI 10.1509/jmkr.37.3.383.18782. Grade A.
- Macé, S., & Neslin, S. A. (2004). The determinants of pre- and postpromotion dips in sales of
  frequently purchased goods. *Journal of Marketing Research*, 41(3), 339–350.
  DOI 10.1509/jmkr.41.3.339.35992. Grade A.
- Desai, P. S., Koenigsberg, O., & Purohit, D. (2010). Forward buying by retailers. *Journal of
  Marketing Research*, 47(1), 90–102. DOI 10.1509/jmkr.47.1.90. Grade A.
- Li, K., & Shankar, V. (2023). A two-step synthetic control approach for estimating causal
  effects of marketing events. *Management Science*. DOI 10.1287/mnsc.2023.4878. Grade A.
- Menchetti, F., & Bojinov, I. (2022). Estimating the effectiveness of permanent price reductions
  for competing products using multivariate Bayesian structural time series models. *Annals of
  Applied Statistics*. DOI 10.1214/21-AOAS1498. Grade A.
- Ning, Y., Ghosal, S., & Thomas, A. (2018). Bayesian method for causal inference in
  spatially-correlated multivariate time series. *Bayesian Analysis*. DOI 10.1214/18-BA1102.
  Grade A.
- Sun, Y., Zheng, E., Jin, Y., Jiang, Y., & Wang, Z. (2019). Estimating promotion effects using
  big data: A partially profiled LASSO model with endogeneity correction. *Decision Sciences*.
  DOI 10.1111/deci.12354. Grade A.
- Epstein, L. D., et al. Dynamic effects of store promotions on purchase conversion. *Journal of
  Business Research*; retrieved as the FGV working-paper version. Grade A for method, with the
  caveat that the retrieved document is not the typeset article.
- Hendel, I., & Nevo, A. The post-promotion dip puzzle: What do the data have to say? CSIO working
  paper via EconStor. Grade A as a working paper; not peer-reviewed in this version.
- Lu, Z. (2017). Consumer stockpiling and sales promotions. DIW Discussion Paper 1680. Grade A as
  a working paper.
- Gardete, P. M., & Cotrim, R. (2025). Dynamic demand elasticities in promotion cycles. Working
  paper. Grade A as a working paper, not yet peer-reviewed; source for the time-varying
  elasticity claim in section 3.2.

### Standards and official guidance (B)

- IAB & IAB Europe (November 2025). *Guidelines for Incremental Measurement in Commerce Media.*
  Ranks method types by causal strength. Grade B as an industry standards body's guidance; no
  legal force.

### Company cases and company-affiliated research (C)

- Tierney, N., Hellmayr, C., Barkimer, J., & West, M. (2023). Multivariate Bayesian dynamic
  modeling for causal prediction. arXiv 2302.03200. Promotion evaluation with 16 treated and 43
  control stores; used for the uncertainty point only. Grade C.
- Artefact. *Forecasting something that never happened: how we estimated past promotions
  profitability.* Consultancy case study; counterfactual forecast trained only on non-promotion
  dates, reporting roughly 90% accuracy. Grade C.
- Vistex. *Trade Promotion and Customer Planning.* Vendor eBook reporting trade spend near 22% of
  revenue, citing a 2017 survey of 110 manufacturers. Grade C.

### Vendor marketing and vendor reference material (D)

- NielsenIQ. *CPG Dictionary: incremental $ lift per week of support.* Vendor reference defining
  incremental lift and noting it can be negative. Grade D.
- Makai. *How to measure incremental sales from retail activations.* Vendor methodology article
  covering randomised, matched-store, DiD, synthetic control, and pre/post designs. Grade D.
- Tellius. *Why 70% of trade promotions lose money.* Vendor article repeating the 72% figure;
  provenance traces to a small number of originals not reproduced in retrievable form. Grade D.
- Kard. *The complete guide to incrementality testing for commerce media.* Vendor guide
  recommending 10–20% control and two to four weeks' duration. Grade D.
- Cozl Labs. *Incrementality testing for offers and discounts*, plus its holdout size calculator.
  Vendor guidance on holdout design and sample size. Grade D.
- TELUS Agriculture & Consumer Goods. *Optimizing trade promotion ROI in the CPG industry.*
  Vendor article reporting trade spend at 15–25% of gross revenue. Grade D.
- MetricGate. *Promotional lift: how to measure campaign impact.* Practitioner article reporting
  a simulation in which naive pre/post comparison inflated lift by 30–50%. Grade D.

> **Note:** No source here defines a universal lift methodology. The peer-reviewed sources
> establish mechanisms and estimators, the standards body a ranking, and the vendor sources what
> practice looks like. The synthesis in sections 3, 5, and 10 is the author's.

## 17. Links

**This module:** [Module 09 index](_index.md) · [Topic 01 — Demand Sensing versus
Forecasting](01_demand-sensing-signals.md) · [Topic 02 — Sales
Decomposition](02_sales-decomposition.md) · [Topic 03 — Promotion Mechanics and Calendar
Data](03_promotion-mechanics.md) · Part II index: [`../_index.md`](../_index.md)

**Planned siblings** (not yet written; listed as text so the links do not break):
`05_cannibalization-halo.md` (Cannibalization, Halo, and Pull-Forward) ·
`06_price-display-effects.md` (Price and Display Effects) · `07_causal-ml-uplift.md` (Causal
Machine Learning and Uplift) · `10_post-launch-learning.md` (Post-Launch Learning and ROI) — all
planned.

**Prerequisites:** [Module 05 index](../../01_foundations/05_experimentation_pilots_evidence/_index.md) ·
[Module 05 topic 02 — Randomisation
Units](../../01_foundations/05_experimentation_pilots_evidence/02_randomisation-units.md) ·
[Module 05 topic 05 — Switchback and Operational Pilot
Designs](../../01_foundations/05_experimentation_pilots_evidence/05_switchback-pilots.md) · [Part I
index](../../01_foundations/_index.md)

**Adjacent method:** [Module 07 topic 04 — Regression with Lags and Fourier
Terms](../07_time_series_forecasting/04_regression-lags-fourier.md) · [Module 07
index](../07_time_series_forecasting/_index.md)

**Module survey:**
[`09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)

> **Note:** Module 06 (machine learning foundations) is deferred in this edition and is not
> referenced here. Where a machine-learning baseline would sit, this topic defers to Module 07 for
> the forecast baseline and to Module 05 for the design.
