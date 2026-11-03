# Overrides and Forecast Value Added

*Measuring whether a planner's adjustment improves the forecast.*

> Module 07 — Time Series Forecasting Foundations
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 09; Module 05 topics 01 and 05; Module 07 topics 01–09
> Version 2026.5

---

## 1. Why this topic exists

The forecast a company acts on is almost never the forecast the model produced. Between the
model run and the purchase order there is at least one person, usually three, and sometimes a
meeting. That person is doing the most consequential forecasting work in the building, and until
someone measures it, nobody knows whether it helps.

Forecast value added (FVA) is the measurement. It compares the forecast that was published
against the forecast the system would have issued anyway, on the same series and the same
horizon, and asks whether the difference made the number better or worse.

Without that comparison, three things go wrong, and all three are common.

**One. The improvement that belongs to the model is credited to the planner.** A team reports
that its overrides improved accuracy by six per cent. The number is real arithmetic. The
comparison underneath it is final forecast against a naive benchmark, so it contains the
statistical model's entire contribution plus the planner's. Subtract the model's contribution and
the planner's share is often negative. This is the single most common error in override reporting,
and it survives because the deck looks rigorous.

**Two. A planner who is right is punished, or one who is wrong is promoted.** Because nobody
measures the counterfactual, the planner who overrode a bad system forecast and was proved right
is indistinguishable from the planner who overrode a good one and got lucky. Promotion, coaching,
and automation decisions are then made on narrative.

**Three. Value is moved rather than created, and the movement is booked as skill.** A planner
pulls next month's volume into this month to close a gap. This month's accuracy improves. Next
month's collapses. Across the horizon the sum is zero, or negative once the plan has been
disturbed. A cell-level report sees only the improvement.

There is a fourth reason, specific to this moment. As statistical baselines and machine-learned
forecasts improve, the human contribution becomes smaller and more contestable. Every automation
business case is now, explicitly or implicitly, an argument about FVA: the system is at least as
good as the planner on most items, therefore let the system run. That argument may be right, but
it is only as good as the measurement behind it, and most organisations do not have the
measurement. They have an override count and an anecdote.

**Why the topic is politically awkward.** Accuracy metrics are bounded below by zero and most
people score in the seventies or eighties. FVA has a natural zero that means "you contributed
nothing" and a negative region that means "you made it worse than doing nothing." It is the only
routine forecast metric that can tell a professional their judgement has been harmful. That is
why the person being measured so often controls the measurement, and why the honest designs in
section 5 are mostly about taking that control away.

This is the last topic of the module. The nine before it built the apparatus: benchmark, method,
level, evaluation. This one asks who is still in the loop after all of that, and whether the
answer is defensible.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** forecast value added and **distinguish** it from forecast accuracy, naming the
   quantity each one measures and the decision each one supports.
2. **Construct** the three-baseline comparison — naive, statistical, final — for a set of
   series-horizon cells, and **decompose** a reported improvement into the model's share and the
   planner's share.
3. **Diagnose** a naive override-accuracy claim by naming the baseline error and the selection
   error it contains.
4. **Design** an honest override measurement: what to log, when to freeze it, how to define the
   population, and how to report a distribution rather than a mean.
5. **Specify** a randomised or switchback trial of override policy, stating the randomisation
   unit, the blinding, the power requirement, and the ethical constraint.
6. **Distinguish** a value-creating override from a value-redistributing one, across both horizons
   and across series.
7. **Compute** FVA under an accuracy metric and under an asymmetric cost function, and
   **explain** why the two can disagree in sign.
8. **Write** the governance rules that keep FVA usable: the level of measurement, the reporting
   format, and the prohibition on rewarding accuracy alone.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a single forecast cell — one series, one horizon, one origin — with three
numbers stacked above it. The distance between the stacks is the whole subject.

### 3.1 The three baselines

```text
            ONE SERIES, ONE HORIZON, THREE BASELINES

   (1) NAIVE ......................... N      "what the business would
       seasonal-naive, or last         .      have done with no
       value, or last cycle's plan     .      forecasting process"
                                        .
        error(N) = 9.2%  ................  the bar almost everyone clears

                                        .
   (2) STATISTICAL ................... S      "the number the planner
       the model, unadjusted,          .      saw on the screen"
       frozen at issue                 .
                                        .
        error(S) = 4.2%  ................  the model's contribution: +5.0 pp

                                        .
   (3) FINAL ......................... F      "what was published and
       after overrides, after the      .      acted on"
       consensus meeting               .
                                        .
        error(F) = 5.4%  ................  the override's contribution: -1.2 pp

   THREE DELTAS, AND THEY ADD UP
   ---------------------------------------------------------------
   process  vs naive  = error(N) - error(F) = +3.8 pp   the whole chain
   model    vs naive  = error(N) - error(S) = +5.0 pp   not the planner's
   override vs system = error(S) - error(F) = -1.2 pp   the planner's

                        3.8  =  5.0  +  (-1.2)
```

Read the bottom line twice. The headline "our forecasting process improved accuracy by 3.8
points" is true. The headline "our overrides improved accuracy by 3.8 points" is false, and
overstates the override contribution by five points. The only number that belongs to the person
who changed the forecast is the third delta, and in this illustration it is negative.

### 3.2 Two questions, one diagram

The diagram answers two different questions, and most reporting conflates them.

- **Is the forecasting process worth running at all?** That is baseline 1 against baseline 3. It
  is the question a chief financial officer asks when deciding whether to fund a planning
  function.
- **Is the human in the loop worth keeping?** That is baseline 2 against baseline 3. It is the
  question a head of demand planning asks when deciding whether to let the system run untouched.

Both questions are legitimate. They require different baselines, and a report that answers the
first while claiming to answer the second is the central failure this topic exists to prevent.

There is a third baseline worth naming because it is the one management often means without
saying so: **last cycle's published forecast**, or the locked plan. The delta from last cycle to
this cycle answers "did we improve since we last committed," which is a business question, not a
statistical one, and it mixes model change, information arrival, and human adjustment into one
number. Useful for a monthly business review; useless for attributing credit.

### 3.3 The zero point

Accuracy has no natural zero. FVA does, and this is what makes it uncomfortable.

| Value | Meaning | What a manager hears |
|---|---|---|
| Positive | The step reduced error | "This is working" |
| Zero | The step changed nothing measurable | "Why are we paying for this step?" |
| Negative | The step increased error | "This step is harmful" |

A planner with a 78% accuracy score is, on the face of it, doing well. A planner with an FVA of
minus 1.2 percentage points is being told their professional judgement detracted from the number
they are paid to produce. The two statements can describe the same person in the same quarter,
because most of the 78% was produced by the model.

> **Note:** FVA measures the marginal contribution of an intervention, not the competence of the
> person making it. A planner who inherits a strong model will show low or negative FVA while
> being excellent, because there is little left to add. A planner who inherits a broken model will
> show high FVA while being ordinary. The metric is only interpretable alongside the quality of
> the baseline it is measured against.

### 3.4 The second axis

Accuracy is one axis; cost is another, and they can disagree in sign. Section 6 carries an example
in which the override makes absolute error worse and total cost better, because the error it adds
falls on the cheap side of an asymmetric cost function. The metric's name promises the second axis;
most implementations measure only the first.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Override | A planner's replacement of the system-generated forecast for a specific series and horizon | Used for any change to a plan, including quantity and timing changes downstream | Numeric, in forecast units |
| Judgemental adjustment | The academic term for an override, including both numeric edits and structured adjustments | Treated as synonymous with bias correction | Numeric delta or replacement value |
| System forecast | The number the statistical or machine-learned model produces before any human edit | Confused with the final published forecast | Units per bucket |
| Statistical baseline | Synonym for system forecast when the model is a classical time-series method | Used loosely for any benchmark | Units per bucket |
| Final forecast | The forecast published after all human steps, including consensus | Treated as "the forecast" without qualification | Units per bucket |
| Consensus forecast | The output of a cross-functional review meeting, which may contain several overrides | Assumed to be a single planner's judgement | Units per bucket, with an owner |
| Naive forecast | A benchmark that carries the last observed value forward | Called a model rather than a floor | Units per bucket |
| Seasonal-naive forecast | A benchmark that carries the same period last year forward | Confused with a seasonal model | Units per bucket |
| Random walk | The formal name for the naive forecast under a stochastic process | Used as an insult rather than a technical term | Units per bucket |
| Benchmark | Any stated reference forecast against which a method is scored | Chosen after seeing results | Units per bucket, plus a definition |
| Forecast value added (FVA) | The change in a forecast performance metric attributable to a specific step or participant | Reported as a single organisation-wide number | Percentage points, or currency |
| FVA stairstep | A chart showing error after each sequential process step, from naive to final | Used as a synonym for FVA itself | Ordered categorical steps |
| Counterfactual forecast | The forecast that would have been published had the step not occurred | Assumed to be observable rather than reconstructed | Units per bucket |
| Shadow forecast | A recorded but unapplied forecast, kept for comparison | Confused with a rejected override | Units per bucket, flagged |
| Blind period | A period in which a step's effect is withheld from the published number | Confused with a holdout in model evaluation | Calendar window |
| Override rate | Share of series-horizon cells that a human edited | Reported without the denominator of reviewed cells | Proportion, 0 to 1 |
| Override magnitude | Size of the edit relative to the system forecast | Reported in absolute units across unequal series | Proportion of system forecast |
| Reason code | A coded label for why an override was made, recorded at the time | Written retrospectively after the outcome is known | Controlled vocabulary |
| Directional override | An override classified as upward or downward relative to the system forecast | Confused with an error direction | Up, down, or none |
| Forecast error | A measure of distance between forecast and actual | Used without stating the metric | Units or percentage |
| Mean absolute percentage error (MAPE) | Mean of absolute error divided by actual, as a percentage | Compared across series with different volumes | Per cent, undefined at zero actual |
| Weighted MAPE (WMAPE) | Sum of absolute errors divided by sum of actuals | Assumed to equal a volume-weighted average of MAPEs | Per cent |
| Mean absolute error (MAE) | Mean of absolute error in forecast units | Compared across series of different scale | Units |
| Mean absolute scaled error (MASE) | MAE divided by the in-sample MAE of a naive benchmark | Computed with an out-of-sample denominator by mistake | Ratio, dimensionless |
| Average relative MAE (AvgRelMAE) | Geometric mean of the ratio of a method's MAE to a benchmark's MAE | Assumed to rank methods the same way as MAPE | Ratio, dimensionless |
| Bias | Mean signed error, showing systematic over- or under-forecast | Confused with accuracy | Units, or per cent of actual |
| Tracking signal | Cumulative bias divided by mean absolute deviation | Used as an override trigger without a threshold rule | Ratio |
| Series-horizon cell | One forecast for one series at one horizon from one origin | Treated as interchangeable with a series | Indexed tuple |
| Horizon | Number of periods between forecast origin and target period | Pooled across without saying so | Integer periods |
| Aggregation level | The hierarchy level at which a forecast is made or scored | Changed between measurement and reporting | Category, item, location |
| Selection effect | Distortion from scoring only the cases where a person chose to act | Described as "the overrides we looked at" | Statistical, not a unit |
| Regression to the mean | The tendency of extreme observations to be followed by less extreme ones | Mistaken for skill in an override applied at a spike | Statistical, not a unit |
| Goodhart's law | When a measure becomes a target, it ceases to be a good measure | Quoted and then ignored | None |
| Algorithm aversion | The tendency to lose confidence in an algorithm faster than in a human after equivalent errors | Confused with distrust of technology generally | Behavioural, measured in choice rates |
| Automation bias | Over-reliance on a system's output, the opposite failure | Assumed not to coexist with aversion | Behavioural |
| Debiasing | A procedure that corrects a systematic direction in adjustments | Confused with removing human judgement | Model or procedure |
| Newsvendor asymmetry | The cost structure in which under- and over-forecasting carry different penalties | Ignored when accuracy is the only metric | Currency per unit, two rates |
| Silent override | A human intervention made through a parameter (safety stock, lead time, allocation) rather than the forecast | Excluded from FVA because the forecast looks untouched | Parameter delta |
| Value-creating override | An override that improves the relevant metric across the full horizon window | Equated with any positive cell-level FVA | Metric-specific |
| Value-redistributing override | An override that moves error between periods, series, or metrics without reducing it | Booked as an improvement | Metric-specific |
| Sandbagging | Deliberately under-forecasting to create headroom | Confused with conservative planning | Bias, in units |
| Padding | Inflating a forecast to protect service or secure supply | Treated as a planning decision rather than a forecast | Bias, in units |
| Immutable forecast archive | A store of as-issued forecasts that cannot be overwritten | Assumed to exist in the planning system | Database object |
| As-of data | Data as it stood at a stated timestamp, including later revisions | Replaced by current values | Timestamped record |
| Forecast KPI | A key performance indicator computed on forecast quality | Assumed to be decision-relevant | Metric with a definition |
| Override governance | The rules governing who may override, how it is recorded, and how it is reviewed | Treated as an IT permissions question | Policy document |

Three of these carry most of the damage when misunderstood.

**Override versus plan change.** An override changes what the company believes will happen. A
plan change alters what the company intends to do. They are recorded in different systems, owned
by different roles, and scored against different things. A planner who lowers the forecast because
supply cannot cover demand has changed the plan and disguised it as a forecast; the accuracy
report will call it an error, correctly, and the organisation will learn the wrong lesson.

**MAPE versus WMAPE.** MAPE averages percentage errors across series, so a small item with a
50% error counts as much as a large item with a 5% error. WMAPE sums absolute errors and divides
by summed actuals, so volume dominates. A planner who improves small items and damages large ones
can improve MAPE while destroying WMAPE. The two numbers are not two views of one thing; they
answer different questions. Section 9 of the error-metrics topic (`09_error-metrics-cost.md`,
planned) treats the choice in full.

**Accuracy FVA versus cost FVA.** Accuracy FVA asks whether error fell. Cost FVA asks whether
the money spent on the consequences fell. They coincide only when the cost of error is symmetric,
which it almost never is in supply chains, where a stock-out usually costs more per unit than a
week of excess inventory.

## 5. The mechanics, step by step

The procedure has five phases. Each step states what goes in, what comes out, and the branch point
where a careless analyst goes wrong.

### Phase A — Capture, before any measurement is possible

**Step 1. Freeze the as-issued system forecast.**

- Input: the model run that produced the number the planner saw.
- Output: an immutable record of that number per series and horizon, with a timestamp and a model
  version.
- Branch point: **does the planning system overwrite forecasts in place?** Most do. When a planner
  edits a forecast cell, the previous value is gone unless the system keeps an audit trail or the
  analyst extracts a snapshot before the edit window opens. Reconstructing the pre-edit value
  afterwards is guesswork, and guesswork is where most FVA programmes quietly die.

**Step 2. Log the override at cell level.**

- Input: the planner's edit.
- Output: a record containing the series, the horizon, the origin, the system value, the final
  value, the timestamp, the role of the editor, a reason code, and the free-text note.
- Branch point: **is the reason code required, or optional?** Optional codes are filled in only
  when the planner has time, which correlates with the overrides that went well. A taxonomy built
  on optional codes describes a biased sample.

**Step 3. Record the actual on the forecast's definition.**

- Input: the demand or shipment history.
- Output: the outcome for each cell, on the same unit, the same bucket, and the same calendar as
  the forecast.
- Branch point: **is the actual a sale or a shipment?** If the forecast is of demand and the
  actual is shipments, then a stock-out suppresses the actual, and the forecast that caused the
  stock-out is scored as accurate. This is censoring, and it rewards under-forecasting. Module 01A
  topic 08 treats the general problem.

**Step 4. Fix the baseline set before the period starts.**

- Input: a policy decision.
- Output: a written statement of which benchmarks will be used and why.
- Branch point: **is the baseline chosen after the results are visible?** If so, it will be the one
  that flatters. The choice must be dated, and the date must precede the outcome.

### Phase B — Align, and decide what the population is

**Step 5. Join on series, horizon, origin, and vintage.**

- Input: the four tables — system forecast, override log, final forecast, actuals.
- Output: one row per cell, with all three forecasts and the actual.
- Branch point: **do the vintages match?** A final forecast issued on Monday joined to an actual
  restated on Friday is a comparison of two different datasets. Every join in this procedure needs
  an as-of timestamp.

**Step 6. Decide what counts as the population.**

Three defensible populations, three different questions:

| Population | Question it answers | What it hides |
|---|---|---|
| All reviewed cells, overridden or not | Did the review process add value? | Nothing, if review coverage is complete |
| Overridden cells only | Did the overrides that were made help? | The cells the planner chose not to touch |
| All cells in the hierarchy | Did the forecast improve overall? | Everything about individual contribution |

The default in most reporting is the second, presented as if it were the third. State which one
you used, and report the override rate alongside it, or the reader cannot tell whether a low FVA
means bad overrides or few of them.

**Step 7. Decide the unit of account.**

- A cell (series, horizon, origin). Fine-grained, noisy, the natural unit of a decision.
- A series across all horizons. Answers whether the planner helped this item.
- An aggregate across series. Answers whether the portfolio helped, and hides everything else.
- Branch point: **is the level of measurement the level of the decision?** A planner decides per
  item per week. A director decides per category per quarter. Both measurements are legitimate;
  a report that measures at one level and is used to manage at another is not.

**Step 8. Handle the hierarchy and the horizon.**

- Output: a rule for whether a fix at item level counts when the category total worsens.
- Branch point: **does the sum of cell-level FVA equal the aggregate FVA?** It does not, because
  errors do not add linearly. A planner can improve every item and worsen the total, by pushing
  errors in the same direction. Publish both, and say which one the decision uses.

### Phase C — Score

**Step 9. Choose the error metric and state the sign convention.**

- Output: a metric, a denominator, and a signed convention written down.
- Branch point: **is positive FVA "error fell" or "error rose"?** Half the industry uses one,
  half the other. Every report needs the definition on its face. This topic uses positive = error
  fell = value added.

**Step 10. Compute the three pairwise deltas.**

- Output: process FVA (naive to final), model FVA (naive to system), override FVA (system to
  final).
- Branch point: **does the arithmetic close?** `process = model + override`, exactly, when the
  three use the same metric and the same cells. If it does not close, one of the three is computed
  on a different population. This is the cheapest integrity check in the whole procedure.

**Step 11. Weight, or do not weight, and say which.**

- Output: a stated weighting rule.
- Branch point: **is the mean dominated by one cell?** In the worked example in section 6, a
  single cell contributes more than the entire net result. An unweighted mean across cells and a
  volume-weighted mean can disagree in sign. Report both when they disagree, and investigate why.

**Step 12. Compute the distribution.**

- Output: share of cells improved, median delta, interquartile range, and the fifth percentile.
- Branch point: **is the mean carrying the whole message?** A mean of −0.4 percentage points could
  be uniform mild harm or a few catastrophes offset by many small gains. The policy response is
  completely different in each case.

**Step 13. Segment.**

Segment by reason code, by direction, by magnitude, by planner role, and by ABC class. The
published findings are consistent enough to make these the first cuts: positive adjustments
behave differently from negative ones, small adjustments differently from large ones, and
information-driven overrides differently from relationship-driven ones.

### Phase D — Interpret

**Step 14. Separate skill from selection.**

- Output: a statement of how the overridden cells differed from the unmodified ones before any
  override occurred.
- Branch point: **were the overridden cells harder?** If the planner intervened where the system
  error was already high, then the overridden population is a selected sample and its raw error
  cannot be compared with the population average. The honest comparison is always within the
  overridden cells, against the system forecast for those same cells.

**Step 15. Separate creation from redistribution.**

Three redistribution patterns account for most false positives:

1. **Between horizons.** Volume pulled from next month into this month. Signature: positive FVA
   at horizon h, negative at h+1, sum approximately zero.
2. **Between series.** A planner fixes the category total by moving volume between items.
   Signature: item-level FVA negative, aggregate positive.
3. **Between metrics.** An upward override reduces MAPE because MAPE penalises over-forecasting
   more than under-forecasting, while WMAPE or bias worsens. Signature: one metric improves, the
   other degrades, and the report quotes one.

**Step 16. Separate accuracy from cost.**

- Output: FVA under the accuracy metric and FVA under the cost function, side by side.
- Branch point: **is the cost function known before the period?** If it is fitted afterwards, it
  will be fitted to the result. Write down the two rates — cost per unit of over-forecast and
  cost per unit of under-forecast — and get them signed off by the function that bears them.

**Step 17. Decide.**

Four responses, and the choice depends on the segment, not the average:

| Finding | Response |
|---|---|
| Information-driven overrides add value; relationship-driven ones do not | Keep the first, coach or restrict the second |
| A specific planner's overrides are negative across segments and periods | Investigate the workflow, not the person; the cause is usually a missing signal or a perverse incentive |
| Overrides are value-neutral and expensive | Consider removing the step and redeploying the time |
| FVA is negative but the cost FVA is positive | The accuracy metric is the wrong target; fix the metric before the people |

### Phase E — Govern

**Step 18. Publish on a fixed cadence, with a fixed format.**

Monthly or quarterly, same cuts, same definitions, with the sample size and the coverage on the
page. A report that changes its cuts whenever the result is uncomfortable is a press release.

**Step 19. Audit a sample of reason codes.**

Pull fifty overrides, read the free text, and ask whether the code matches the note. Coders drift,
and a taxonomy nobody audits becomes a set of labels that mean "I did something."

**Step 20. Retire the metric before it becomes a target.**

The moment FVA appears in a bonus formula, three behaviours appear: overrides concentrate on easy
cells, positive overrides disappear, and the record stops being a record. Use FVA to manage the
process and to design the workflow. Use it on individuals only with the protections in section 10,
and preferably not at all.

### 5.1 What makes the counterfactual hard

The system forecast is an observable counterfactual only if nothing else changed. Three things
usually do.

**The consensus meeting.** If a planner's override is followed by a sales override and a finance
override, the published number contains three judgements. Attribution requires a stairstep, not
a two-point comparison, and each step needs its own frozen value. Most systems log only the last
edit.

**The silent override.** Planners also intervene through parameters: a safety-stock multiplier, a
lead-time override, an allocation rule, a manual order. These do not touch the forecast, so a
numeric FVA sees nothing, while the actual intervention — often the more consequential one — goes
unmeasured.

**Reflexivity.** The forecast triggers a purchase, the purchase changes availability, availability
changes what customers can buy, and the recorded demand is partly a function of the forecast. The
system is not predicting an exogenous series. It is participating in one. Any accuracy measure
inherits this, and FVA inherits it too.

### 5.2 Three designs for an honest measurement

| Design | What is withheld or randomised | What it supports | Main weakness |
|---|---|---|---|
| Shadow log | Nothing; the override is applied and the unapplied value kept | A same-cell comparison | The planner still chooses when to override |
| Blind period | Overrides recorded but not published for a window | A policy comparison over time | Confounded by anything else that changed |
| Randomised application | Each reviewed cell assigned to apply or not apply the recorded override | A causal estimate | Needs volume, consent, and an exemption rule |

The third design is the only one that supports a causal claim, and almost nobody runs it. Randomise
within strata, ABC class and horizon, because error scales differ sharply across them; score the
shadow forecast rather than the planner's intention, which is unobservable once the outcome is
known. The binding constraint is ethical: you cannot withhold an override the planner believes is
needed for a customer commitment or a safety obligation. Exempt those, record the exemption, and
report the exemption rate, because a high rate means the trial is measuring the overrides nobody
cared about. The design logic is generic; see
[`01_claim-hypothesis-design.md`](../../01_foundations/05_experimentation_pilots_evidence/01_claim-hypothesis-design.md)
and, for alternating-period designs where cell-level randomisation is impractical,
[`05_switchback-pilots.md`](../../01_foundations/05_experimentation_pilots_evidence/05_switchback-pilots.md).

## 6. Worked example

An illustrative case, internally consistent but constructed. A beverage distributor forecasts ten
item-location series over a four-week horizon. The numbers below are one quarter's worth of
reviewed cells, ten of them, chosen to be readable rather than representative.

### 6.1 The data

Units are cases of twelve. `N` is the seasonal-naive forecast, `S` the system forecast frozen at
issue, `F` the final published forecast, `A` the actual.

| Cell | Series | Horizon (wk) | N | S | F | A |
|---|---|---:|---:|---:|---:|---:|
| 1 | Sparkling 330ml / DC-North | 1 | 500 | 420 | 460 | 455 |
| 2 | Sparkling 330ml / DC-South | 1 | 300 | 310 | 290 | 330 |
| 3 | Still 500ml / DC-North | 1 | 820 | 780 | 780 | 772 |
| 4 | Still 500ml / DC-South | 2 | 160 | 155 | 180 | 170 |
| 5 | Cola 330ml / DC-North | 2 | 690 | 640 | 700 | 610 |
| 6 | Cola 330ml / DC-South | 2 | 210 | 225 | 225 | 237 |
| 7 | Tonic 200ml / DC-North | 3 | 470 | 505 | 480 | 520 |
| 8 | Tonic 200ml / DC-South | 3 | 95 | 90 | 120 | 105 |
| 9 | Multipack / DC-North | 4 | 1,240 | 1,150 | 1,150 | 1,120 |
| 10 | Multipack / DC-South | 4 | 380 | 375 | 400 | 395 |

Seven of the ten cells were overridden (cells 1, 2, 4, 5, 7, 8, 10). Three were not (3, 6, 9).

### 6.2 The three-baseline comparison

| Cell | Error N | Error S | Error F | FVA (S to F) |
|---|---:|---:|---:|---:|
| 1 | 45 | 35 | 5 | +30 |
| 2 | 30 | 20 | 40 | −20 |
| 3 | 48 | 8 | 8 | 0 |
| 4 | 10 | 15 | 10 | +5 |
| 5 | 80 | 30 | 90 | −60 |
| 6 | 27 | 12 | 12 | 0 |
| 7 | 50 | 15 | 40 | −25 |
| 8 | 10 | 15 | 15 | 0 |
| 9 | 120 | 30 | 30 | 0 |
| 10 | 15 | 20 | 5 | +15 |
| **Sum** | **435** | **200** | **255** | **−55** |

Sum of actuals: `455 + 330 + 772 + 170 + 610 + 237 + 520 + 105 + 1120 + 395 = 4,714` cases.

```text
WMAPE(naive)    = 435 / 4,714 = 9.23%
WMAPE(system)   = 200 / 4,714 = 4.24%
WMAPE(final)    = 255 / 4,714 = 5.41%

process  FVA = 9.23 - 5.41 = +3.82 pp
model    FVA = 9.23 - 4.24 = +4.99 pp
override FVA = 4.24 - 5.41 = -1.17 pp

check: 4.99 + (-1.17) = 3.82
```

The arithmetic closes. That is the first thing to verify in any such table, because it fails
whenever the three numbers were computed on different populations.

### 6.3 The slide that would be shown

The quarterly review slide reads: *"Planner overrides improved forecast accuracy by 3.8
percentage points."* Every number on it is correct. The attribution is wrong by five points. The
honest version is: the forecasting process improved accuracy by 3.8 points, the model supplied
5.0 of those, and the overrides removed 1.2.

### 6.4 The distribution, not the mean

Three cells improved (1, 4, 10), three worsened (2, 5, 7), four were unchanged (3, 6, 8, 9). Of
the seven overridden cells, three improved, three worsened, one was unchanged.

The mean signed FVA is `-55 / 10 = -5.5` cases per cell. That number is nearly meaningless: cell
5 alone contributes −60 cases, so without it the net would be +5. The single largest override,
an upward adjustment of 60 cases on a series that then came in 90 cases below the final forecast,
is the entire finding. Any report that leads with the mean and buries the outlier has told the
reader nothing actionable.

### 6.5 The selection question

Were the overridden cells harder than the untouched ones? Compare the system forecast's error
before any override:

```text
overridden cells   1,2,4,5,7,8,10 : 35+20+15+30+15+15+20 = 150 / 7 = 21.4 cases
untouched cells    3,6,9          :  8+12+30             =  50 / 3 = 16.7 cases
```

The planner intervened where the system was, on average, 28% worse. That is rational behaviour,
and it is also why the overridden cells cannot be compared with the portfolio average. The only
valid comparison is inside the overridden cells, against the system forecast for those same
cells, which is what column FVA does.

### 6.6 The cost axis

Suppose the business states its cost structure in advance: excess stock costs EUR 2.00 per case
per week, and a shortfall costs EUR 7.50 per case in expediting and substitution. The asymmetry is
typical of a distributor with fast expedite options and cheap storage. Cost per cell is

```text
cost = 2.00 x max(0, forecast - actual) + 7.50 x max(0, actual - forecast)
```

| Cell | Cost N (EUR) | Cost S (EUR) | Cost F (EUR) | Cost FVA (EUR) |
|---|---:|---:|---:|---:|
| 1 | 90.00 | 262.50 | 10.00 | +252.50 |
| 2 | 225.00 | 150.00 | 300.00 | −150.00 |
| 3 | 96.00 | 16.00 | 16.00 | 0 |
| 4 | 75.00 | 112.50 | 20.00 | +92.50 |
| 5 | 160.00 | 60.00 | 180.00 | −120.00 |
| 6 | 202.50 | 90.00 | 90.00 | 0 |
| 7 | 375.00 | 112.50 | 300.00 | −187.50 |
| 8 | 75.00 | 112.50 | 30.00 | +82.50 |
| 9 | 240.00 | 60.00 | 60.00 | 0 |
| 10 | 112.50 | 150.00 | 10.00 | +140.00 |
| **Sum** | **1,651.00** | **1,126.00** | **1,016.00** | **+110.00** |

```text
process  cost FVA = 1,651.00 - 1,016.00 = +635.00 EUR  (+38.5%)
model    cost FVA = 1,651.00 - 1,126.00 = +525.00 EUR  (+31.8%)
override cost FVA = 1,126.00 - 1,016.00 = +110.00 EUR  (+9.8%)

check: 525.00 + 110.00 = 635.00
```

The sign has flipped. On absolute error the overrides destroyed value (−1.17 percentage points).
On the stated cost function they created EUR 110 on a base of EUR 1,126, about 9.8%. The reason is
in the last column of the first table: the net effect of the overrides was upward by 135 cases
(`40 − 20 + 25 + 60 − 25 + 30 + 25`), and in a cost regime where a shortfall costs 3.75 times an
excess, an upward shift is generally rewarded.

Cell 8 is the clearest case. The override changed a 15-case under-forecast into a 15-case
over-forecast: absolute error unchanged, FVA zero, and cost fell from EUR 112.50 to EUR 30.00.
An accuracy-only report scores that override as worthless. The business would score it as
valuable. Both are correct answers to different questions.

### 6.7 By reason code

The reason codes were recorded at the time of the edit, before the outcome was known.

| Reason class | Cells | Unit FVA (cases) | Cost FVA (EUR) |
|---|---|---:|---:|
| Known event or new information (promotion, listing) | 1, 4, 8, 10 | +50 | +567.50 |
| Sales team input (no verifiable event) | 2, 5, 7 | −105 | −457.50 |
| **Total** | **10** | **−55** | **+110.00** |

This split is the one practitioners most often report, and it matches the published pattern:
overrides justified by specific, checkable information behave differently from overrides carrying
a relationship or a feeling. The second class is not worthless by definition, but it needs
evidence before it is trusted, and in this sample it has none.

### 6.8 What the example does not prove

It is ten cells. The confidence interval on a −1.17 percentage point estimate from ten
observations spans zero comfortably. The cost rates are assumptions. The reason codes are part of
the construction rather than an independent record; in a real dataset the equivalent risk is
retrospective coding, which is the failure section 5 tells you to design against. Read the example
as a demonstration of method, not as evidence about planners.

## 7. Code, query, or calculation

### 7.1 The formulas

For cell `i`, with system forecast `S_i`, final forecast `F_i`, actual `A_i`, and naive forecast
`N_i`, define the quantities as follows. Positive FVA means error fell.

| Quantity | Definition | Unit |
|---|---|---|
| Cell error, system | `eS_i = abs(S_i - A_i)` | forecast units |
| Cell error, final | `eF_i = abs(F_i - A_i)` | forecast units |
| Cell error, naive | `eN_i = abs(N_i - A_i)` | forecast units |
| Cell FVA | `FVA_i = eS_i - eF_i` | forecast units, positive = value added |
| Cell FVA, percentage | `100 x FVA_i / A_i`, undefined at a zero actual | percentage points |
| Weighted error | `WMAPE(x) = 100 x sum(eX_i) / sum(A_i)` | per cent |
| Process FVA | `WMAPE(N) - WMAPE(F)` | percentage points |
| Model FVA | `WMAPE(N) - WMAPE(S)` | percentage points |
| Override FVA | `WMAPE(S) - WMAPE(F)` | percentage points |
| Identity | `process = model + override` | must close to rounding |

Two things to check every time. The identity must hold to rounding. And the sign convention must
be printed next to the number, because a reader who assumes the opposite will invert your
conclusion.

### 7.2 The query that builds the comparison

One structured query language (SQL) block. It assumes a warehouse with an archived system
forecast, an override log, and a final forecast, all keyed by series, horizon, and origin. Names
will need adjusting.

```sql
with sys as (
    select series_id, origin_id, horizon, model_version, system_forecast
    from forecast.system_snapshot            -- frozen at issue, never updated
    where origin_id = date '2026-06-01'
),
fin as (
    select series_id, origin_id, horizon, final_forecast, override_flag
    from forecast.final_published
    where origin_id = date '2026-06-01'
),
act as (
    select series_id, origin_id, horizon, actual_units
    from forecast.actuals_asof
    where asof_ts = timestamp '2026-09-30 23:59:59'
)
select s.series_id, s.horizon,
       s.system_forecast, f.final_forecast, a.actual_units, f.override_flag,
       abs(s.system_forecast - a.actual_units) as err_system,
       abs(f.final_forecast  - a.actual_units) as err_final,
       abs(s.system_forecast - a.actual_units)
         - abs(f.final_forecast - a.actual_units) as fva_units
from sys s
join fin f using (series_id, origin_id, horizon)
join act a using (series_id, origin_id, horizon)
order by s.series_id, s.horizon;
```

Expected output for the worked example: ten rows, `fva_units` summing to −55, and `override_flag`
true on seven of them. The naive column is omitted here because it lives in a separate benchmark
table; join it on the same keys to complete the three-baseline comparison.

### 7.3 Two hand checks

**Check the identity on the worked example.** `200 − 255 = −55` and `435 − 200 = 235`, so
`435 − 255 = 180 = 235 + (−55)`. The unit identity holds. In percentage terms the same relation
holds to rounding: `3.82 = 4.99 − 1.17`.

**Check that the mean is not the message.** Sort the ten cell deltas: `−60, −25, −20, 0, 0, 0, 0,
+5, +15, +30`. Median `0`, mean `−5.5`, range 90 cases. Reporting the mean alone describes a
distribution that does not exist in any single week.

## 8. Data, standards, and regulatory context

### 8.1 There is no FVA standard

No ISO, no IAS, no national regulator defines forecast value added. The Supply Chain Operations
Reference (SCOR) model published by the Association for Supply Chain Management (ASCM) includes
forecast accuracy among its reliability metrics, but it measures the process output, not the
marginal contribution of a human step, and it is a reference model rather than a binding standard.
Grade C for SCOR as a normative reference.

The consequence is that every organisation invents its own definition. That is acceptable only if
the definition is written down and stable. An FVA figure without a stated baseline, metric, cell
definition, and sign convention is not a measurement; it is a number with a story attached.

### 8.2 The archive is the real data requirement

| Object | Required content | Why it fails in practice |
|---|---|---|
| System forecast snapshot | Value per series, horizon, origin, model version, timestamp | Planning systems overwrite in place |
| Override log | Editor role, timestamp, before and after values, reason code, free text | Codes optional; free text unstructured |
| Final published forecast | The value that drove the decision, with its release time | Revisions after release are not versioned |
| Actuals as-of | The outcome as it stood at scoring time | Later restatements silently change history |
| Calendar and unit definitions | Bucket, week rule, unit of measure | Changed mid-year without a mapping |

None of these is regulated. All of them are prerequisites, and the first is the one most often
missing. A planning system that does not preserve the pre-edit number makes honest FVA
impossible, and no amount of statistical sophistication recovers it.

### 8.3 Where forecast overrides become an accounting control

If the forecast feeds a financial statement line, the override becomes an internal-control
matter rather than a planning-quality matter.

| Instrument | How it touches the override | Grade |
|---|---|---|
| IAS 2 / ASC 330, Inventories | Excess and obsolete provisions are driven by forecast-based demand assumptions; an optimistic override can delay a write-down | B |
| IAS 36, Impairment of Assets | Impairment testing uses management's approved cash flow forecasts; overrides to those forecasts are audit evidence | B |
| Sarbanes-Oxley Act 2002, sections 302 and 404 | Where forecasts feed disclosure or internal control over financial reporting, changes to the number require documented authorisation | B |
| Local statutory audit requirements | Material forecast assumptions are tested; the override trail is part of the evidence | B |

This is the strongest argument for an override log that most planners have never heard. The
control does not care whether the override was skilful. It cares whether the change was
authorised, recorded, and reviewable.

### 8.4 Employment and privacy

If FVA is used to rate an individual, it becomes employee monitoring.

- The General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, Article 88, permits
  member states to legislate specific safeguards for processing in the employment context. The
  national implementations differ.
- In several member states, works councils or employee representatives have consultation rights
  over systems that monitor individual performance.
- Performance data derived from a system log is personal data, subject to access, retention, and
  purpose-limitation rules.

> **Note:** This is educational description, not legal advice. Before publishing a named
> individual FVA report, consult employment counsel in each jurisdiction. The default that avoids
> most of the problem is to report FVA by process step and by segment, never by person.

### 8.5 What is convention rather than requirement

- The choice of naive baseline. Seasonal-naive is conventional for weekly data; nothing requires it.
- The sign convention. Either is defensible; consistency is what matters.
- The reporting cadence. Monthly is common; quarterly is often more stable.
- The stairstep format, and the single summary number. Both conventional, neither mandated, and
  section 6 shows why the second is usually wrong.

## 9. Industry practice and case evidence

### 9.1 The published findings

| Study | Design | Finding | Grade |
|---|---|---|---|
| Fildes, Goodwin, Lawrence and Nikolopoulos (2009), *International Journal of Forecasting* | 60,000+ forecasts, four United Kingdom supply-chain firms (pharma, food, household products, retail) | About 75% of forecasts were adjusted; adjustments improved accuracy in three of four firms, with MAPE down 3.6 points on average; small adjustments often damaged accuracy; more than a third of upward adjustments in the non-retail firms were wrong-sided | A |
| Davydenko and Fildes (2013), *International Journal of Forecasting* | Re-analysis of adjustment accuracy using relative measures | Different error metrics rank adjustments differently; with a relative measure, about 57% of adjustments improved accuracy (about 53.5% of upward and 66% of downward adjustments) | A |
| Fildes, Goodwin and De Baets (2025), *International Journal of Forecasting* 41(2), 649–669 | Pooled re-analysis of about 147,000 forecasts from six studies | Adjustments improved accuracy or bias for only just over half of stock-keeping units (SKUs); upward adjustments were more likely to worsen; large downward adjustments tended to help; evidence that forecasters used information unavailable to the algorithm was weak; a debiasing procedure helped | A |
| Morlidge (2014), *Foresight* 33 | Large multi-company sample reported as more than 300,000 forecasts | 52% of forecasts failed to beat a random-walk benchmark | B |
| Seifert, Markoff and Spooner (2024), I by IMD | Practitioner commentary | "For the majority, often 80% or more, intervention by planners reduces the accuracy of the forecast" | C |

The 80% and the 52% figures are not the same claim, and neither is the same as the peer-reviewed
"just over half." The IMD figure is presented in the source as a typical result, not as a
controlled finding; treat it as an illustration of the direction. The Morlidge figure concerns
whether the published forecast beat a naive benchmark at all, which is the process question, not
the override question. Only the academic studies isolate the adjustment's contribution, and they
put the share of SKUs improved at somewhere around one half to two thirds, depending on metric and
sample.

Read together, the finding to internalise is this: **negative override value on a substantial
minority of series is the normal published result, not a scandal.** A programme that reports it
has probably measured something real. A programme that reports uniform improvement has probably
measured the wrong baseline.

### 9.2 Named practice

Gilliland's *Foresight* paper on FVA (2013) is the paper that codified the method for
practitioners: the stairstep chart, the choice of naive benchmark, and the histogram of value
added across items rather than a single average. The same paper reports case notes from Newell
Rubbermaid, Tempur-Pedic, Amway, and Cisco, and states that at Cisco FVA appeared on senior
management dashboards. Grade B for the method; grade C for the case notes, which are practitioner
reports rather than audited outcomes.

The claim worth taking from those notes is not that FVA improved accuracy. It is that FVA changed
the conversation from "how accurate are we" to "which steps earn their place." That is a
process-design question, and it is answerable in a way that an accuracy target is not.

### 9.3 Why the override exists at all

Dietvorst, Simmons and Massey (2015), *Journal of Experimental Psychology: General* 144(1),
114–126, ran five studies in which participants chose between an algorithm's forecasts and a
human's. People who saw the algorithm perform were less confident in it and less likely to choose
it over an inferior human forecaster, even after seeing it outperform the human. The mechanism
was differential loss of confidence after an equivalent error: humans are presumed to be able to
learn, algorithms are presumed not to. Grade A.

Their follow-up work (2016, *Management Science*) found that people will use an imperfect
algorithm if they can modify its output, even slightly. Grade A.

This is the most useful piece of evidence in the topic for anyone who has tried to remove an
override. The override is not only a forecasting act. It is the price of the planner's trust in
the system. A programme that deletes the override without addressing the trust will produce
workarounds — the silent override, the manual purchase order, the padded safety stock — that are
harder to measure than the one it removed.

### 9.4 What companies do, against what the evidence supports

| Practice | Observed frequency | Evidence position | Gap |
|---|---|---|---|
| Report override accuracy against a naive baseline | Common | Wrong attribution; overstates planner contribution | Use the system forecast as the baseline |
| Report a single average FVA | Common | Hides distribution; usually dominated by outliers | Report share improved, median, worst decile |
| Log reason codes | Occasional | Codes are the highest-value segmentation available | Make them mandatory and audit them |
| Preserve the pre-edit forecast | Rare | Without it, FVA cannot be computed at all | Archive at issue |
| Run a controlled override trial | Very rare | The only design that supports a causal claim | Randomise application, not recording |
| Use FVA for individual appraisal | Occasional | Creates gaming and legal exposure | Report by process step and segment |

The last row is where most of the organisational damage occurs. It is also the row with the
least evidence behind it in either direction: nobody has published a clean study of what happens
to a planning team when FVA becomes a personal score. What can be said is that the incentive to
abstain is immediate and rational — a planner who overrides nothing has an FVA of exactly zero,
which is better than a negative number — and an incentive structure that rewards doing nothing is
a design defect.

### 9.5 What practitioners argue about

1. **Which baseline** — naive answers "is the process worth running"; the system forecast answers
   "is the human worth keeping." Both are right for their question.
2. **Cells or aggregates** — cell-level scoring is noisy and can punish a correct portfolio
   decision; aggregate scoring hides item-level failure.
3. **Whether to block negative-FVA overrides** — the 2009 counterfactual is in-sample, untested as
   an intervention, and would also remove legitimate event-driven upward adjustments.
4. **KPI or diagnostic** — as a KPI it focuses attention, and Goodhart's law applies within a
   budget cycle.
5. **Accuracy or cost** — section 6 shows the two can disagree in sign on the same data.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Naive-baseline attribution | The FVA slide has no system-forecast column | Final is compared with naive, so the model's contribution is credited to the planner | Overrides look valuable when they are not | Print all three baselines on every slide |
| Scoring only overridden cells | "We scored 412 overrides" with no count of reviewed cells | The population is selected by the planner's own decision to act | FVA is uninterpretable and coverage is invisible | Report override rate alongside FVA |
| Reconstructed overrides | Override timestamps cluster at month end or at 23:59 | The pre-edit value was overwritten and rebuilt from memory or from the final | The counterfactual is a fiction | Freeze the system forecast at issue; make the log append-only |
| Metric switching | Last quarter MAPE, this quarter WMAPE, no restatement | The metric is chosen after the result is seen | Comparisons across periods are meaningless | Fix the metric in the reporting charter |
| Mean-only reporting | One bullet: "average FVA +0.4%" | A few large cells dominate; the distribution is unknown | Policy is set on an artefact | Publish share improved, median, worst decile |
| Override count as a target | "Reduce overrides by 30%" appears in objectives | Planners stop overriding rather than override better | Accuracy and service fall while the metric improves | Never target the count; target the outcome |
| FVA as an individual rating | Planner names appear in the FVA report | The measured person controls the reason codes and the timing | Gaming and legal exposure | Report by step and segment, not by person |
| Horizon-blind measurement | FVA computed on the week-one published number only | Pull-forward looks like skill at short horizons | Value is redistributed, not created | Score the full horizon window and the sum |
| Redistribution across series | Item FVA negative, category FVA positive, no explanation | Volume moved between items to fix an aggregate | Portfolio appears improved; item service degrades | Report both levels and the reconciliation |
| Cost function fitted after the fact | The cost ratio is first mentioned in the results meeting | The weights are chosen to produce a preferred sign | Cost FVA is circular | Agree the two rates in advance, in writing |
| Silent overrides excluded | Forecast accuracy stable while inventory and service swing | Intervention happens through parameters, not the forecast | The largest human effect is unmeasured | Log parameter changes and manual orders as interventions |
| Survivorship in the log | Overrides rejected by a manager are absent from the record | Only applied edits are stored | The measured set is the approved set, not the judged set | Record proposals, decisions, and rejections separately |
| Aggregate-only reporting | "Category FVA is +2%" and no item detail | Errors cancel in aggregation | Item-level harm is hidden | Publish at the decision level, aggregate as a summary |

Two deserve a note. **Naive-baseline attribution** is the most common and the most forgivable,
because the person producing the slide usually believes they are comparing the final forecast with
the model; the structural fix is to force all three numbers onto the page. **Override count as a
target** is the most damaging: the fastest way to improve the count is to stop overriding, and the
fastest way to improve measured accuracy is to stop intervening on the hard items. FVA as an
individual rating carries legal exposure as well as behavioural risk; see section 8.4.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails |
|---|---|---|
| The as-issued system forecast is recoverable | It is the counterfactual | FVA cannot be computed; any substitute is a reconstruction with unknown bias |
| The actual is measured on the forecast's definition, unit, and bucket | Errors are only meaningful within one definition | A unit or calendar mismatch produces error that is an artefact of the join |
| Demand is not censored by the forecast itself | Otherwise the forecast partly causes the outcome | Under-forecasting is rewarded; stock-outs score as accurate |
| The override is the only human step, or all steps are recorded | Attribution requires separating contributions | The measured delta belongs to the whole chain, not the planner |
| The cost function is known before the period and stable | Cost FVA depends on it entirely | The cost weights can be chosen to produce any sign |
| Cells are comparable across the period | Mix shifts change the weighted metric | An apparent improvement is a change in what was forecast |
| The scoring window is long enough for sampling noise to be small | FVA is an estimate, not a measurement | A one-quarter result is read as a verdict |
| The measurement does not change override behaviour | Otherwise the observed FVA is a response to being observed | The metric decays as soon as it is used |
| The baseline is fixed before the period | Otherwise it is selected to flatter | The result is unfalsifiable |
| Overrides are independent across cells | Standard errors and significance depend on it | One event affecting fifty cells is counted as fifty observations |
| The final forecast is what actually drove the decision | Otherwise accuracy is disconnected from consequence | You measure a number nobody used |

Two limits are worth stating plainly.

**FVA measures forecast quality, not business value.** The name promises more than the metric
delivers. An override that worsens weighted error but prevents a stock-out at a strategic account
may be the right decision, and accuracy FVA will record it as harmful. Measure business value
separately, against the decision's own cost or service outcome; where the two disagree, the
business outcome governs.

**FVA cannot resolve the reflexivity problem.** The forecast changes the world it is scored
against, so even a perfect counterfactual does not tell you what would have happened under a
different forecast. It tells you what happened under this one: enough to compare processes that
were run, not enough to simulate processes that were not.

## 12. Exercises

### Exercise 1 — Three baselines by hand (calculation, 40 minutes)

You are given twelve series-horizon cells with `N`, `S`, `F`, and `A` in cases. Compute
`WMAPE(N)`, `WMAPE(S)`, `WMAPE(F)`, and the two FVA deltas. State which of the three numbers a
slide claiming "overrides improved accuracy by 6%" would have quoted.

**Deliverable:** a table of the three WMAPEs and two deltas, with the identity check shown.
**Acceptance criteria:** correct arithmetic to two decimal places; identity closes; the misquoted
number is identified and the size of the overstatement stated in percentage points.

### Exercise 2 — Design the override log (design, 45 minutes)

Design the schema for an override log in a planning system that currently overwrites forecasts in
place.

**Deliverable:** a field list with types, a statement of what is immutable, and a retention rule.
**Acceptance criteria:** every field needed to reconstruct the three-baseline comparison is
present; the pre-edit value is captured without relying on the planner; the design states how a
rejected or reverted override is recorded.

### Exercise 3 — Critique the slide (critique, 35 minutes)

A slide reads: "Planner overrides improved forecast accuracy by 6% versus benchmark." Write the
five questions you would ask before accepting it.

**Deliverable:** five questions, each with the specific error it tests for.
**Acceptance criteria:** at least one question concerns the baseline, one the population, one the
metric, one the horizon, and one the archive; each question is answerable from data, not opinion.

### Exercise 4 — Classify the reasons (application, 50 minutes)

Take twenty free-text override notes (use your own organisation's, or construct twenty plausible
ones). Classify each into: known event, new information, judgement, or target or relationship
pressure.

**Deliverable:** the classification table, plus a note on how many were ambiguous.
**Acceptance criteria:** a written rule for each class; ambiguous cases identified rather than
forced; at least one class defined so that a second coder could apply it.

### Exercise 5 — Accuracy FVA against cost FVA (calculation, 45 minutes)

Using the data from Exercise 1, apply an asymmetric cost function with an over-forecast cost of
EUR 2.00 per case and an under-forecast cost of EUR 7.50 per case. Compute cost FVA for each cell
and in total.

**Deliverable:** a table of unit FVA and cost FVA side by side, with the sign comparison stated.
**Acceptance criteria:** arithmetic correct; at least one cell identified where the signs differ;
a one-paragraph explanation of why.

### Exercise 6 — Design the trial (design, 60 minutes)

Design a trial to estimate the causal effect of planner overrides on forecast error. You may
withhold the application of some overrides for a period, but you may not withhold the planner's
ability to record them.

**Deliverable:** a protocol stating the randomisation unit, the stratification, the outcome, the
duration, the power requirement, and the ethical constraint.
**Acceptance criteria:** randomisation is at a level that avoids contamination between cells; the
protocol states what is blinded and what cannot be; the power requirement is computed, not
asserted; the ethical constraint names the case in which withholding an override would cause harm
and the rule for exempting it.

### Exercise 7 — Find the redistribution (analysis, 45 minutes)

You are given override effects by horizon for one series. Identify whether the pattern is
value-creating or value-redistributing, and across what.

**Deliverable:** a horizon-by-horizon table, the sum across horizons, and a verdict.
**Acceptance criteria:** the sum across the horizon window is computed, not just the per-horizon
values; the verdict names the redistribution axis (horizon, series, or metric); the alternative
explanation of a genuine level shift is considered and rejected or accepted on evidence.

### Exercise 8 — Write the charter (design, 50 minutes)

Write a one-page charter governing the FVA report for a demand planning function.

**Deliverable:** the metric definition, the baseline set, the population, the level, the cadence,
the audience, and the prohibitions.
**Acceptance criteria:** every term a reader could interpret two ways is defined; the charter
states explicitly that FVA is not used in individual appraisal, or states the protections if it
is; the charter names who may change the definition and how a change is announced.

### Exercise 9 — Read the distribution (critique, 40 minutes)

You are shown a histogram of cell-level FVA: 45% of cells positive, 40% negative, 15% at zero,
median +0.3 percentage points, mean −0.1, fifth percentile −14 points. Decide whether to change
the override policy, and what change.

**Deliverable:** a recommendation with the evidence it rests on and the evidence it does not.
**Acceptance criteria:** the recommendation does not rest on the mean; the worst decile is
addressed; the recommendation distinguishes a coaching response from a restriction; the
limitation of a single period is stated.

### Exercise 10 — The module in one page (synthesis, 40 minutes)

Write one page explaining how the ten topics of this module fit together, addressed to a
colleague who has read none of them.

**Deliverable:** a one-page narrative with a single diagram.
**Acceptance criteria:** all ten topics are named and placed in sequence; the through-line
(benchmark, method, level, human, measured honestly) is explicit; no topic is described with a
claim the topic itself would not support.

## 13. Self-check answers

**Exercise 1.** The procedure is: sum the absolute errors for each forecast, divide by the sum of
actuals, multiply by 100. Then the three deltas are pairwise differences. A slide claiming a six
per cent improvement is almost certainly quoting `WMAPE(N) − WMAPE(F)`, the process delta, which
includes the model's contribution. The overstatement is exactly the model delta, `WMAPE(N) −
WMAPE(S)`. In the worked example the process delta was 3.82 points, the model delta 4.99, and the
override delta −1.17, so a slide quoting the process number and attributing it to overrides
overstates the override contribution by 5.0 points and reverses its sign.

**Exercise 2.** Minimum fields: `series_id`, `location_id`, `origin_id` (the forecast run), the
horizon, the `model_version`, `system_forecast_value`, `system_forecast_ts`, `proposed_value`,
`applied_value`, `decision` (applied, rejected, reverted), `editor_role`, `editor_id` (restricted),
`decision_ts`, `reason_code`, `reason_text`. Immutability: append-only, no update or delete grants,
with the pre-edit value written by a database trigger at the moment of edit rather than by the
planner. Retention: at least the length of the longest audit and appraisal cycle, and not less
than the financial record retention requirement. The rejected-override record is the part most
designs miss, and it is what allows you to distinguish "the planner did not see it" from "the
planner saw it and was overruled."

**Exercise 3.** The five questions:

1. *Which baseline?* If the answer is naive, the number measures the process, not the override.
2. *Which population?* All reviewed cells, or only the cells where someone chose to act? Without
   the denominator, the share is uninterpretable.
3. *Which metric, and does it treat over- and under-forecasting symmetrically?* MAPE does not, and
   an upward override can improve MAPE while worsening bias.
4. *Which horizon?* A short-horizon-only number can reward pull-forward.
5. *Is the pre-edit forecast archived, or reconstructed?* If reconstructed, the comparison is
   against a memory.

**Exercise 4.** A workable rule set: *known event* names a dated, external, verifiable occurrence
(promotion start, holiday, competitor exit, customer listing). *New information* cites a document
or a system record not available to the model (a signed contract, a confirmed order book, a
customer communication). *Judgement* states a belief with no cited evidence ("feels low," "the
sales team is optimistic"). *Target or relationship pressure* references a plan, a budget, a
commitment, or a person's expectation. Expect 20–30% ambiguity, mostly between judgement and
target pressure; the useful test is whether the note names an artefact. A second coder should
reach the same class on at least three quarters of the unambiguous cases; below that, the
taxonomy needs another pass.

**Exercise 5.** Cost per cell is `c_o × max(0, F − A) + c_u × max(0, A − F)`. In the worked
example the total cost FVA is +EUR 110 against a unit FVA of −55 cases, because the overrides were
net upward and the under-forecast penalty is 3.75 times the over-forecast penalty. Cell 8 is the
clean demonstration: unchanged absolute error, cost down from EUR 112.50 to EUR 30.00. The general
point is that unit FVA and cost FVA agree only when `c_o = c_u`; when they disagree, the decision
should follow the cost function, provided the cost function was fixed before the outcome was
known.

**Exercise 6.** A defensible protocol: randomise at the cell level within strata defined by ABC
class and horizon, because cells in different strata have very different error scales and
different cost consequences. For each reviewed cell, draw an application indicator; where it is
zero, publish the system forecast and record the planner's proposal in the shadow log. The outcome
is the error difference between the applied and the shadow forecast against the actual, scored on
the pre-declared metric and cost function. Duration: long enough that the number of independent
events, not the number of cells, gives adequate power; fifty cells affected by one promotion are
one observation. Blinding: the planner cannot be blinded to whether the override was applied, so
blind the scoring and the analyst, and pre-register the analysis. Ethics: exempt any override
where the planner judges that withholding it would breach a customer commitment or a safety
requirement, and record the exemption; the exemption rate is itself a finding. Cross-reference
Module 05 topics 01 and 05 for the design logic and the switchback structure.

**Exercise 7.** Redistribution across horizons has a signature: positive FVA at the short horizon,
negative at the adjacent horizon, and a sum across the window that is close to zero or negative.
Redistribution across series has a different signature: the aggregate improves while the
item-level cells that moved worsen. Redistribution across metrics has a third: the quoted metric
improves while bias or WMAPE degrades in the same direction. A genuine level shift looks like
improvement at every horizon, sustained beyond the window, with no offsetting deterioration
anywhere. The test is always the sum, never the individual cell.

**Exercise 8.** The charter should fix: the metric (say WMAPE at the item-location level, with the
cost metric as a secondary), the three baselines and their order, the population (all reviewed
cells, with the override rate reported), the level (measure at item-location-week, report at
item-location and category), the cadence (monthly, with a quarterly distribution review), the
audience (process owners, not individuals), and the prohibitions (not in individual appraisal; not
a target for override counts; not restated retroactively without an announced change). It should
also name the definition owner and the change procedure, because a definition that can be edited
silently is not a definition.

**Exercise 9.** The mean is not the evidence; the distribution is. A median of +0.3 with a fifth
percentile of −14 says that the typical override is roughly neutral and a small tail is
catastrophic. The policy response is therefore targeted, not global: identify the cells in the
worst decile and ask what they have in common — usually a class of override, a planner workflow,
or a missing signal — and address that. Do not restrict overrides generally, because the median
says the typical one is harmless and the 45% positive share says some are useful. Do not conclude
from one period; require the same pattern in two consecutive periods before changing policy.

**Exercise 10.** A strong one-page answer places the topics in four movements. First, the
benchmark (topic 01): before any method, establish what "doing nothing" produces, because every
later claim is relative to it. Second, the methods (topics 02 to 04) and the level at which they
are applied (topic 05), with intermittent demand (topic 06) as the case where the method choice
is dominated by the classification, and combination (topic 07) as the correction for the fact
that selection is unreliable. Third, the evaluation (topics 08 and 09): rolling origins to
manufacture replication, and an error metric chosen for the cost of the decision rather than for
convenience. Fourth, the human (topic 10): the last unmeasured step, measurable only against the
forecast the system would have issued, and only honestly if the measurement is designed before
the outcome is known. The through-line is that each stage is a claim, and the discipline is
asking what each claim is compared against.

## 14. Interview questions

**1. "What is forecast value added, and what does it measure that accuracy does not?"**

A strong answer defines it as the change in a performance metric attributable to a specific step,
notes that accuracy is a level and FVA is a marginal contribution, and states that FVA requires a
counterfactual while accuracy does not. A weak answer treats FVA as a synonym for accuracy or
describes it as "accuracy after overrides."

**2. "A planner tells you their overrides improved accuracy by 6%. What do you ask first?"**

A strong answer asks which baseline. Final against naive credits the model; final against the
system forecast is the only comparison that isolates the planner. It then asks which population
and which metric. A weak answer asks for the sample size and stops, which addresses precision but
not validity.

**3. "Why can FVA be negative, and why is that not evidence that the planner is bad?"**

A strong answer explains that the model already captures most of the signal, so the marginal
contribution of a human edit is small and can easily be negative; that a good planner with a good
model will show low FVA; and that the published literature finds negative value on a substantial
minority of series as a normal result. A weak answer treats negative FVA as a performance problem
to be managed.

**4. "Which baseline would you use?"**

A strong answer names the question first: naive for "is the process worth running," system
forecast for "is the human worth keeping," last cycle for "did we improve since we committed,"
and says that a report must state which question it answers. A weak answer names one baseline as
correct in all cases.

**5. "How do you measure an override that prevents a stock-out but worsens weighted error?"**

A strong answer separates forecast quality from business value: record the accuracy FVA as
negative and the cost or service outcome as positive, and let the decision follow the outcome. It
notes that the cost function must be declared in advance. A weak answer either reports the
negative FVA as a failure or argues that service overrides should be excluded from measurement,
which removes the most interesting cases from view.

**6. "How would you design a trial of overrides?"**

A strong answer randomises the application of overrides at the cell level within strata, records
the unapplied proposals in a shadow log, pre-registers the metric and cost function, powers the
design on independent events rather than cells, and states the ethical exemption. A weak answer
proposes a before-and-after comparison, which is confounded by everything else that changed.

**7. "What is a value-redistributing override?"**

A strong answer gives the three axes — horizon, series, metric — and the signature on each: an
offsetting negative at the adjacent horizon, item-level harm under aggregate improvement, and one
metric improving while another degrades. A weak answer defines it vaguely as "moving demand
around" without naming how to detect it.

**8. "Should you ban overrides with negative FVA?"**

A strong answer says no, for three reasons: the estimate is noisy at the cell level; the class
contains legitimate event-driven adjustments that the metric cannot distinguish from hunches; and
banning produces workarounds that are harder to see. It recommends segmenting by reason code and
coaching, with restriction only where a segment is consistently negative across periods. A weak
answer says yes, or says the question is not answerable.

**9. "How would you report FVA to a steering committee?"**

A strong answer reports the share of cells improved, the median, the worst decile, the sample
size, and the coverage, with all three baselines on the page, and explicitly says what the report
does not measure. A weak answer reports an average.

**10. "What is the difference between accuracy FVA and cost FVA?"**

A strong answer explains the asymmetric cost function, gives the condition under which they agree
(symmetric costs), and says the two can differ in sign on the same data. A weak answer treats cost
FVA as accuracy FVA converted to money at a single rate.

**11. "Why is FVA politically difficult?"**

A strong answer names the zero point: it is the only routine forecast metric whose zero means
"you contributed nothing" and whose negative region means "you made it worse." It notes that the
person measured often controls the measurement, and that the fix is structural — report by
process and segment, not by person. A weak answer calls it "a sensitive topic" without explaining
the mechanism.

**12. "What did this module add up to?"**

A strong answer gives the arc: a benchmark to establish the floor, methods to beat it, a level at
which to make the claim, an evaluation that manufactures replication and scores against the
decision's cost, and a human whose contribution is measured against the forecast the system would
have issued. It ends with the honest version of the claim: most of the value is in the first four,
and the fifth is where organisations most often fool themselves. A weak answer lists the topics
without a through-line.

## 15. Key terms

- **Override** — a human replacement of a system forecast for one series and horizon.
- **Judgemental adjustment** — the academic term for an override.
- **System forecast** — the unadjusted model output the planner saw.
- **Final forecast** — the published number after all human steps.
- **Consensus forecast** — the output of a cross-functional review, possibly several overrides deep.
- **Naive benchmark** — the last value carried forward; the floor for any method.
- **Seasonal-naive benchmark** — the same period last year; the floor when seasonality exists.
- **Forecast value added (FVA)** — the metric change attributable to one process step.
- **Stairstep** — the chart of error after each sequential step from naive to final.
- **Counterfactual** — the forecast that would have been published without the step.
- **Shadow forecast** — a recorded but unapplied forecast, kept for comparison.
- **Override rate** — share of reviewed cells that a human edited.
- **Reason code** — a controlled label for why an override was made, recorded at the time.
- **Series-horizon cell** — one forecast for one series at one horizon from one origin.
- **Population** — the set of cells the metric is computed over.
- **Selection effect** — distortion from scoring only the cases a person chose to act on.
- **Regression to the mean** — extreme values tending to be followed by less extreme ones.
- **WMAPE** — weighted mean absolute percentage error, volume-weighted by construction.
- **MAPE** — mean absolute percentage error, unweighted and asymmetric.
- **Bias** — mean signed error, showing systematic over- or under-forecast.
- **AvgRelMAE** — geometric mean of MAE ratios against a benchmark.
- **Value-creating override** — one that reduces error or cost across the full horizon window.
- **Value-redistributing override** — one that moves error between periods, series, or metrics.
- **Silent override** — intervention through a parameter rather than the forecast.
- **Algorithm aversion** — faster loss of confidence in algorithms than in humans after equal errors.
- **Debiasing** — a procedure correcting a systematic direction in adjustments.
- **Override governance** — the rules on who may override, how it is recorded, and how it is reviewed.

## 16. Sources

**Peer-reviewed and academic**

- Fildes, R., Goodwin, P., and De Baets, S. (2025). "Forecast value added in demand planning."
  *International Journal of Forecasting*, 41(2), 649–669. DOI: 10.1016/j.ijforecast.2024.07.006.
  Grade A. Pooled re-analysis of about 147,000 forecasts from six studies.
- Fildes, R., Goodwin, P., Lawrence, M., and Nikolopoulos, K. (2009). "Effective forecasting and
  judgmental adjustments: an empirical evaluation and strategies for improvement in supply-chain
  planning." *International Journal of Forecasting*, 25(1), 3–23.
  DOI: 10.1016/j.ijforecast.2008.11.010. Grade A.
- Davydenko, A., and Fildes, R. (2013). "Measuring forecasting accuracy: the case of judgmental
  adjustments to SKU-level demand forecasts." *International Journal of Forecasting*. Grade A.
  Cited here for the finding that metric choice changes the ranking of adjustments.
- Goodwin, P., and Fildes, R. (2011). "Forecasting in supply chain companies: should you trust
  your judgment?" *OR Insight*. DOI: 10.1057/ori.2011.5. Grade A.
- Dietvorst, B. J., Simmons, J. P., and Massey, C. (2015). "Algorithm aversion: people erroneously
  avoid algorithms after seeing them err." *Journal of Experimental Psychology: General*, 144(1),
  114–126. DOI: 10.1037/xge0000033. Grade A.
- Dietvorst, B. J., Simmons, J. P., and Massey, C. (2016). "Overcoming algorithm aversion: people
  will use imperfect algorithms if they can (even slightly) modify them." *Management Science*.
  Grade A. Cited without a DOI because one was not confirmed in this research pass.

**Practitioner and analyst**

- Gilliland, M. (2013). "FVA: A reality check on forecasting practices." *Foresight: The
  International Journal of Applied Forecasting*, Issue 29. Grade B for the method; the case notes
  it reports are Grade C. The paper that codified the stairstep and the distribution-based
  reporting convention for practitioners.
- Morlidge, S. (2014). "Forecast quality in the supply chain." *Foresight*, 33, 26–31. Grade B.
  Reported figure: 52% of more than 300,000 forecasts failed to beat a random walk. Accessed in
  this pass through secondary summaries, not the original text.
- Seifert, R. W., Markoff, R., and Spooner, M. (2024). "How a new approach to demand planning can
  redefine success." I by IMD, 5 August 2024. Grade C. The "80% or more" figure is presented as a
  typical result, not as a controlled study finding, and should not be quoted as one.
- *Demand Forecasting Best Practices*, Chapter 12 ("Forecast value added"). Manning. Grade C.
  Practitioner treatment; used here only for framing.
- Institute of Business Forecasting glossary entry on forecast value added. Grade D. Used for the
  definition only, not for evidence.
- Demand-planning.com (2018), "What is forecast value added analysis?" Grade D. Practitioner
  how-to; the Morlidge figure is quoted at second hand.

**Standards and official**

- IAS 2 *Inventories*; ASC 330 *Inventory*. Grade B.
- IAS 36 *Impairment of Assets*. Grade B.
- Sarbanes-Oxley Act of 2002, sections 302 and 404. Grade B.
- Regulation (EU) 2016/679 (General Data Protection Regulation), Article 88. Grade B.
- ASCM SCOR Digital Standard. Grade C. A reference model, not a binding standard.

**Note on citations.** Percentages are reported as the sources state them and were not
independently re-computed. Where a figure reached this text through a secondary summary, the text
says so. No citation was invented; where a DOI could not be confirmed, the entry omits it.

## 17. Links

- Module map: [`_index.md`](_index.md)
- First topic in this module: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- Aggregation level: [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md)
- Intermittent series, where overrides are most common: [`06_intermittent-demand.md`](06_intermittent-demand.md)
- Combination, the alternative to selection: [`07_forecast-combination.md`](07_forecast-combination.md)
- Evaluation protocol: [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md)
- Error metrics and cost: [`09_error-metrics-cost.md`](09_error-metrics-cost.md)
- Trial design: [`../../01_foundations/05_experimentation_pilots_evidence/01_claim-hypothesis-design.md`](../../01_foundations/05_experimentation_pilots_evidence/01_claim-hypothesis-design.md)
- Switchback and alternating-period designs: [`../../01_foundations/05_experimentation_pilots_evidence/05_switchback-pilots.md`](../../01_foundations/05_experimentation_pilots_evidence/05_switchback-pilots.md)
- Power and minimum detectable effect: [`../../01_foundations/05_experimentation_pilots_evidence/06_power-mde.md`](../../01_foundations/05_experimentation_pilots_evidence/06_power-mde.md)
- Measurement error and censoring: [`../../01_foundations/01A_foundations_units_time_assumptions/08_measurement-error-bias.md`](../../01_foundations/01A_foundations_units_time_assumptions/08_measurement-error-bias.md)
- Error metric definitions: [`../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Part map: [`../_index.md`](../_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
