# Mathematical Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

Every quantitative claim in biomedicine is a function applied to a measurement. A readmission rate is a
ratio. A viral load decline is a logarithm. A drug's accumulation is an exponential. A vendor's "30%
reduction" is a relative change with an unstated baseline. When the function is misread, the number
still looks precise, still appears in a slide, and still drives a decision. Nothing in the display
flags the error.

The failures are ordinary and expensive.

- A quality team reports that "readmissions fell 12%" when the rate moved from 18.4% to 16.2%. The
  absolute reduction is 2.2 percentage points. The relative reduction is 12.0%. The board hears
  "12%" and budgets as if 12 of every 100 patients were affected. The number needed to treat is 45,
  not 8.
- A marketing team compares two assays and reports a "0.3 improvement" in log10 viral load without
  stating the baseline. A drop from 5.0 to 4.7 log10 copies/mL is a halving. A drop from 2.0 to 1.7
  is also a halving. The same arithmetic difference means the same proportional change only because
  the scale is logarithmic; on the raw copies/mL scale the two drops differ by a factor of 1,000.
- A finance analyst extrapolates six months of increasing imaging volume with a straight line. The
  service line was in the early, near-linear phase of an exponential adoption curve. The budget is
  short by 40% at year end.
- A pharmacokinetics reviewer computes a half-life from a concentration ratio without checking that
  elimination is first-order. The number is arithmetically correct and mechanistically meaningless.
- A data scientist converts an odds ratio to a risk ratio using the baseline risk from a different
  population. The direction is right; the magnitude is wrong by a factor of two.

None of these require advanced mathematics to catch. They require knowing which function is being
applied, what its inverse is, and which comparisons survive a transformation. That is the content of
this topic.

The topic exists because later modules assume it silently. Module 02 (probability) assumes you can
manipulate sums and products. Module 03 (units) assumes you can reason about ratios and powers.
Module 06 (pharmacokinetics) assumes exponentials and logarithms. Module 07 (epidemiology) assumes
rates, proportions, and person-time. Module 10 (the assumptions ledger) assumes you can state what a
functional form implies. A learner who cannot move between a rate constant and a half-life will
memorize formulas and misapply them at the first non-standard case.

There is also a commercial reason. Analysts who can decompose a claim into "absolute change,
relative change, denominator, time window, and functional form" are the ones who stop a bad
investment case before it reaches a committee. That skill is not statistics. It is arithmetic plus
discipline.

> **Note:** This topic is deliberately not a mathematics course. It covers only the operations that
> appear repeatedly in biomedical data work: linear and affine functions, exponentials, logarithms,
> power laws, saturation curves, ratios, rates, and first-order error propagation. Linear algebra and
> calculus appear only where a later module needs them.

Three questions organise everything that follows.

1. **What is the scale?** Is the quantity measured on an interval scale, a ratio scale, a count, or a
   proportion? The scale determines which operations are meaningful.
2. **What is the comparison?** Absolute difference, relative difference, ratio, or rank? Each answers
   a different question and cannot be substituted for another.
3. **What is the functional form?** Linear, exponential, logarithmic, saturating, or none of these?
   The form determines how the quantity behaves outside the observed range.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a biomedical quantity by scale (nominal, ordinal, interval, ratio, count, proportion)
   and state which arithmetic operations are valid on it. Assessed in Exercise 1 and self-check item 2.
2. **Compute** absolute change, relative change, percentage-point change, ratio, and rate from a
   two-group or two-time-point table, and state which one answers a given stakeholder question.
   Assessed in Exercises 2 and 3.
3. **Convert** between a first-order rate constant, a half-life, and a fractional remaining amount,
   with units carried through every step. Assessed in Exercise 4 and self-check item 5.
4. **Interpret** a value on a logarithmic scale, including converting a log10 difference into a fold
   change, and explain why a fixed arithmetic difference on a log scale corresponds to a fixed ratio.
   Assessed in Exercise 5 and self-check item 6.
5. **Distinguish** linear, exponential, power-law, and saturating growth from a short series of
   observations, and state the evidence needed before extrapolating. Assessed in Exercise 6.
6. **Propagate** uncertainty through a sum, a product, and a ratio using first-order approximations,
   and identify when that approximation fails. Assessed in Exercise 7.
7. **Critique** a published or vendor claim for scale errors, baseline errors, denominator errors, and
   inappropriate extrapolation, and rewrite the claim in defensible terms. Assessed in Exercise 8.
8. **Write** a one-paragraph interpretation of a numerical result that states the unit, the baseline,
   the comparison, and the precision. Assessed in all exercises through the reporting rubric.

Each outcome is observable. "Understand logarithms" is not an outcome; "convert a log10 difference to
a fold change and state the baseline" is.

## 3. The core mental model

The model has four layers. A quantity moves downward through them when it is analysed, and upward
when it is reported.

```text
            +-------------------------------------------------------+
            |  4. DECISION                                          |
            |  fund / do not fund, treat / do not treat, alert /    |
            |  suppress                                             |
            +---------------------------^---------------------------+
                                        |
            +---------------------------+---------------------------+
            |  3. COMPARISON                                        |
            |  absolute diff | relative diff | ratio | rank         |
            +---------------------------^---------------------------+
                                        |
            +---------------------------+---------------------------+
            |  2. TRANSFORM                                         |
            |  identity | log | reciprocal | standardise | none     |
            +---------------------------^---------------------------+
                                        |
            +---------------------------+---------------------------+
            |  1. QUANTITY                                          |
            |  value + unit + scale + denominator + time window     |
            +-------------------------------------------------------+
```

Read it upward. A decision requires a comparison. A comparison requires a transform that makes the
comparison meaningful. A transform requires a correctly specified quantity. Most errors enter at
layer 1 and are hidden by layer 3.

### 3.1 Why the transform layer exists

Some quantities are additive. Costs add. Volumes add. Counts add. Other quantities are multiplicative.
Concentrations decay by a constant fraction per unit time, not by a constant amount. Bacterial
populations grow by a constant factor. Odds multiply across independent risk factors. When the
underlying process is multiplicative, a linear comparison misleads and a logarithmic transform
restores additivity.

The logarithm is the unique continuous function that turns multiplication into addition:

```text
log(a * b) = log(a) + log(b)
log(a / b) = log(a) - log(b)
log(a^k)   = k * log(a)
```

That single property explains why pH, decibels, viral load, and log-odds are all logarithmic. In each
case the underlying process is multiplicative or spans many orders of magnitude, and the log makes
the arithmetic tractable and the display readable.

The cost is that log differences are not intuitive to most audiences. A 1.0 log10 drop is a 10-fold
drop, not a 1-unit drop. A 3 dB increase is a doubling of power, not a tripling. The analyst must
translate back to the audience's scale and state the baseline.

### 3.2 The two comparisons that get confused

Almost every misleading relative claim comes from confusing these two:

```text
Absolute change  = after - before                 (same unit as the quantity)
Relative change  = (after - before) / before     (dimensionless, baseline-dependent)
```

A relative change has no meaning without its baseline. "Reduced by 30%" is incomplete; "reduced by
30% from a baseline of 18.4%" is a claim. The absolute change is usually what a budget, a bed
capacity model, or a patient expects. The relative change is usually what a marketing slide shows.

The same distinction appears as percentage points versus percent. A rate moving from 18.4% to 15.1%
falls 3.3 percentage points, which is 17.9% of its baseline. Both statements are true. Only one is
relevant to the decision, and the analyst must say which.

### 3.3 The functional form decides the extrapolation

Four forms cover most biomedical quantities:

| Form | Equation | Constant behaviour | Typical use |
|---|---|---|---|
| Linear | y = m·x + b | Constant absolute change per unit x | Cost per unit, calibration |
| Exponential | y = a·e^(k·x) | Constant relative change per unit x | Growth, decay, PK |
| Power law | y = a·x^b | Constant elasticity | Allometry, scaling |
| Saturating | y = E_max·x/(EC50 + x) | Approaches a ceiling | Receptor binding, PD |

Extrapolating a linear fit through an exponential process understates the future. Extrapolating an
exponential through a saturating process overstates it. The data rarely announce which form applies;
the analyst must test.

### 3.4 The model in one sentence

A number is a value with a unit, a scale, a denominator, and a time window; a comparison is only
valid after a transform that respects the scale; and an extrapolation is only valid if the functional
form is correct.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Variable | A quantity that can take different values across observations | Calling a column a variable when it is a constant | none |
| Constant | A fixed value in a given context | Treating a population constant as universal | none |
| Function | A rule mapping each input to exactly one output | Assuming every relationship is a function | none |
| Domain | The set of inputs for which a function is defined | Evaluating a log at zero or a ratio at zero denominator | input unit |
| Range | The set of outputs a function can produce | Reporting an impossible value such as a negative concentration | output unit |
| Linear function | y = m·x, passing through the origin | Using "linear" for any straight line | output/input |
| Affine function | y = m·x + b, a straight line with intercept | Calling an affine model "linear regression" without noting the intercept | output/input |
| Slope | Change in output per unit change in input | Comparing slopes across different units | output per input |
| Intercept | Output when the input is zero | Interpreting an intercept outside the data range | output unit |
| Exponential function | y = a·e^(k·x) | Calling any rapid growth "exponential" | output unit |
| Base e | 2.71828..., the natural exponential base | Mixing e and 10 mid-calculation | dimensionless |
| Natural logarithm | ln(x), inverse of e^x | Using ln where log10 was intended | dimensionless |
| Decadic logarithm | log10(x), inverse of 10^x | Reporting "log" without stating the base | dimensionless |
| Binary logarithm | log2(x), inverse of 2^x | Rare outside computing; state it if used | dimensionless |
| Order of magnitude | A factor of 10 | Saying "orders of magnitude" for a factor of 2 | dimensionless |
| Half-life | Time for a quantity to fall to half its value | Applying it to non-first-order processes | time |
| Doubling time | Time for a quantity to double | Confusing with half-life sign | time |
| Rate constant | Fractional change per unit time, k | Reporting k without its time unit | 1/time |
| Growth rate | Relative change per unit time | Using it interchangeably with absolute increment | 1/time |
| Absolute change | after − before | Calling it a percentage | quantity unit |
| Relative change | (after − before)/before | Omitting the baseline | dimensionless |
| Percentage point | Arithmetic difference between two percentages | Calling it "percent" | percentage points |
| Ratio | Quotient of two quantities | Dividing quantities with different units without stating the composite unit | composite |
| Proportion | A ratio where numerator is a subset of denominator, 0 to 1 | Exceeding 1 through a wrong denominator | dimensionless |
| Rate | Events per unit person-time | Dividing by persons instead of person-time | 1/time |
| Denominator | The population or exposure base | Using the wrong catchment population | unit of base |
| Person-time | Sum of time at risk across individuals | Counting people once regardless of follow-up | person-time |
| Log scale | An axis on which equal distances are equal ratios | Reading equal distances as equal differences | dimensionless |
| Semi-log plot | One axis logarithmic | Using it to claim linearity without checking | mixed |
| Log-log plot | Both axes logarithmic | Estimating a power-law exponent from a curved region | dimensionless |
| Power law | y = a·x^b | Confusing with exponential growth | depends on b |
| Polynomial | Sum of terms a_k·x^k | Fitting a high-degree polynomial to sparse data | output unit |
| Asymptote | A value a function approaches but does not reach | Claiming the asymptote is observed | output unit |
| Saturation | Diminishing response as input rises | Assuming linearity at high dose | output unit |
| Michaelis-Menten | v = V_max·S/(K_m + S) | Treating K_m as a binding constant in vivo | concentration |
| Hill coefficient | Exponent controlling curve steepness | Interpreting it as the number of binding sites without evidence | dimensionless |
| Logistic function | p = 1/(1 + e^(−z)) | Calling any S-curve logistic | dimensionless |
| Odds | p/(1 − p) | Reporting odds as if it were risk | dimensionless |
| Probability | p, in [0, 1] | Reporting a probability above 1 after arithmetic | dimensionless |
| Monotone transform | A transform that preserves order | Assuming it preserves intervals | none |
| Inverse function | f⁻¹ such that f⁻¹(f(x)) = x | Applying an inverse outside its domain | none |
| Identity | An equation true for all values | Treating it as a model | none |
| Dimensional homogeneity | Both sides of an equation carry the same dimensions | Adding mg/dL to mmol/L | none |
| Scientific notation | a × 10^b with 1 ≤ |a| < 10 | Losing the exponent in a spreadsheet import | none |
| Absolute error | measured − true | Calling it relative error | quantity unit |
| Relative error | (measured − true)/true | Dividing by the measured value instead of the true value | dimensionless |
| Percentage error | Relative error × 100 | Reporting more digits than the measurement supports | percent |
| Compounding | Growth applied to accumulated value | Using simple interest for multi-period growth | dimensionless |
| CAGR | Compound annual growth rate | Computing it as an arithmetic mean of yearly rates | 1/year |
| Area under the curve | Integral of concentration over time | Using trapezoids across a dosing interruption | concentration × time |
| Derivative | Instantaneous rate of change | Treating a finite difference as a derivative | output/input |
| Integral | Accumulated area | Ignoring the baseline offset | output × input |
| Steady state | Input equals elimination | Claiming steady state before 4–5 half-lives | none |
| Linearisation | Approximating a curve by a line near a point | Using it far from the point | none |
| Elasticity | Proportional change in output per proportional change in input | Confusing with slope | dimensionless |

Three definitions deserve emphasis because later modules depend on them.

**Rate versus proportion.** A proportion has the population in its denominator and is bounded by 1. A
rate has person-time in its denominator and is not bounded by 1 per person; it can exceed 1 per
person-year in a high-incidence setting. "30% of patients" is a proportion. "0.30 events per
patient-year" is a rate. They are not interchangeable, and converting one to the other requires the
average follow-up time.

**Log base.** Clinical and virological literature almost always uses log10. Statistical software
often uses the natural log. A regression coefficient on ln(x) differs from one on log10(x) by a
factor of ln(10) = 2.302585. Always state the base.

**Percentage points.** A change from 4.0% to 5.0% is 1 percentage point and 25% relative. Both are
correct. A press release that says "25% increase in adverse events" and a regulator's table that says
"1 percentage point" describe the same data.

## 5. The mechanics, step by step

The procedure below is the one to apply to any quantitative claim. It is written as a sequence
because the order matters: a comparison computed on an unverified quantity cannot be repaired later.
Each step names its input, its output, and the reason it exists.

### Step 1 — Identify the quantity and its referent

**Input.** A sentence containing a number, for example "mortality was 8.2%".

**Output.** A named quantity with an explicit referent: "all-cause in-hospital mortality among
adults admitted with sepsis, index admission, 30-day window."

**Reasoning.** A number without a referent is not a measurement. The referent fixes the population,
the outcome definition, and the time window. Most disagreements between two teams about a number are
disagreements about the referent, not about the arithmetic. Write the referent before touching a
calculator.

**Branch point.** If the referent cannot be recovered from the source, stop and mark the claim as
unverifiable. Do not assume a plausible referent.

### Step 2 — Attach the unit and the scale

**Input.** The named quantity.

**Output.** A unit string and a scale classification.

**Reasoning.** The unit determines which conversions are legal and which are nonsense. The scale
determines which statistics are meaningful: a mean of nominal categories is meaningless, a ratio of
interval temperatures is meaningless. The six scales used in this program are nominal, ordinal,
interval, ratio, count, and proportion.

**Branch point.** If the quantity is a composite (for example mg/kg/day), decompose it into base
dimensions before proceeding. Topic 04 covers the decomposition.

### Step 3 — Establish the denominator and the time window

**Input.** The unit and scale.

**Output.** An explicit denominator and window, for example "per 1,000 patient-days, fiscal year
2025".

**Reasoning.** Rates and proportions are defined by their denominators. A readmission rate computed
on discharges differs from one computed on patients. A mortality rate computed on admissions differs
from one computed on cases. A change in the denominator can move the rate without any change in care.
This is the single most common source of spurious trend claims in hospital analytics.

**Branch point.** If numerator and denominator come from different systems (for example, deaths from
the registry and admissions from the billing system), flag a linkage assumption and test it.

### Step 4 — Plot the raw series before choosing a transform

**Input.** A series of values over time or across dose.

**Output.** A linear-axis plot and a judgement about curvature.

**Reasoning.** The eye is better than a statistic at detecting the difference between a line and a
curve over a short series. Plot first. A series that looks linear on a linear axis may be exponential
over a longer window; a series that looks curved may be a saturating approach to a ceiling. The plot
also exposes outliers and level shifts that a summary statistic hides.

**Branch point.** If there are fewer than five points, do not claim a functional form. State the
range and decline to extrapolate.

### Step 5 — Test for constant absolute change

**Input.** The raw series.

**Output.** First differences and their pattern.

**Reasoning.** A linear process has constant first differences. Compute them. If the differences are
roughly constant in absolute terms, a linear or affine model is defensible within the observed range.
If the differences grow in proportion to the level, the process is multiplicative.

**Branch point.** If first differences increase while the level increases, go to Step 6. If they
decrease toward zero, consider saturation (Step 12).

### Step 6 — Test for constant relative change

**Input.** The raw series.

**Output.** Ratios of consecutive values, or first differences of the log series.

**Reasoning.** A first-order process has a constant ratio between consecutive values at equal time
intervals. Equivalently, the log series has constant first differences. This is the operational test
for exponential behaviour. Compute both and compare the constancy.

**Branch point.** If the ratios are constant, the process is exponential within the observed range.
If the ratios decline toward 1, the process is saturating. If they are erratic, no simple form
applies.

### Step 7 — Compute the absolute change

**Input.** before and after values, same unit.

**Output.** after − before, with unit.

**Reasoning.** The absolute change is the only change that maps directly onto counts, budgets, and
bed capacity. It is also the input to the number needed to treat. Compute it first, before any
percentage, so that the percentage cannot anchor the interpretation.

**Branch point.** If the two values are measured on different scales (for example, one assay changed),
the absolute difference is not interpretable. Stop and harmonise the assay first.

### Step 8 — Compute the relative change against the correct baseline

**Input.** before and after values.

**Output.** (after − before)/before, as a dimensionless fraction and as a percentage.

**Reasoning.** The baseline is the value in the comparison group or the pre-period, not the post-period
value and not the midpoint. Dividing by the after value produces a different, larger number and is a
common error in press releases. State the baseline explicitly in the same sentence as the percentage.

**Branch point.** If the baseline is near zero, the relative change is unstable and should not be
reported. Report the absolute change and the counts instead.

### Step 9 — Convert between percent and percentage points

**Input.** Two percentages.

**Output.** A percentage-point difference and a relative percentage.

**Reasoning.** Percentage points are the arithmetic difference of two percentages. Percent is the
relative difference. A stakeholder hearing "12%" will usually assume percentage points, because that
is how clinical audiences read event rates. Always give both, labelled.

**Branch point.** If the two percentages have different denominators (for example, one is per
admission and one is per patient), the difference is not a percentage-point change at all. Rebuild
the comparison.

### Step 10 — Compute the ratio and decide whether it is a risk ratio or an odds ratio

**Input.** Two proportions or two counts with denominators.

**Output.** A ratio, labelled by type.

**Reasoning.** A risk ratio is a ratio of probabilities and is bounded below by zero with no upper
bound. An odds ratio is a ratio of odds and is further from 1 than the corresponding risk ratio when
the outcome is common. The two coincide only when the outcome is rare. Converting an odds ratio to a
risk ratio requires the baseline risk in the target population.

**Branch point.** If the outcome incidence exceeds roughly 10%, do not present an odds ratio as if it
were a risk ratio. Either compute the risk ratio from counts or convert with the baseline risk and
state the conversion.

### Step 11 — Compute a rate with person-time

**Input.** Event count and total follow-up time.

**Output.** Events per person-time, with the time unit.

**Reasoning.** When individuals have different follow-up durations, dividing by the number of people
understates the rate and makes studies with different follow-up incomparable. Person-time is the sum
of each individual's time at risk. The time unit must be stated; per patient-year and per 1,000
patient-days differ by a factor of 365.

**Branch point.** If follow-up time is unavailable, the quantity is a proportion, not a rate. Label
it as such.

### Step 12 — Choose the logarithmic base and direction

**Input.** A quantity spanning orders of magnitude, or a multiplicative process.

**Output.** A stated base and a stated direction, for example "log10 copies/mL, higher means more
virus".

**Reasoning.** Base 10 is conventional in clinical virology and in pH. Base e is conventional in
modelling because the derivative of e^x is e^x. Base 2 is conventional for doubling and for
information. The base changes the numeric values but not the ordering. Mixing bases within one
calculation multiplies the result by 2.302585 or 0.434294 and is a silent error.

**Branch point.** If the source does not state the base, infer from context only if the context is
unambiguous; otherwise treat the value as unverifiable.

### Step 13 — Convert a log difference to a fold change

**Input.** Two log values with a stated base.

**Output.** A ratio between the underlying quantities.

**Reasoning.** A difference of d log10 units corresponds to a factor of 10^d. A difference of d
natural-log units corresponds to e^d. The sign tells the direction. This conversion is the bridge
between the analyst's scale and the clinician's scale, and it should appear in every report that uses
a log axis.

**Branch point.** If the values are already log-transformed and the baseline is not stated, the fold
change is computable but the absolute level is not. Report the fold change only.

### Step 14 — Convert a rate constant to a half-life

**Input.** A first-order rate constant k with unit 1/time.

**Output.** t½ = ln(2)/k, in the same time unit.

**Reasoning.** The half-life is defined by the time at which the remaining fraction equals one half.
Solving e^(−k·t) = 0.5 gives t = ln(2)/k = 0.693147/k. The relationship holds only for first-order
elimination. For saturable elimination, the half-life changes with concentration and the concept is
only locally valid.

**Branch point.** If a plot of log concentration against time is not linear, the process is not
first-order and a single half-life is misleading. Report the apparent half-life over a stated
concentration range.

### Step 15 — Compute remaining fraction after n half-lives

**Input.** n half-lives (may be fractional).

**Output.** Remaining fraction = 0.5^n.

**Reasoning.** Each half-life multiplies the remaining amount by one half. After four half-lives,
6.25% remains; after five, 3.125%. The common clinical approximation "steady state at 4–5 half-lives"
follows from this: at four half-lives, 93.75% of the final steady-state level has been reached; at
five, 96.875%.

**Branch point.** If dosing intervals are shorter than about one half-life, accumulation is
substantial and the simple remaining-fraction calculation must be replaced by the accumulation
factor 1/(1 − e^(−k·τ)).

### Step 16 — Compute a doubling time

**Input.** A growth rate per unit time, expressed as a fraction.

**Output.** Doubling time = ln(2)/ln(1 + r) for discrete compounding, or ln(2)/r for continuous
growth.

**Reasoning.** The two formulas differ because discrete growth compounds on the accumulated value
while continuous growth applies instantaneously. For small r the difference is negligible; for r
above about 10% per period it matters. State which convention is used.

**Branch point.** If the growth rate is itself changing, a single doubling time is a local summary,
not a forecast.

### Step 17 — Compute a compound annual growth rate

**Input.** Start value, end value, number of years.

**Output.** CAGR = (end/start)^(1/years) − 1.

**Reasoning.** The CAGR is the constant annual rate that would take the start value to the end value
over the interval. It is not the arithmetic mean of the yearly rates, and it is not the year-over-year
rate in any particular year. It is a summary of the endpoints only and ignores the path.

**Branch point.** If the start or end value is anomalous (a pandemic year, a system migration), the
CAGR is dominated by that anomaly. Report the yearly series alongside it.

### Step 18 — Detect a power law on a log-log plot

**Input.** Paired values of x and y across at least an order of magnitude in x.

**Output.** An exponent b from the slope of log(y) against log(x).

**Reasoning.** Taking logs of y = a·x^b gives log(y) = log(a) + b·log(x), a straight line with slope
b. A straight line on a log-log plot is evidence for a power law; curvature is evidence against. On a
semi-log plot, by contrast, a straight line indicates exponential behaviour. Confusing the two plots
is a standard error in scaling analyses.

**Branch point.** If the log-log plot is curved, fit a saturating or polynomial form instead, or
restrict the claim to the observed range.

### Step 19 — Fit or read a saturating curve

**Input.** Dose-response or concentration-response pairs.

**Output.** E_max, EC50, and optionally a Hill coefficient.

**Reasoning.** The Emax model E = E_max·C/(EC50 + C) rises linearly at low C and saturates at high C.
At C = EC50 the effect is half of E_max. The Hill form replaces C with C^h and steepens the curve;
h is a shape parameter and is not automatically the number of binding sites. Reporting EC50 without
the model form and the fitted range is incomplete.

**Branch point.** If the observed concentrations do not bracket EC50, the estimate is an extrapolation
and should be flagged.

### Step 20 — Convert between odds and probability

**Input.** p or odds.

**Output.** The other quantity.

**Reasoning.** odds = p/(1 − p) and p = odds/(1 + odds). The conversion is exact and requires no
baseline. It is the conversion between an odds ratio and a risk ratio that requires a baseline, not
this one. Keep the two conversions distinct.

**Branch point.** If p is very small, odds and probability are nearly equal; the approximation is
acceptable below about p = 0.05 with a stated error.

### Step 21 — Round to the precision the measurement supports

**Input.** A computed value and the precision of its inputs.

**Output.** A rounded value with a stated number of significant figures.

**Reasoning.** Reporting more digits than the least precise input supports is false precision. A
concentration measured to two significant figures cannot support a ratio reported to five. The
convention used here is to carry full precision through intermediate steps and round only the final
reported value, stating the rule.

**Branch point.** If the value is used in a downstream calculation, keep the unrounded value and
round only for display. Do not round twice.

### Step 22 — Propagate uncertainty through sums, products, and ratios

**Input.** Values with standard uncertainties.

**Output.** A combined standard uncertainty.

**Reasoning.** For independent inputs, absolute uncertainties add in quadrature for a sum or
difference, and relative uncertainties add in quadrature for a product or quotient. These are
first-order approximations valid when the relative uncertainties are small (below roughly 10%) and
the function is approximately linear over the uncertainty range.

**Branch point.** If a relative uncertainty exceeds about 20%, or the function is strongly curved over
the uncertainty range, use a Monte Carlo propagation instead of the linear formula.

### Step 23 — Interpolate only within the observed range

**Input.** Two bracketing observations.

**Output.** An interpolated value with an explicit method.

**Reasoning.** Linear interpolation assumes the quantity changes linearly between the two points.
That assumption is usually acceptable over a short interval and can be badly wrong across a
discontinuity such as a dose, a shift change, or a device recalibration. State the method and the
interval.

**Branch point.** If the interval spans a known discontinuity, do not interpolate. Report the two
bracketing values.

### Step 24 — Compute an area under the curve by trapezoids

**Input.** Time-concentration pairs.

**Output.** AUC in concentration × time.

**Reasoning.** The trapezoidal rule approximates the integral as the sum of trapezoid areas:
AUC = Σ (C_i + C_(i+1))/2 × (t_(i+1) − t_i). It is exact for linear segments and approximate for
curved ones. The units are the product of the axes, which is why AUC is reported in mg·h/L or
nmol·min/L.

**Branch point.** If sampling is sparse around the peak, the trapezoidal AUC underestimates the true
exposure. State the sampling schedule.

### Step 25 — Check the algebra by inversion and bound the answer

**Input.** A rearranged formula and a computed result.

**Output.** A verification that substitution returns the original equation, and a plausibility bound.

**Reasoning.** Algebraic slips are cheap to catch: solve for the variable, substitute back, and confirm
the identity. Every quantity also has a plausible range. A creatinine clearance of 400 mL/min, a
half-life of 0.2 hours for a drug known to persist for days, or a probability of 1.4 all indicate an
error before any statistical check.

**Branch point.** If the substitution fails or the result lies outside the bound, find the error before
reporting. If the bound itself is uncertain, report the discrepancy rather than suppressing it.

### Step 26 — State the transform and the comparison in the report

**Input.** The final value.

**Output.** A reporting sentence containing unit, baseline, comparison, transform, and precision.

**Reasoning.** The reader cannot recover the transform from the number. "Readmissions fell from 18.4%
to 15.1% (3.3 percentage points; 17.9% relative; n = 2,500 per arm)" is a complete statement. The
sentence is longer than a slide headline and it survives scrutiny.

**Branch point.** If the sentence cannot be written without an unknown, the analysis is incomplete.

### 5.1 The branch map

```text
Claim
  |
  +-- referent unknown? ----------> mark unverifiable
  |
  +-- scale nominal/ordinal? -----> use counts, medians, ranks; no ratios
  |
  +-- denominator unclear? -------> rebuild numerator/denominator from source
  |
  +-- first differences constant? -> linear/affine, extrapolate with caution
  |
  +-- ratios constant? -----------> exponential; use log scale
  |
  +-- ratios falling to 1? -------> saturating; fit Emax
  |
  +-- x spans >1 order? ----------> log-log check for power law
  |
  +-- baseline near zero? --------> report absolute change and counts only
  |
  +-- outcome common (>10%)? -----> risk ratio, not odds ratio
  |
  +-- relative uncertainty >20%? -> Monte Carlo propagation
  |
  +-- extrapolation requested? ---> state form, range, and failure mode
```

## 6. Worked example

A regional health system is deciding whether to fund a remote monitoring program for heart failure
discharge. The vendor cites a trial in which 30-day readmissions fell from 18.4% to 15.1%. The
system's own baseline readmission rate is 12.0%, and its finance office uses a fully loaded cost of
$16,200 per readmission. The program costs $1,450 per enrolled patient. The system expects to enrol
2,500 patients per year.

The case is carried through five calculations: the published effect, the local expected effect, the
financial case, a sensitivity analysis, and two scale translations that appear in the same decision
packet.

### 6.1 Reconstruct the published effect from counts

The trial reports percentages. Convert them back to counts so that every later step has an integer
base.

```text
Control arm:       2,500 patients, 18.4% readmitted
  events = 0.184 x 2500 = 460 readmissions

Intervention arm:  2,500 patients, 15.12% readmitted
  events = 0.1512 x 2500 = 378 readmissions
```

The intervention percentage is 15.12% rather than exactly 15.1% because 378/2,500 = 0.15120. A
published figure of 15.1% is consistent with 377 or 378 events; the analyst should not invent
precision that the publication does not provide. For this example, 378 is used and the resulting
difference is reported to two significant figures.

```text
Absolute risk reduction (ARR)
  = 0.1840 - 0.1512
  = 0.0328
  = 3.28 percentage points

Relative risk reduction (RRR)
  = 0.0328 / 0.1840
  = 0.17826...
  = 17.8%

Number needed to treat (NNT)
  = 1 / 0.0328
  = 30.4878...
  = 31 patients (round up; a fractional patient is not treatable)
```

The vendor's headline is "30% relative reduction." The reconstructed relative reduction is 17.8%.
The discrepancy matters: if the vendor used a different baseline or a different endpoint definition,
the analyst needs the raw counts, not the percentage. The reconstruction is the first check on any
inherited claim.

### 6.2 Apply the effect to the local baseline

The trial baseline of 18.4% is higher than the system's 12.0%. A relative effect is often assumed to
transport while the absolute effect does not. Test that assumption rather than assuming it.

```text
Local ARR at constant relative effect
  = 0.1200 x 0.17826
  = 0.021391...
  = 2.14 percentage points

Local NNT
  = 1 / 0.021391
  = 46.75
  = 47 patients
```

If instead the absolute effect transported unchanged (3.28 percentage points), the local rate would
fall from 12.0% to 8.72%, an implausible 27.3% relative reduction for a program that showed 17.8% in
the trial. The constant-relative-effect assumption is the more defensible default for a first pass,
but it is an assumption and belongs in the ledger.

```text
Expected readmissions avoided per year
  = 0.021391 x 2500
  = 53.48
  = 53 readmissions (expected count; retain 53.48 for cost arithmetic)
```

### 6.3 Build the financial case

```text
Program cost
  = $1,450 x 2,500
  = $3,625,000

Readmission cost avoided
  = 53.4783 x $16,200
  = $866,347.83

Net cost
  = $3,625,000 - $866,347.83
  = $2,758,652.17

Net cost per readmission avoided
  = $2,758,652.17 / 53.4783
  = $51,584.55
```

The program does not save money at these inputs. It costs about $51,600 per readmission avoided. A
committee that wants a cost-saving program should reject it; a committee that values the readmission
avoided at more than $51,600, or that counts the downstream effects of a readmission (post-acute
placement, patient burden, penalty exposure), may fund it. The arithmetic does not make that value
judgement, but it makes the judgement possible.

Note the difference between the crude and the expected-value cost per event. Dividing $3,625,000 by
53 (the rounded count) gives $68,396, which overstates the cost by 32%. Retaining the expected count
through the division avoids rounding artefacts in ratio calculations.

### 6.4 Run a one-way sensitivity analysis

The two inputs most likely to be wrong are the local baseline and the transported relative effect.
Vary each across a plausible range and recompute the net cost per avoided readmission.

| Baseline | RRR | ARR (pp) | Avoided per 2,500 | Net cost per avoided |
|---:|---:|---:|---:|---:|
| 12.0% | 17.8% | 2.14 | 53.5 | $51,584 |
| 12.0% | 10.0% | 1.20 | 30.0 | $120,833 |
| 12.0% | 25.0% | 3.00 | 75.0 | $32,133 |
| 18.4% | 17.8% | 3.28 | 82.0 | $28,007 |
| 8.0% | 17.8% | 1.43 | 35.7 | $85,477 |

Check one row by hand. At a 10.0% relative effect on a 12.0% baseline:

```text
ARR  = 0.1200 x 0.1000 = 0.0120
Avoided = 0.0120 x 2500 = 30.0
Net cost = 3,625,000 - (30.0 x 16,200) = 3,625,000 - 486,000 = 3,139,000
Net cost per avoided = 3,139,000 / 30.0 = 104,633.33
```

The table row says $120,833. That row is wrong, and the hand check caught it. The correct value is
$104,633. This is exactly why the procedure requires a hand check on at least one table cell: a
spreadsheet copied across columns propagates one bad formula into every row. Correcting the whole
table gives:

| Baseline | RRR | ARR (pp) | Avoided per 2,500 | Net cost per avoided |
|---:|---:|---:|---:|---:|
| 12.0% | 17.8% | 2.14 | 53.5 | $51,584 |
| 12.0% | 10.0% | 1.20 | 30.0 | $104,633 |
| 12.0% | 25.0% | 3.00 | 75.0 | $32,133 |
| 18.4% | 17.8% | 3.28 | 82.0 | $28,007 |
| 8.0% | 17.8% | 1.43 | 35.7 | $85,477 |

The decision is dominated by the relative effect. At a 10% effect the program costs over $100,000 per
readmission avoided; at a 25% effect it costs about $32,000. The committee should therefore fund a
pilot that measures the local effect rather than committing to the full rollout on the vendor's
number. That is a decision the arithmetic supports.

### 6.5 Translate two logarithmic quantities in the same packet

The same decision packet contains a biomarker result and a drug exposure estimate. Both use
logarithms, and both are easy to misread.

**Biomarker.** Median procalcitonin fell from 1.94 ng/mL in the usual-care arm to 0.52 ng/mL in the
guided arm.

```text
Fold change = 1.94 / 0.52 = 3.7308
log10 difference = log10(1.94) - log10(0.52)
                 = 0.28780 - (-0.28400)
                 = 0.57180
Check: 10^0.57179 = 3.7307   (agrees to four significant figures)

Natural-log difference = ln(3.7308) = 1.3166
```

A report that says "procalcitonin decreased by 0.57" is meaningless without the base. A report that
says "decreased by 0.57 log10 units, a 3.7-fold reduction" is complete.

**Drug exposure.** A drug has an elimination half-life of 6.0 hours. The clinical question is how
much remains 24 hours after the last dose, assuming first-order elimination and no further dosing.

```text
k = ln(2) / t_half
  = 0.693147 / 6.0 h
  = 0.115525 /h

Number of half-lives in 24 h
  = 24.0 / 6.0 = 4.0

Remaining fraction
  = 0.5^4 = 0.0625 = 6.25%

Check via the exponential:
  e^(-0.115525 x 24) = e^(-2.77259) = 0.06250
```

The two routes agree. The clinical interpretation is that roughly 94% of the drug has been
eliminated, which is why a missed dose 24 hours later does not simply "top up" the previous level.
If the same drug were given every 6 hours, accumulation would be governed by
1/(1 − e^(−k·τ)) = 1/(1 − 0.5) = 2.0, so the steady-state trough would be twice the single-dose
trough. That accumulation factor is the correct calculation for a repeated-dose regimen; the
remaining-fraction calculation alone is not.

### 6.6 The growth extrapolation that failed

The imaging service line recorded monthly volumes of 1,000, 1,040, 1,082, and 1,125 scans in the
first four months, a consistent 4.0% monthly growth. A linear trend fitted to the first three points
has a slope of about 41 scans per month. Extrapolating that line to month 24 gives:

```text
Linear at month 24 = 1,000 + 40.8 x 24 = 1,979 scans
Exponential at month 24 = 1,000 x 1.04^24 = 2,563 scans
Shortfall of the linear forecast = (2,563 - 1,979) / 2,563 = 22.8%
```

The first three points cannot distinguish a line from an exponential, because over a short interval
an exponential is nearly linear. The fourth point does not settle it either. Only a longer series, or
a mechanistic reason to expect compounding, supports the exponential. The honest conclusion at four
points is a range, not a point forecast.

### 6.7 What the example demonstrates

| Step | Operation | Error prevented |
|---|---|---|
| Reconstruct from counts | Percentages back to integers | Inherited headline error |
| Absolute before relative | ARR first | Anchoring on the larger number |
| Constant-relative transport | Apply RRR to local baseline | Transporting absolute effect blindly |
| Expected counts in ratios | Retain 53.48 | 32% overstatement from rounding |
| One-way sensitivity | Vary the two fragile inputs | False certainty from a point estimate |
| Log base stated | log10, fold change | "0.57" with no meaning |
| Half-life as fraction | 0.5^n and e^(−kt) | Dosing-interval error |
| Form before extrapolation | Linear versus exponential | 22.8% capacity shortfall |

## 7. Code, query, or calculation

The script below reproduces every number in section 6 using only the Python standard library. It is
written to be run as-is and to print a fixed report. The assertions at the end fail loudly if the
arithmetic drifts.

```python
"""Topic 01 arithmetic check.

Reproduces section 6: absolute and relative effects, NNT, budget impact,
one-way sensitivity, log conversions, half-life, and growth extrapolation.
Standard library only.
"""

from math import log, log10, exp

def pct(x: float) -> str:
    return f"{x * 100:.2f}%"

def money(x: float) -> str:
    return f"${x:,.2f}"

# --- 6.1 Published effect, reconstructed from counts -------------------------
n_per_arm = 2500
control_events = 460          # 18.40% of 2500
intervention_events = 378     # 15.12% of 2500
control_rate = control_events / n_per_arm
intervention_rate = intervention_events / n_per_arm
arr = control_rate - intervention_rate
rrr = arr / control_rate
nnt = 1 / arr

print("6.1 Published effect")
print(f"  control rate        {pct(control_rate)}")
print(f"  intervention rate   {pct(intervention_rate)}")
print(f"  ARR                 {arr * 100:.2f} percentage points")
print(f"  RRR                 {pct(rrr)}")
print(f"  NNT                 {nnt:.2f} -> {int(-(-1 // arr))} patients")

# --- 6.2 Local baseline ------------------------------------------------------
local_baseline = 0.120
local_arr = local_baseline * rrr
local_nnt = 1 / local_arr
avoided = local_arr * n_per_arm

print("\n6.2 Local baseline")
print(f"  local ARR           {local_arr * 100:.2f} percentage points")
print(f"  local NNT           {local_nnt:.2f} -> {int(-(-1 // local_arr))} patients")
print(f"  avoided per year    {avoided:.2f}")

# --- 6.3 Financial case ------------------------------------------------------
cost_per_patient = 1450.0
readmission_cost = 16200.0
program_cost = cost_per_patient * n_per_arm
savings = avoided * readmission_cost
net_cost = program_cost - savings
net_per_avoided = net_cost / avoided
crude_per_avoided = program_cost / round(avoided)

print("\n6.3 Financial case")
print(f"  program cost        {money(program_cost)}")
print(f"  cost avoided        {money(savings)}")
print(f"  net cost            {money(net_cost)}")
print(f"  net per avoided     {money(net_per_avoided)}")
print(f"  crude per avoided   {money(crude_per_avoided)}")

# --- 6.4 Sensitivity ---------------------------------------------------------
print("\n6.4 Sensitivity (net cost per readmission avoided)")
for baseline in (0.08, 0.12, 0.184):
    for effect in (0.10, rrr, 0.25):
        a = baseline * effect
        avoided_i = a * n_per_arm
        net_i = program_cost - avoided_i * readmission_cost
        per_i = net_i / avoided_i if avoided_i else float("nan")
        print(f"  baseline {pct(baseline):>7}  RRR {pct(effect):>6}  "
              f"ARR {a*100:4.2f} pp  avoided {avoided_i:5.1f}  {money(per_i):>12}")

# --- 6.5 Logarithms ----------------------------------------------------------
pct_usual, pct_guided = 1.94, 0.52
fold = pct_usual / pct_guided
d_log10 = log10(pct_usual) - log10(pct_guided)
d_ln = log(fold)

print("\n6.5 Logarithms")
print(f"  fold change         {fold:.4f}")
print(f"  log10 difference    {d_log10:.5f}")
print(f"  antilog check       10**d = {10 ** d_log10:.4f}")
print(f"  ln difference       {d_ln:.4f}")

half_life_h = 6.0
k = log(2) / half_life_h
remaining = 0.5 ** (24.0 / half_life_h)
accumulation = 1 / (1 - exp(-k * half_life_h))

print(f"  k                   {k:.6f} /h")
print(f"  remaining at 24 h   {remaining:.4f}")
print(f"  accumulation q6h    {accumulation:.4f}")

# --- 6.6 Growth extrapolation ------------------------------------------------
v0, growth = 1000.0, 0.04
linear_slope = 40.8  # fit through months 0-2 of a 4%/mo series
linear_24 = v0 + linear_slope * 24
expo_24 = v0 * (1 + growth) ** 24
shortfall = (expo_24 - linear_24) / expo_24

print("\n6.6 Growth")
print(f"  linear at 24 mo     {linear_24:.1f} scans")
print(f"  exponential at 24   {expo_24:.1f} scans")
print(f"  shortfall           {shortfall * 100:.1f}%")

# --- Invariants --------------------------------------------------------------
assert abs(arr - 0.0328) < 1e-12
assert round(rrr, 4) == 0.1783
assert 30 < nnt < 31
assert abs(remaining - 0.0625) < 1e-12
assert abs(accumulation - 2.0) < 1e-9
assert abs(10 ** d_log10 - fold) < 1e-9
print("\nAll assertions passed.")
```

**Expected output.**

```text
6.1 Published effect
  control rate        18.40%
  intervention rate   15.12%
  ARR                 3.28 percentage points
  RRR                 17.83%
  NNT                 30.49 -> 31 patients

6.2 Local baseline
  local ARR           2.14 percentage points
  local NNT           46.75 -> 47 patients
  avoided per year    53.48

6.3 Financial case
  program cost        $3,625,000.00
  cost avoided        $866,347.83
  net cost            $2,758,652.17
  net per avoided     $51,584.55
  crude per avoided   $68,396.23

6.4 Sensitivity (net cost per readmission avoided)
  baseline   8.00%  RRR 10.00%  ARR 0.80 pp  avoided  20.0   $165,050.00
  baseline   8.00%  RRR 17.83%  ARR 1.43 pp  avoided  35.7    $85,476.83
  baseline   8.00%  RRR 25.00%  ARR 2.00 pp  avoided  50.0    $56,300.00
  baseline  12.00%  RRR 10.00%  ARR 1.20 pp  avoided  30.0   $104,633.33
  baseline  12.00%  RRR 17.83%  ARR 2.14 pp  avoided  53.5    $51,584.55
  baseline  12.00%  RRR 25.00%  ARR 3.00 pp  avoided  75.0    $32,133.33
  baseline  18.40%  RRR 10.00%  ARR 1.84 pp  avoided  46.0    $62,604.35
  baseline  18.40%  RRR 17.83%  ARR 3.28 pp  avoided  82.0    $28,007.32
  baseline  18.40%  RRR 25.00%  ARR 4.60 pp  avoided 115.0    $15,321.74

6.5 Logarithms
  fold change         3.7308
  log10 difference    0.57180
  antilog check       10**d = 3.7308
  ln difference       1.3166
  k                   0.115525 /h
  remaining at 24 h   0.0625
  accumulation q6h    2.0000

6.6 Growth
  linear at 24 mo     1979.2 scans
  exponential at 24   2563.3 scans
  shortfall           22.8%

All assertions passed.
```

**Assumptions stated by the script.**

1. The two arms have equal size. Unequal arms require the pooled or weighted formulas in Topic 02.
2. The relative effect is assumed constant across baselines. This is the transportability assumption
   discussed in section 6.2 and tested in the sensitivity block.
3. Costs are fully loaded and fixed per event. A real budget impact model varies cost with length of
   stay and post-acute placement.
4. Elimination is first-order and the dosing interval equals one half-life. Both are stated in the
   output rather than assumed silently.
5. Growth is constant at 4.0% per month. The example exists to show the divergence between forms, not
   to forecast the service line.

## 8. Data, standards, and regulatory context

Mathematical conventions are not all matters of taste. Several are fixed by standards, and a few are
fixed by regulation because a misreported effect size can alter prescribing or reimbursement.

### 8.1 Binding standards

| Standard | Body | What it fixes | Binding status |
|---|---|---|---|
| ISO 80000-2:2019 | ISO/IEC | Mathematical signs, symbols, and notation, including log bases and exponential notation | Voluntary standard; contractual when cited |
| SI Brochure, 9th edition (2019) | BIPM | SI units and prefixes used in every formula | Treaty-level for signatories; de facto global |
| JCGM 100:2008 (GUM) | JCGM | Vocabulary and method for expressing measurement uncertainty | Voluntary; referenced by accreditation bodies |
| ISO 5725 | ISO | Accuracy and precision of measurement methods | Voluntary; used in lab accreditation |
| UCUM | Regenstrief | Machine-readable unit codes | Required by HL7 FHIR profiles that adopt it |
| ICH E9(R1) | ICH | Estimand framework: population, variable, intercurrent events, summary | Binding for ICH-region drug submissions |
| CONSORT 2010 | CONSORT Group | Reporting of absolute and relative effects with baselines | Journal requirement, not law |

The distinction between binding and conventional matters. ISO 80000-2 is not law in most
jurisdictions, but a contract that cites it makes it binding for that project. ICH E9(R1) is
incorporated into regulation in the EU, US, and Japan, so its requirement to define the estimand
before analysis has legal force for those submissions. CONSORT is a journal requirement; a paper that
violates it is not illegal, it is unpublishable in most major journals.

### 8.2 What the standards constrain

**ISO 80000-2** fixes notation, not interpretation. It specifies that logarithms without a stated
base are ambiguous and that the natural logarithm is written ln. It does not tell an analyst whether
a relative reduction should be reported as a percentage or a proportion. That choice is governed by
reporting guidelines and by the audience.

**The SI Brochure** fixes the unit system. It specifies that unit symbols are printed upright, that
there is a space between a number and its unit, and that prefixes may not be combined. These rules
matter in machine-readable reports because a parser that expects "mg/dL" will fail on "mg/dl" if the
lookup is case-sensitive. The SI does not define clinical units such as mmHg or IU; those are
accepted for use with the SI but are not SI units.

**The GUM** fixes how uncertainty is expressed: a measurand, a value, a standard uncertainty, and a
coverage factor. It endorses the first-order propagation used in Step 22 for linear functions and
requires Monte Carlo methods when the linear approximation is inadequate (GUM Supplement 1). This is
the standards basis for the branch point at Step 22.

**ICH E9(R1)** is the strongest constraint on the arithmetic in this topic. It requires the sponsor to
define the estimand: the population, the treatment condition, the variable, the handling of
intercurrent events, and the population-level summary. A relative risk reduction is a summary; the
estimand determines whether it is even the right summary. A trial that reports a 17.8% relative
reduction without stating how deaths and discontinuations were handled has not defined its estimand,
and the number cannot be interpreted.

**CONSORT** requires that both absolute and relative effects be reported with confidence intervals
and that the baseline risk be given. This is the reporting standard that would have prevented the
vendor headline in section 6 from being ambiguous.

### 8.3 What the standards do not constrain

Nothing in these standards specifies:

- Which baseline to use for a relative effect. That is a scientific choice.
- Whether an odds ratio or a risk ratio is appropriate. That follows from the design.
- How many significant figures to report. The GUM gives principles; the analyst applies them.
- Whether a logarithmic transform is clinically meaningful. That is a domain judgement.
- Whether a functional form is correct outside the observed range. No standard can supply that.

A common regulatory failure is to treat a reporting standard as a validation standard. CONSORT makes
a paper transparent; it does not make the underlying trial valid. ISO 80000-2 makes notation
unambiguous; it does not make the arithmetic correct.

### 8.4 Regulatory consequences of arithmetic errors

Three settings where the arithmetic in this topic has direct regulatory weight:

1. **Drug labelling.** Dose adjustment tables in labelling are derived from clearance and exposure
   calculations. An error in a half-life or an accumulation factor becomes a dosing instruction. The
   FDA's guidance on pharmacokinetics in patients with impaired hepatic or renal function expects
   these calculations to be shown and justified.
2. **Diagnostic reporting.** A lab result reported in the wrong unit is a reportable safety event
   under CLIA in the US and under IVDR in the EU. The arithmetic is trivial; the consequence is not.
3. **Health economic submissions.** Reimbursement dossiers to NICE or to a national payer require an
   incremental cost-effectiveness ratio with a stated perspective, time horizon, discount rate, and
   sensitivity analysis. An undiscounted or single-point ICER is routinely rejected.

> **Note:** The regulatory content in this section is educational. It is not legal advice, and a
> submission-grade calculation requires review by a qualified regulatory professional.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Hospital quality reporting.** US hospitals report readmissions to the Centers for Medicare &
Medicaid Services (CMS) under the Hospital Readmissions Reduction Program (HRRP), established by the
Affordable Care Act and applied to payments from fiscal year 2013. The program compares an observed
readmission rate to an expected rate derived from a risk model. The arithmetic is a ratio of observed
to expected events, and small changes in the risk model move the penalty. Documented in CMS's
published methodology; the operational lesson is that the denominator and the risk adjustment are
the contested parts, not the subtraction.

**Value assessment.** NICE in England uses a cost-effectiveness threshold commonly cited as £20,000
to £30,000 per quality-adjusted life year (QALY). The Institute for Clinical and Economic Review
(ICER) in the US publishes health-benefit price benchmarks with a commonly cited range of $100,000 to
$150,000 per QALY. These are policy thresholds, not measurements, and they are applied to incremental
cost-effectiveness ratios computed as ΔCost/ΔQALY. A ratio with a near-zero denominator is unstable,
which is why both organisations require sensitivity analysis.

**Diabetes software.** Glucose conversion between mg/dL and mmol/L is a live interoperability
problem. The long-published factor is 18.0182 (mg/dL per mmol/L), derived from a molar mass of about
180.182 g/mol. The modern IUPAC molar mass of anhydrous D-glucose is 180.156 g/mol, which gives
18.0156. Open-source diabetes projects have moved between these values: Nightscout adopted 18.01559
in 2025 and xDrip followed in a 2026 nightly release, while many device vendors and clinical
documents still print 18.0182. The discrepancy is about 0.015% and clinically negligible at typical
glucose values, but it produces visible display mismatches between apps and is therefore a real
interoperability issue rather than a theoretical one. This program uses 18.0182 as the stated
clinical convention and notes the alternative where it matters.

**Clinical laboratories.** Assay harmonisation is an ongoing effort. Two labs measuring the same
analyte with different methods can produce results that differ by more than the clinical decision
limit, which is why external quality assessment schemes exist. The arithmetic consequence is that a
ratio computed across labs may compare methods rather than patients.

### 9.2 Case evidence

**The EHR alert that fired on 40% of patients.** A commonly reported operational pattern, documented
in the clinical decision support literature, is an alert whose positive predictive value collapses
because the baseline prevalence in the deployed population is far below the prevalence in the
development sample. The arithmetic is the Bayes update in Topic 02; the point here is that the same
sensitivity and specificity produce a different PPV at a different baseline. Inference: the specific
institution is not named because the pattern is generic; the mechanism is documented.

**The imaging forecast that missed.** Service-line forecasts built by linear regression on 6 to 12
months of data are common in hospital finance. When the underlying adoption is exponential, the
forecast understates volume, and the error compounds: a 4% monthly growth versus a linear fit is
22.8% short at 24 months by the calculation in section 6.6. This is arithmetic, not a case report.

**The vendor AUC claim.** A vendor reports an area under the receiver operating characteristic curve
(AUROC) of 0.89 on a random split of retrospective data. A random split assumes exchangeability
between training and test observations. When the same patient contributes multiple encounters, a
random split leaks patient identity and inflates AUROC. The arithmetic of AUROC is covered in Module
06; the assumption is statistical and belongs in the ledger.

> **Note:** Named organisations above are cited for documented policies (CMS HRRP, NICE threshold
> range, ICER benchmark range, Nightscout/xDrip conversion values). The operational patterns in 9.2
> are labelled as inference where no single named case is asserted.

### 9.3 The practical pattern that separates strong teams

Teams that avoid these errors share three habits:

1. They reconstruct percentages into counts before comparing anything.
2. They state the baseline in the same sentence as the relative effect.
3. They plot before they model, and they decline to extrapolate from fewer than five points.

None of these habits requires software. All three require the discipline that this topic is designed
to install.

## 10. Failure modes and red flags

The table lists the errors that survive review because the output looks plausible. Each row names a
signal that can be checked before the number reaches a decision.

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Relative change reported without a baseline | Slide says "up 30%" and no starting value appears | The denominator was dropped when the percentage was computed | Stakeholders assume percentage points and overestimate impact | Require the baseline in the same sentence; report ARR alongside RRR |
| 2 | Percentage points called "percent" | Two percentages subtracted and the result labelled "%" | Ambiguity in the word "percent" | A 3.3-point change is read as 3.3% of patients | Always write "percentage points" for arithmetic differences |
| 3 | Dividing by the post-period value | Relative change computed as (after−before)/after | Wrong baseline chosen from the table column order | Effect overstated, sometimes by more than 20% | Write the formula with "before" in the denominator before computing |
| 4 | Odds ratio presented as a risk ratio | Outcome incidence above 10% and an OR near 2 | Odds exceed risk when the outcome is common | Effect overstated by a factor that grows with incidence | Convert with the baseline risk or compute the risk ratio from counts |
| 5 | Log base unstated | Report says "a 0.57 log reduction" | Base 10 and base e differ by 2.302585 | Reader cannot recover the fold change; cross-study comparison fails | State the base in the axis label, table header, and sentence |
| 6 | Log difference read as a linear difference | "0.3 better on the log scale" treated as 0.3 units of the raw quantity | The log scale compresses ratios, not amounts | Magnitude misjudged by orders of magnitude | Convert to fold change and state the absolute baseline |
| 7 | Half-life applied to non-first-order elimination | Log concentration versus time is curved | Saturable clearance makes t½ concentration-dependent | Dose interval wrong at high concentrations | Plot log C versus t; report apparent half-life over a stated range |
| 8 | Accumulation ignored on repeated dosing | Trough concentration predicted from single-dose data | Repeated doses sum with the accumulation factor | Trough underpredicted by up to 2-fold or more | Compute 1/(1−e^(−k·τ)) and state the dosing interval |
| 9 | Linear extrapolation of exponential growth | Forecast uses a straight line through early adoption data | Short series cannot distinguish linear from exponential | Capacity or budget shortfall that compounds | Test constant ratios; state the form; give a range |
| 10 | Exponential extrapolation of a saturating process | Projection exceeds the physical ceiling of the system | Early growth is nearly exponential; saturation appears later | Overinvestment, unused capacity | Fit Emax or logistic; identify the ceiling and the observed range |
| 11 | Power law read from a semi-log plot | Straight line on semi-log called a power law | Semi-log linearity indicates exponential, not power law | Wrong exponent, wrong scaling conclusion | Use log-log for power laws; state which plot was used |
| 12 | False precision in the reported result | Result has five significant figures from two-significant-figure inputs | Full calculator precision copied into the report | Credibility loss; spurious differences treated as real | Round at the end to the least precise input; state the rule |
| 13 | Ratio with a near-zero denominator | Cost per case avoided when avoided is 2 | Small denominator amplifies both value and error | Unstable estimate presented as a point value | Report counts and absolute costs; use a range or decline the ratio |
| 14 | Rate computed with persons instead of person-time | Follow-up durations vary widely across patients | Denominator omits exposure time | Rate underestimated; studies incomparable | Compute person-time; state the time unit |
| 15 | Units dropped during a multi-step calculation | Intermediate result has no unit written next to it | Unit tracking abandoned after the first step | Silent factor-of-1,000 error (mg vs g, mL vs L) | Write units at every step; apply the dimensional check in Topic 04 |
| 16 | Rounding twice | Intermediate value rounded, then used in a ratio | Display rounding applied to working values | Small but systematic bias in totals | Carry full precision; round only the final report |

Three of these deserve a longer note because they recur in every analytics review.

**Row 4, odds versus risk.** An odds ratio of 2.0 with a baseline risk of 30% corresponds to a risk
ratio of about 1.54. The conversion uses RR = OR/(1 − p0 + p0·OR) = 2/(1 − 0.3 + 0.6) = 2/1.3 =
1.538. Presenting the OR as an RR overstates the effect by 30%. At a baseline of 5%, the same OR
corresponds to an RR of about 1.90, an overstatement of 5%. The error scales with baseline risk,
which is why the red flag is "common outcome plus odds ratio."

**Row 9, linear extrapolation.** The mechanism is that a short window of an exponential is
approximately linear: 1.04^t ≈ 1 + 0.04t for small t. The approximation error grows quadratically.
An analyst who fits a line to months 0 to 3 is not wrong about those months; the error is entirely in
the extrapolation.

**Row 13, near-zero denominator.** The ratio ΔCost/ΔEffect is unstable when ΔEffect approaches zero,
and it changes sign when ΔEffect crosses zero. Health economic guidelines require that such cases be
presented as dominance or as a range, not as a point estimate. The same instability affects
cost-per-case-avoided calculations in operational dashboards.

## 11. Assumptions and limits

Every calculation in this topic rests on assumptions. The ledger below states them, why they are
needed, what breaks them, and how to test them.

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The numerator and denominator refer to the same population and period | A rate is meaningless if its parts describe different groups | Different catchment areas, different fiscal calendars, late-arriving records | Rebuild both from one source; reconcile counts to a published total |
| 2 | The measurement scale supports the arithmetic used | Means and ratios are undefined on nominal or ordinal scales | Averaging diagnosis codes or pain ranks | Classify the scale before computing; use counts and medians for ordinal data |
| 3 | The baseline is stable and adequately measured | Relative change divides by the baseline | Small or noisy baseline; baseline measured in a different assay | Report the baseline with a confidence interval; check assay comparability |
| 4 | The functional form holds within the observed range | Interpolation and extrapolation depend on the form | Curvature, thresholds, saturation, regime change | Plot; test constant differences and constant ratios; split the range |
| 5 | The relative effect transports across populations | Applying trial results to a local baseline | Effect modification by severity, adherence, or care pathway | Compare case mix; run a local pilot; test interaction in the trial data |
| 6 | Elimination is first-order | Half-life and accumulation factors assume it | Saturable metabolism, zero-order drugs, renal failure | Log-linear plot; dose-normalised AUC comparison |
| 7 | Measurements are independent | Uncertainty propagation assumes independent inputs | Shared calibration, repeated measures on one patient | Check the covariance structure; use correlated propagation |
| 8 | Uncertainties are small enough for a linear approximation | First-order propagation ignores curvature | Relative uncertainty above about 20%; strongly nonlinear function | Compare with Monte Carlo; widen the interval |
| 9 | Interpolation is valid between bracketing points | Linear interpolation assumes a straight segment | Discontinuities at dose changes, shift changes, recalibration | Identify known discontinuities; do not interpolate across them |
| 10 | The reported precision matches the inputs | Significant figures communicate real resolution | Extra digits from software; unit conversion adding digits | Round to the least precise input; state the convention |
| 11 | The comparison group is the right counterfactual | Absolute and relative effects are defined against a comparator | Comparator differs in severity or era | Check the comparator definition; consider active comparators |
| 12 | The time window is long enough to capture the effect | Short windows miss delayed outcomes and include transient effects | 30-day window for a 90-day outcome; seasonality | Justify the window from the mechanism; test alternative windows |
| 13 | Costs are measured from a stated perspective | Cost per event depends on whose costs are counted | Payer versus provider versus societal perspective | State the perspective; list included and excluded costs |
| 14 | No systematic drift over the observation period | Trend and growth estimates assume a stable process | Device recalibration, coding change, policy change | Plot residuals over time; test for a level shift |

> **Note:** Assumption 5 is the one most often violated silently. A trial conducted in academic
> centres with high adherence rarely transports a relative effect unchanged to a community setting
> with lower adherence. The constant-relative-effect default in section 6.2 is a convenience, not a
> law.

The limits of this topic are equally explicit:

- It does not cover statistical inference. Whether an observed difference is distinguishable from
  noise is Topic 02.
- It does not cover measurement uncertainty in depth. The GUM treatment, calibration, and limit of
  detection are Topic 03.
- It does not cover dimensional analysis of composite clinical units. That is Topic 04.
- It does not cover the biological mechanisms behind the quantities. That is Topic 05.
- It does not make any claim about which functional form is biologically correct. It provides tests
  that can reject a form, not proofs that a form is true.

## 12. Exercises

Eight exercises follow, ordered from recall to design. Each states a deliverable, acceptance criteria,
and an estimated time. Work in a script or a spreadsheet, and keep the intermediate steps; the
acceptance criteria check the steps, not only the final number.

### Exercise 1 — Classify the scale (recall and application)

Take the following variables from a hospital dataset and classify each as nominal, ordinal, interval,
ratio, count, or proportion: blood type; pain score 0–10; body temperature in °C; serum creatinine in
mg/dL; number of prior admissions; percentage of beds occupied; NYHA heart failure class; length of
stay in days.

For each, state one arithmetic operation that is valid and one that is not.

**Deliverable.** A table with columns: variable, scale, valid operation, invalid operation, one-sentence
justification.

**Acceptance criteria.** All eight variables classified correctly; the invalid operation is genuinely
invalid (for example, a mean of blood type); the justification names the property of the scale that
decides the case.

**Time.** 25 minutes.

### Exercise 2 — Absolute and relative effects from counts (calculation)

A trial reports 812 readmissions among 9,400 control patients and 690 among 9,350 intervention
patients.

Compute: the two risks, the absolute risk reduction in percentage points, the relative risk
reduction, the number needed to treat, and the expected number of readmissions avoided per 10,000
patients treated.

**Deliverable.** A calculation sheet showing each formula, the substitution, the intermediate value,
and the final value with units.

**Acceptance criteria.** Risks reported to two decimal places as percentages; ARR in percentage
points; RRR as a percentage with the baseline stated; NNT rounded up to a whole patient; the per-10,000
figure derived from the ARR, not from the RRR.

**Time.** 30 minutes.

### Exercise 3 — Rewrite a misleading sentence (critique)

A press release states: "The program reduced readmissions by 12%, from 18.4% to 16.2%, a
transformative improvement in patient outcomes."

Rewrite the sentence in defensible form. Identify every defect: the relative/absolute confusion, the
missing counts, the missing time window, the promotional adjective, and any claim that the data do not
support.

**Deliverable.** A 120-word rewrite plus a bullet list of defects.

**Acceptance criteria.** The rewrite states the absolute change in percentage points and the relative
change with its baseline; the word "transformative" is removed or justified; at least four defects are
named; the rewrite does not introduce a new unsupported claim.

**Time.** 25 minutes.

### Exercise 4 — Half-life, remaining fraction, and accumulation (calculation)

A drug has a first-order elimination half-life of 8.0 hours and is given every 8 hours.

Compute: the elimination rate constant in /h; the fraction remaining 24 hours after a single dose;
the accumulation factor at steady state for the 8-hour interval; the steady-state trough as a multiple
of the single-dose trough. Then state what changes if the half-life is actually 20 hours because the
patient has renal impairment.

**Deliverable.** Four computed values with units, plus a three-sentence interpretation.

**Acceptance criteria.** k = 0.0866 /h to three significant figures; remaining fraction 0.125;
accumulation factor 2.00; the renal impairment case shows a larger accumulation factor and the
calculation is repeated, not asserted; the interpretation states the assumption of first-order
elimination.

**Time.** 30 minutes.

### Exercise 5 — Log scale translation (calculation)

A viral load falls from 4.8 log10 copies/mL to 2.1 log10 copies/mL. A second patient's load falls from
2.9 log10 copies/mL to 0.2 log10 copies/mL.

Compute the fold change for each patient, the arithmetic difference in log10 units for each, and the
absolute difference in copies/mL for each. Explain why the log difference is the same for both
patients while the absolute difference is not.

**Deliverable.** A table of six values plus a paragraph.

**Acceptance criteria.** Fold changes 501.2 and 501.2 (to four significant figures); both log
differences 2.70; the absolute differences computed as 10^4.8 − 10^2.1 and 10^2.9 − 10^0.2; the
explanation states that a log scale converts ratios into differences.

**Time.** 25 minutes.

### Exercise 6 — Identify the functional form (application and design)

You are given four short series:

- A: 100, 110, 121, 133, 146 (annual)
- B: 100, 120, 140, 160, 180 (annual)
- C: 10, 40, 90, 160, 250 (dose 1 to 5)
- D: 0.50, 0.75, 0.88, 0.94, 0.97 (dose 1 to 5)

For each, compute first differences and consecutive ratios, propose a functional form, and state what
additional data would be required before extrapolating five periods ahead.

**Deliverable.** A table with series, first differences, ratios, proposed form, and the evidence needed.

**Acceptance criteria.** A identified as exponential with ratio 1.10; B as linear with constant
difference 20; C as quadratic or power law with increasing differences and a log-log check; D as
saturating with ratios approaching 1; every extrapolation statement names a range and a failure mode.

**Time.** 40 minutes.

### Exercise 7 — Propagate uncertainty (calculation)

Clearance is computed as CL = Dose/AUC. The dose is 500 mg with a standard uncertainty of 5 mg. The AUC
is 62.5 mg·h/L with a standard uncertainty of 3.1 mg·h/L.

Compute CL, its standard uncertainty by first-order propagation, and its 95% coverage interval using
a coverage factor of 2. Then state whether the linear approximation is adequate and why.

**Deliverable.** A calculation with units at every step and a one-paragraph adequacy judgement.

**Acceptance criteria.** CL = 8.00 L/h; relative uncertainties 1.0% and 5.0%; combined relative
uncertainty 5.1%; standard uncertainty 0.41 L/h; interval 7.18 to 8.82 L/h; the judgement references
the 5% relative uncertainty and the near-linear form of the quotient.

**Time.** 30 minutes.

### Exercise 8 — Audit a vendor claim (design)

A vendor white paper claims: "Our AI triage tool cuts emergency department length of stay by 42%,
saving $2.1M annually per hospital." The paper reports a before-after comparison at one site, a
6-month pre-period, a 6-month post-period, and a $900 average cost per emergency visit.

Build a one-page audit that: reconstructs the implied absolute change; states what baseline and
denominators are missing; names three alternative explanations for the decline; specifies the
additional data required; and rewrites the claim as a testable hypothesis.

**Deliverable.** A one-page audit with a reconstruction table, three alternative explanations, a data
request list, and a rewritten hypothesis.

**Acceptance criteria.** The reconstruction shows the arithmetic and states the implied minutes saved;
at least three alternatives are named from {seasonality, coding change, staffing change, regression to
the mean, patient mix, measurement change}; the rewritten hypothesis specifies population, comparator,
outcome, time window, and expected effect size; no number is invented.

**Time.** 50 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Variable | Scale | Valid operation | Invalid operation |
|---|---|---|---|
| Blood type | Nominal | Counting, mode | Mean, ratio |
| Pain score 0–10 | Ordinal | Median, rank comparison | Mean treated as interval, ratio |
| Temperature °C | Interval | Difference, mean of differences | Ratio ("twice as hot") |
| Creatinine mg/dL | Ratio | Ratio, mean, coefficient of variation | None of the standard operations is invalid |
| Prior admissions | Count | Sum, mean, rate per patient | Treating as continuous normal without checking |
| Bed occupancy % | Proportion | Mean across wards weighted by beds | Unweighted mean of ward percentages |
| NYHA class | Ordinal | Median, transition matrix | Mean class |
| Length of stay days | Ratio (count-like) | Mean, median, total bed-days | Ratio when the denominator includes zero-length stays |

The justification rule: interval scales have an arbitrary zero, so ratios are meaningless; ordinal
scales have ordered categories with unequal spacing, so means assume spacing that the scale does not
provide.

### Answer to Exercise 2

```text
Control risk     = 812 / 9,400   = 0.086383 = 8.64%
Intervention risk = 690 / 9,350  = 0.073797 = 7.38%

ARR = 0.086383 - 0.073797 = 0.012586 = 1.26 percentage points
RRR = 0.012586 / 0.086383 = 0.14570 = 14.6%
NNT = 1 / 0.012586 = 79.45 -> 80 patients
Avoided per 10,000 = 0.012586 x 10,000 = 125.9 -> 126 readmissions
```

The NNT rounds up because a fraction of a patient cannot be treated; rounding down would understate
the number needed. The per-10,000 figure uses the ARR, not the RRR: applying 14.6% to 10,000 would
give 1,460, which is the number of readmissions in the control group, not the number avoided.

### Answer to Exercise 3

Defensible rewrite: "In a single-site before-after evaluation, 30-day readmissions fell from 18.4% to
16.2%, an absolute reduction of 2.2 percentage points and a relative reduction of 12.0% against the
18.4% baseline. The evaluation did not include a concurrent control group, so the decline cannot be
attributed to the program alone."

Defects: (1) "reduced by 12%" without stating that 12% is relative; (2) no counts or sample size;
(3) no time window; (4) "transformative" is promotional and unsupported; (5) the causal attribution
implied by "the program reduced" is not supported by an uncontrolled design.

### Answer to Exercise 4

```text
k = ln(2) / 8.0 h = 0.693147 / 8.0 = 0.086643 /h

Fraction remaining at 24 h = 0.5^(24/8) = 0.5^3 = 0.125 (12.5%)

Accumulation factor at tau = 8 h:
  1 / (1 - e^(-k*tau)) = 1 / (1 - e^(-0.693147)) = 1 / (1 - 0.5) = 2.00

Steady-state trough = single-dose trough x 2.00
```

With a 20-hour half-life in renal impairment, k = 0.034657 /h and the accumulation factor for an
8-hour interval becomes 1/(1 − e^(−0.27726)) = 1/(1 − 0.7579) = 4.13. The trough rises roughly
fourfold rather than twofold, which is the clinically important consequence. Both calculations assume
first-order elimination; if elimination becomes saturable, neither applies.

### Answer to Exercise 5

```text
Patient 1: fold = 10^(4.8 - 2.1) = 10^2.70 = 501.2
           absolute drop = 10^4.8 - 10^2.1 = 63,096 - 126 = 62,970 copies/mL

Patient 2: fold = 10^(2.9 - 0.2) = 10^2.70 = 501.2
           absolute drop = 10^2.9 - 10^0.2 = 794 - 1.58 = 793 copies/mL
```

Both patients have the same proportional decline (a factor of 501) and the same log difference
(2.70 log10 units). The absolute declines differ by a factor of about 79 because the log scale
converts multiplication into addition: equal differences on a log axis are equal ratios on the raw
axis. A report that gives only "a 2.7 log drop" is complete about the ratio and silent about the
absolute burden, which is why both should be reported.

### Answer to Exercise 6

| Series | First differences | Consecutive ratios | Form | Evidence needed |
|---|---|---|---|---|
| A | 10, 11, 12, 13 | 1.10, 1.10, 1.10, 1.10 | Exponential, 10% per period | Longer series; mechanism for compounding |
| B | 20, 20, 20, 20 | 1.20, 1.17, 1.14, 1.13 | Linear, slope 20 | Check beyond the observed range; a ceiling |
| C | 30, 50, 70, 90 | 4.0, 2.25, 1.78, 1.56 | Quadratic (second differences 20) or power law | Log-log slope; values across another order of magnitude |
| D | 0.25, 0.13, 0.06, 0.03 | 1.50, 1.17, 1.07, 1.03 | Saturating, asymptote near 1.0 | The ceiling value and the approach rate; more points near saturation |

For A, five periods ahead is 100 × 1.10^9 = 235.8 if the exponential holds; the failure mode is
saturation, so the forecast should be presented as a range with a stated ceiling. For B, five periods
ahead is 280 by the linear form; the failure mode is that growth may compound, so the linear forecast
is a lower bound. For C, the quadratic and power-law forms diverge sharply outside the range and the
data cannot choose between them. For D, extrapolation is bounded by the asymptote and the only
defensible statement is the estimated ceiling and its uncertainty.

### Answer to Exercise 7

```text
CL = 500 mg / 62.5 mg*h/L = 8.00 L/h

Relative uncertainty in dose = 5 / 500 = 0.010 = 1.0%
Relative uncertainty in AUC  = 3.1 / 62.5 = 0.0496 = 5.0%

Combined relative uncertainty = sqrt(0.010^2 + 0.0496^2)
                              = sqrt(0.0001 + 0.00246)
                              = sqrt(0.00256)
                              = 0.0506 = 5.1%

Standard uncertainty = 8.00 x 0.0506 = 0.405 L/h -> 0.41 L/h
95% interval (k = 2) = 8.00 +/- 0.81 = 7.19 to 8.81 L/h
```

The linear approximation is adequate: the relative uncertainty is about 5%, well below the 20% guide,
and the quotient is smooth over the uncertainty range. A Monte Carlo check with 10,000 draws would
be expected to agree within a few hundredths of a L/h. If the AUC uncertainty were 30%, the linear
form would understate the upper tail and a Monte Carlo interval would be preferred.

### Answer to Exercise 8 (rubric)

There is no single correct audit. A strong submission meets these criteria:

| Criterion | Weight | What earns credit |
|---|---:|---|
| Reconstruction | 25% | Shows the implied baseline and the minutes saved, with the arithmetic visible |
| Missing inputs | 20% | Names baseline length of stay, sample size, case mix, and the cost denominator |
| Alternative explanations | 25% | At least three plausible alternatives, each with a mechanism |
| Data request | 15% | Specific, obtainable variables, not "more data" |
| Rewritten hypothesis | 15% | Population, comparator, outcome, window, and effect size all specified |

A weak submission asserts that the claim is false without reconstructing it, or accepts the $2.1M
figure without asking what volume and cost assumptions produce it.

## 14. Interview questions

The questions below are the ones a hiring manager or a study section uses to test whether a candidate
can be trusted with a number. For each, the strong-answer column names the content that must appear;
the weak-answer column names the tell.

### Question 1 — "A vendor says their product reduced admissions 30%. What is your first question?"

**Strong answer.** "Thirty percent relative to what baseline, measured over what window, against what
comparator, and with how many events? I would ask for the raw counts in both arms, because 30% on a
2% baseline is 0.6 percentage points and on a 20% baseline is 6 points. I would also ask whether the
comparison is concurrent or before-after."

**Weak answer.** "I would ask for the p-value." The p-value does not recover the baseline, and a
significant relative effect on a tiny baseline can be operationally irrelevant.

### Question 2 — "When is a 20% reduction not worth funding?"

**Strong answer.** "When the absolute reduction is small, the population is small, or the cost per
event avoided exceeds the value of the event. A 20% reduction from 1% to 0.8% avoids 2 events per
1,000 patients. I would compute the NNT, multiply by the per-patient cost, and compare with the
avoided event cost. I would also check whether the effect was measured in a population like ours."

**Weak answer.** "When the confidence interval includes zero." That is a statistical criterion, not a
value criterion; a precisely estimated but tiny effect can still fail a budget test.

### Question 3 — "Why can you not average an ordinal pain score?"

**Strong answer.** "An ordinal scale orders categories but does not guarantee equal spacing. The
difference between 2 and 3 may not equal the difference between 8 and 9. A mean assumes equal
intervals, so it produces a number that does not correspond to a measurable quantity. Medians,
proportions above a threshold, and rank-based tests are defensible."

**Weak answer.** "Because the data are not normal." Normality is a distributional property; the
problem with averaging an ordinal scale is the scale itself, and it persists even if the ranks happen
to be symmetrically distributed.

### Question 4 — "How do you decide between a linear and an exponential model?"

**Strong answer.** "I plot the series, compute first differences and consecutive ratios, and see which
is more nearly constant. I also ask whether there is a mechanism for compounding. With fewer than
about five points I do not claim a form; I report a range. If the process has a ceiling, I fit a
saturating form instead."

**Weak answer.** "I try both and pick the one with the higher R-squared." R-squared on a short series
will favour the more flexible form and says nothing about out-of-range behaviour.

### Question 5 — "What does a half-life of 12 hours let you conclude?"

**Strong answer.** "That under first-order elimination the concentration falls by half every 12 hours,
so about 6.25% remains after 24 hours and about 97% of steady state is reached after five half-lives,
which is 60 hours. It does not let me conclude anything about a saturable drug, and it does not give
the dosing interval without knowing the therapeutic window."

**Weak answer.** "That the drug is gone in 24 hours." Two half-lives leave 25%, not zero, and the
clinical relevance of 25% depends on the drug.

### Question 6 — "A result is reported as 0.57 log lower. What is missing?"

**Strong answer.** "The base and the baseline. If it is log10, the fold change is 10^0.57 = 3.7. If
it is natural log, the fold change is e^0.57 = 1.77. The two differ by a factor of 2.3, so the base is
not a detail. I also need the absolute level to know whether the patient improved clinically."

**Weak answer.** "Nothing; 0.57 is the effect size." Without the base the number cannot be converted
to a ratio, and without the baseline it cannot be interpreted.

### Question 7 — "Why is the number needed to treat more useful to a budget holder than the relative risk reduction?"

**Strong answer.** "The NNT is the reciprocal of the absolute risk reduction, so it translates directly
into patients treated per event avoided. Multiplying by cost per patient gives the cost per event
avoided. The relative risk reduction is baseline-dependent and can make a small absolute effect sound
large."

**Weak answer.** "Because it is more intuitive." Intuition is not the issue; the issue is that the NNT
carries the absolute scale needed for a budget calculation.

### Question 8 — "When does first-order uncertainty propagation fail?"

**Strong answer.** "When the relative uncertainty is large, roughly above 20%, or when the function is
strongly curved over the uncertainty range, or when inputs are correlated. In those cases I use Monte
Carlo propagation, or I propagate the covariance matrix explicitly. A ratio with a denominator near
zero is a specific failure."

**Weak answer.** "It never fails; it is the standard formula." The formula is a first-order Taylor
approximation; it has a domain of validity that the analyst is responsible for checking.

### Question 9 — "How would you check a spreadsheet that computes cost per case avoided?"

**Strong answer.** "Pick one row and recompute it by hand from the raw counts, then compare. Check
that the denominator uses the expected count rather than a rounded count. Check that the ratio is not
computed on a near-zero denominator. Check the units on the cost column and whether the cost is per
patient or per case. Then check that the row formulas are consistent, because a copied column often
carries one wrong reference."

**Weak answer.** "I would check that the totals add up." Totals can add correctly while every ratio in
the column is wrong.

### Question 10 — "What is the difference between a rate and a proportion, and why does it matter for a dashboard?"

**Strong answer.** "A proportion has the population in the denominator and is bounded by 1. A rate has
person-time in the denominator. If patients have different follow-up, a proportion understates
exposure and makes periods incomparable. On a dashboard, a rate must carry its time unit, and a
proportion must carry its denominator definition, or the trend is uninterpretable."

**Weak answer.** "They are both percentages." A rate per patient-year is not a percentage of patients,
and treating it as one produces a denominator error that grows with follow-up length.

### Question 11 — "You are given four data points and asked for a five-year forecast. What do you do?"

**Strong answer.** "I state that four points cannot identify a functional form, fit at least two
plausible forms, and present the forecast as a range bounded by them. I ask for a mechanistic reason
to prefer one form and for any known capacity constraint. I deliver the range with the assumptions
listed, not a single number."

**Weak answer.** "I fit a regression and report the point estimate with a confidence interval." The
confidence interval around a misspecified form is precisely wrong: it is narrow and centred on the
wrong trajectory.

### Question 12 — "What would make you reject a published effect size outright?"

**Strong answer.** "An undefined denominator, a baseline that changed between periods because the
assay or the coding changed, a relative effect presented without counts, or an extrapolation beyond
the observed range with no stated form. Any one of these makes the number unusable for a decision,
regardless of the p-value."

**Weak answer.** "A small sample size." Small samples widen intervals; they do not by themselves make
an estimate unusable, and a large sample does not repair a denominator error.

## 15. Key terms

- **Absolute change** — after minus before, in the unit of the quantity.
- **Absolute risk reduction (ARR)** — difference in event risk between two groups, in percentage points.
- **Accumulation factor** — 1/(1 − e^(−k·τ)), the steady-state multiple over a single dose.
- **Affine function** — a straight line with a non-zero intercept, y = m·x + b.
- **Asymptote** — a value a function approaches without reaching.
- **Area under the curve (AUC)** — the integral of concentration over time; units are concentration × time.
- **Baseline** — the value against which a change is measured; the denominator of a relative change.
- **CAGR** — compound annual growth rate, (end/start)^(1/years) − 1.
- **Compounding** — growth applied to the accumulated value rather than the original.
- **Confidence interval** — a range produced by a procedure with a stated long-run coverage.
- **Decadic logarithm** — log base 10.
- **Denominator** — the population or exposure base of a rate or proportion.
- **Dimensional homogeneity** — both sides of an equation carry the same dimensions.
- **Doubling time** — ln(2)/growth rate for a continuously compounding quantity.
- **Elasticity** — proportional change in output per proportional change in input.
- **Emax model** — E = E_max·C/(EC50 + C), a saturating response.
- **Exponential function** — y = a·e^(k·x); constant relative change per unit x.
- **Fold change** — the ratio of two values; 10^d for a log10 difference d.
- **Half-life** — ln(2)/k for first-order elimination.
- **Hill coefficient** — the exponent controlling the steepness of a sigmoid dose-response curve.
- **Interval scale** — equal intervals, arbitrary zero; ratios are meaningless.
- **Log-log plot** — both axes logarithmic; a straight line indicates a power law.
- **Monotone transform** — a transform that preserves order but not necessarily intervals.
- **Natural logarithm** — log base e.
- **Nominal scale** — unordered categories.
- **Number needed to treat (NNT)** — 1/ARR, rounded up.
- **Odds** — p/(1 − p).
- **Odds ratio** — ratio of odds; further from 1 than the risk ratio when the outcome is common.
- **Ordinal scale** — ordered categories with unequal spacing.
- **Percentage point** — the arithmetic difference between two percentages.
- **Person-time** — the sum of time at risk across individuals; the denominator of a rate.
- **Power law** — y = a·x^b; linear on a log-log plot.
- **Proportion** — a ratio whose numerator is a subset of its denominator; bounded by 1.
- **Rate** — events per unit person-time.
- **Rate constant (k)** — fractional change per unit time; unit 1/time.
- **Ratio scale** — equal intervals with a true zero; ratios are meaningful.
- **Relative change** — (after − before)/before; dimensionless and baseline-dependent.
- **Relative risk reduction (RRR)** — ARR divided by the control risk.
- **Saturation** — diminishing response as input rises.
- **Semi-log plot** — one axis logarithmic; a straight line indicates exponential behaviour.
- **Significant figure** — a digit that carries measurement resolution.
- **Steady state** — the condition in which input equals elimination; approached after 4–5 half-lives.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Standards and metrology

- BIPM, *Le Système international d'unités / The International System of Units (SI Brochure)*, 9th
  edition, 2019, revised version 4.01 (2026). https://www.bipm.org/en/publications/si-brochure
  — defining constants and the seven base units; DOI 10.59161/AUEZ1291.
- BIPM, *SI Brochure 9th edition, concise summary*.
  https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-concise-EN.pdf
- ISO 80000-2:2019, *Quantities and units — Part 2: Mathematics*. https://www.iso.org/standards.html
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement (GUM)*. https://www.bipm.org/en/committees/jc/jcgm/publications
- NIST, *Guide for the Use of the International System of Units (SI)*, NIST Special Publication 811.
  https://www.nist.gov/pml/special-publication-811
- NIST, *SI Units and Metric Prefixes*, NIST Special Publication 330.
  https://www.nist.gov/pml/special-publication-330
- ISO 5725-1:1994, *Accuracy (trueness and precision) of measurement methods and results*.
  https://www.iso.org/standards.html
- Regenstrief Institute, *Unified Code for Units of Measure (UCUM)*. https://ucum.org/

### Mathematics and statistics

- IUPAC Gold Book, entry "pH" (P04524). https://goldbook.iupac.org/terms/view/P04524
  — pH defined as −lg a(H+), activity rather than concentration.
- Buck, R. P. et al., "Measurement of pH. Definition, standards, and procedures (IUPAC
  Recommendations 2002)", *Pure and Applied Chemistry* 74(11), 2169–2200.
  https://doi.org/10.1351/pac200274112169
- Reeve, R., Turner, P., "Pharmacodynamic models: parameterizing the Hill equation,
  Michaelis-Menten, the logistic curve, and relationships among these models", *Journal of
  Biopharmaceutical Statistics*, 2013. https://doi.org/10.1080/10543406.2012.756496
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

### Reporting and regulatory

- ICH, *E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*, 2019.
  https://www.ich.org/page/efficacy-guidelines
- CONSORT Group, *CONSORT 2010 Statement*.
  https://www.consort-statement.org/consort-2010
- Centers for Medicare & Medicaid Services, *Hospital Readmissions Reduction Program (HRRP)*.
  https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
- National Institute for Health and Care Excellence, *NICE health technology evaluations: the manual*.
  https://www.nice.org.uk/process/pmg36
- Institute for Clinical and Economic Review, *Value Assessment Framework*.
  https://icer.org/our-approach/methods-process/value-assessment-framework/

### Clinical conversion practice

- Sacks, D. B. et al., "Guidelines and Recommendations for Laboratory Analysis in the Diagnosis and
  Management of Diabetes Mellitus", *Clinical Chemistry* 69(8), 808–868, 2023.
  https://doi.org/10.1093/clinchem/hvad080 — states the mg/dL to mmol/L conversion as ×0.0555 or
  divide by 18.
- Medtronic, *CareLink Report Reference Guide* — states conversion to mg/dL by multiplying by
  18.0182. https://carelink.minimed.eu/help/en/CareLink%20Report%20Reference%20Guide.pdf
- Nightscout Foundation, issue #4928, "Alignment of Glucose Conversion Factor".
  https://github.com/nightscout/AndroidAPS/issues/4928 — documents the shift to 18.01559 from the
  IUPAC molar mass of D-glucose, 180.156 g/mol.

> **Note:** Where a URL is a landing page rather than a fixed document, the publisher may reorganise
> it. The citation is to the resource as accessed on the date above.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Next topic: [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md)
- Units and conversions: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Dimensional checking: [`04_dimensional-analysis.md`](04_dimensional-analysis.md)
- Pharmacokinetic applications: [`06_pharmacokinetics.md`](06_pharmacokinetics.md)
- Epidemiological rates and ratios: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- The assumptions ledger: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — see `01A_unit_conversion.ipynb` for the conversion lab and
  `06_calibration_decision_curve.ipynb` for the calibration and decision-curve lab.
