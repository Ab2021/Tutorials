> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 08 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01, 01A.02, 02.08

# Measurement Error, Accuracy, Precision, and Bias

---

## 1. Why this topic exists

A cycle-count report says inventory accuracy is 98.4%. A planner says the system shows 400 units
of an item that is not on the shelf. Both statements are correct, and they are about different
quantities. The report measured the share of counted locations whose recorded quantity fell
within a tolerance. The planner measured whether a pick could be completed. Neither is
"accuracy" in the sense the word carries in a laboratory, and the distance between them is where
inventory decisions go wrong.

Four words are used as if they were one:

| Word | What it describes | What it is not |
|---|---|---|
| Accuracy | Closeness of a measured value to the true value | A percentage you can put on a dashboard |
| Trueness | Closeness of the mean of many replicates to a reference | The same as precision |
| Precision | Closeness of repeated measurements to each other | Evidence of correctness |
| Resolution | The smallest change the instrument can distinguish | A statement about error |

A measurement can be precise and wrong. Two counters can agree with each other and both miss the
same damaged pallet. A measurement can be true on average and useless for a single decision:
errors cancel across a thousand locations and any one location is off by a case. A measurement
can have fine resolution and poor accuracy: a scale reading to one gram, mounted on a vibrating
dock, is precise about a value that is wrong.

The vocabulary determines four operational things: **comparability** (a conformance rate at ±5%
on an A-weighted sample is not comparable with ±1% on a random sample); **decision support** (a
tolerance finer than the uncertainty produces pass/fail labels that are partly random);
**model validity** (features and labels are measurements, so error in a feature biases a
regression); and **budget** (measurement is an expense whose return is the reduction in decision
loss). Without that last estimate, measurement investment is either neglected or gold-plated.

**The 98.4% that could not see a 6% hole.** A distribution centre counts 150 locations a month,
all A items, with a tolerance of ±5% of recorded quantity. The reported hit rate is 98.4%.
Quantity-weighted, the same population is 93.8% accurate, because the two largest errors sit in
the two locations that hold most of the stock. The number was computed correctly and answered a
question nobody was asking.

**The promotion that looked weak.** Point-of-sale (POS) scans miss items during high-traffic
promotions: queue overflow, items scanned under a neighbouring barcode, substitutions at the
till. Recorded sales understate true sales precisely when demand is highest. A promotion
evaluation that uses recorded sales as its outcome learns that promotions are less effective
than they are, and the next budget cycle cuts the wrong programme.

**The availability coefficient that was 36% too small.** A model regresses weekly sales on
recorded on-hand inventory. The recorded on-hand is measured with error whose variance is about
half the variance of true stock. The estimated slope is attenuated by the reliability ratio, 0.64
in this case. The model says a hundred extra units on the shelf produce eighteen extra units of
weekly sales; the corrected figure is twenty-eight. The investment case is understated by more
than a third, and every additional year of data makes the wrong estimate more precise.

This topic sits underneath three others. Topic 01A.01 fixes the unit, which determines what is
being counted. Topic 01A.02 fixes the container, which determines the resolution: a count in
cases cannot resolve a unit. Topic 02.08 supplies the data-quality dimensions and the idea of a
reference truth. This topic supplies the measurement theory that makes those dimensions
testable: what error is, how it decomposes, how it propagates, and when a measurement is fit to
support a decision.

> **Note:** The central difficulty in inventory measurement is that the reference is another
> measurement. There is no kilogram artefact for "units on hand". The physical count is itself a
> measurement with resolution, repeatability, and bias. Every claim of inventory accuracy is
> therefore a claim about agreement between two measurements, and can only be as strong as the
> argument that one of them is better.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** accuracy, trueness, precision, and resolution, and **state** which one a given
   number actually measures.
2. **Decompose** a measurement error into systematic and random components, and **explain** why
   the decomposition determines whether averaging helps.
3. **Build** an uncertainty budget for a warehouse or transactional measurement, classifying
   components as Type A or Type B, combining them, and reporting an expanded uncertainty with
   its coverage factor.
4. **Compute** the resolution contribution to uncertainty and **decide** whether a stated
   tolerance is coarser than the measurement can resolve.
5. **Design** a cycle-count sample that estimates a population accuracy with a stated interval,
   and **explain** why an exception-triggered sample estimates something else.
6. **Quantify** attenuation bias from a reliability ratio, **correct** an observed regression
   slope, and **state** the condition under which the correction is legitimate.
7. **Diagnose** whether an error in a response variable or a regressor is more damaging for a
   stated model, and **predict** the direction of the resulting bias.
8. **Estimate** the cost of measuring against the cost of not measuring, and **recommend** a
   stopping point with the assumption that drives it.

## 3. The core mental model

Hold one picture: every number in a supply chain dataset is the output of a measurement process,
and that process has a truth it cannot see, a reference it compares against, an instrument with
finite resolution, and a record that someone wrote at some time.

```text
   Each panel is one measurement process.  +  marks the true value.
   o  marks one reading.

        A  ACCURATE AND PRECISE          B  PRECISE, BIASED

             |    o o                        |    o o
             |    o o                        |    o o
          ---+---                         ---+---
             |                               |
        trueness high, precision high   trueness low, precision high

        C  ACCURATE, IMPRECISE           D  BIASED AND IMPRECISE

             |  o                            |  o
             |      o                       |      o
          ---+---                         ---+---
             |   o                           |   o
             |        o                     |        o
        trueness high, precision low    trueness low, precision low
```

Panel A is the goal and is rare. Panel B is the most dangerous in practice, because repeated
agreement builds confidence: two counters, the same count sheet, the same missed pallet. Panel C
is tolerable for portfolio decisions, where errors cancel, and intolerable for item-level
replenishment. Panel D is what a bad process looks like when nobody has checked it against a
reference.

Resolution is a separate axis, and it sets a floor under precision:

```text
   recorded
   quantity
      ^
   36 |                    +---------+
      |                +---+
   24 |            +---+
      |        +---+
   12 |    +---+
      |+---+
    0 +---------+---------+---------+---------> true quantity
          6        18        30        42

   step d = 12 eaches.  Quantisation error is uniform on [-6, +6].
   Standard deviation from resolution alone = d / sqrt(12) = 3.46 eaches.
```

A count in cases of twelve cannot report a number between 12 and 24. That does not make it
wrong; it makes it coarse. Whether the coarseness matters depends entirely on the tolerance of
the decision, which is why resolution and tolerance must always be stated together.

Error enters a decision through the measurement stack:

```text
   TRUE STATE  (what is physically there, at an instant)
       |  definition: quantity, unit, boundary, timestamp
       v
   REFERENCE   <---- traceability: calibration or documented procedure
       |  comparison
       v
   INSTRUMENT  resolution -> repeatability -> reproducibility -> drift
       |  reading
       v
   RECORD      who writes it, when, append or overwrite
       |  feature or label
       v
   MODEL       sum: bias adds linearly, random error as sqrt(n)
               ratio: relative errors combine in quadrature
               regression: error in X attenuates the slope
       v
   DECISION
```

Six properties follow, and they explain most of what comes later.

**Property 1 — Accuracy belongs to the result, not the instrument.** The International Vocabulary
of Metrology (VIM) defines measurement accuracy as the closeness of agreement between a measured
value and a true value of the measurand, and states that it is not a quantity and has no numerical
value (JCGM 200:2008, 2.13). An instrument has a bias, a repeatability, a resolution, and a drift;
it does not have an "accuracy" that transfers to every use. A warehouse that says its scanners are
99% accurate has said nothing about any particular count.

**Property 2 — Trueness and precision are separable, and both are needed.** Trueness is about the
mean of replicates; precision is about their spread (VIM 2.14, 2.15). A process can be precise and
biased, as when two counters use the same wrong unit, or true on average and imprecise, as when
errors cancel across a thousand locations but any single location is unreliable. The first damages
replenishment; the second damages item-level promises.

**Property 3 — Resolution is a floor, not a measure of quality.** Quantisation error is bounded by
half a step and has standard deviation `d / sqrt(12)` under a uniform assumption. If the tolerance
is finer than the resolution, the measurement cannot support the decision, however careful the
operator.

**Property 4 — Bias does not average out; random error does.** Averaging `n` independent replicates
reduces the random component by `sqrt(n)` and leaves the systematic component unchanged. Ten
thousand records with a 2% undercount produce a very precise 2% undercount. "We have a lot of data"
is not an answer to a measurement problem.

**Property 5 — Random error in a regressor does not average out either.** Classical error in a
regressor biases the slope toward zero by the reliability ratio, and more data makes the attenuated
estimate more precise, not less biased. Section 5.14 derives this.

**Property 6 — Error in the response behaves differently from error in the regressor.** Classical
error in the response leaves the slope unbiased and widens the interval; classical error in a
regressor biases the slope. The asymmetry is a diagnostic: a significant effect smaller than theory
or experience predicts is a candidate for attenuation, not evidence of a weak effect.

Three propagation regimes cover almost every supply chain calculation. In a **sum**, independent
random errors grow as `sqrt(n)` so relative error falls, while bias adds linearly so relative bias
persists. In a **product or ratio**, relative errors combine in quadrature and a shared bias
partially cancels. In a **regression**, random error in a regressor biases the slope toward zero,
and the direction of a systematic error depends on its correlation with the regressor.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Measurement | Process of experimentally obtaining quantity values reasonably attributable to a quantity | Used to mean any data point | Procedure identifier |
| Measurand | The quantity intended to be measured, including its boundary and time | Conflated with the database column | Quantity kind plus unit |
| True value | Value consistent with the definition of the measurand | Treated as knowable in practice | Same unit as the measurand |
| Reference value | Value accepted as a basis of comparison for a quantity | Assumed exact | Same unit as the measurand |
| Accuracy | Closeness of agreement between a measured value and a true value | Reported as a percentage | None; it is not a quantity (VIM 2.13) |
| Trueness | Closeness of the mean of an infinite number of replicates to a reference value | Used as a synonym for accuracy | None; expressed operationally as bias in units |
| Precision | Closeness of agreement among replicate measurements under stated conditions | Used as a synonym for accuracy | Standard deviation in units |
| Resolution | Smallest change in a quantity that produces a perceptible change in indication | Confused with precision | Same unit as the measurand |
| Repeatability | Precision under the same operator, instrument, location, and short interval | Called reproducibility | Standard deviation in units |
| Reproducibility | Precision under changed operator, instrument, location, or time | Used for repeats by one person | Standard deviation in units |
| Repeatability limit (r) | Difference between two single results that is exceeded with 5% probability | Confused with tolerance | Units; conventionally `2.77 s_r` |
| Measurement error | Measured value minus true value | Called "noise" | Units |
| Systematic error | Error component that remains constant or varies predictably | Assumed to cancel across records | Units |
| Random error | Error component that varies unpredictably across replicates | Assumed to be the only error present | Standard deviation in units |
| Bias | Estimate of systematic error; mean error over replicates | Used for any error, including random | Units or percent of reference |
| Drift | Bias that changes with time or use | Treated as random | Units per period |
| Measurement uncertainty | Non-negative parameter characterising dispersion of values attributed to the measurand | Called "error", or reported without a coverage factor | Same unit as the measurand |
| Standard uncertainty | Uncertainty expressed as a standard deviation | Called a confidence level | Units |
| Type A evaluation | Uncertainty evaluated by statistical analysis of observations | Treated as the only valid kind | Standard deviation |
| Type B evaluation | Uncertainty evaluated from specifications, calibration, or judgement | Ignored because it is not statistical | Standard deviation |
| Combined standard uncertainty | Quadrature sum of sensitivity-weighted components | Computed as a simple average | Units |
| Expanded uncertainty | Combined standard uncertainty multiplied by a coverage factor | Reported without `k` | Units |
| Coverage factor (k) | Multiplier on the combined uncertainty; `k = 2` gives roughly 95% coverage | Omitted from reports | Dimensionless |
| Tolerance | Permitted limits of error for a stated purpose | Confused with uncertainty | Units or percent |
| Conformance | Whether a result lies within a stated tolerance | Reported as accuracy | Pass/fail, or proportion |
| Metrological traceability | Property of a result whereby it relates to a reference through an unbroken documented chain of calibrations, each contributing uncertainty | Claimed without documentation | Chain identifier |
| Calibration | Operation that establishes the relation between an indication and reference values | Confused with adjustment | Calibration record |
| Reference standard | Measurement standard used as a reference for other measurements | Assumed to be exact | Artefact or procedure identifier |
| Hit rate | Share of counted locations whose error falls within tolerance | Reported as quantity accuracy | Proportion |
| Quantity accuracy | One minus the ratio of summed absolute quantity error to physical quantity | Usually not reported | Proportion |
| Inventory record accuracy | Agreement between recorded and physical quantity at a stated time | Treated as agreement with truth | Proportion within tolerance |
| Cycle count | Periodic count of a defined subset of locations | Treated as a full audit | Count event |
| Blind count | Count performed without the recorded quantity visible to the counter | Rarely verified | Procedure identifier |
| Attenuation bias | Downward bias in a regression slope caused by random error in a regressor | Called "noise" or "weak effect" | Slope units |
| Reliability ratio | Ratio of true variance to observed variance in a regressor; equals one minus the error variance share | Assumed to be one | Dimensionless, 0 to 1 |
| Classical measurement error | Additive, mean-zero error independent of the true value | Assumed for all error | Model assumption |
| Berkson error | Error independent of the observed value rather than the true value | Treated as classical | Model assumption |
| Misclassification | Error in a categorical variable, described by a confusion matrix | Analysed as if it were additive | Confusion matrix |

Three distinctions deserve expansion.

**Accuracy versus conformance.** In metrology, accuracy is not a number. "99% inventory accuracy"
is a conformance rate: the share of measurements within a stated tolerance of a stated reference.
It is a statistic about a tolerance and a sample, not a property of the data, and it can be
improved by widening the tolerance or changing the sample without any change in the physical
world.

**Tolerance versus uncertainty.** Tolerance is what the decision permits; uncertainty is what the
measurement cannot resolve. When they are comparable, a binary pass/fail contains a large
arbitrary component. A measurement is adequate when its expanded uncertainty is comfortably
smaller than the tolerance, often a factor of three.

**Reference value versus true value.** The true value is defined by the measurand and is not
knowable. The reference value is what you compare against, and it carries its own uncertainty. In
inventory, the reference is a count procedure, which is why a blind dual count is worth more than
a single count by the person who owns the location.

## 5. The mechanics, step by step

This section builds a measurement from the definition forward, propagates error into a decision,
audits the process, then prices it. The order matters: a measurement designed before the
tolerance is known is designed for the wrong question.

### 5.1 Three questions before any measurement

1. **What is the measurand?** Quantity kind, unit, boundary, and timestamp. "Inventory on hand at
   06:00 on 30 September, in eaches, owned by the distribution centre, excluding consignment" is
   a measurand. "Stock" is not.
2. **Against what reference, with what uncertainty?** A number without a named reference is an
   assertion.
3. **What tolerance does the decision require?** Accuracy is undefined without a purpose.

If any cannot be answered, stop. The number will otherwise be used as if it could.

### Phase A — Define the measurand and the reference

**Step 1. Fix the quantity, unit, boundary, and time.**

- The unit determines resolution: eaches, cases of twelve, layers, pallets.
- The boundary determines inclusion: owned, consigned, in transit, in quality hold, at a
  third-party site.
- The timestamp determines what is compared. A snapshot at 23:59 and a count at 14:00 differ by a
  day of transactions. That is a definitional mismatch, not error, and it is often larger than
  the error being studied.

**Step 2. Choose the reference and state its uncertainty.**

| Domain | Usual reference | What it can establish | What it cannot |
|---|---|---|---|
| Mass | Calibrated weight, traceable chain | Bias of a scale at a stated load | Accuracy on a vibrating dock |
| Volume | Calibrated measure or flow meter | Bias over a flow range | Accuracy for a different product viscosity |
| Count | Blind full count; dual independent count | Agreement between count and record | Truth, if both share a unit or boundary error |
| Inbound receipt | Supplier advance ship notice plus dock scan | Agreement with what was declared and scanned | Physical quantity, if the scan is a licence-plate scan of a pallet |
| Sales | POS scan plus returns reconciliation | Recorded sales | True demand, if items go unscanned or are substituted |
| Location | Satellite positioning or fixed read point | Position within a stated radius | Identity of the item at that position |
| Time | Coordinated Universal Time (UTC) synchronised clock | Ordering and duration | Whether the event was recorded when it occurred |

The table is the practical form of traceability. In mass and volume, the reference is an artefact
with a calibration history. In count and sales, the reference is a procedure. A procedure can be
traceable if it is written, blind, and independently repeated. It cannot be traceable if the
person counting is the person whose work is being counted.

**Step 3. State the tolerance and its source.**

- Derive it from the decision: the smallest quantity error that would change an order, a
  replenishment trigger, or a financial posting.
- Write the derivation. "±2% because the reorder quantity moves in pallets of 60 on a base of
  3,000" is a derivation. "±2% because that is the policy" is a convention.
- If the tolerance and the measurement uncertainty are within a factor of three, the decision
  rule needs a guard band, discussed in step 15.

### Phase B — Characterise the instrument or process

**Step 4. Quantify resolution.**

Quantisation error is bounded by half a step. Under a uniform distribution over the step, the
standard deviation is `d / sqrt(12)`.

```text
d = 24 eaches (one case)
u_resolution = 24 / sqrt(12) = 6.93 eaches

If tolerance is +/-2% on a location holding 300 eaches, tolerance = +/-6 eaches.
Resolution alone (6.93) exceeds the tolerance (6). The count cannot support the decision.
```

**Step 5. Measure repeatability.**

- Count the same unchanged location at least ten times, by the same person, with the same
  procedure, within a short interval.
- Compute the standard deviation `s_r`. The repeatability limit is `r = 2.77 s_r`, the difference
  between two single results that is exceeded with 5% probability, where 2.77 is `1.96 sqrt(2)`.
- Repeatability is a floor: no process can be more reproducible than it is repeatable.

**Step 6. Measure reproducibility.**

- Repeat the experiment across operators, shifts, or sites.
- Compute `s_R`. The ratio `s_R / s_r` locates the problem. A ratio near one means the
  instrument or procedure dominates. A ratio well above one means the operator or the local
  practice dominates, and buying a better instrument will not help.
- In warehouse counting, the ratio is usually between two and four, because counting is a
  judgement-laden procedure: what counts as damaged, what counts as in the location, what to do
  with a partially opened case.

**Step 7. Test drift and linearity.**

- Drift is bias changing with time. Check the same reference at the start and end of a shift, and
  across weeks.
- Linearity is bias changing with level. Check the reference at low, middle, and high values. A
  scale that reads correctly at 10 kg and 0.5% high at 1,000 kg has a level-dependent bias, and
  correcting it with a single offset makes one end worse.
- A bias that is stable and known is a correction. A bias that moves is an uncertainty. Treating
  the second as the first produces a correction that is wrong in a way nobody monitors.

**Step 8. Compare against an independent reference.**

- Use a reference whose error is known to be smaller than the process being tested. A calibrated
  weight against a scale; a full blind count against a cycle count; a weighed pallet against a
  case count.
- Record the difference, not just the pass/fail. The signed differences are the bias evidence; the
  absolute differences are the precision evidence.
- If no independent reference exists, say so. The result is then a consistency check between two
  measurements, and should be labelled as such.

### Phase C — Build the uncertainty budget

**Step 9. List the components and combine them.**

For a result `y = f(x_1, ..., x_n)`, the combined standard uncertainty for independent inputs is:

```text
u_c^2(y) = sum over i of (df/dx_i)^2 * u^2(x_i)
```

In practice, list each component, estimate its standard uncertainty, and add the squares. The
sensitivity coefficient is often one, as when every component is already expressed in units of
the measurand.

| Component | Type | Standard uncertainty (eaches) | Variance |
|---|---|---:|---:|
| Count resolution, 12-each step | B | 3.46 | 12.0 |
| Repeat count, same operator | A | 9.00 | 81.0 |
| Receipt timing, uniform 0–3 h at 40 units/h | B | 34.64 | 1,200.0 |
| Unrecorded damage, mean 12, SD 6 | B | 6.00 | 36.0 |
| Combined | | 36.46 | 1,329.0 |

```text
u_c = sqrt(12.0 + 81.0 + 1,200.0 + 36.0) = sqrt(1,329.0) = 36.46 eaches
U  = 2 * 36.46 = 72.9 eaches at approximately 95% coverage
```

**Step 10. Separate bias from uncertainty.**

The damage component has a mean of 12 units. That is not a dispersion; it is a bias. If the
direction and magnitude are known, correct the record and remove the mean from the budget. If
they are not known, keep the component as uncertainty and say that the result is not
bias-corrected.

This is the most commonly botched step in practice. Teams build a large uncertainty and conclude
that the measurement is imprecise, when in fact it is biased and could be corrected. Precision
and trueness require different remedies: precision needs a better procedure, bias needs a
correction or a calibration.

**Step 11. Report the result as a value with an interval and a coverage factor.**

"1,200 ± 73 eaches, k = 2, approximately 95% coverage" is a complete statement. "1,200 eaches,
accurate" is not. "1,200 eaches, ±6%" is incomplete without the coverage factor.

### Phase D — Propagate to the decision

**Step 12. Sums and averages.**

- Independent random errors add in quadrature: the standard deviation of a sum of `n` independent
  measurements with equal error SD `s` is `s sqrt(n)`, so relative error falls as `1/sqrt(n)`.
- Correlated systematic errors add linearly, so relative bias does not fall with `n`.
- Consequence: a portfolio total can be accurate to 0.3% while every line is accurate to 5%,
  provided the errors are independent. If they share a cause, the total inherits it.

**Step 13. Products and ratios.**

- Relative standard uncertainties combine in quadrature: `(u_z/z)^2 = sum (u_xi/xi)^2`.
- For a ratio, a shared bias partially cancels; independent biases add.
- Consequence: fill rate, cost per unit, and percentage error all inherit denominator error. A
  fill rate against a forecast denominator carries the forecast error inside it.

**Step 14. Regression and attenuation.**

Suppose the true relationship is `y = a + b x* + e`, and the regressor is measured with classical
error `x = x* + u`, where `u` has mean zero, variance `sigma_u^2`, and is independent of `x*` and
`e`. The ordinary least squares slope converges to:

```text
plim b_hat = b * ( sigma_x*^2 / (sigma_x*^2 + sigma_u^2) ) = b * lambda
lambda = reliability ratio, between 0 and 1
```

The observed slope is the true slope multiplied by the share of observed variance that is real.
The bias is toward zero, never away, and it does not shrink with sample size. Three consequences
follow.

- **A null result may be an attenuation result.** If the reliability ratio is 0.5, a true effect
  of 0.4 is estimated at 0.2, and a study powered for 0.4 is underpowered for what it will
  actually see.
- **A significant but small result may be correct but understated.** Correct it, and widen the
  interval to reflect the uncertainty in `lambda`.
- **Correcting requires an external estimate of `lambda`.** Estimating `lambda` from the same
  data and then correcting with it is circular. Valid sources are repeated measurements on a
  subsample, a validation study against a better instrument, or a known error variance from the
  measurement process.

**Step 15. Compare uncertainty with tolerance and set the decision rule.**

- If `U` is much smaller than the tolerance, a simple pass/fail is adequate.
- If `U` is comparable to the tolerance, a simple pass/fail misclassifies a predictable share of
  items near the limits. The remedy is a guard band: treat results within `U` of the limit as
  indeterminate, or shift the acceptance limit inward by `U`.
- If `U` exceeds the tolerance, the measurement cannot support the decision. Either improve the
  measurement or restate the decision at a coarser grain.

The guard band is not statistical decoration. Without it, a process that is perfectly in control
will still produce apparent failures at a rate set by the ratio of uncertainty to tolerance, and
the organisation will respond by adjusting good stock.

### Phase E — Audit the measurement process

**Step 16. Design the cycle-count sample.**

- **Frame.** Every countable location, including the ones nobody wants to count. A frame that
  excludes bulk storage describes the accessible population, not the warehouse.
- **Selection.** Random or stratified random. Exception-triggered counts are a diagnostic, not an
  estimate: they over-sample suspected locations and describe the exception population.
- **Blindness.** The counter must not see the recorded quantity, which anchors the count most
  strongly where the counter is accountable for the variance.
- **Tolerance.** Fixed in advance, applied to the signed difference, and reported with `U` beside it.
- **Two grains.** Report the hit rate and the quantity accuracy; they answer different questions.
- **Frequency.** Allocate counts by value times error rate. A high-value item with a stable process
  needs fewer counts than a medium-value item with a broken one.

**Step 17. Measure the measurement.**

- Keep a control sample counted twice, independently, every period. Its disagreement rate is the
  reproducibility of the count process and the denominator of every accuracy claim.
- Reconcile the sampled estimate to a periodic full count; the gap is the sampling error made visible.
- Chart signed differences. A trend is drift; a step is a process change; a stable band is bias and
  noise together.

### Phase F — Price the measurement

**Step 18. Compare the cost of measuring with the expected loss from not measuring.**

- The cost of measurement is direct: labour, equipment, downtime, and opportunity cost.
- The expected loss is the probability of an error large enough to change the decision,
  multiplied by the cost of the changed decision, summed over outcomes.
- The value of a better measurement is the reduction in that expected loss, not the improvement
  in an accuracy metric.
- Three stopping rules are defensible: stop when `U` is small against the tolerance; stop when
  marginal cost exceeds marginal reduction in expected loss; stop when no decision can change,
  in which case the value is zero regardless of quality.

The third rule saves the most money. A monthly full count of 4,800 locations that changes no
replenishment parameter is a ritual, not a control.

## 6. Worked example

Ravenhill Distribution is a regional distribution centre (DC) holding 4,800 stock-keeping
unit-location combinations (SKU-locations) and 120,000 units. It runs a cycle-count programme and
reports inventory accuracy monthly. All figures are illustrative and internally consistent.

### 6.1 Setting

| Property | Value |
|---|---|
| SKU-locations | 4,800 |
| Physical units on hand; value at cost | 120,000; €960,000 |
| Unit cost; unit margin | €8; €3 |
| Annual throughput; carrying cost rate | 1,400,000 units; 22% |
| Cycle counts per month; tolerance | 150; ±2% of recorded quantity |
| Reported accuracy; management target | 97.3%; 99.0% |

### 6.2 Two grains, two answers

The count programme reports the share of counted locations within tolerance. Of 150 counts in the
month, 146 pass.

```text
hit rate = 146 / 150 = 0.9733 = 97.3%
```

A full stocktake in the same month finds 120,000 units physically and 112,000 in the system. The
difference is concentrated: two high-volume locations hold 40,000 units each in reality and are
recorded 10% low, an error of 4,000 units each. The remaining 148 locations hold 40,000 units and
reconcile exactly.

```text
quantity accuracy = 1 - (4,000 + 4,000) / 120,000
                  = 1 - 8,000 / 120,000 = 0.9333 = 93.3%
```

The gap is 4.0 percentage points, and both numbers are correct. The hit rate counts locations
equally; the quantity accuracy counts units. Two locations out of 150 carry all of the quantity
error because they hold two thirds of the stock. Management sees 97.3% and asks for one more
point. The planner experiences 93.3% on the two items that matter most.

> **Note:** The two-grain gap is not a reporting trick. It is the difference between a
> location-weighted and a unit-weighted population, and it grows with the concentration of stock.
> Any accuracy target stated without its grain is a target that can be met by changing the
> denominator.

### 6.3 The sample is not the population

The 150 monthly counts are allocated by ABC class: A items are counted twelve times a year, B
items four times, C items once. With 20% of locations in A, 30% in B, and 50% in C, the share of
counts is:

```text
A: 0.20 x 12 = 2.4   -> 2.4 / 4.1 = 58.5% of counts
B: 0.30 x  4 = 1.2   -> 1.2 / 4.1 = 29.3%
C: 0.50 x  1 = 0.5   -> 0.5 / 4.1 = 12.2%
total effort = 4.1
```

Suppose the true location-level pass rates are 99% for A, 96% for B, and 91% for C. The expected
reported rate is the count-weighted average:

```text
reported = 0.585(0.99) + 0.293(0.96) + 0.122(0.91)
         = 0.579 + 0.281 + 0.111 = 0.971 = 97.1%
```

The population rate, weighting each location once, is:

```text
population = 0.20(0.99) + 0.30(0.96) + 0.50(0.91)
           = 0.198 + 0.288 + 0.455 = 0.941 = 94.1%
```

The programme is not lying. It is reporting the accuracy of the locations it counts, which are
the locations it counts most because they are the most valuable. That is a defensible
operational choice. It is not an estimate of warehouse accuracy, and it cannot be compared with
a site that counts randomly.

> **Note:** ABC cycle counting optimises for detecting and correcting errors where they cost
> most. It does not optimise for estimating the error rate. If the same programme is used for
> both purposes, one of them is wrong. Separate the control sample from the assurance sample.

### 6.4 The tolerance is finer than the measurement

Take one A-item location holding 1,200 units. The uncertainty budget from section 5.9 applies:
resolution 3.46 eaches, repeatability 9.00, receipt timing 34.64, unrecorded damage 6.00. The
combined standard uncertainty is 36.46 eaches and the expanded uncertainty is 72.9 eaches at
`k = 2`.

```text
tolerance half-width = 0.02 x 1,200 = 24 eaches
expanded uncertainty = 72.9 eaches
ratio U / tolerance = 72.9 / 24 = 3.0
```

The measurement cannot resolve the tolerance by a factor of three. The consequence is concrete.
Treat the combined error as approximately normal with mean zero and standard deviation 36.46.
The probability that a location with no true error at all falls outside the ±24 tolerance is:

```text
P(|e| > 24) = 2 x (1 - Phi(24 / 36.46))
            = 2 x (1 - Phi(0.658))
            = 2 x (1 - 0.745) = 0.510
```

Half of perfectly correct locations fail the tolerance on any given count. The organisation
observes a stream of failures, adjusts stock that was right, and widens the tolerance informally
until the failures stop. Both responses make the measurement worse.

The normal approximation overstates the tails: with the actual bounded components the rate is
lower, though still large. The conclusion is unchanged.

### 6.5 Bias, and what it does to replenishment

The system understates the two large locations by 10% each and the population by 6.67%. The bias
is systematic, not random: receipts are scanned at putaway rather than at the dock, so units sit
physically in the location for up to three hours before the record agrees with them, and damage
is written off monthly rather than at the event.

A systematic understatement is not a precision problem. Averaging more counts does not remove it,
and neither does a longer history. It propagates into replenishment as a late trigger:

- The system believes it holds 112,000 units when it holds 120,000.
- Reorder points computed on the recorded position fire when the true position is already 8,000
  units lower than intended.
- The affected items turn over four times a year, so 8,000 units of understatement exposes about
  32,000 units of demand a year.
- At a 25% lost-sale share and €3 margin, that is `32,000 x 0.25 x 3 = €24,000` a year.
- Expedites caused by the late triggers add an estimated €18,240 a year at four per month and
  €380 each.

The total exposure is about €42,240 a year, before counting the planner time spent reconciling.

### 6.6 Attenuation in the availability model

The commercial team asks whether more stock on the shelf sells more product. An analyst regresses
weekly store sales on recorded on-hand inventory, with store and week fixed effects, using 40
stores over 104 weeks. The estimated coefficient is 0.18 units of weekly sales per unit of
on-hand, with a 95% interval of 0.10 to 0.26.

Recorded on-hand is measured with error. A reliability study counts 60 locations twice, blindly,
and decomposes the observed variance across locations:

```text
observed variance  = 44,944
error variance     = 12,544
true variance      = 44,944 - 12,544 = 32,400
reliability ratio  = 32,400 / 44,944 = 0.721
```

Under classical measurement error, the observed slope is the true slope times the reliability
ratio:

```text
observed 0.18 = true slope x 0.721
true slope    = 0.18 / 0.721 = 0.250
```

The true effect is 39% larger than the estimate. A hundred extra units on hand produce 25 extra
units of weekly sales, not 18. The investment case for holding more stock was understated by more
than a quarter, and it was understated in the direction that makes a real effect look marginal.

Three qualifications belong beside the correction.

1. The reliability ratio is estimated from 60 observations and carries its own uncertainty. The
   corrected interval is wider than the naive `[0.10/0.721, 0.26/0.721] = [0.139, 0.361]`.
2. The classical error model requires the error to be independent of true stock and of sales. If
   high-selling stores are also the stores with the most counting errors, the direction of the
   bias is no longer determined.
3. If the error is Berkson rather than classical, meaning the recorded value is a target that the
   true value varies around, attenuation does not follow the same formula.

> **Note:** The correction is legitimate because the reliability ratio came from an independent
> repeated-measures study. Estimating the error variance from the regression residuals and then
> correcting with it is circular: the residuals contain the very attenuation being estimated.

### 6.7 The decision the analysis supports

Two interventions compete for the same budget.

| Intervention | Annual cost | Effect | Annual benefit |
|---|---:|---|---:|
| Dock scanning and event-level damage write-off | €20,521 | Removes about 70% of the 6.67% bias | €29,568 |
| Monthly full count of all 4,800 locations | €85,000 | Removes sampling error, not bias | Near zero on bias |
| Do nothing | €0 | Bias persists | −€42,240 |

Dock scanning amortises €46,000 of capital over five years at 8%, or €11,521 a year, and adds
€9,000 of operating cost. Break-even requires removing 48.6% of the bias, so the case survives a
substantial shortfall but not a claim that scanning fixes everything.

The full count is the instructive row. It would improve the accuracy metric, because sampling error
disappears, and leave the bias untouched, because the bias is created at receipt and damage, not at
counting. It is the most expensive option and the least useful.

## 7. Code, query, or calculation

The following Python fragment reproduces the four calculations that drive the example. It is
deliberately short: the arithmetic is the point, and every quantity is stated in the units used
above.

```python
import math
from statistics import NormalDist

# 1. Uncertainty budget for one 1,200-unit location, in eaches
u = [12 / math.sqrt(12), 9.0, 120 / math.sqrt(12), 6.0]
u_c = math.sqrt(sum(x * x for x in u))
U = 2 * u_c
print(f"u_c={u_c:.1f}  U(k=2)={U:.1f}  U/1200={U/1200:.2%}")

# 2. False-fail rate at a +/-2% tolerance when the true error is zero
tol = 0.02 * 1200
p_fail = 2 * (1 - NormalDist().cdf(tol / u_c))
print(f"tolerance=+/-{tol:.0f}  false-fail={p_fail:.1%}")

# 3. Cycle-count interval with finite population correction
n, N, p = 150, 4800, 0.9733
se = math.sqrt(p * (1 - p) / n * (N - n) / (N - 1))
print(f"95% CI: {p - 1.96*se:.3f} to {p + 1.96*se:.3f}")

# 4. Attenuation correction from an external reliability ratio
beta_hat, lam = 0.18, 0.721
print(f"corrected beta={beta_hat/lam:.3f}")
```

Expected output: `u_c=36.5`, `U(k=2)=72.9`, `U/1200=6.08%`, `tolerance=+/-24`,
`false-fail=51.0%`, `95% CI: 0.948 to 0.999`, `corrected beta=0.250`.

The budget assumes independent components; if receipt timing and damage share a cause, the
combined uncertainty exceeds the quadrature sum. The false-fail rate also depends on the
distribution, not only the standard deviation, so confirm it by simulation when the decision is
sensitive to it.

## 8. Data, standards, and regulatory context

Three documents define the vocabulary, and they do not agree with each other in every detail.
Knowing which one a claim is using tells you what the claim can mean.

### 8.1 The International Vocabulary of Metrology

The VIM (JCGM 200:2008, 3rd edition) is the reference for the words in this topic. It is freely
available, which is unusual for a metrology standard and worth knowing.

| Clause | Term | Definition in brief | Consequence for supply chain work |
|---|---|---|---|
| 2.13 | Measurement accuracy | Closeness of agreement between a measured value and a true value; explicitly not a quantity, no numerical value | "98% accuracy" is a conformance rate, not accuracy |
| 2.14 | Measurement trueness | Closeness of the mean of infinite replicates to a reference value | Operationally reported as bias |
| 2.15 | Measurement precision | Closeness of agreement among replicate measurements under stated conditions | Reported as a standard deviation, with conditions named |
| 2.16 | Measurement error | Measured value minus true value | Signed, in units; not "noise" |
| 2.17 | Systematic error | Error component that remains constant or varies predictably | Does not average out |
| 2.18 | Bias | Estimate of systematic error | A number you can correct |
| 2.19 | Random error | Error component that varies unpredictably | A number you can average down |
| 2.26 | Measurement uncertainty | Non-negative parameter characterising dispersion of values attributed to the measurand | Wider than error; includes what you do not know |
| 2.30 | Standard uncertainty | Uncertainty expressed as a standard deviation | The input to a budget |
| 2.35 | Expanded uncertainty | Combined standard uncertainty times a coverage factor | Always report the factor |
| 2.38 | Coverage factor | Multiplier, commonly 2 or 3 | `k = 2` is roughly 95%, not exactly |

The vocabulary's most useful discipline is negative: it forbids saying that a value is accurate.
What you store is a conformance rate, a bias estimate, a standard deviation, or an uncertainty.

### 8.2 ISO 5725 and the accuracy of a measurement method

ISO 5725-1:2023 defines accuracy as a combination of trueness and precision, where trueness is
expressed as bias and precision is expressed as standard deviation under stipulated conditions.
The series separates two extremes of precision:

| Term | Conditions | What it captures |
|---|---|---|
| Repeatability | Same method, operator, instrument, location, short interval | The irreducible noise of the procedure |
| Intermediate precision | Same method, changed operator, day, or equipment within one site | The local practice component |
| Reproducibility | Same method, different sites, operators, and equipment | The between-site component |

Two derived quantities are useful in a warehouse. The repeatability limit `r = 2.77 s_r` is the
difference between two single results exceeded with 5% probability, where 2.77 is `1.96 sqrt(2)`;
the reproducibility limit `R` uses `s_R` in the same formula. If two counters disagree by more
than `r`, repeatability alone does not explain it.

> **Note:** ISO 5725 and the VIM use "accuracy" differently. ISO 5725 treats it as a combined
> property of trueness and precision; the VIM says it is not a quantity at all. Both are current.
> In a report, state which usage you mean rather than assuming the reader shares it.

### 8.3 The GUM and the uncertainty budget

The Guide to the Expression of Uncertainty in Measurement (GUM, JCGM 100:2008) supplies the
method used in section 5.9. Its most misunderstood point is that Type A and Type B are methods of
evaluation, not kinds of error. Type A is evaluation by statistical analysis of observations;
Type B is evaluation by any other means, including specifications, calibration certificates, and
judgement. Neither label maps onto random versus systematic.

The combination rule for independent inputs is the quadrature sum of sensitivity-weighted
variances, and `U = k u_c`, with `k = 2` giving roughly 95% coverage for a normal distribution. The
GUM also warns against double counting: a component already in a calibration certificate should not
be added again from a datasheet.

### 8.4 Traceability, and what replaces it in counting

Metrological traceability relates a result to a reference through an unbroken, documented chain
of calibrations, each contributing its own uncertainty. For mass and volume, the chain ends at a
national metrology institute. For counts and sales, there is no artefact at the end of the chain.
What replaces it is a procedure:

- a written definition of the measurand and the boundary;
- a blind count, so the counter cannot anchor on the record;
- an independent second count on a subsample;
- a periodic full count as a reference;
- and a record of the disagreement rate between them.

That procedure is weaker than a calibration chain, and it is stronger than an assertion. The
honest statement is "agreement with a blind dual count, whose own disagreement rate is 2.1%",
not "accurate to 99%".

### 8.5 Event data and identifiers

EPCIS 2.0, the GS1 standard for event data, records what happened to which object, where, when,
and why. An observation event at a read point is a measurement of presence, and it has the same
structure as any other: a read rate below 100% is a precision figure, and systematic non-reads
caused by metal, liquid, or tag placement are a bias at item level. A portal that reads 98% of
tags is not 98% accurate; it is 98% sensitive, with an unknown bias in which 2% it misses.

Identifiers matter for the same reason. A pack-size change creates a new Global Trade Item Number
(GTIN), so recording a new pack under an old GTIN is a misclassification, not a rounding error.
Substituting one item's barcode for another at the till is the same failure in the sales data,
and it biases item-level demand in both directions while leaving the total nearly correct.

### 8.6 What regulation supplies, and what it does not

Legal metrology regimes set tolerances for prepackages and require specified average-quantity
rules; traceability regimes require event records with a location and a timestamp. Neither
supplies the tolerance for your replenishment decision, and neither certifies that your inventory
record is right. As in Topic 02.08, a compliance check is a floor: a dataset can satisfy every
retention requirement and still be unfit for the decision at hand.

## 9. Industry practice and case evidence

### 9.1 What the evidence shows

| Source | Finding | Grade | What it establishes |
|---|---|---|---|
| DeHoratius and Raman (2008), *Management Science* | 370,000 records across 37 stores of one retailer; 65% did not match physical stock; estimated lost revenue above 1% of North American retail sales and 3% of gross profit | A | Record inaccuracy is large, common, and economically material |
| Hardgrave, Aloysius and Goyal (2013), *Production and Operations Management* | Radio-frequency identification (RFID) reduced inventory record inaccuracy by about 26% on average, ranging from no significant effect to 81% by category | A | The benefit of better measurement is real but highly contingent |
| Buckel and Thiesse (2013) | About 58,000 items; RFID cycle counts erred on 1.22% of items against 1.17% for physical counts; human error caused 38% of RFID errors, technical 19%, upstream tagging 29% | B | A better instrument does not remove process error |
| Bertolini et al. (2015), *Journal of RF Technologies* | RFID counts detected out-of-stocks invisible to both barcode counts and the system; barcode double reads can overstate stock | A | Barcode measurement has both false negatives and false positives |
| GS1 US, EPC/RFID Retail Supply Chain Data Exchange Study | Eight brand owners and five retailers; 69% of inbound orders contained errors; EPC/RFID reconciliation raised order accuracy above 99.9%; item-level RFID raised store accuracy from about 63% to 95% | C for the study, D for the uplift figures | The gap between declared and actual shipment is large; vendor-consortium numbers need the protocol |
| Oldham et al. (2001), inventory integrity benchmarking | 18 warehouses; one reached 99.9% record accuracy; best-in-class sites did not use tolerances at all | C | The 99.9% benchmark is protocol-dependent |
| Oliver Wight survey, 410 manufacturers | 20% of cycle-counting firms reached 98% or better; more than 60% were in the 90–97% band | C | Typical performance, if the sample is representative |
| APICS cycle-count guidance | Tolerances of roughly 0.5%, 1%, and 5% for A, B, and C items | C | A convention, not a measurement standard |
| Glock, Syntetos and Rekik (2025) | Inventory record inaccuracy affects 50–70% of stock-keeping units at any time; recommends expressing tolerance against physical stock and reporting a composite of binary accuracy and a quantity error measure | A− | The metric design problem is still open in the literature |

### 9.2 The benchmark is a protocol, not a property

The most quoted number in this field is "world-class inventory accuracy is 99.9%". It is not a
property of good warehouses. It is the result of a protocol: count every location, compare
exactly, and count a mismatch as a failure. A programme with a ±5% tolerance and an A-weighted
sample will report 99% on a population that a full exact count would place near 94%. Both sites
can say "99%" and mean different things, and neither is lying.

Three questions make a benchmark usable: what tolerance, what sampling frame, and what reference.
Omit any of the three and the figure describes a programme, not the inventory.

### 9.3 The instrument is not the process

The RFID evidence is the cleanest demonstration in the supply chain literature that measurement
quality is a process property. Buckel and Thiesse found RFID and physical counts with nearly
identical error rates, and attributed most RFID errors to human and upstream tagging causes
rather than to the technology. Hardgrave and colleagues found effects ranging from zero to 81%
by category, which is what one expects when the instrument removes one component of a
multi-component error.

The practical reading: before buying a better instrument, decompose the error. If the variance is
dominated by operator judgement, a scanner will not help. If it is dominated by timing, a faster
scanner will. Section 5.6 gives the ratio that tells you which.

Inventory errors are also usually signed. Receipt timing understates stock during putaway;
unrecorded damage overstates it; substitution at the till redistributes demand between items
without changing the total. Record the sign and its stability: a stable sign is a correction
waiting to be made, an unstable sign an uncertainty to be bounded.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Accuracy claimed with no reference | "99% accurate" with no named comparison | The number is a conformance rate against an unnamed standard | Unfalsifiable claim; no path to improvement | Name the reference and its uncertainty |
| Precision reported as accuracy | Repeated counts agree; physical stock still differs | Agreement among replicates says nothing about the mean | Confident wrong decisions | Report bias and SD separately |
| Resolution finer than tolerance assumed | Counts in cases; tolerance in eaches | Quantisation error alone exceeds the decision tolerance | Pass/fail labels partly random | Compare `d/sqrt(12)` with the tolerance before designing |
| Tolerance widened until failures stop | Tolerance changed after a bad month | The metric is being managed, not the process | Accuracy appears to improve; stock does not | Freeze the tolerance; log every change |
| Cycle-count sample not random | Counts concentrated in A items and problem locations | The sample estimates a different population | Reported accuracy cannot be compared or extrapolated | Separate control and assurance samples |
| Hit rate reported as quantity accuracy | High hit rate; large value discrepancies | Locations weighted equally; units concentrated | The number management sees is not the number planners experience | Report both grains, always |
| Timing mismatch counted as error | Errors spike at month end | Record and count refer to different instants | False defects; real defects hidden | Align cut-offs; state the timestamp |
| Bias averaged as if random | More data; same discrepancy | Systematic error does not shrink with `n` | Precision without accuracy | Correct the mean; do not widen the interval |
| Scan undercount correlated with demand | Promotional lift looks weak; totals reconcile | Missed scans rise when volume rises | Demand model learns promotions are ineffective | Audit scans during peaks; model the error |
| Attenuation read as a weak effect | Significant but small coefficient; low `R^2` | Random error in a regressor biases the slope toward zero | Underinvestment in the true driver | Estimate the reliability ratio; correct |
| Reliability ratio estimated from the same data | Correction changes little | The residuals contain the attenuation being estimated | Circular correction; false confidence | Use repeated measures or a validation subsample |
| Level-dependent bias corrected with a constant | One end of the range improves; the other worsens | Linearity untested | Error moved, not removed | Test at three levels before correcting |

Three deserve a note.

**Tolerance widened until failures stop** is the most common corruption, because it is done with
good intentions: a team facing a 40% false-fail rate widens the tolerance, the metric improves, and
nothing in the warehouse changes. Improve the measurement or restate the decision, and record why.

**Scan undercount correlated with demand** is the most damaging for forecasting, because the error
is largest where the model is most sensitive. A constant 2% undercount is an offset; an undercount
that rises with traffic changes the shape of the demand curve and cannot be offset.

**Attenuation read as a weak effect** is the most expensive for analytics, because it produces a
plausible finding that argues against a real relationship. A coefficient that is significant,
directionally correct, and smaller than experience predicts is a measurement hypothesis to test.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Error is additive and classical | Attenuation and averaging results depend on it | Multiplicative error, thresholds, or error correlated with the true value | Plot error against the true value on a validation sample |
| Error is independent of the true value | The reliability ratio is a simple variance share | Errors larger for large or fast-moving items | Stratify the error estimate by size and velocity |
| The reference is better than the instrument | Otherwise the comparison measures the reference | Two procedures with similar error; a count used as truth | Estimate the reference's own repeatability |
| Errors are independent across records | Sums and portfolio metrics improve with `n` | A shared cause: one cut-off, one operator, one unit error | Correlate errors within shift or operator |
| Bias is stable over the measurement window | A correction estimated once can be applied | Process change, new system, seasonal tempo | Chart signed differences over time |
| The sample is random and the frame complete | An interval requires a probability sample | Exception-triggered counts; excluded bulk storage | Compare sample composition to the frame |
| The tolerance reflects the decision | Accuracy is undefined without a purpose | Tolerance copied from policy or a template | Ask what action changes at the tolerance |
| Resolution is much finer than the tolerance | Otherwise pass/fail is partly arbitrary | Case-level counts against unit-level tolerances | Compute `d/sqrt(12)` and compare |
| Record and count refer to the same instant | Otherwise the difference is definitional | Snapshot at 23:59, count at 14:00 | Align cut-offs; quantify the transaction gap |
| Units are consistent through the chain | A unit error swamps every other component | Each versus case versus layer confusion | Reconcile one record by hand, unit by unit |

Two limits of the framing are worth stating plainly. **The framework describes error, not truth.**
Every method here bounds how wrong a number can be; none recovers the true value. When the
decision needs truth rather than a bound — a recall, a regulatory filing, a controlled substance —
the answer is a better reference, not a better model. **Measurement economics is local.** The cost
of a count, the margin on a lost sale, and the value of a service level differ by site and
contract. The worked example's €42,240 is a teaching figure; what transfers is the structure of
the calculation, not the number.

## 12. Exercises

### Exercise 1 — Rewrite five claims as measurements (critique, 45 minutes)

Take these five statements: "our inventory accuracy is 99.2%"; "the forecast is 95% accurate";
"the scale is accurate to 0.1 g"; "RFID gives a 99% read rate, so our stock data is accurate";
"supplier lead-time data is 98% accurate".

**Deliverable:** for each, a rewritten statement naming the measurand, the reference, the
tolerance, the sample or population, and the interval or coverage factor.
**Acceptance criteria:** at least one rewrite separates resolution from accuracy; at least one
separates a read rate from a record accuracy; at least one states that the original claim is not
falsifiable as written.

### Exercise 2 — Build an uncertainty budget (calculation, 50 minutes)

A location holds 1,200 units. Components: resolution 24 units; repeatability `s_r = 14` units;
receipt timing uniform over 0–4 hours at 30 units per hour; unrecorded damage with mean 9 units
and standard deviation 4.

**Deliverable:** `u_c`, `U` at `k = 2`, the variance share of the dominant component, and what to
do with the damage component.
**Acceptance criteria:** every conversion shown; the dominant component identified by variance
share; correcting a mean distinguished from widening an interval.

### Exercise 3 — Design the count sample and report the interval (calculation, 45 minutes)

A site has 4,800 locations. A random sample of 200 finds 188 within tolerance.

**Deliverable:** the point estimate, the 95% interval with a finite-population correction, and the
sample size needed for ±2 percentage points.
**Acceptance criteria:** the correction applied and its effect stated; the sample size derived
rather than asserted; one paragraph on why an exception-triggered sample invalidates the interval.

### Exercise 4 — Hit rate versus quantity accuracy (calculation, 40 minutes)

Of 150 counted locations, two hold 40,000 units each and are recorded 10% low; the other 148 hold
40,000 units in total and reconcile exactly. Unit cost is €8 and the carrying rate is 22%.

**Deliverable:** the hit rate, the quantity accuracy, the value of the quantity error, and the
annual carrying-cost effect.
**Acceptance criteria:** both grains computed and the gap explained in one sentence; the value
denominator stated; the answer says which number a planner experiences.

### Exercise 5 — Correct an attenuated slope (calculation, 60 minutes)

A regression of weekly sales on recorded on-hand gives a slope of 0.18 with a 95% interval of
0.10 to 0.26. An independent repeated-measures study estimates the reliability ratio at 0.64.

**Deliverable:** the corrected slope, the corrected interval, the attenuation percentage, and two
conditions under which the correction would be invalid.
**Acceptance criteria:** the correction divides by the reliability ratio, not multiplies; the
interval is corrected consistently; both conditions are specific to the error model.

### Exercise 6 — Design a reference for scan undercount (design, 50 minutes)

Design a study to estimate the undercount in point-of-sale scans during promotional weeks.

**Deliverable:** a one-page protocol stating the reference, the sampling frame, the estimator, the
interval, and what the result cannot establish.
**Acceptance criteria:** the reference is independent of the scan data; the frame includes peak
days; the protocol acknowledges the audit's own measurement error; the blind spot is stated.

### Exercise 7 — Repeatability and reproducibility (calculation, 60 minutes)

Three operators each count the same unchanged pallet ten times. Within-operator standard
deviation is 8 units; the standard deviation between operator means is 22 units.

**Deliverable:** `s_R`, the ratio `s_R / s_r`, the repeatability limit `r`, and the reproducibility
limit `R`.
**Acceptance criteria:** the combination in quadrature; the ratio interpreted operationally; the
recommendation distinguishing a training fix from an instrument fix.

### Exercise 8 — Cost of measuring against cost of not measuring (memo, 45 minutes)

Using the Ravenhill figures, evaluate dock scanning: €46,000 of capital amortised over five years
at 8%, plus €9,000 a year operating cost, removing 70% of a €42,240 annual exposure. Then test the
decision at 40% removal.

**Deliverable:** annual cost, annual benefit, net benefit, break-even removal share, and the
sensitivity result.
**Acceptance criteria:** the annuity factor shown; the break-even solved rather than guessed; the
recommendation changes or holds explicitly at the lower effectiveness.

### Exercise 9 — Red-team a vendor claim (critique, 40 minutes)

A vendor states that its RFID system delivers 99.9% inventory accuracy.

**Deliverable:** six questions that would make the claim evaluable, and a one-paragraph verdict
on what the claim can and cannot mean.
**Acceptance criteria:** at least one question distinguishes read rate from record accuracy; at
least one addresses the population and sample; at least one addresses exclusions such as
misplaced or substituted items.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
resolution:  24 / sqrt(12) = 6.928   -> variance  48.0
repeatability: 14                    -> variance 196.0
timing:      30 x 4 = 120 range; 120 / sqrt(12) = 34.641 -> variance 1,200.0
damage:      standard deviation 4    -> variance  16.0

u_c = sqrt(48.0 + 196.0 + 1,200.0 + 16.0) = sqrt(1,460.0) = 38.21 units
U   = 2 x 38.21 = 76.4 units at k = 2
```

Timing contributes `1,200 / 1,460 = 82.2%` of the variance; the remedy is a process change, such
as scanning at the dock, not a better counter. The damage mean of 9 units is a bias to be
corrected from the scrap log; its standard deviation of 4 belongs in the budget, its mean does
not. Adding 9 to `u_c` confuses a bias with a dispersion.

**Exercise 3 arithmetic.**

```text
p = 188 / 200 = 0.940
se = sqrt( p(1-p)/n x (N-n)/(N-1) )
   = sqrt( 0.94 x 0.06 / 200 x 4,600 / 4,799 )
   = sqrt( 0.000282 x 0.9585 ) = 0.01644

95% interval = 0.940 +/- 1.96 x 0.01644 = 0.940 +/- 0.0322
             = 0.908 to 0.972, or 90.8% to 97.2%
```

For ±2 percentage points, ignoring the correction first:

```text
n0 = 1.96^2 x 0.94 x 0.06 / 0.02^2 = 3.8416 x 0.0564 / 0.0004 = 541.7 -> 542
n  = n0 / (1 + (n0 - 1)/N) = 542 / (1 + 541/4,800) = 487.1 -> 488
```

The correction saves 54 counts, about 10%. Report the interval rather than the point estimate:
near 1 the binomial variance is small in absolute terms but the logit scale is stretched.

**Exercise 4 arithmetic.**

```text
hit rate          = 148 / 150 = 0.9867 = 98.7%
quantity accuracy = 1 - 8,000 / 120,000 = 0.9333 = 93.3%
gap               = 5.3 percentage points
value of error    = 8,000 x EUR 8 = EUR 64,000
carrying effect   = 64,000 x 0.22 = EUR 14,080 per year
```

The planner experiences the quantity figure, because a pick either finds the units or does not.
The hit rate is the right number for deciding where to send the next counter.

**Exercise 5 arithmetic.**

```text
corrected slope = 0.18 / 0.64 = 0.281
corrected interval = [0.10/0.64, 0.26/0.64] = [0.156, 0.406]
attenuation = 1 - 0.64 = 36%
```

The observed estimate is 64% of the true effect. The correction is invalid if the error is
correlated with sales, as when high-volume stores also have the most counting errors, or if the
error is Berkson rather than classical. Dividing interval endpoints by an estimated `lambda`
understates the corrected width; a proper interval accounts for the uncertainty in `lambda` and
is wider than `[0.156, 0.406]`.

**Exercise 7 arithmetic.**

```text
s_R = sqrt(8^2 + 22^2) = sqrt(64 + 484) = sqrt(548) = 23.41 units
ratio = 23.41 / 8 = 2.93
r = 2.77 x 8 = 22.2 units
R = 2.77 x 23.41 = 64.8 units
```

The ratio of nearly three says the operator component dominates the within-operator noise. A
better scale would reduce `s_r` and barely move `s_R`; the intervention is standard work.

**Exercise 8 arithmetic.**

```text
annuity factor = (1 - 1.08^-5) / 0.08 = 3.9927
annual capital = 46,000 / 3.9927 = EUR 11,521
annual cost    = 11,521 + 9,000 = EUR 20,521

benefit at 70% = 0.70 x 42,240 = EUR 29,568
net benefit    = 29,568 - 20,521 = EUR 9,047 per year

break-even removal = 20,521 / 42,240 = 0.486 = 48.6%
at 40% removal: benefit = 16,896 ; net = 16,896 - 20,521 = -EUR 3,625
```

The case holds at the central assumption and fails at the lower one. The investment is sensitive
to a parameter nobody has measured: the share of the understatement that reaches a lost sale. The
next study should estimate that share, not re-estimate the bias.

**Exercise 1 rubric.** A strong answer rewrites each claim with five elements: measurand, reference,
tolerance, sample, and interval. It identifies "99.2% inventory accuracy" as a conformance rate needing
a tolerance and a sample; "95% accurate" as needing a metric, horizon, and denominator; "accurate to
0.1 g" as a resolution statement that says nothing about bias; a 99% read rate as sensitivity, not record
accuracy; and lead-time accuracy as needing its reference named. A weak answer adds "and the standard
deviation" without naming a reference.

**Exercise 6 rubric.** A strong design uses a physical audit as the reference: for a random sample of
store-days, count shelf quantity at opening, reconcile receipts and returns, and compare expected with
scanned sales. It stratifies by day type so peak days are included and reports an interval. It
acknowledges the audit's own error and states that the design estimates the gap between physical and
recorded flow, not true demand. A weak design compares POS data with the retailer's own system, which
shares the upstream errors.

**Exercise 9 rubric.** A strong answer asks: what is the tolerance; what is the reference; is 99.9% a tag
read rate, an order accuracy, or a record accuracy; over what population and sample; and what is excluded,
including misplaced items, substitutions, and present but unreadable tags. It concludes that the claim may
be true for a defined protocol and is not comparable with a tolerance-based cycle count. A weak answer
accepts the number or rejects RFID without engaging with the measurement definition.

## 14. Interview questions

**1. "What is the difference between accuracy and precision, and why does it matter?"**

A strong answer separates trueness (mean against a reference) from precision (spread of replicates),
notes that a precise biased process produces confident wrong decisions, and gives an example such as
two counters who agree and both miss the same pallet. A weak answer gives the target analogy and stops.

**2. "Your warehouse reports 99% inventory accuracy. What do you ask?"**

A strong answer asks for the tolerance, the reference, the sampling frame, the grain (location or
quantity), and the interval, and notes that 99% with a ±5% tolerance on A items is not comparable with
99% exact on a random sample. A weak answer accepts the number or asks only which system produced it.

**3. "Distinguish accuracy, trueness, and resolution."**

A strong answer gives the VIM structure: accuracy is closeness to truth and is not a quantity; trueness
is the closeness of the mean to a reference; resolution is the smallest distinguishable change and sets
a floor under precision. A weak answer treats all three as synonyms for "how good the number is".

**4. "A vendor claims 99.9% inventory accuracy with RFID. What are the red flags?"**

A strong answer separates tag read rate from record accuracy, asks for the reference and the population,
and asks what is excluded: misplaced items, substitutions, unreadable tags. It notes that a 99.9% figure
usually implies an exact-match protocol. A weak answer accepts the number or dismisses RFID entirely.

**5. "What happens to a regression when the regressor is measured with random error?"**

A strong answer states attenuation: the slope is multiplied by the reliability ratio, biased toward
zero, and the bias does not shrink with sample size, so a significant but small coefficient may be an
attenuated true effect. A weak answer says the estimate becomes noisy, which is the response-variable case.

**6. "What happens when the response is measured with random error?"**

A strong answer says the slope stays unbiased in a linear model and the standard errors widen, so power
falls, adding that nonlinear models and non-classical error break this result. A weak answer applies
attenuation to both sides of the equation.

**7. "How would you estimate a reliability ratio?"**

A strong answer uses repeated measurements on a subsample, a validation study against a better
instrument, or a known error variance from the measurement process, and never the residuals of the model
being corrected. A weak answer says to use `R^2` or to assume a value.

**8. "How would you design a cycle-count sample?"**

A strong answer specifies a complete frame, random or stratified random selection, blind counting,
a pre-registered tolerance, and reporting at both location and quantity grain. It separates the control
sample, which targets risk, from the assurance sample, which estimates the population rate. A weak answer
says to count the A items more often, which is a control policy rather than an estimator.

**9. "The tolerance is ±2% and the expanded uncertainty is ±6%. What do you do?"**

A strong answer says the measurement cannot support the decision, then chooses between improving the
process and restating the decision at a coarser tolerance, and explains the false-fail rate that follows
from leaving it unchanged. A weak answer widens the tolerance without recording why.

**10. "How do you decide how much measurement is enough?"**

A strong answer compares the marginal cost of precision with the marginal reduction in expected decision
loss, and stops when the measurement can no longer change a decision; `U` well below the tolerance is
usually sufficient. A weak answer says to measure as accurately as possible, which has no stopping rule.

## 15. Key terms

- **Accuracy** — closeness of a measured value to the true value; not a quantity and not a percentage.
- **Trueness** — closeness of the mean of replicates to a reference value.
- **Precision** — closeness of repeated measurements to each other under stated conditions.
- **Resolution** — smallest change in a quantity that produces a perceptible change in indication.
- **Repeatability** — precision under one operator, instrument, location, and short interval.
- **Reproducibility** — precision under changed operators, instruments, locations, or times.
- **Measurement error** — measured value minus true value.
- **Systematic error** — error component that is constant or varies predictably.
- **Random error** — error component that varies unpredictably across replicates.
- **Bias** — estimate of systematic error, usually the mean error over replicates.
- **Drift** — bias that changes with time or use.
- **Measurement uncertainty** — parameter characterising the dispersion of values attributed to a measurand.
- **Standard uncertainty** — uncertainty expressed as a standard deviation.
- **Type A and Type B** — evaluation by statistical analysis and by other means, respectively.
- **Expanded uncertainty** — combined uncertainty multiplied by a coverage factor.
- **Coverage factor** — multiplier `k`, commonly 2 for roughly 95% coverage.
- **Tolerance** — permitted limits of error for a stated purpose.
- **Conformance** — whether a result lies within tolerance; what "accuracy" usually means in practice.
- **Traceability** — relating a result to a reference through a documented chain of calibrations.
- **Hit rate** — share of counted locations within tolerance.
- **Quantity accuracy** — one minus the ratio of absolute quantity error to physical quantity.
- **Attenuation bias** — downward bias in a regression slope caused by random error in a regressor.
- **Reliability ratio** — true variance divided by observed variance in a regressor.
- **Classical measurement error** — additive, mean-zero error independent of the true value.
- **Berkson error** — error independent of the observed value rather than the true value.

## 16. Sources

**Primary standards and official guidance**

- JCGM 200:2008, *International Vocabulary of Metrology (VIM)*, 3rd edition, clauses 2.13–2.19, 2.26, 2.30, 2.35, 2.38. https://jcgm.bipm.org/vim/en/ ; DOI: 10.59161/jcgm200-2012 (Grade B)
- JCGM 100:2008, *Guide to the Expression of Uncertainty in Measurement (GUM)*. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf ; DOI: 10.59161/jcgm100-2008e (Grade B)
- ISO 5725-1:2023, *Accuracy (trueness and precision) of measurement methods and results — Part 1*. https://www.iso.org/standard/69418.html (Grade B)
- ISO 5725-2 to -6, the remaining parts: estimation, intermediate precision, trueness, alternatives, application. (Grade B)

**Peer-reviewed and technical**

- DeHoratius, N. and Raman, A. (2008). "Inventory Record Inaccuracy: An Empirical Analysis." *Management Science*, 54(4), 627–641. DOI: 10.1287/mnsc.1070.0789 (Grade A)
- Hardgrave, B. C., Aloysius, J. A., and Goyal, S. (2013). "RFID-Enabled Visibility and Retail Inventory Record Inaccuracy." *Production and Operations Management*. DOI: 10.1111/poms.12010 (Grade A)
- Hutcheon, J. A., Chiolero, A., and Hanley, J. A. (2010). "Random measurement error and regression dilution bias." *BMJ*, 340, c2289. https://www.bmj.com/content/340/bmj.c2289 (Grade A)
- Bertolini, M., Bottani, E., Romagnoli, G., and Vignali, G. (2015). "The impact of RFID technologies on inventory accuracy in apparel retailing." *Journal of RF Technologies*. DOI: 10.3233/rft-150069 (Grade A)
- Buckel, T. and Thiesse, F. (2013). "The Performance and Reliability of a RFID Cycle-Count." *International Conference on Mobile Business*. https://aisel.aisnet.org/icmb2013/ (Grade B)
- Glock, C. H., Syntetos, A. A., and Rekik, Y. (2025). "On the measurement of inventory record inaccuracies." Working paper, Cardiff University. (Grade A−)
- Pischke, J.-S. "Lecture Notes on Measurement Error." LSE, EC524. https://econ.lse.ac.uk/staff/spischke/ec524/merr.pdf (Grade A−)
- Frost, C. and Thompson, S. G. (2000). "Correcting for Regression Dilution Bias." *JRSS Series A*, 163(2), 173–189. (Grade A)

**Industry, practitioner, and vendor**

- Oldham, K. et al. (2001). *Benchmarking for Best Practices in Inventory Integrity*. University of Arkansas. https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/7/795/files/2016/08/ierc_2001c_oldham.pdf (Grade C)
- Rossetti, M. D., Collins, T., and Kurgund, R. "Inventory Cycle Counting — A Review." University of Arkansas. https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/7/795/files/2016/08/ierc_2001_a_kurgund.pdf (Grade C)
- GS1 US, *EPC/RFID Retail Supply Chain Data Exchange Study* (Project Zipper). https://documents.gs1us.org (Grade D)
- Oliver Wight survey of 410 manufacturers and APICS cycle-count tolerance guidance, as reported in the benchmarking literature above. (Grade C, secondary)
- SCDigest (2009), "Measuring Inventory Accuracy — No Clear Answer, Experts Say." https://www.scdigest.com/assets/On_Target/09-11-03-1.php (Grade E)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or official, C company case or industry study, D vendor marketing, E press or anecdote. Working papers and teaching notes are A−. Vendor-consortium figures are reported as stated and not independently verified; the protocol behind each figure matters more than the figure.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, a forward text reference pending the file: `07_hierarchy-design.md`
- Earlier topics: [`05_lead-time-definitions.md`](05_lead-time-definitions.md), [`06_currency-and-fx.md`](06_currency-and-fx.md)
- Foundations: [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md), [`02_container-loading.md`](02_container-loading.md)
- Data-quality dimensions assumed here: [`../02_supply_chain_data_landscape/08_data-quality-dimensions.md`](../02_supply_chain_data_landscape/08_data-quality-dimensions.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Next topics, not yet on disk: `09_forecast-error-metrics.md` (bias and dispersion in forecast evaluation), `10_intermittent-metrics.md`, `11_demand-lead-time-distributions.md`.
- Return to the part map: [`../_index.md`](../_index.md)


