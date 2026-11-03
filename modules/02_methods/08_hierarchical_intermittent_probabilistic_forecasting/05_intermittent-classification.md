# Intermittent Demand: Classification and Methods

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01A.10, 01A.11, 07.06, 08.01–08.04
> Version 2026.5

---

## 1. Why this topic exists

Every large planning system contains a router. It is rarely called that. It is a table of rules,
a configuration screen, or a stored procedure that reads a demand history, computes two numbers,
and assigns the series to a method. The router decides that a bearing gets single exponential
smoothing (SES) and a gearbox gets the Syntetos–Boylan approximation (SBA). Nobody in the
planning meeting has seen the rule, and everybody's service level depends on it.

The router exists because the alternative does not scale. A mid-sized industrial distributor
carries 40,000 item-location combinations; a hospital supply chain carries more; a grocery
retailer carries millions. In such a portfolio the majority of series are slow-moving. In the
illustrative portfolio used in section 6, seven of twelve items — 58% of the item count — sit in
the two sporadic cells, and they account for 4% of the units. No team can hand-pick a method for
each, and no team can afford to give them all the same method, because the same method is
expensive in different ways in different cells.

Three failures recur, and they are the reason this topic is separate from the two that precede it.

**Failure one: the classification is treated as a description of the item.** It is not. It is a
description of a *measurement* — a particular series, at a particular level of aggregation, in a
particular time bucket, over a particular window, with a particular treatment of zeros. Change any
of those four and the class can change. A gearbox is not lumpy in the way that steel is ferrous.
It is lumpy at one location, on a weekly bucket, over the last year, with stock-out weeks retained.
At the regional level on a monthly bucket it may be smooth, and both statements are correct.

**Failure two: the cut-offs are treated as constants of nature.** The values 1.32 and 0.49 that
appear in almost every implementation were derived from a comparison of two specific estimators on
one sample of automotive series. They have been re-derived since, and the re-derivation found
different numbers. That history is not a scandal; it is how a convention is made. But an analyst
who does not know it will defend a rounded constant against a business that needs a different one.

**Failure three: the class is computed once.** A new product passes through lumpy, intermittent,
erratic, and smooth in its first two years. An end-of-life item decays in a way the grid cannot
see at all, because the grid measures interval and dispersion and not direction. A label assigned
at launch and never refreshed will misroute an item for the rest of its life, and nobody will
notice until the service number moves.

This topic does five things. It restates the classification as a routing decision with a loss
function. It reviews what each cell implies about method families, including the places where the
literature disagrees. It prices the cost of a misroute with arithmetic. It shows why hierarchy and
lifecycle make static classification fail. And it sets out the operational machinery — versioned
rules, exception queues, override logs, drift monitors — that a router needs before it is allowed
to run unattended.

It does not re-derive the metrics. The average demand interval (ADI), the squared coefficient of
variation (CV²), the four cells, and the failure of percentage errors on zero-heavy series are
defined and motivated in [Module 01A topic 10](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md).
The mechanics of Croston's method, SBA, and the Teunter–Syntetos–Babai (TSB) method are developed
in [Module 07 topic 06](../07_time_series_forecasting/06_intermittent-demand.md). This file takes
those as given and works on the decision that sits between them: which method a given series
should receive, at what level, at what time, and who is accountable when the answer is wrong.

> **Note:** Classification is a model-selection device. It is not a taxonomy of demand, and the
> cell label is not an output that anyone should report to a customer. The output is the method,
> and the method is only as good as the backtest behind it. A router with no backtest is a
> preference.

## 2. Learning outcomes

By the end of this topic you can:

1. **Reproduce** the four-cell classification from raw demand, stating the bucket, the observation
   window, the zero-handling rule, and the dispersion convention, and **show** how each choice
   moves a borderline series between cells.
2. **Explain** where the cut-offs came from, why they were later re-derived, and **state** the
   conditions under which they are and are not transferable to a new portfolio.
3. **Route** each cell to a method family with a stated justification, and **name** at least one
   published disagreement about the routing for each cell.
4. **Quantify** the cost of a misroute in units, service level, and currency, using a stated demand
   model and a stated inventory policy.
5. **Demonstrate** with arithmetic how aggregation changes ADI and CV², and **predict** the
   direction of the class change when a series is rolled up or split down a hierarchy.
6. **Diagnose** class drift across a product lifecycle, and **specify** a reclassification cadence
   and a set of drift triggers.
7. **Design** a routing rule with version control, class-conditional evaluation, an exception
   queue, and an override log, and **defend** it against the objection that a single ensemble
   would be simpler.
8. **Critique** a vendor or internal routing table for unstated conventions, untested cut-offs, and
   missing governance.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three pictures. The first is the grid and the method families inside it. The second is the
pipeline that produces the grid entry, because most classification errors are made upstream of
the grid, not in it. The third is the hierarchy, because the same item occupies several grid
positions at once and only one of them is the one you are forecasting.

### 3.1 The grid, with methods assigned

```text
                       CV2 of nonzero demand size (dimensionless)
                       low: < 0.49            high: >= 0.49
                     +---------------------+---------------------+
   ADI < 1.32        |       SMOOTH        |       ERRATIC       |
   frequent          |  SES, ETS, ARIMA,   |  damped ETS, robust |
   (~3+ weeks in 4)  |  seasonal ETS,      |  ETS, forecast      |
                     |  regression         |  combinations       |
                     +---------------------+---------------------+
   ADI >= 1.32       |    INTERMITTENT     |        LUMPY        |
   sporadic          |  SBA, Croston, TSB, |  SBA, TSB, hurdle   |
   (< ~3 weeks in 4) |  empirical bootstrap|  (compound Poisson),|
                     |  of lead-time demand|  empirical bootstrap|
                     +---------------------+---------------------+

        ADI = periods in window / periods with demand   units: periods
        CV2 = (sd of nonzero sizes / mean nonzero size)^2   units: none
```

In the grid, SES is single exponential smoothing; ETS is the error–trend–seasonal family of
exponential smoothing models; ARIMA is the autoregressive integrated moving average family;
SBA and TSB are defined in section 4.

Read the grid as four different statistical problems wearing the same clothes. In the top-left
corner the problem is estimating a level. In the top-right it is estimating a level and a
dispersion. In the bottom-left it is estimating a rate of occurrence. In the bottom-right it is
estimating a rate, a size distribution, and the interaction between them. A method that solves
the wrong one of those four problems is not merely less accurate; it is answering a question
nobody asked.

The two axes are not symmetric, and that asymmetry matters later. ADI is a property of the
*bucket*: it counts periods, and a period is whatever you decided a period is. CV² is closer to a
property of the *process*, because it is computed only on the periods where demand occurred and
is therefore immune to how many zero periods you inserted between them. This is why temporal
aggregation moves series vertically down the grid far more than it moves them horizontally.

### 3.2 The pipeline that produces a grid entry

```text
  raw series: item x location x bucket x window
        |
        v
  [1] mask artificial zeros (stock-outs, allocations, credit blocks)
        |
        v
  [2] compute ADI and CV2 at the forecasting level and bucket
        |
        v
  [3] place in cell; record distance to the nearest cut-off
        |
        v
  [4] method pool for that cell  ->  [5] backtest inside the cell
        |                                    |
        |                                    v
        |                          [6] route; stamp the rule version
        |                                    |
        +------------------------------------+
                                             |
                                             v
                            [7] monitor drift and overrides
                                             |
                                             +--- reclassify ---> [2]
```

Six of the seven steps are not classification. Step 1 decides whether the zeros are real. Step 2
decides what the series *is*. Step 4 encodes a policy about which methods may compete. Step 5
decides whether the policy survives contact with the data. Step 6 makes the decision auditable.
Step 7 accepts that the decision expires. Only step 3 is the part that appears in the textbook
diagram, and it is the part least likely to be wrong.

### 3.3 The same item, five grid positions

| Level | Bucket | ADI | CV² | Cell |
|---|---|---:|---:|---|
| Item-location | Week | 13.00 | 0.85 | Lumpy |
| Item-location | Month | 3.00 | 0.85 | Intermittent |
| Item-region (8 locations) | Week | 1.60 | 0.26 | Intermittent |
| Item-national (16 locations) | Week | 1.16 | 0.13 | Smooth |
| Item-national | Month | 1.00 | 0.13 | Smooth |

Every row describes the same physical demand. Four of the five are defensible measurements of it.
The mistake is to compute the class at one level and apply the method at another, which is the
single most common structural defect in hierarchical forecasting systems and the subject of
section 5.

### 3.4 A taxonomy of the decision, not of the demand

The grid answers one question: *which family of estimators is likely to have the lowest decision
loss on this series?* It does not answer whether the item is important, whether the item is
forecastable, whether the item should be stocked, or whether the demand is real. Those are
separate decisions with separate evidence, and conflating them produces the familiar pathology of
a critical spare part treated as unimportant because its cell says "lumpy".

Hold the model accordingly: **the cell is an input to a routing decision, the routing decision is
a model, and the model is validated on the decision it feeds.** Everything in the rest of this
file follows from that sentence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Demand occurrence | Whether any demand is observed in a period | Treated as a fixed property of the item | Binary per period; 1 = demand |
| Demand size | Quantity demanded in a period where demand occurred | Averaged over all periods including zeros | Units per occurrence |
| ADI | Average demand interval: periods in the window divided by periods with demand | Computed as the mean of observed gaps only, which drops censored runs | Periods per demand |
| CV² | Squared coefficient of variation of nonzero demand sizes | Computed on the full series including zeros | Dimensionless |
| Demand classification | Assignment of a series to one of four cells by ADI and CV² | Treated as a stable attribute of the stock-keeping unit (SKU) | Cell label plus rule version |
| Cut-off | The ADI and CV² values that separate cells | Treated as universal constants | ADI in periods; CV² dimensionless |
| Smooth | Cell with ADI < 1.32 and CV² < 0.49 | Used to mean "high volume" | Cell label |
| Erratic | Cell with ADI < 1.32 and CV² ≥ 0.49 | Confused with lumpy | Cell label |
| Intermittent | Cell with ADI ≥ 1.32 and CV² < 0.49 | Used as an umbrella term for all four cells | Cell label |
| Lumpy | Cell with ADI ≥ 1.32 and CV² ≥ 0.49 | Used as a synonym for intermittent | Cell label |
| Intermittency rate | Share of periods in the window with zero demand | Confused with ADI, which is its reciprocal | Proportion, 0–1 |
| Router | The rule that maps a series to a method family | Treated as infrastructure rather than as a model | Rule identifier plus version |
| Method family | A group of estimators sharing an assumption about occurrence | Treated as a single method | Label |
| Method pool | The set of candidate methods a cell is allowed to draw from | Left implicit in tooling | List of method identifiers |
| Class-conditional evaluation | Accuracy computed inside a cell | Replaced by portfolio-level accuracy | Metric plus cell |
| Static classification | A label assigned once and not refreshed | Defended as "stability" | Date of assignment |
| Class drift | Movement of a series between cells over time | Discovered only when service fails | Sequence of labels |
| Reclassification cadence | How often labels are recomputed | Set by a system batch job, not by need | Weeks |
| Boundary distance | How close a series sits to a cut-off | Ignored in reporting | ADI periods; CV² units |
| Observation window | The history used to compute ADI and CV² | Chosen by data availability | Weeks |
| Bucket | The time unit of the series | Never stated in the method documentation | Day, week, month |
| Aggregation level | The entity level at which the series is defined | Assumed not to affect the class | Item, item-location, item-region |
| Hierarchy | The set of aggregation levels and their relationships | Treated as a reporting device only | Tree structure |
| Reconciliation | Making forecasts coherent across levels | Treated as cosmetic | Method plus constraints |
| Disaggregation | Splitting a parent forecast down to children | Assumed to preserve the child's class | Method |
| Structural zero | A period with no demand because none was possible | Counted as a demand zero | Boolean flag |
| Artificial zero | A period with no sale because no stock was available | Counted as a demand zero | Stock-out flag |
| Censoring | Demand hidden by unavailability | Ignored in measurement | Share of periods censored |
| Hurdle model | Occurrence probability multiplied by conditional size | Used as a synonym for Croston | Two-part model |
| Compound Poisson | Poisson arrivals with independently drawn sizes | Assumed to imply constant sizes | Arrivals per period; units |
| SBA | Syntetos–Boylan Approximation: Croston multiplied by (1 − α/2) | Applied without stating the smoothing constant α | α dimensionless, 0–1 |
| TSB | Teunter–Syntetos–Babai: smooths occurrence probability every period | Applied to stationary demand without testing | α and β dimensionless |
| Croston | Separate smoothing of intervals and sizes, forecast as their ratio | Used as the baseline for every series | α dimensionless |
| SES | Single exponential smoothing | Applied to lumpy series "because it is robust" | α dimensionless |
| ADIDA | Aggregate–disaggregate intermittent demand approach | Confused with plain temporal aggregation | Aggregation level m |
| Supersession | Replacement of one item by another in the catalogue | Ignored when building history | Predecessor link |
| New product introduction (NPI) | The launch phase of an item | Classified with mature-item rules | Launch date |
| End of life (EOL) | The retirement phase of an item | Classified with growth-phase rules | Retirement date |
| Cold start | A series with too little history to classify | Classified anyway, from two observations | Count of periods |
| Override | A human change to the routed method | Not logged, so not reviewable | User identifier plus reason code |

Four of these need expansion, because they are where the arguments happen.

**ADI versus intermittency rate.** They are reciprocals of one another, and both are reported.
An intermittency rate of 0.75 is an ADI of 1.33. The convention matters because the cut-off of
1.32 is an ADI, and 1/1.32 = 0.758, so the widely quoted threshold is very nearly "demand occurs
in fewer than three periods in four". Stated that way it sounds like a rule of thumb, which is
what it is, and the authority of the decimal places evaporates.

**Stock-out zeros versus true zeros.** A zero is evidence about occurrence only if the item could
have been sold. If the shelf was empty, the zero measures availability, not demand. Feeding such
zeros into the classifier inflates both ADI and CV², pushing items toward the lumpy cell, and
then the lumpy method forecasts a low rate that justifies holding less stock, which produces more
stock-outs. The loop is self-reinforcing, and it is one of the most consequential data defects in
the field.

**Static versus dynamic classification.** Static means the label is computed once, typically at
item creation or at a system migration. Dynamic means it is recomputed on a schedule. Almost every
commercial implementation is static with a manual refresh, which is a governance fact worth
establishing before you trust a class label in a report.

**Classification accuracy versus decision loss.** You can measure a classifier by how often its
cell agrees with an expert's label, or by what the routed method costs. The first is easy and
almost meaningless, because expert labels are themselves produced by a rule. The second is hard
and is the only one that matters. Section 9 returns to this.

## 5. The mechanics, step by step

This section runs the pipeline of section 3.2 in order. The steps are numbered because the order
matters: almost every defect that shows up at step 3 or 4 was created at step 1 or 2, and almost
every complaint about the routing table is really a complaint about a convention chosen in step 2
that nobody wrote down.

### Step 1 — Fix the measurement frame

Before a single number is computed, write down four things:

- **Level.** Item-location, item-region, item-national, category. This is the level at which the
  forecast will be consumed, which is not necessarily the level at which the demand was recorded.
- **Bucket.** Day, week, four-week period, calendar month. State the calendar convention: ISO 8601
  weeks, the retailer's 4-4-5 calendar, or the fiscal month. A bucket that starts on a different
  day produces a different series from identical events.
- **Window.** The number of periods used to compute ADI and CV². Fifty-two weeks, 104 weeks, or
  "all available history" are three different measurements. A window that includes a promotion
  or a stock-out period is a window that includes a distortion.
- **Zero treatment.** Which periods are excluded, which are retained, and how partial
  availability is handled.

**Branch point.** If the level of measurement differs from the level of consumption, stop and
resolve it. A class computed at item-national and applied at item-location is not a classification
of the item-location series; it is a classification of something else, borrowed.

The reason to be pedantic here is arithmetic. With a 52-week window, ADI = 52/k where k is the
number of weeks with demand. The cut-off of 1.32 falls between k = 39 (ADI 1.333) and k = 40 (ADI
1.300). One additional demand week flips the cell. Any window shorter than 52 makes the flip
easier: at 26 weeks the same cut-off falls between k = 19 and k = 20, and a single order moves an
item across the line.

### Step 2 — Mask artificial zeros

A zero in a demand series has three possible causes, and only one of them is information about
demand:

1. **No demand existed.** The customer did not want the item. This is the zero the classifier
   wants.
2. **Demand existed but could not be served.** The item was out of stock, the customer was on
   credit hold, the line was down. This zero is censored demand, and including it biases the
   occurrence rate downward and the size dispersion upward.
3. **The item was not orderable.** Discontinued, not yet launched, or excluded from the catalogue
   for the period. This is a structural zero and should not be in the window at all.

Masking requires an availability history at the same grain as the demand history. That is a data
engineering requirement before it is a forecasting one, and it is frequently unmet: the warehouse
management system knows the stock position, the demand history table does not. Where availability
cannot be reconstructed, the honest move is to state that the classification is computed on
uncensored data and to expect an upward bias in ADI.

**Branch point.** If more than roughly a fifth of the zero periods were censored, the computed
class is not trustworthy and the appropriate output is an exception, not a label.

### Step 3 — Compute ADI and CV²

Here ADI = n / k, in periods per demand, and CV² = (s / m)², dimensionless, where n is the
number of periods in the window after masking, k is the number of periods with demand, m is the
mean of the nonzero sizes, and s is their standard deviation. Three conventions must be stated
because each moves the answer:

- **Sample or population standard deviation.** With k = 4, dividing by k − 1 instead of k inflates
  CV² by a factor of 4/3. An item at CV² = 0.40 becomes 0.53 and crosses the cut-off. Most
  statistical software defaults to the sample form; most of the classification literature does not
  state which it used.
- **Excluding or including the final interval.** ADI computed as the mean of observed gaps treats
  the run of zeros after the last demand as missing. In a right-censored window that understates
  ADI for items that have just stopped selling, which is exactly the EOL case.
- **Integer or fractional sizes.** A series recorded in cases but forecast in eaches has a
  different CV² from the same physical demand recorded in eaches, if rounding to cases compresses
  small orders.

A useful discipline is to compute both the sample and the population form and to flag any series
whose cell differs between them. In the worked example, one item in twelve does.

### Step 4 — Place the series and record the boundary distance

Apply the cut-offs and record not only the cell but the distance to the nearest cut-off, expressed
in the units of the axis. This single field does more for operational trust than any amount of
documentation, because it distinguishes a series that is confidently lumpy from one that is lumpy
by 0.01 of ADI and will be intermittent next month.

**Branch point.** A series within a small tolerance of a boundary — say 5% of the cut-off value —
should be routed to the method that is *robust across the boundary* rather than to the method that
is optimal on one side. In practice this means a method whose performance degrades gracefully when
the assumption fails, which is usually the simpler of the two candidates or a combination.

### Step 5 — Bind each cell to a method pool

The pool is a policy decision, and it should be written down as one. A defensible default:

| Cell | Primary family | Acceptable alternatives | Not admissible without evidence |
|---|---|---|---|
| Smooth | SES, ETS, seasonal ETS, ARIMA | regression with drivers, combinations | Croston-type methods, which discard information |
| Erratic | ETS with damping, robust ETS, combinations | ARIMA, regression | naive level-only smoothing |
| Intermittent | SBA, Croston, TSB | empirical bootstrap of lead-time demand | SES, moving average |
| Lumpy | SBA, TSB, hurdle or compound Poisson, bootstrap | combinations of the above | SES, moving average, naive |

The "not admissible" column is where the value is. A method pool that admits everything is not a
policy; it is a search, and a search over a portfolio of 40,000 series will find spurious winners.

**Branch point.** If the pool for a cell contains fewer than two methods, the cell has no
challenger and its accuracy cannot be attributed to the classification. Record that as a known
limitation.

### Step 6 — Backtest inside the cell

Evaluate each candidate on the series it would actually be applied to, using a rolling origin, a
horizon equal to the decision's protection period, and a metric that matches the decision. Two
traps:

- **The sparse-cell problem.** If a cell contains 30 series, a win rate estimated on those 30 has
  a standard error of roughly 7 percentage points at a true rate of 80%. A routing table tuned on
  a cell that small is tuned on noise. Report the interval, not the point.
- **Selection leakage.** Choosing the method after seeing the test-period error and then reporting
  that error as out-of-sample performance overstates accuracy. Hold out a final period that the
  selection never sees, or accept that the reported number is optimistic and say so.

### Step 7 — Route and stamp the rule version

Every forecast artefact should carry the identifier of the rule that produced it: rule name,
version, effective date, and the conventions from step 1. Without this, a change in accuracy
cannot be attributed to a change in method, and the router becomes unfalsifiable.

### Step 8 — Monitor drift

Class drift has two causes and they need different responses.

- **Genuine drift.** The item's demand process changes: a new product matures, a spare part's
  installed base grows, a promotion ends, a competitor exits. The response is reclassification on
  a schedule, plus a trigger when the rolling ADI or CV² crosses a boundary.
- **Measurement drift.** The process did not change; the data did. A new stock-out flag was
  introduced, a location was added to the hierarchy, a rounding rule changed in the enterprise
  resource planning (ERP) system. The response is a data-quality investigation, not a
  reclassification.

A practical monitor plots rolling ADI and CV² with the cut-offs drawn as horizontal lines. Series
that cross and re-cross repeatedly are not drifting; they are borderline, and they should be
routed to the robust default from step 4 rather than reclassified every month.

### Step 9 — Reconcile with the hierarchy

Classification and reconciliation interact in both directions. Upward, aggregation changes the
class of the parent, so a bottom-up forecast of a lumpy family can be reconciled against a smooth
parent and inherit a correction that assumes occurrence is certain. Downward, a smoothed parent
disaggregated to children spreads a level estimate over series whose problem is occurrence.

The governing rule is simple and widely violated: **classify at the level at which the method will
run, and never let a reconciliation step change the method.** If the reconciled bottom-up total
must equal a smooth national forecast, that is a constraint on the numbers, not a licence to
replace the children's methods.

### Step 10 — Govern the exceptions

The router will be wrong. The question is whether the organisation can see where. Four artefacts
make the difference:

- An **exception queue** of series that fail a validity test: too few observations, a cell
  disagreement between sample and population dispersion, a censoring rate above threshold, a
  boundary distance inside tolerance.
- An **override log** recording who changed which series to which method, when, and why, with a
  reason code.
- A **class-conditional scorecard** so that accuracy is reported per cell rather than only in
  aggregate.
- A **change log** for the rule itself, with the backtest that justified each version.

### 5.1 Where the cut-offs came from, and why that matters

The constants are usually presented as empirical findings. They are better understood as the
solution to a specific comparison, rounded.

Syntetos, Boylan and Croston (2005) compared Croston's method with the SBA correction and derived
the region in which one has lower mean squared error than the other. The boundary that falls out
of their equations is a curve in the (ADI, CV²) plane, not a rectangle. The published values of
1.32 and 0.49 are approximations to that curve, chosen so that a planner can apply a rule with two
comparisons rather than a formula. The paper's own recommendation was to use Croston's method in
the erratic region and SBA elsewhere — a recommendation about *two estimators*, not about demand
in general.

Kostenko and Hyndman (2006) re-derived the same boundary from the same equations and reported
that the exact values are 4/3 and 1/2 rather than 1.32 and 0.49, that the true boundary is
non-linear, and that a diagonal approximation collapses the four cells into effectively two. They
also observed that Croston's method cannot be optimal for any demand series, a result they
attribute to earlier work. Syntetos, Boylan and Croston replied accepting the technical correction
while defending the four-cell scheme as a practical rule that a planner can apply and audit.

Three lessons follow, and they are the intellectual core of this topic.

1. **The numbers are a rounded algebra result under a specific loss.** Mean squared error, two
   estimators, i.i.d. sizes, independent intervals. Change any of those and the boundary moves.
2. **A simpler rule can beat an exact one on adoption.** The four-cell scheme survived because
   planners can use it. That is a real advantage and it is not a scientific one.
3. **Re-deriving the cut-offs on your own portfolio is legitimate and is what the original authors
   recommended.** The method is straightforward: for each series, compute the error of each
   candidate method, then find the ADI and CV² values at which the ranking flips.

### 5.2 Three conventions that flip more series than the cut-offs do

In practice, the choice between 1.32 and 4/3 changes almost nothing. These three change a great
deal:

| Convention | Option A | Option B | Effect on a borderline series |
|---|---|---|---|
| Dispersion denominator | divide by k | divide by k − 1 | Inflates CV² by k/(k−1); 33% at k = 4 |
| Zero treatment | include all zeros | mask stock-out zeros | Moves ADI down and CV² down; can move two cells |
| Window length | 52 weeks | 104 weeks | Halves the sampling error of ADI and CV²; changes the answer for trending items |

An analyst who argues about 1.32 versus 1.33 while leaving these three unstated has the priorities
backwards.

## 6. Worked example

An industrial distributor — illustrative, but internally consistent — carries twelve
representative item-location series. All are weekly, all cover the same 52 complete weeks, all
are measured in eaches, and all stock-out weeks have been masked. The dispersion convention is the
population standard deviation, dividing by k. Two items sit deliberately on the ADI boundary.

### 6.1 The portfolio

| Item | Description | k | ADI | Mean size | SD (pop) | CV² | Cell | Nearest boundary |
|---|---|---:|---:|---:|---:|---:|---|---|
| A-1001 | Bearing 6204 | 50 | 1.040 | 120.0 | 9.60 | 0.006 | Smooth | 0.28 ADI inside |
| A-1002 | Seal kit | 48 | 1.083 | 40.0 | 12.00 | 0.090 | Smooth | 0.40 CV² inside |
| A-1003 | Gasket, nitrile | 40 | 1.300 | 60.0 | 15.00 | 0.063 | Smooth | **0.02 ADI** |
| B-2101 | Hydraulic hose 2 m | 46 | 1.130 | 25.0 | 19.00 | 0.578 | Erratic | 0.09 CV² over |
| B-2102 | Filter cartridge | 44 | 1.182 | 18.0 | 14.40 | 0.640 | Erratic | 0.15 CV² over |
| C-3301 | Pump assembly | 13 | 4.000 | 3.0 | 1.20 | 0.160 | Intermittent | 2.68 ADI over |
| C-3302 | Motor 1.5 kW | 9 | 5.778 | 5.0 | 2.00 | 0.160 | Intermittent | 4.46 ADI over |
| C-3303 | Gearbox | 6 | 8.667 | 2.0 | 0.80 | 0.160 | Intermittent | 7.35 ADI over |
| C-3304 | Coupling | 39 | 1.333 | 8.0 | 2.00 | 0.063 | Intermittent | **0.01 ADI** |
| D-4401 | Valve actuator | 4 | 13.000 | 6.0 | 5.52 | 0.847 | Lumpy | 11.68 ADI over |
| D-4402 | Sensor module | 7 | 7.429 | 12.0 | 18.00 | 2.250 | Lumpy | 1.76 CV² over |
| D-4403 | Circuit board rev C | 3 | 17.333 | 10.0 | 11.00 | 1.210 | Lumpy | 0.72 CV² over |

Portfolio totals: 12,808 eaches per year. The three smooth items carry 10,320 of them, or 80.6%.
The seven intermittent and lumpy items carry 546, or 4.3%, and account for 58% of the item count.
This is the shape of almost every real portfolio, and it explains both why classification is
necessary and why it is chronically under-resourced: the cells where the decision is hardest are
the cells where the units are fewest.

### 6.2 The arithmetic, item by item

**A-1003, the borderline smooth item.** k = 40, n = 52, so ADI = 52/40 = 1.300. Sizes: mean 60.0,
population standard deviation 15.0, so CV² = (15/60)² = 0.0625. Both axes are inside the smooth
cell, but the ADI margin is 1.32 − 1.30 = 0.02. One additional demand week in the window would
give k = 41, ADI = 1.268, still smooth; one fewer would give k = 39, ADI = 1.333, and the item
becomes intermittent. The cell is one order away from changing.

**C-3304, its mirror image.** k = 39, ADI = 52/39 = 1.3333. Same CV² of 0.0625. The item is
intermittent by 0.0133 of ADI — that is, by a single week's demand in a 52-week window. Its
sizes are small and stable; the method that suits it is an occurrence-aware one, but the amount
of occurrence information in the series is thin. Any method applied here should be tested on both
sides of the boundary.

**D-4401, a genuinely lumpy item.** k = 4, ADI = 13.0. The four nonzero sizes are 15, 2, 1, and 6
eaches, totalling 24. Mean 6.0. Deviations are +9, −4, −5, 0; the sum of squares is 81 + 16 + 25 +
0 = 122; population variance 122/4 = 30.5; standard deviation 5.523; CV² = 30.5/36 = 0.847. Under
the sample convention the variance would be 122/3 = 40.67 and CV² = 1.130. Both are lumpy, but the
same series would land on opposite sides of the 0.49 cut-off if the sizes were less dispersed —
which is precisely why the convention must be stated.

**D-4402, where the size distribution dominates.** k = 7, ADI = 7.429. Mean size 12.0, standard
deviation 18.0, so CV² = (18/12)² = 2.25. The mean absolute deviation from 12 is smaller than the
standard deviation suggests, because one occurrence of roughly 60 eaches does most of the work.
A method that estimates the mean size will be dominated by that observation for years. This is
the case for a bootstrap of the size distribution rather than a parametric assumption.

### 6.3 The bucket changes ADI and leaves CV² alone

Take D-4401 at weekly resolution: 4 demands in 52 weeks. Aggregate to calendar months, and suppose
the four demands fall in four different months. The monthly series has n = 12, k = 4, so ADI =
12/4 = 3.0. The nonzero monthly sizes are the same four numbers, so CV² is unchanged at 0.847. The
series has moved from the lumpy cell to the intermittent cell without any change in the underlying
demand, purely because a month is a longer unit than a week.

```text
  weekly:   n = 52, k = 4   ADI = 13.00   CV2 = 0.85   LUMPY
  monthly:  n = 12, k = 4   ADI =  3.00   CV2 = 0.85   INTERMITTENT
```

The lesson is general: **ADI is not scale-invariant, so a cut-off quoted without a bucket is
incomplete.** A routing table imported from a monthly planning system into a weekly one will
under-route series to occurrence-aware methods, because every ADI has been divided by roughly 4.3.

### 6.4 Aggregation moves the class, and the parent lies about the children

Now pool D-4401 across locations. Model each location as independent, with weekly occurrence
probability 6/52 = 0.115 (slightly higher than the single location's 4/52, because the example
is constructed to show the mechanism), nonzero mean 6.0, and CV² 0.85. Pooling across m locations:

```text
  m locations   P(demand in week)   ADI     pooled CV2   cell
  ------------------------------------------------------------
       1              0.115        8.70       0.85      LUMPY
       8              0.624        1.60       0.26      INTERMITTENT
      16              0.858        1.16       0.13      SMOOTH
```

The pooled occurrence probability is 1 − (1 − 0.115)^m, and the pooled CV² falls because the sum
of independent sizes has a smaller relative dispersion than its components. The item has not
changed. Its position in the grid has changed twice.

This produces the characteristic hierarchical error. The national series is smooth, so the
national forecast uses a level-tracking method and is accurate. The location series are lumpy, so
the location forecasts use occurrence-aware methods and are noisy. Reconciliation forces the
disaggregated children to sum to the smooth parent, and in doing so it imposes a level on series
that have no level. The national number is right and the local numbers are wrong in a way that
looks like noise rather than method failure.

> **Note:** Pooling reduces CV² only when the component demands are imperfectly correlated. In a
> portfolio where locations rise and fall together — a national promotion, a shared customer, a
> common season — the pooled CV² can stay high while ADI falls, and the item moves to erratic
> rather than smooth. The direction of the CV² move is an empirical question, not a theorem.

### 6.5 A product that changes class three times

A new product on monthly buckets, eighteen months of history, classified retrospectively in
rolling twelve-month windows:

| Phase | Months | k of n | ADI | CV² | Cell |
|---|---|---:|---:|---:|---|
| Launch | 1–6 | 3 of 6 | 2.00 | 0.95 | Lumpy |
| Ramp | 7–12 | 5 of 12 | 2.40 | 0.62 | Lumpy, drifting |
| Growth | 13–18 | 9 of 18 | 2.00 | 0.40 | Intermittent |
| Maturity (projected) | 19–30 | 12 of 12 | 1.00 | 0.15 | Smooth |

A label assigned at month 6 and never refreshed says lumpy. The item becomes intermittent around
month 15 and smooth around month 20. For the eighteen months after the label was set, the item
runs an occurrence-aware method that is designed to be conservative about whether demand happens,
on a series where demand happens every month. The method does what it was asked; the router asked
the wrong question.

The same mechanism runs in reverse at end of life, with one extra twist. A declining item that
still sells every month has ADI near 1.0 and low CV². It is classified smooth. It is also dying.
The grid cannot see the decline, because ADI and CV² measure interval and dispersion, not trend.
An EOL item needs a lifecycle flag from the item master, not a better classifier.

### 6.6 What a misroute costs, in units and in service

Take D-4401. Protection period is a four-week lead time plus a one-week review, so five weeks.
Demand rate is 24/52 = 0.462 eaches per week. Model arrivals as Poisson with λ = 4/52 = 0.0769 per
week, and sizes drawn from the empirical set {1, 2, 6, 15} with equal probability. Over five weeks
the expected number of arrivals is 0.385.

Three policies, three outcomes:

| Policy | Calculation | Stock level | Cycle service level |
|---|---|---:|---:|
| Trust the smooth point forecast | 0.462 × 5 = 2.31, rounded | 3 | 0.81 |
| Point forecast plus z·σ·√L from the raw series | 2.31 + 1.645 × 2.214 × √5 | 11 | 0.91 |
| Exact compound-Poisson 95% quantile | solve from the arrival and size distributions | 15 | 0.97 |

The middle row is what most planning systems do when a smooth method is routed a lumpy series.
It is not obviously wrong, and it is wrong: it holds eight units of stock, misses the service
target by four points, and does so because the normal approximation treats a highly skewed size
distribution as if it were symmetric. The top row is what happens when the point forecast is taken
literally and no safety stock is added. The bottom row is the number the decision actually needs:
the smallest stock level that clears the 95% target. Because the distribution is discrete, that
level overshoots to about 97%, which is itself worth knowing — the quantile function of a lumpy
series moves in jumps.

At a unit cost of EUR 900 and 40 stocking locations, the difference between the middle and bottom
policies is 160 units of inventory, or EUR 144,000, deployed in a way that does not reach the
service target. That is the price of a routing decision, and it is the number to put in front of
whoever owns the router.

## 7. Code, query, or calculation

The query below computes the two axes and the cell at item-location level, with stock-out weeks
masked and a minimum-occurrence guard. It is written for a warehouse with a weekly demand table
and a boolean stock-out flag. Expected output for the coupling item is one row: `39, 1.333,
0.063, intermittent`.

```sql
-- Weekly demand, item x location, 52 complete ISO weeks; stock-out weeks masked.
with w as (
  select item_id, location_id, units,
         case when stockout_any_day then null else units end as d
  from demand.weekly_item_location
  where week_start between date '2025-10-27' and date '2026-10-19'
),
s as (
  select item_id, location_id,
         count(d)                            as n_periods,
         count(*) filter (where d > 0)       as k_nonzero,
         avg(d) filter (where d > 0)         as mean_size,
         stddev_pop(d) filter (where d > 0)  as sd_size
  from w group by item_id, location_id
)
select item_id, location_id, k_nonzero,
       round(n_periods::numeric / k_nonzero, 3)          as adi,
       round(power(sd_size / mean_size, 2), 3)           as cv2,
       case when n_periods::numeric / k_nonzero < 1.32
                 and power(sd_size / mean_size, 2) < 0.49 then 'smooth'
            when n_periods::numeric / k_nonzero < 1.32   then 'erratic'
            when power(sd_size / mean_size, 2) < 0.49     then 'intermittent'
            else 'lumpy' end                              as cell
from s
where k_nonzero >= 4;   -- below four occurrences, neither axis is estimable
```

Four things to notice about this query, each of which is a decision:

- `count(d)` excludes masked weeks from the denominator, which is the intended treatment of
  censoring but lowers ADI relative to an unmasked computation. Document it.
- `stddev_pop` fixes the dispersion convention at division by k. Switch to `stddev_samp` and some
  series change cell.
- The comparison uses the unrounded ratio. Rounding to two decimals first can flip a series whose
  ADI is 1.3167 to the intermittent cell, because 1.32 is not less than 1.32.
- The guard at four occurrences is a policy, not a statistical law. Below four, ADI is dominated by
  the window boundary and CV² is estimated from almost nothing. Those series belong in an
  exception queue, not in a cell.

## 8. Data, standards, and regulatory context

No standard mandates the Syntetos–Boylan classification. No regulator prescribes a cut-off. That
absence is itself important, because it means every threshold in the router is a convention that
the organisation is free to change and accountable for defending. What standards do govern is the
data on which the classification depends, and that is where they bite.

### 8.1 Standards that constrain the inputs

| Standard | Governs | What it constrains for classification | Grade |
|---|---|---|---|
| GS1 GTIN (global trade item number) | Trade item identity | The series key; a re-coded item breaks its own history | B |
| GS1 GLN (global location number) | Location identity | The location key; site renumbering silently splits a series | B |
| GS1 EPCIS 2.0 (Electronic Product Code Information Services) | Event data on objects | Evidence of on-hand and out-of-stock events, hence censoring | B |
| ISO 8601 (International Organization for Standardization) | Date and week representation | Bucket boundaries; week 53 handling; Monday-start weeks | B |
| ANSI X12 852 (American National Standards Institute) | Product activity data | The sell-through feed; weekly and by location if configured | B |
| 4-4-5 retail calendar | Retail period boundaries | A bucket that is not a calendar month; ADI shifts accordingly | C |
| ISA-95 | Enterprise–control integration | Production confirmations, not demand; sets the granularity of supply events | B |
| SCOR DS | Process reference model | Vocabulary for plan, source, make, deliver, return | C |

Three of these deserve comment.

**Item and location identity are the silent failure.** A GTIN change at a packaging revision, or a
new GLN after a warehouse move, creates a new series with no history. The new series is
unclassifiable and the old one is dead. Every classification system needs a supersession or
predecessor link, and most do not have one that is maintained. The practical effect is that the
items with the most turbulent histories — repackaged, relocated, reformulated — are the ones whose
history is truncated, and they are routed on the basis of a few months of data.

**EPCIS is the only general standard that can evidence a stock-out.** If the warehouse emits
object-level events with a business step for stocking and a disposition for unavailable stock,
then censoring can be reconstructed at the grain of the demand series. If it does not, censoring
must be inferred from inventory snapshots, which is a coarser and more error-prone route. This is
the strongest argument for an event layer in a supply chain that already has orders and invoices:
without it, the classifier cannot tell a zero from a stock-out.

**ISO 8601 week numbering looks trivial and is not.** A 53-week year occurs roughly every five to
six years. A 52-week window that spans one has an extra week, which changes ADI by about 2% for
every series in the portfolio on the same date. If the window is defined by date subtraction
rather than by week count, the effect is smaller but still present near year boundaries.

### 8.2 What the accounting and audit framework touches

| Requirement | Where | Effect on the forecasting process | Grade |
|---|---|---|---|
| IAS 2 / ASC 330 | Inventory valuation | Forecast assumptions feed net realisable value tests; the demand view must be defensible | B |
| IFRS audit evidence | Audit engagement standards | Forecast inputs and overrides must be documented and attributable | B |
| Sarbanes–Oxley Section 404 | U.S. listed companies | Internal control over forecast inputs, including manual overrides | B |
| Local record-retention rules | Varies by jurisdiction | Demand history and method versions must be retained for the stated period | B |

The consequence is procedural rather than statistical. An override of the router that cannot be
attributed to a person, a date, and a reason is a control weakness, not merely a modelling
preference. Teams that treat the override log as bureaucracy tend to discover its value during an
audit or a service failure, whichever comes first.

### 8.3 What is convention rather than requirement

- The values 1.32 and 0.49 are conventions, derived from one comparison on one sample and revised
  since.
- The four-cell scheme itself is a convention. Two cells, six cells, and continuous scores are all
  defensible.
- The choice of 52 weeks as a window is a convention inherited from annual planning cycles.
- The rule "SBA for intermittent, Croston for erratic" is a recommendation about two estimators,
  not a standard.
- The practice of reporting a class label on a dashboard is a convention with no external
  authority, and it is often a mistake.

An analyst who can separate the requirement (identity, event evidence, auditability) from the
convention (thresholds, windows, cell names) can challenge a convention without appearing to
challenge the system. That distinction is the difference between a productive redesign and a
political fight.

## 9. Industry practice and case evidence

### 9.1 The research–practice gap is documented, not anecdotal

Bacchetti and Saccani (2012) reviewed the spare-parts classification and forecasting literature
and then examined ten case studies of durable-goods manufacturers and their service networks. The
finding that matters here is structural: the theoretical apparatus — multi-criteria
classification, demand-pattern categorisation, Croston-type methods — is well developed, and
adoption in the cases was partial and uneven. The authors attribute the gap to model complexity,
to contingency factors that the models ignore, and to differing maturity of spare-parts practices
across companies. Grade A for the review; the ten cases are not a random sample and should not be
read as a prevalence estimate.

The same authors, with Plebani and Syntetos (2012), proposed an empirically driven hierarchical
classification of stock-keeping units, which is the closest thing in the literature to a
defence of classifying within a hierarchy rather than at a single level. Grade A.

### 9.2 The routing rule is contested in the literature

| Question | Position A | Position B | Evidence |
|---|---|---|---|
| Does classification improve accuracy over a single method? | Yes, when the portfolio is heterogeneous and the cell is large enough to validate | Not reliably; combinations of methods match or beat selection | Peer-reviewed comparisons, Grade A |
| SBA or TSB? | SBA is the safer default for stationary intermittent demand | TSB dominates when items can stop being ordered | Mixed; grade A studies disagree |
| Are the 1.32/0.49 cut-offs usable? | Yes, as a practical rule with a stated loss | No, the true boundary is non-linear and the numbers are rounded | Published exchange, Grade A |
| Should cut-offs be re-derived locally? | Yes, on the portfolio and the decision at hand | No, the sampling error on a small portfolio is larger than the gain | Methodological argument |

The honest summary is that the literature supports the *discipline* of routing — measure the
series, state the convention, validate inside the class, monitor drift — and does not support any
particular routing table as universally correct. A team that adopts the discipline and derives its
own table is doing what the original authors recommended. A team that adopts the table and skips
the discipline is doing the opposite.

Two specific empirical findings sharpen this.

- Teunter and Duncan (2009) compared intermittent-demand methods on Royal Air Force data and
  showed that traditional per-period error measures are inappropriate for the purpose, because
  they do not reflect the inventory consequences. Grade A.
- Kourentzes (2014) evaluated error metrics as cost functions for model and parameter selection
  and found that conventional metrics perform poorly in that role, and that elaborate model
  selection barely outperforms using a single sensible model. Grade A. This is the strongest
  published argument that the router matters less than the choice of a robust default.

Against that, Petropoulos and Kourentzes (2014) compared classification-based selection with
forecast combinations on intermittent demand and found combinations competitive or better,
evaluated with inventory proxies. Grade A. If combinations win, the router's value is reduced to
choosing a pool, not a method — a materially weaker claim for the whole apparatus.

### 9.3 What implementations actually look like

Commercial planning suites expose some version of this classifier, usually as a two-axis
segmentation with configurable thresholds and a per-segment method assignment. Enterprise resource
planning and advanced planning systems commonly ship a default rule and a manual override. The
vendor documentation describes the mechanism; it does not, and cannot, validate the thresholds on
your portfolio. Grade B for what the software does, and no grade at all for whether it works on
your data.

Practitioner implementations found in the public record show the same pattern: a segmentation
step, a per-segment model pool, and a walk-forward evaluation. A retail study of two Chinese
e-commerce datasets (2022) classified series with the 0.49 and 1.32 thresholds and found that
selecting the best past performer worked better for intermittent and lumpy series, while
performance-weighted combinations worked better for smooth and erratic ones. Grade A for the
study; the finding is dataset-specific and has not been replicated at scale.

A single-warehouse study of more than 16,000 items (2020) reported TSB outperforming Croston, SBA,
SES, and moving average, with Croston and SBA performing worst overall. Grade B: a journal paper,
but one warehouse, one product mix, and one error metric. It is a useful counterweight to the
default assumption that SBA is always the safe choice, not a basis for a portfolio-wide rule.

### 9.4 The two things practitioners argue about most

**Whether to classify at all.** The combination argument says: skip the router, run three or four
methods on every series, and combine. It is simple, it removes a failure point, and the evidence
is competitive. The counter-argument is operational: combinations are harder to explain to a
planner, harder to override meaningfully, and produce a forecast whose behaviour under extreme
inputs nobody has characterised. Both positions are defensible; the choice is about governance as
much as accuracy.

**Whether the class should drive the service target.** It should not. Service targets come from
criticality, cost of shortage, and contractual commitment, not from the shape of the demand
history. A lumpy C-class item that stops a production line needs a higher service level than a
smooth A-class item with substitutes. Class determines *how to forecast*; criticality determines
*how much to hold*. Systems that conflate them systematically under-serve the slow-moving items
that matter most.

## 10. Failure modes and red flags

| Failure | Red flag in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Classifying at the wrong level | Class is computed nationally and applied per location | Aggregation lowers ADI and CV² | Local series run level-tracking methods | Classify where the method runs; record the level |
| Using a window shorter than the phenomenon | Half the portfolio has fewer than 20 observations | ADI estimated on a handful of orders | Cells flip monthly; method churn | Minimum-occurrence guard; exception queue |
| Retaining stock-out zeros | Stock-out flag exists in the warehouse but not in the demand table | Censoring inflates ADI and CV² | Items drift to lumpy; stock falls; more stock-outs | Mask availability; report censoring rate |
| Unstated dispersion convention | Two analysts get different cells for the same item | Sample versus population denominator | Disputes nobody can resolve | Publish the convention with the rule |
| Static label from launch | Class date predates the item's maturity | Lifecycle drift | Mature items run launch-phase methods | Scheduled reclassification; drift triggers |
| Using a rule built for two estimators as a general selector | The rule is applied to choose among gradient-boosted models | Out-of-domain use of a derived boundary | Selection with no theoretical or empirical basis | Restrict the rule to its provenance; evaluate alternatives directly |
| Cell-level backtest on a tiny cell | "The lumpy cell improved 6%" with 24 series | Sampling error dominates | Chasing noise; unstable configuration | Report intervals; pool cells or accept uncertainty |
| Aggregation then disaggregation | Parent is smooth, children are lumpy | Reconciliation imposes a level | Local forecasts systematically wrong | Constrain numbers, not methods |
| Unlogged overrides | Planner says "we changed that item years ago" | No attribution or review | Router unfalsifiable; audit exposure | Override log with reason codes |
| Borderline series reclassified every cycle | Same item changes cell in consecutive reports | Boundary distance inside tolerance | Forecast instability; planner distrust | Robust default for borderline; hysteresis band |
| Class used to set service targets | Lumpy items get lower service levels by policy | Conflation of forecast method with criticality | Critical spares under-held | Separate criticality from demand class |
| No method version on the forecast artefact | Accuracy changed and nobody knows why | Rule changed without a record | Cannot attribute or roll back | Stamp rule identifier and version |

Two of these are worth expanding, because they are the ones that survive audit.

**Retaining stock-out zeros** is the failure that manufactures its own evidence. The zero looks
like a demand observation, so the item looks slower than it is, so the classifier moves it toward
lumpy, so the method forecasts a low rate, so the stocking policy holds less, so the item stocks
out more. Every step in the loop is locally reasonable and the loop has no natural brake. The only
detectable signature is a rising censoring rate in a cell whose population is not otherwise
changing.

**Using a rule built for two estimators as a general selector** is the failure of the decade. The
1.32/0.49 boundary was derived for a comparison between Croston and SBA under squared error. It
has been reused to choose among exponential smoothing variants, machine-learning models, and
gradient-boosted ensembles, where the loss surface and the estimator bias are entirely different.
The rule still appears to work, in the weak sense that routed models beat unrouted ones on a
portfolio average, because almost any segmentation beats no segmentation when the portfolio is
heterogeneous. That is not validation of the boundary.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test |
|---|---|---|---|
| Demand is observable at the forecasting level | Every axis is computed from observed demand | Shipments proxy for demand; the class describes the proxy | Compare sell-in and sell-out series for the same items |
| Zeros are demand zeros unless proven otherwise | Occurrence is the quantity being measured | Censoring inflates ADI and CV² | Compute the censoring rate per cell per quarter |
| The demand process is stationary within the window | ADI and CV² are single summaries | Trending or seasonal series get a meaningless average | Plot rolling ADI and CV²; test for trend |
| Occurrence and size are independent | Croston-type decompositions assume it | Large orders arrive at particular times; bias in the ratio | Correlate interval and size across the window |
| The empirical size distribution is representative | Bootstrap and quantile methods resample it | One large order dominates; tail is understated | Compare quantiles across windows |
| The bucket matches the decision's review period | ADI is defined per bucket | A weekly series routed with monthly thresholds | Recompute the class at the decision bucket |
| The cut-offs transfer to this portfolio | They are applied as if universal | Boundary is in the wrong place for the cost structure | Re-derive on a holdout; compare cell membership |
| The method pool contains a method suited to each cell | Routing presumes a destination | Cell routes to nothing better than the default | Inspect the pool per cell; count challengers |
| The evaluation window represents the decision horizon | Backtest error is used to choose methods | Methods chosen for the wrong horizon | Rolling-origin backtest at the protection period |
| Class membership is stable enough to act on | A method is committed for a period | Churn; forecasts change without demand changing | Measure label changes per item per quarter |

Two limits deserve plain statement.

**The grid cannot see trend.** ADI and CV² are interval and dispersion statistics. A series that
sells every month in steadily declining quantities is smooth. So is a series that sells every
month in steadily growing quantities. Lifecycle state has to come from the item master, from a
fitted trend, or from a separate test — not from the classifier.

**The grid cannot see criticality.** A lumpy item and a smooth item can be equally critical, and
the grid will route them to different methods while saying nothing about how much stock either
needs. Any design that lets the cell label propagate into service targets has crossed from
forecasting into policy without saying so.

## 12. Exercises

### Exercise 1 — Hand-classify twelve series (application, 40 minutes)

Use the twelve series in section 6, or substitute twelve from your own portfolio. Compute ADI and
CV² by hand for at least four of them, showing the intermediate sums.

**Deliverable:** a table with k, ADI, mean size, standard deviation, CV², cell, and boundary
distance for each series.
**Acceptance criteria:** the dispersion convention is stated; at least one item's arithmetic is
shown line by line; any series within 5% of a cut-off is marked.
**Estimated time:** 40 minutes.

### Exercise 2 — Convention sensitivity (application, 45 minutes)

Recompute the same twelve series three ways: population standard deviation; sample standard
deviation; and a 26-week window instead of 52.

**Deliverable:** a table showing which series change cell under each convention, with the
magnitude of the change.
**Acceptance criteria:** every change is explained by the arithmetic, not asserted; the number of
cell changes is reported as a count and a share of the portfolio; a recommendation states which
convention the organisation should adopt and why.
**Estimated time:** 45 minutes.

### Exercise 3 — The aggregation ladder (application, 60 minutes)

Take one item with at least eight locations. Compute its class at item-location, item-region, and
item-national, on the same bucket and window.

**Deliverable:** a three-level table of ADI, CV², and cell, plus a one-paragraph explanation of the
direction of the movement.
**Acceptance criteria:** the same window and bucket at every level; the ADI movement is decomposed
into occurrence-count and period-count effects; at least one sentence states what a reconciled
forecast would do to the lowest level.
**Estimated time:** 60 minutes.

### Exercise 4 — Class drift over a lifecycle (application, 50 minutes)

Construct or obtain an eighteen-month monthly history for a new product. Classify it in rolling
twelve-month windows and plot the class path.

**Deliverable:** a table of window end date, k, ADI, CV², and cell, plus a timeline diagram.
**Acceptance criteria:** at least four windows; the point at which the label should have changed is
identified; a reclassification trigger is proposed with a specific threshold.
**Estimated time:** 50 minutes.

### Exercise 5 — Price the misroute (calculation, 60 minutes)

For the lumpy item in section 6, compute the stock level and cycle service level under the three
policies in section 6.6, then repeat with a size distribution of your choosing.

**Deliverable:** a policy table with stock levels, service levels, and the currency value of the
difference at a stated unit cost and location count.
**Acceptance criteria:** the arrival model and size distribution are stated; the quantile is derived
rather than looked up; the currency figure carries units and a stated holding-cost assumption.
**Estimated time:** 60 minutes.

### Exercise 6 — Critique a routing table (critique, 40 minutes)

Obtain a segmentation table from a vendor deck, a system configuration screen, or an internal
design document. Audit it against the checklist in sections 5 and 8.

**Deliverable:** a one-page critique listing every unstated convention, plus three questions you
would ask the owner.
**Acceptance criteria:** each criticism names the missing convention and the direction of its
effect; at least one criticism is about governance rather than statistics; the questions are
answerable by a named role.
**Estimated time:** 40 minutes.

### Exercise 7 — Design the router governance (design, 60 minutes)

Write the operating specification for a router covering 20,000 series: rule versioning, exception
queue, override log, drift monitoring, and change control.

**Deliverable:** a two-page specification with roles, triggers, and artefact fields.
**Acceptance criteria:** every artefact has an owner role and a retention period; the change-control
step requires a backtest; the exception criteria are numeric; the specification states what happens
when the router and a planner disagree.
**Estimated time:** 60 minutes.

### Exercise 8 — Classification versus combination (critique, 90 minutes)

Argue both sides of the question: should the organisation route series to method families, or run a
small ensemble on everything? Use at least three published sources.

**Deliverable:** a two-page memo with a recommendation and a proposed test.
**Acceptance criteria:** at least one source supports each side; the recommendation states the
conditions under which it would reverse; the test specifies the metric, the horizon, and the
decision it informs.
**Estimated time:** 90 minutes.

### Exercise 9 — Censoring audit (application, 45 minutes)

Take a demand extract and a stock-position extract for the same period. Estimate the share of zero
weeks that were stock-outs.

**Deliverable:** a censoring rate by cell, with the join logic and its failure modes.
**Acceptance criteria:** the join is at the correct grain; partial-week stock-outs are handled
explicitly; the effect on ADI is recomputed with and without masking.
**Estimated time:** 45 minutes.

### Exercise 10 — Defend the router (design, 30 minutes)

Present your routing decision to a sceptical planning manager in five minutes, then answer three
questions: why not one method, why these thresholds, and who is accountable when it is wrong.

**Deliverable:** a five-slide outline and a one-page question-and-answer brief.
**Acceptance criteria:** the answer to "why these thresholds" cites the provenance and the local
validation; the accountability answer names a role and a process; no answer relies on the phrase
"industry best practice".
**Estimated time:** 30 minutes.

## 13. Self-check answers

**Exercise 1.** For A-1003: k = 40, ADI = 52/40 = 1.300; mean 60.0; population standard deviation
15.0; CV² = 0.0625; smooth, with an ADI margin of 0.02. For D-4401: k = 4, ADI = 13.0; sizes 15, 2,
1, 6; sum 24; mean 6.0; sum of squared deviations 122; variance 30.5; standard deviation 5.523;
CV² = 0.847; lumpy. A correct answer marks both A-1003 and C-3304 as borderline and states the
dispersion convention.

**Exercise 2.** The expected finding is that the sample convention changes at most one or two
series in twelve, and that it changes them only where k is small. With k = 4, the factor is 4/3,
so any item with CV² between 0.37 and 0.49 flips to erratic or lumpy. Shortening the window to 26
weeks changes more: ADI becomes 26/k, the cut-off falls between k = 19 and k = 20, and items with
moderate counts move. The recommendation should be to fix one convention, publish it, and use the
longest window the data supports, because the sampling error of ADI falls as the window grows.

**Exercise 3.** Pooling lowers ADI because more periods contain at least one demand, and lowers
CV² when component demands are imperfectly correlated. A typical result is lumpy at location,
intermittent at region, smooth or erratic nationally. The decomposition should note that the ADI
change comes from two sources: more periods with demand (numerator effect) and, if the window
changes with the level, a different period count (denominator effect). Only the first is
legitimate. The reconciled forecast imposes the parent's total on the children and must not be
allowed to change their methods.

**Exercise 4.** A plausible path is lumpy at launch (three of six months), lumpy or drifting
through ramp, intermittent as monthly occurrence becomes regular, and smooth at maturity. The
label should have changed when the rolling twelve-month ADI first fell below about 1.6 while CV²
fell below 0.5 — early enough to anticipate the crossing, not so early that noise triggers it. A
sensible trigger is two consecutive windows in the new cell, or a boundary-distance rule combined
with a hysteresis band.

**Exercise 5.** With λ = 0.0769 per week over five weeks, the expected arrival count is 0.385. The
probability of no arrival is exp(−0.385) = 0.681. The 95% quantile of the compound distribution is
15 units, because a single occurrence can be 15 units and occurs with probability 0.262 × 0.25 =
0.066. A stock level of 11 gives about 0.91; a level of 3 gives about 0.81. At EUR 900 per unit and
40 locations, the top-to-middle gap is 8 units per location, 320 units, EUR 288,000; the
middle-to-bottom gap is 4 units per location, 160 units, EUR 144,000. The second figure is the cost
of the normal-approximation error alone, holding the service target fixed.

**Exercise 6.** Look for: the window length; the dispersion denominator; whether stock-out zeros
are masked; the level at which the class is computed; whether the cut-offs were derived locally;
whether any cell has a challenger method; whether the rule is versioned; whether overrides are
logged. Three good questions: "What window and bucket produced these labels?"; "Which zeros were
masked as stock-outs, and how do you know?"; "What was the last backtest of the routing rule, on
what horizon, and who signed it off?"

**Exercise 7.** The specification should include a rule identifier and semantic version; an
effective date; a backtest record; a minimum-occurrence threshold; a boundary-distance tolerance;
an exception queue with an owner and a service-level target for clearing it; an override log with
user, timestamp, old method, new method, and reason code; a monthly drift report; and a change
board that must see a rolling-origin evaluation before a rule change ships. Where the router and a
planner disagree, the planner wins for the current cycle and the disagreement is logged for review
— otherwise the planner will route around the system entirely.

**Exercise 8.** The pro-routing case: heterogeneous portfolios, published evidence that class
membership predicts which method wins, explainability to planners, and a clean override story. The
anti-routing case: combinations are competitive in published comparisons, the router adds a
failure point, the cut-offs were derived for another purpose, and model selection barely beats a
robust default. A strong memo recommends routing with a small pool per cell rather than a single
method, tests it against a combination benchmark, and states that the recommendation reverses if
the combination wins by more than the cost of the added governance.

**Exercise 9.** The censoring rate is the share of zero item-location-weeks where on-hand stock was
zero or negative at some point in the week. Typical results in distribution businesses run from a
few percent to more than a quarter, concentrated in the slow-moving cells. Recomputing ADI with
masked weeks lowers it; the size of the fall is the measure of how much the classification was
being driven by availability rather than demand.

**Exercise 10.** The five slides: the portfolio shape and why one method fails it; the measurement
frame and its conventions; the routing table and its provenance; the backtest and its uncertainty;
the governance and the accountability. On thresholds, cite their origin, state that they are a
rounded boundary from a two-estimator comparison, and show the local re-derivation. On
accountability, name the demand-planning owner and the change board.

## 14. Interview questions

**1. "What is the Syntetos–Boylan classification and what is it for?"**

A strong answer describes two axes, four cells, and — critically — states that it is a
model-selection device derived from a comparison of two estimators, not a taxonomy of demand. It
mentions that the cut-offs are conventions. A weak answer recites 1.32 and 0.49 and calls them
thresholds for intermittent demand, revealing that the candidate has memorised the table without
the provenance.

**2. "Why is ADI not comparable across time buckets?"**

A strong answer explains that ADI counts periods, so lengthening the period lowers it
mechanically, and gives an example: four demands in 52 weeks is ADI 13 weekly and ADI 3 monthly.
A weak answer says ADI is "an average interval" and does not engage with the unit.

**3. "A series has ADI 1.31 and CV² 0.50. Which cell, and what do you do?"**

A strong answer places it in erratic, notes that both coordinates are within a hair of a
boundary, and refuses to commit without the boundary distance, the conventions, and a backtest.
It suggests routing to a method that is robust across the boundary. A weak answer confidently
names a cell and a method.

**4. "Why might classifying at item-national level be wrong?"**

A strong answer explains that aggregation reduces ADI and usually reduces CV², so a lumpy
item-location series can appear smooth nationally, and that a national method applied locally
answers the wrong question. A weak answer says national data is "more stable" and stops.

**5. "How do stock-outs corrupt the classification?"**

A strong answer distinguishes structural, true, and censored zeros, explains that censored zeros
inflate ADI and CV², and describes the self-reinforcing loop through the stocking policy. A weak
answer says "bad data causes bad forecasts".

**6. "Where did 1.32 and 0.49 come from?"**

A strong answer describes the Croston-versus-SBA comparison, the automotive sample, the
non-linear boundary, and the Kostenko–Hyndman re-derivation to 4/3 and 1/2. It states that the
rounded values survived for usability. A weak answer says they are "empirical thresholds from the
literature" and cannot say what they separate.

**7. "How would you evaluate a routing rule?"**

A strong answer says: on decision loss, not classification accuracy — stock, service, cost — with a
rolling-origin backtest inside each cell, intervals on the cell-level estimates, and a holdout the
selection never saw. A weak answer says "compare accuracy before and after".

**8. "Your router sends an item to SBA and the planner disagrees. What happens?"**

A strong answer has a process: the planner's method is used for the cycle, the override is logged
with a reason code, and the disagreement is reviewed against the backtest. It notes that
suppressing overrides produces workarounds outside the system. A weak answer says the system
should win because it is objective.

**9. "Why do you need a separate lifecycle flag if you have a classifier?"**

A strong answer explains that ADI and CV² measure interval and dispersion, not direction, so a
declining item that sells every month is smooth. It mentions NPI cold start and EOL decline as
cases the grid cannot see. A weak answer says the classifier handles it because the class changes.

**10. "What is the strongest argument against classification-based routing?"**

A strong answer gives the combination argument: published comparisons find ensembles competitive
or better, routing adds a failure point, and model selection frequently fails to beat a robust
default. It then states what evidence would change its mind. A weak answer defends routing
because it is "best practice".

**11. "You have 40,000 series and a cell with 24 members. Can you tune the method for that cell?"**

A strong answer computes the standard error — roughly 8 percentage points at a true win rate of
80% with n = 24 — and says no, or says yes only with shrinkage toward the portfolio default. A
weak answer says yes if the backtest shows an improvement.

## 15. Key terms

- **ADI** — average demand interval, in periods per demand; the vertical axis of the grid.
- **CV²** — squared coefficient of variation of nonzero demand sizes; the horizontal axis.
- **Smooth cell** — frequent demand, stable size; a level-estimation problem.
- **Erratic cell** — frequent demand, unstable size; a variance problem.
- **Intermittent cell** — sporadic demand, stable size; an occurrence problem.
- **Lumpy cell** — sporadic demand, unstable size; occurrence and size together.
- **Router** — the rule that maps a series to a method family; a model in its own right.
- **Method pool** — the admissible candidates for a cell, and the policy that admits them.
- **Class-conditional evaluation** — accuracy measured inside a cell, with its own uncertainty.
- **Boundary distance** — how far a series sits from a cut-off, in axis units.
- **Borderline series** — a series inside a tolerance of a cut-off; route to a robust default.
- **Hysteresis band** — the dead zone that prevents a series from flipping cells on noise.
- **Class drift** — genuine or measurement-induced movement between cells over time.
- **Static classification** — a label assigned once; the default in most systems.
- **Dynamic classification** — a label recomputed on a schedule or on a trigger.
- **Artificial zero** — a zero caused by unavailability; censored demand.
- **Structural zero** — a zero caused by the item not being orderable.
- **Censoring rate** — the share of zero periods that were stock-outs.
- **Cold start** — a series too short to classify; an exception, not a cell.
- **Supersession** — the predecessor link that lets a new item inherit a history.
- **NPI** — new product introduction; the phase where classification is least reliable.
- **EOL** — end of life; the phase the grid cannot detect, because it has no direction.
- **SBA** — Syntetos–Boylan Approximation, Croston corrected by (1 − α/2).
- **Croston** — separate smoothing of intervals and sizes, forecast as their ratio.
- **TSB** — Teunter–Syntetos–Babai; smooths the occurrence probability every period.
- **Hurdle model** — occurrence probability multiplied by a conditional size distribution.
- **ADIDA** — aggregate–disaggregate intermittent demand approach; changes the bucket, not the method.
- **Override log** — the record of human changes to the routed method, with reason codes.
- **Rule version** — the identifier stamped on every forecast produced by a given router.
- **Decision loss** — the cost the forecast is meant to reduce; the only valid routing criterion.

## 16. Sources

**Peer-reviewed**

- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2005). "On the categorization of demand
  patterns." *Journal of the Operational Research Society*, 56(5), 495–503. DOI:
  10.1057/palgrave.jors.2601841. Grade A.
- Kostenko, A. V., and Hyndman, R. J. (2006). "A note on the categorization of demand patterns."
  *Journal of the Operational Research Society*, 57(10), 1256–1257. Author copy at
  robjhyndman.com. Grade A.
- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2006). "Reply to Kostenko and Hyndman."
  *Journal of the Operational Research Society*, 57, 1257–1258. DOI: 10.1057/palgrave.jors.2602182.
  Grade A.
- Teunter, R. H., and Duncan, L. (2009). "Forecasting intermittent demand: a comparative study."
  *Journal of the Operational Research Society*. DOI: 10.1057/palgrave.jors.2602569. Grade A.
- Petropoulos, F., and Kourentzes, N. (2014). "Forecast combinations for intermittent demand."
  *Journal of the Operational Research Society*. DOI: 10.1057/jors.2014.62. Grade A.
- Wallström, P., and Segerstedt, A. (2010). "Evaluation of forecasting error measurements and
  techniques for intermittent demand." *International Journal of Production Economics*, 128(2),
  625–636. Grade A.
- Bacchetti, A., and Saccani, N. (2012). "Spare parts classification and demand forecasting for
  stock control: investigating the gap between research and practice." *Omega*, 40(6), 722–737.
  DOI: 10.1016/j.omega.2011.06.008. Grade A.
- Bacchetti, A., Plebani, F., Saccani, N., and Syntetos, A. A. (2012). "Empirically-driven
  hierarchical classification of stock keeping units." *International Journal of Production
  Economics*, 143(2), 263–274. Grade A.
- Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., and Assimakopoulos, V. (2011).
  "An aggregate–disaggregate intermittent demand approach (ADIDA) to forecasting." *Journal of the
  Operational Research Society*, 62(3), 544–554. DOI: 10.1057/jors.2010.32. Grade A.
- Babai, M. Z., Ali, M. M., and Nikolopoulos, K. (2012). "Impact of temporal aggregation on
  stock-control performance for intermittent demand items." *Omega*, 40(6), 713–721. Grade A.
- Boylan, J. E., and Babai, M. Z. (2016). "On the performance of overlapping and non-overlapping
  temporal demand aggregation approaches." *International Journal of Production Economics*, 181,
  136–144. Grade A.
- Rostami-Tabar, B., Babai, M. Z., and Syntetos, A. A. (2022). "To aggregate or not to aggregate:
  forecasting of finite autocorrelated demand." *Journal of the Operational Research Society*. DOI:
  10.1080/01605682.2022.2118631. Grade A.
- Kourentzes, N., Rostami-Tabar, B., and Barrow, D. K. (2017). "Demand forecasting by temporal
  aggregation: using optimal or multiple aggregation levels?" *Journal of Business Research*, 78,
  1–9. Grade A.
- "Dynamic model selection based on demand pattern classification in retail sales forecasting."
  *Mathematics*, 10(17), 3179 (2022). Grade A.
- Solis, A., Longo, F., Nicoletti, L., and Yemialyanava, K. (2013). "Applying modeling and
  simulation to evaluate statistical accuracy and inventory control performance of lumpy demand
  forecasting methods." *MAS 2013 proceedings*. Grade B: conference proceedings, simulation-heavy.
- Doszyn, M. (2020). "Accuracy of intermittent demand forecasting systems in the enterprise."
  *European Research Studies Journal*. Grade B: single warehouse, more than 16,000 items.
- Aktas, E., Chomachaei, F., and Golmohammadi, A. (2025). "Life cycle prediction: a comparison of
  methods for lighting products." Cranfield University repository. Grade A: verify the final
  journal version before citing the reported accuracies.

**Preprints and working papers**

- Svetunkov, I., and Sroginis, A. "Why do zeroes happen? A model-based approach for demand
  classification." arXiv:2504.05894. Grade A for method, with the preprint caveat: not yet
  peer-reviewed at the time of writing.
- Kourentzes, N. (2014). "On intermittent demand model optimisation and selection." Lancaster
  University eprint 69640. Grade A.

**Standards and official**

- GS1 General Specifications, current edition: GTIN, GLN, and the EPCIS 2.0 event standard. Grade B.
- ISO 8601, date and time representation, including week numbering. Grade B.
- ANSI X12 852, product activity data. Grade B.
- ISA-95 / IEC 62264, enterprise–control system integration. Grade B.
- SCOR Digital Standard, ASCM. Grade C: a reference framework, not a regulation.
- IAS 2 *Inventories*; ASC 330 *Inventory*. Grade B.

**Practitioner, vendor, and commentary**

- Oracle, *Demand Planning Implementation and User's Guide*, product lifecycle and event modelling
  sections. Grade B for documented behaviour; no grade for efficacy.
- AWS Supply Chain, "Improving demand planning for products with varying lifecycles," blog.
  Grade D.
- Svetunkov, I., "Intermittent demand classifications: is that what you need?" and "SBC is not for
  you!", OpenForecast, 2024 and 2025. Grade D: expert commentary, not peer-reviewed.
- Umbrex, *Inventory Management Playbook*, demand segmentation section. Grade D.
- Covirage, "Demand forecasting over product life cycle stages," 2026. Grade D.

**Note on citations.** Bibliographic details are given as the sources were returned by search and
have not been checked against publisher records line by line. Where a DOI is given it was returned
with the record; where a venue or year is uncertain, the entry says so rather than supplying a
false precision. The 1.32 and 0.49 values, the 4/3 and 1/2 re-derivation, and the four-cell scheme
are reported from the published exchange described in section 5.1; read the original papers before
relying on the constants in a regulated context.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Previous topic in this module: [`04_temporal-cross-temporal.md`](04_temporal-cross-temporal.md)
- Next topic in this module: [`06_compound-poisson-hurdle.md`](06_compound-poisson-hurdle.md)
- Hierarchy structures, which set the levels this topic classifies at: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- Aggregation consistency, which governs what reconciliation may change: [`02_aggregation-consistency.md`](02_aggregation-consistency.md)
- Probabilistic forecasts, which consume the distribution this routing assumes: [`07_probabilistic-forecasts.md`](07_probabilistic-forecasts.md)
- The method mechanics this topic routes to: [`../07_time_series_forecasting/06_intermittent-demand.md`](../07_time_series_forecasting/06_intermittent-demand.md)
- The metrics this topic assumes: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Lead-time demand distributions, the bridge to inventory policy: [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Part I index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Part II index: [`../_index.md`](../_index.md)
- Module survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
