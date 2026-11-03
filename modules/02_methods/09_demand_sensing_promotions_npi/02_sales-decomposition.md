# Sales Decomposition — baseline, seasonality, promotion, distribution, and price separated explicitly

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topic 04; Module 09 topic 01
> Version 2026.5

---

## 1. Why this topic exists

A sales number is not a fact. It is a sum of causes that have been added together and reported as
one figure, and almost every argument in a commercial review is really an argument about which
cause owns which part of it.

A brand closes the quarter nine percent above last year. The commercial director credits the new
promotional plan. The category manager credits a distribution gain in the discount channel.
Finance credits a list-price increase taken in February. The demand planner says the quarter was
soft and only looked strong because Easter fell three weeks later. All four are describing the
same nine percent, and at most one of them can be substantially right. Without an explicit
decomposition the argument is settled by seniority, and next quarter's plan is built on whoever
won.

The cost is not rhetorical. Four decisions depend on the split, each with a different owner:

- A **promotion** that produced 300 units of lift justifies repeating it. One that produced 60
  units of lift and 240 units of pulled-forward demand does not.
- A **distribution** gain that produced 55 units is a sales-force achievement and belongs in the
  account plan. The same 55 units misread as promotion lift will be charged to trade spend and
  the account work will be cut.
- A **price** increase that cost 49 units is a decision already taken. The same 49 units attributed
  to "the market" will be re-litigated every month.
- A **baseline** drifting down under a stable headline is the only signal that the brand is
  weakening. It is also the component most easily hidden, because nobody owns it.

**Why decomposition is hard rather than merely tedious.** Each component is defined by a
counterfactual, and a counterfactual is a choice, not a measurement. "Baseline" means "what we
would have sold if we had not promoted". That sentence requires you to say what "not promoted"
means for price, shelf availability, competitor response, and the shoppers who would have bought
something else. Every one of those choices changes the number. Two competent analysts can
decompose the same week and disagree by thirty percent on the promotion contribution, and both can
be right about their own question.

A second difficulty sits on top, and it is what this topic is really about. **The components are
correlated in the data by construction.** Retailers do not promote at random. They promote when
volume is soft, on items where the margin allows it, with a display the store agreed to, at a
price the shopper will notice, with extra stock in the back room. The promotion week is
simultaneously a price week, a display week, a feature week, and a stock week. A regression with
all of those columns will fit beautifully and will not have separated anything, because the
columns do not vary independently. This is the identification problem, and it is not a technicality
that better software solves. It is a property of how promotions are executed.

> **Note:** This topic does not re-teach seasonal regression. Module 07 topic 04 covers Fourier
> terms, distributed lags, and regressor-path projection, and it is assumed. What is new here is
> the *partition*: which components exist, how they are defined before they are estimated, and how
> the resulting table is made to sum and to be audited.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the five components — baseline, seasonality, promotion, distribution, and price/mix
   — in writing, with a stated counterfactual for each, before any model is estimated.
2. **Diagnose** whether a series demands an additive or a multiplicative form, using three named
   diagnostics, and **state** the evidence each provides.
3. **Build** a confounding table that records which drivers co-move, and **identify** which
   components the available variation cannot separate.
4. **Place** a problem on the five-rung ladder — ratio-to-moving-average, STL (seasonal-trend
   decomposition using locally weighted regression, LOESS), regression with dummies, state-space,
   hierarchical Bayesian — and **justify** the rung by what it buys and costs.
5. **Compute** numeric and weighted distribution, **convert** each into an availability factor, and
   **explain** why they disagree when a launch expands into smaller stores.
6. **Separate** price from mix using a Laspeyres, Paasche, or Fisher index, and **demonstrate** the
   index-number trap in which a mix shift is reported as a price change.
7. **Produce** a table whose component contributions sum to the observed actual in every period,
   with a stated attribution convention where the model is multiplicative.
8. **Read** a structured residual and **name** which component it implicates, rather than filing it
   under "other".

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a single observed number — one stock-keeping unit (SKU), one week — at the top, a
ladder of increasingly demanding methods descending from it, and a hard break in the middle where
two components stop being separable with the data in hand.

```text
                       OBSERVED SALES  y(t)        units per SKU-week
                                 |
        +------------------------+------------------------+
        |                        |                        |
   [ RUNG 1 ]               [ RUNG 2 ]               [ RUNG 3 ]
   ratio-to-moving-         STL (LOESS)              regression with
   average; classical       additive, robust,        promo / holiday
   Census lineage           time-varying season      dummies + Fourier
        |                        |                        |
        +------------------------+------------------------+
                                 |
                           [ RUNG 4 ]
                           state-space / UCM
                           time-varying baseline;
                           components carry variance
                                 |
                           [ RUNG 5 ]
                           hierarchical Bayesian
                           partial pooling across
                           store / SKU / region
                                 |
                                 v
              +--------------------------------------+
              |     COMPONENT TABLE, PER PERIOD      |
              |   B + S + D + P + Pr + e  =  y       |
              |   every row sums to the actual       |
              +--------------------------------------+

   >> IDENTIFICATION BREAK — sits between rung 2 and rung 3
   ------------------------------------------------------------
   In a single series, promotion weeks are also the weeks with
   a price cut, a feature, a display, and a stock build. The
   columns [promotion], [price], [display] are collinear:
   rank(X) < k, and the split among the three is arbitrary.
   Rungs 3 and above separate them ONLY when the design carries
   variation that is not common to all three — a store panel,
   a split promotion calendar, or an experiment.
```

Four properties of this picture explain most of what follows.

**Property 1 — The ladder is ordered by what it can separate, not by accuracy.** Rungs 1 and 2
produce a trend and a seasonal shape. They cannot produce a promotion effect, because no variable
identifies one: a promotion week is simply a week where the remainder is large. Rung 3 introduces
covariates and therefore the possibility of separation, but only if they vary independently. Rungs
4 and 5 add time-varying parameters and partial pooling, which improve the baseline and the
seasonality and do nothing at all for identification. Buying a better model does not buy
identification. Only the design does.

**Property 2 — Every rung produces a decomposition; only some produce an auditable one.** Rung 1
will report a seasonal index for every week of the year from two years of data. Those indices are
computed, not estimated, and the promotion weeks have contaminated them. The table sums. The table
is wrong in a way the sum cannot reveal. Additivity is a property of arithmetic, not of truth.

**Property 3 — The baseline is the residual of everything you declined to model.** Analysts tend
to estimate the baseline as "what is left after seasonality and promotion", which makes the
baseline the garbage bin. The professional convention is the reverse: define the baseline first,
as a counterfactual object with a stated definition, and let the other components be deviations
from it. A baseline defined last is a residual with a better name.

**Property 4 — Distribution enters multiplicatively and dominates early in a product's life.**
Seasonality, promotion, and price move demand by percentages of a base that exists because the
product is on a shelf. Availability does not move the percentage; it moves the base. Going from
62% to 65% numeric distribution is a 4.8% change in the ceiling on everything else. For a mature
product that is a rounding error next to a promotion. For a launch in its first two quarters it is
the largest term in the table, and it is the term most often missing, because it lives in a
syndicated distribution file rather than in the sales table.

### 3.1 Why the table must sum

A decomposition that does not sum is a set of correlations with a common label. The summing
constraint forces three disciplines that are otherwise easy to skip.

It forces you to **allocate the interaction**: either report contributions in logs, where they add
exactly, or choose an attribution rule in units and state it.

It forces you to **keep a residual**. A table that sums by construction has no error term, and a
model with no error term cannot be diagnosed. A zero residual in every period means the residual
was absorbed into whichever component was estimated last.

It forces the **definitions to be frozen**. If the baseline means one thing in January and another
in June, the table still sums in both months and the difference between them is meaningless.

### 3.2 Three baseline conventions, three different numbers

The same week can be decomposed into a baseline share of 62%, 48%, or 71% by three functions in the
same company. All three are arithmetically correct; what differs is the boundary of the bucket.

| Convention | Baseline contains | Baseline excludes | Typical owner | Typical share |
|---|---|---|---|---|
| Demand planning | Trend, seasonality, regular price, current distribution | Temporary promotion, display, feature | Demand planning | 60–75% |
| Marketing mix modelling | Trend, season, intercept, often media carry-over | Price, promotion, distribution (treated as steerable) | Marketing analytics | 45–60% |
| Finance variance | Last year or plan re-priced to current volume | Everything that moved | FP&A | 70–85% |

The gap between the first two rows is the definitional dispute that runs through the field, and it
is not a disagreement about method. It is a disagreement about which decision the number informs. A
marketing budget decision wants price and distribution outside the baseline because marketing can
influence both. A replenishment decision wants them inside because the planner cannot change either
next month.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect
this topic exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Sales | The observed quantity or value cleared in a period, after returns and cancellations | Used to mean demand, shipments, or orders | Units per SKU-store-week, or currency |
| Decomposition | The partition of an observed series into components with stated definitions that sum to it | Used to mean any factor analysis | Vector of components plus residual, per period |
| Component contribution | The part of the observed value attributed to one component in one period | Reported as a coefficient | Units or currency, per period |
| Baseline | The counterfactual sales under a stated set of conditions with named drivers switched off | Estimated last, from whatever is left | Units per period, counterfactual stated |
| Base volume | Retail-measurement term for sales excluding temporary promotion, at observed distribution and regular price | Used interchangeably with baseline without stating the counterfactual | Units |
| Seasonality | The recurring calendar pattern at a fixed period, net of trend and one-off events | Used for any within-year variation, including holidays | Ratio (multiplicative) or units (additive) |
| Seasonal index | The seasonal factor for one position in the cycle | Reported without the period it applies to | Ratio, mean 1 over a full cycle |
| Trend | The slow-moving level, net of seasonality | Confused with the annual total | Units per period |
| Residual | The part of the observation that no component explains | Used as a dumping ground labelled "other" | Units, or log units |
| Promotion lift | Incremental volume attributable to a promotional event after baseline removal | Equated with promoted volume | Units per event |
| Incremental volume | Volume that would not have occurred without the intervention | Confused with the promoted total | Units |
| Temporary price reduction (TPR) | A short-dated reduction from the regular shelf price | Treated as a price change | Percent off regular, and duration in days |
| Feature | Promotion of an item in a retailer circular, leaflet, or app | Used to mean any advertising | Binary flag, plus share of page or position |
| Display | Additional in-store placement beyond the normal shelf | Confused with shelf facings | Binary flag, plus display type code |
| Numeric distribution (ND) | Share of stores in the universe carrying the item | Used as the only availability driver | Percent of stores |
| Weighted distribution (WD) | Share of category volume covered by stores carrying the item | Confused with ND | Percent of all-commodity volume (ACV) or category volume (PCV) |
| All-commodity volume (ACV) | Total store sales across all categories, used as a store weight | Treated as the item's own sales | Currency per store per period |
| Product category volume (PCV) | Store sales within the relevant category, an alternative weight | Mixed with ACV inside one weighting | Currency per store per period |
| Total distribution points (TDP) | Breadth times depth: the sum of WD across items carried | Reported as if it were weighted distribution | Percentage points summed across items |
| Availability factor | The multiplier converting a baseline at reference distribution to achieved distribution | Ignored, or folded into the baseline | Ratio, 1.0 at reference distribution |
| Rate of sale (ROS) | Sales per carrying store per period | Computed on total stores rather than carrying stores | Units per store per week |
| Price elasticity | Proportional volume change per proportional price change | Quoted from a regression on realised price | Dimensionless, typically negative |
| Mix effect | The part of a value or unit change caused by a shift in what sold | Reported as a price effect | Currency, or percent of base value |
| Average selling price (ASP) | Revenue divided by units | Reported as a price index | Currency per unit |
| Laspeyres index | A price index using base-period quantities as weights | Used without stating the base period | Ratio, base = 1 |
| Paasche index | A price index using current-period quantities as weights | Compared directly with a Laspeyres index | Ratio, current weights |
| Fisher index | The geometric mean of the Laspeyres and Paasche indices | Treated as the only correct index | Ratio |
| Additive decomposition | A form in which components sum to the observation | Chosen without testing | Units |
| Multiplicative decomposition | A form in which components multiply to the observation | Chosen because retail data "looks seasonal" | Ratios |
| Log-additive attribution | Reporting multiplicative contributions as logs, where they sum exactly | Converted to units without stating the conversion | Natural log units |
| Shapley attribution | Averaging a component's marginal contribution over all orderings | Presented as the unique correct answer | Units per period |
| Identification | The property that available variation can distinguish parameters from one another | Used to mean "the model converged" | Property of the design, not the estimator |
| Confounding | Two drivers moving together so their separate effects cannot be recovered | Used loosely for any correlation | Property of the data-generating process |
| Counterfactual | The unobserved state against which the observed is compared | Left implicit | A stated scenario, not a number |
| STL | Seasonal-Trend decomposition using LOESS: an iterative smoother yielding trend, seasonal, and remainder | Treated as a causal model | Three series, additive |
| Unobserved components model (UCM) | A state-space model with latent level, trend, seasonal, and regression effects | Confused with a regression with ARMA errors | State vector plus variance parameters |
| Hierarchical Bayesian | A model in which unit-level parameters are drawn from a group-level distribution | Used as a synonym for "regularised" | Prior plus hyperprior, with posterior intervals |
| Phantom spike | A spurious baseline peak appearing when the estimator is correlated with promotion activity | Mistaken for a real demand event | Diagnostic artifact |

Three carry most of the interpretive weight, and each is a place where a competent analyst loses
an argument to a less competent one.

**Baseline versus base volume.** Base volume is a retail-measurement term with a specific, narrow
counterfactual: no temporary price reduction, no feature, no display, at observed distribution and
regular price. Baseline is a general term whose counterfactual must be stated every time it is
used. The two coincide only when the definitions match. A demand-planning baseline that also
removes a permanent price change is not base volume, and comparing it with a syndicated base volume
series produces a gap that is entirely definitional.

**ND versus WD.** Numeric distribution counts stores; weighted distribution counts volume. A brand
at 65% ND and 80% WD is in two thirds of stores accounting for four fifths of category volume. A
brand at 65% ND and 45% WD is in many small stores and absent from the large ones, which is a
distribution emergency behind a respectable headline. The ratio WD/ND is a diagnostic in its own
right.

**Identification versus estimation.** An estimator that converges has estimated *something*. It has
not necessarily estimated the parameter you named. If promotion, price, and display always
co-occur, a regression returns three coefficients that are stable, plausible, and jointly
arbitrary: any split reproducing the fitted values is equally consistent with the data. Standard
errors on those coefficients are not wrong arithmetically. They describe the noise conditional on a
design that cannot answer the question.

## 5. The mechanics, step by step

The order matters. Phase A defines the object; Phase B chooses the form; Phase C establishes what
can be separated at all; Phase D estimates; Phase E reports. Doing B before A models the wrong
quantity. Doing D before C gives a confident answer to an unanswerable question.

### Phase A — Define the components before estimating anything

**Step 1. Name the decision the decomposition feeds.** A promotion-ROI review, a trade-spend
reallocation, a monthly demand review, and a launch post-mortem need different definitions and
different buckets. The decision determines which counterfactual matters, and the counterfactual
determines everything else.

**Step 2. Fix the unit, the bucket, and the level of the hierarchy.** Units or currency;
store-week, SKU-week, or account-week; national, regional, or store. A national decomposition
averages away the store-level variation that identifies the components, which is why national
decompositions are so often stable and unfalsifiable. Choose the lowest level at which the decision
is made and the data are trustworthy, then aggregate upward.

**Step 3. Write one sentence of counterfactual per component.** Finish "this component is the
difference between what happened and what would have happened if ..." for each. If you cannot
finish it, the component is not defined and no amount of data rescues it.

| Component | Counterfactual sentence | What is switched off |
|---|---|---|
| Baseline | "...if nothing had changed and no promotion had run" | Everything except a stated reference configuration |
| Seasonality | "...if the calendar were at its average position" | The calendar, not the level |
| Promotion | "...if the promotion had not run, at the same price and shelf position" | TPR, feature, display, and the stock build |
| Distribution | "...if the same number and quality of stores had carried the item" | Availability, weighted by store size |
| Price/mix | "...if prices and the sales composition had stayed at the reference" | Regular price changes and product mix |

**Step 4. Decide which component is the anchor.** One must be the reference from which the others
are measured, and the choice is consequential. Anchoring on the baseline answers "how much of this
week was structural and how much was pushed". Anchoring on last year answers "what changed".
Anchoring on plan answers "did we hit it". These are different tables. State the anchor.

**Step 5. Freeze and version the definitions.** Write them into a document with a version number
and an effective date. A decomposition whose baseline definition changed in May cannot be compared
across the year, and nobody will notice, because every month's table still sums to its own actual.

### Phase B — Choose between additive and multiplicative, with evidence

**Step 6. Plot the seasonal amplitude against the level.** Compute the within-year peak-to-trough
swing for each year and the year's mean level. If the swing is flat while the level doubles, the
form is additive. If the swing roughly doubles with the level, it is multiplicative. This is the
most informative plot in the topic and it takes ten minutes.

**Step 7. Regress the log absolute residual on the log fitted value.** Fit the additive form, take
the residuals, and regress `ln|e|` on `ln(yhat)`. A slope near zero says absolute error and an
adequate additive form. A slope near one says proportional error and a multiplicative form. A
slope in between, which is common, says report both.

> **Note:** A slope near one is not proof of multiplicative seasonality in count data. Poisson and
> negative-binomial counts have variance proportional to the mean by construction. If your series
> counts transactions rather than summing continuous quantity, this diagnostic is confounded with
> the count distribution, and Step 6 should carry the argument instead.

**Step 8. Profile a Box-Cox transformation.** Read the confidence interval for the exponent rather
than the point estimate. An interval covering one supports the multiplicative form; an interval
covering zero supports the additive. The point estimate always looks decisive and the interval
often is not.

**Step 9. Choose the attribution convention if the form is multiplicative.** The factors multiply,
so contributions do not add in units. Three conventions are in use and they give different numbers
for the same week.

1. **Log-additive.** Report each `ln(factor)`. These sum exactly to the log ratio of actual to
   baseline and are order-free. Convert to percentages by exponentiating each term. Cleanest
   convention, hardest to explain to a planner who wants units.
2. **Sequential substitution.** Apply the factors one at a time in a stated order, recording each
   step's change in units. Contributions sum exactly to the actual and depend on the order, and the
   dependence is not small: section 6 shows a price contribution moving forty percent between two
   orderings.
3. **Shapley.** Average each component's marginal contribution over all orderings. Removes the order
   dependence at the cost of a computation and an explanation. For four components there are
   twenty-four orderings: trivial to compute, hard to explain in a meeting without a diagram.

Choose one, state it in the table footer, and never mix conventions across periods.

**Step 10. Fix the reference configuration.** In a multiplicative model every factor is relative to
something. Seasonality of 1.28 is relative to the annual mean; distribution of 1.048 to a reference
distribution; price of 0.968 to a reference price. If the references drift between periods, the
components move for reasons unrelated to demand. Store them beside the table.

### Phase C — Establish what can be identified

**Step 11. Build the confounding table.** List every driver that moves during the window and record
which ones co-move. This is the most valuable artefact in the topic: it tells you in advance which
components you cannot separate, and it costs an afternoon rather than a quarter.

| Driver | Moves when | Co-moves with | Independent variation available? |
|---|---|---|---|
| Promotion flag | Planned, seasonal | Price, display, feature, stock | Only if the calendar varies support |
| Shelf price | Promotion and list changes | Promotion, display | The list changes are the independent part |
| Display | Store agreement | Promotion, feature | Only across stores with different compliance |
| Feature | Retailer circular | Promotion, display | Only where feature runs without display |
| Distribution | Launch, delisting, reset | Seasonality, price | Yes, at reset dates |
| Stock availability | Supply, promotion build | Promotion | Only where out-of-stocks are recorded |

**Step 12. Check the rank and condition number of the design you intend to use.** Exact collinearity
is the easy case: the software drops a column or errors. The dangerous case is near-collinearity,
where the software returns coefficients and the split is determined by a handful of observations.
Report the condition number. Above about thirty, the coefficients on the collinear block are
unstable, and a bootstrap will spread wide enough to include both "promotion did everything" and
"price did everything".

**Step 13. Find or create variation that is not common to all drivers.** Five routes, in descending
order of evidential quality:

1. **Experiment.** A controlled in-store or geo test with randomised support. Module 05 topic 04
   covers geo experiments; Module 05 topic 03 covers cluster designs.
2. **Split calendar.** The same discount ran sometimes with feature, sometimes with display,
   sometimes with neither, in different stores or periods.
3. **Panel variation.** Store-level data where compliance differs for reasons unrelated to demand,
   such as store size or regional execution.
4. **Exogenous price.** List-price changes and cost-driven resets that were not a response to
   sell-through. These identify a price effect, not a promotion effect.
5. **Restriction.** Estimate the price elasticity on non-promoted weeks only and apply it to
   promoted weeks, stating the extrapolation.

**Step 14. State what is not identified, and report it as a bundle.** If promotion, price, and
display always move together, the honest output is a single "promotional bundle" component with a
stated composition, not three components with arbitrary splits. A correctly labelled bundle
answers "what did the event contribute". A falsely split one will be used to set the depth of the
next discount.

### Phase D — Estimate

**Step 15. Choose the rung and record why.**

| Rung | Method | Buys | Costs | Use when |
|---|---|---|---|---|
| 1 | Ratio-to-moving-average | Transparency, no software | Contaminated seasonal, endpoint loss, no identification | Teaching, or a first look |
| 2 | STL | Robust seasonal, time-varying shape, outlier resistance | Additive only, no covariates, no identification | A seasonal profile for a stable series |
| 3 | Regression with dummies and Fourier terms | Separation, if the design permits; covariates | Needs identification; specification risk | You have a calendar and a panel |
| 4 | State-space / UCM | Time-varying baseline, component uncertainty, missing data | Estimation complexity; identification unchanged | The baseline itself is moving |
| 5 | Hierarchical Bayesian | Partial pooling, honest component uncertainty | Compute, priors, explainability | Many stores or SKUs with thin histories |

**Step 16. Estimate distribution as a separate driver, not inside the baseline.** Enter availability
as a multiplicative factor or as a log regressor, never as an adjustment folded into the baseline.
Distribution changes have a different owner, cost, and lead time from everything else in the table.
Section 6 works the arithmetic; Module 09 topic 09 treats assortment and distribution in depth.

**Step 17. Separate price from mix before reporting either.** A price effect is a change in the
price of a fixed basket. A mix effect is a change in the basket. Reporting their sum as "price" is
the index-number trap, and it is common because ASP is the number that appears in the sales table.
Section 6.6 shows ASP rising 25% while every price rises 5%.

**Step 18. Read the residual before accepting the fit.** The residual is a diagnostic, not a bin.
Section 5.1 gives the reading rules. Structure in the residual means a component is misassigned, and
the response is a specification change rather than a wider label.

### Phase E — Report, reconcile, and audit

**Step 19. Produce the table that sums, with a footer.** Every row sums to the actual. The footer
states the form, the attribution convention, the reference configuration, the component definitions
with their version number, and the residual's treatment.

**Step 20. Reconcile to the financial view.** The unit and value decompositions are different
objects. Units times price is value, but value also contains mix, and the ledger is net of trade
spend while scanner data is gross. Reconcile line by line, expect a trade-spend bridge, and remember
that a decomposition which cannot be walked to the P&L will be dismissed by finance, correctly.

**Step 21. Run the five audit tests and log the results.** Row sum within tolerance; definition
stability against the previous version; restatement under last period's parameters; sign
discipline against observed ND, WD, and price; residual structure against section 5.1. Log a pass
or a fail per test with its consequence. A log with five passes usually means the tests were drawn
loosely. Section 13 Answer 7 shows the log format.

### 5.1 Reading a structured residual

A residual with structure is a component asking to be moved. Each pattern points at a specific
repair.

| Residual pattern | What it implicates | Repair |
|---|---|---|
| Spikes at promotion flags | Promotion specification too coarse: depth or support missing | Split by support and depth; topic 03 |
| Spike immediately after a promotion | Pull-forward not modelled | Add the give-back lag; topic 05 |
| Seasonal autocorrelation at lag 52 or 12 | Seasonal basis insufficient, or moving holiday unmodelled | Raise the harmonic order; add an event-keyed window |
| Residual correlated with ND or WD change | Distribution omitted or mis-specified | Add availability as a driver |
| Residual correlated with price change | Price entered as ASP rather than as an index | Rebuild the price series |
| Residual variance rising with the level | Wrong form: additive fitted to multiplicative data | Re-estimate in logs |
| Residual grouped by store or region | Hierarchy missing | Move to rung 5, or add store effects |
| Step change at a date | Structural break: reset, delisting, competitor entry | Add a break term or split the sample |
| Residual mean far from zero | Baseline reference mis-set | Re-anchor the reference configuration |
| Residual white but large | Nothing wrong with the specification; the model is weak | Report the interval honestly and stop |

The last row matters as much as the others. A large white residual is not a failure of the
decomposition; it is the decomposition reporting that the predictable share of this series is
small. The professional response is a wide interval, not more components.

## 6. Worked example

One ambient grocery SKU (stock-keeping unit) in one region, weekly. All figures are illustrative
but internally consistent and reproducible by hand.

### 6.1 The situation

Week 34 of the second year. Observed sales are 1,486 units. The region has 1,000 stores in the
universe. Numeric distribution rose from 62% to 65% at a July reset. A feature and display
promotion ran in week 34 at 20% off regular price. Regular price rose 3% at the start of the
quarter. The week sits on a seasonal peak.

### 6.2 The components, defined first

| Component | Definition | Counterfactual | Reference |
|---|---|---|---|
| Baseline B | Level with no promotion, at reference distribution and reference price, de-seasonalised | No promotion, 62% ND, pre-increase price | 1,095 units/week |
| Seasonality S | Ratio of the week's expected level to the annual mean, from a three-harmonic fit on non-promoted weeks | Calendar at its annual mean | 1.000 |
| Distribution D | Ratio of achieved availability to reference availability | 62% ND, 78% WD | 1.000 |
| Promotion P | Multiplier for TPR, feature, and display together | No TPR, no feature, no display | 1.000 |
| Price Pr | Multiplier from the regular price change at elasticity −1.1 | Pre-increase price | 1.000 |

The baseline is stated first and numerically; that ordering is what stops it becoming the residual.

### 6.3 The distribution arithmetic

The reset took ND from 62% to 65% and WD from 78% to 80%. On 1,000 stores, 620 carrying stores
became 650: a ratio of 1.0484. The weighted measure implies 1.0256. The two disagree, and the
disagreement is information rather than error.

```text
ND ratio   = 650 / 620 = 1.0484      stores carrying, +4.84%
WD ratio   =  80 /  78 = 1.0256      share of category volume, +2.56%

implied size of the average new store, indexed to the existing average:
  (80 - 78) / (650 - 620 in ND points) = 2.00 / 3.00 = 0.667
```

A new store carries about two thirds of the average store's category volume. Which ratio enters
the model depends on the mechanism you believe. If volume scales with the store's category size,
use WD and the factor is 1.0256. If stores are homogeneous and the item simply needs to be present,
use ND and the factor is 1.0484. The honest treatment is to enter both and let the data split the
contribution, which is possible only when the two move differently, as they do here. When they move
together, report one availability component. This week uses the ND-based factor of 1.0484, and the
residual will show whether that was right.

### 6.4 The multiplicative decomposition

```text
B  = 1,095.0 units/week
S  = 1.042      week sits 4.2% above the annual mean
D  = 1.0484     ND 62% -> 65%
P  = 1.280      TPR 20% + feature + display, from the promotion model
Pr = 0.968      3% regular price rise at elasticity -1.1:  1.03^(-1.1) = 0.968

predicted = 1,095.0 x 1.042 x 1.0484 x 1.280 x 0.968 = 1,482.1 units
actual    = 1,486.0 units
residual  = +3.9 units, or +0.26% of actual
```

The elasticity check: `1.03^(-1.1) = exp(-1.1 x 0.029559) = exp(-0.032515) = 0.9680`. A 3% price
increase costs 3.2% of volume at this elasticity, about 49 units on a running base of 1,531.

### 6.5 The table that sums, in units

Sequential substitution in the order baseline, seasonality, distribution, promotion, price.

| Line | Computation | Units | Running total |
|---|---|---:|---:|
| Baseline | stated reference | 1,095.0 | 1,095.0 |
| Seasonality | 1,095.0 x 0.042 | +46.0 | 1,141.0 |
| Distribution | 1,141.0 x 0.0484 | +55.2 | 1,196.2 |
| Promotion | 1,196.2 x 0.280 | +334.9 | 1,531.1 |
| Price | 1,531.1 x (−0.032) | −49.0 | 1,482.1 |
| Residual | actual minus predicted | +3.9 | 1,486.0 |
| **Total** | | **1,486.0** | equals actual |

The table sums, and it is still not unique. Reversing the order moves the price contribution from
−49.0 to −35.1 units, forty percent, because the promotion multiplier is applied to a different
base, and promotion moves from 334.9 to 324.2. The total is identical, as it must be. No estimator
fixes this; it is a property of decomposing a product into a sum. Section 7 computes both orderings
and the Shapley average that removes the dependence.

### 6.6 The index-number trap in the value view

The same SKU in value rather than units, two years, two pack sizes. This is where a mix shift is
most often reported as a price change.

| Item | Year 1 price | Year 1 units | Year 2 price | Year 2 units |
|---|---:|---:|---:|---:|
| Pack A | 2.00 | 600 | 2.10 | 400 |
| Pack B | 5.00 | 200 | 5.25 | 300 |
| Total | | 800 | | 700 |
| Value | | 2,200 | | 2,415 |
| ASP | | 2.750 | | 3.450 |

ASP rose from 2.750 to 3.450, or 25.45%. Every individual price rose by exactly 5%. The gap is mix:
the expensive pack went from 25% of units to 43%.

```text
value change    2,415 - 2,200                       = +215
price variance  (2.10-2.00)x400 + (5.25-5.00)x300
              = 40 + 75                            = +115
volume variance (400-600)x2.00 + (300-200)x5.00
              = -400 + 500                         = +100
check           115 + 100                          = +215   exact

of which:
  quantity      700 units at base mix, base price: 1,925 - 2,200  = -275
  mix           2,300 - 1,925                                    = +375
  check         -275 + 375                                       = +100   exact
```

Read the last lines carefully. Unit volume fell 12.5%, and the constant-price value of that volume
fell by 275; mix added 375 because the units that remained were the expensive ones. A report
showing "volume −275, price +115" has hidden the largest effect, and "ASP +25.5%" has attributed it
to pricing. Section 13 Answer 5 works the index forms where the two prices move differently.

> **Note:** Only a mismatched pairing — a Laspeyres price index with a Paasche quantity index —
> decomposes value exactly. State the pairing in the table footer.

### 6.7 The annual table

The same decomposition for four quarters, in units. Component columns are contributions, not
levels, except baseline, which is the reference level.

| Quarter | Baseline | Seasonality | Distribution | Promotion | Price | Residual | Actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 13,900 | −820 | +140 | +1,240 | −410 | +95 | 14,145 |
| Q2 | 14,100 | +610 | +310 | +980 | −430 | −60 | 15,510 |
| Q3 | 14,300 | +1,120 | +560 | +1,610 | −440 | +110 | 17,260 |
| Q4 | 14,400 | −910 | +180 | +2,050 | −450 | −70 | 15,200 |

The planner reads the residual first: Q1 and Q3 carry positive unexplained volume, Q2 and Q4
negative, and the pattern tracks the promotion weeks one quarter back — a pull-forward signature,
and the reason the next calendar is re-cut rather than re-forecast.

## 7. Code, query, or calculation

The code below is a hand decomposition of one SKU-week into five components, written in Python
with only the standard library so it runs anywhere and shows the arithmetic rather than hiding it
behind a package call. It assumes the multiplicative factors from section 6.4, that contributions
are requested in units rather than logs, and that the attribution convention must be stated. It
prints the sequential contributions under one ordering, the log contributions, and the Shapley
average, so the order dependence is visible in the output rather than asserted in prose.

```python
import math
from itertools import permutations
base = 1095.0                                      # units/week, reference config
mult = {"seasonality": 1.042, "distribution": 1.0484,
        "promotion": 1.280, "price": 0.968}        # ratios, not units
def seq(order):
    lvl, out = base, {}
    for k in order:
        out[k] = lvl * (mult[k] - 1.0)             # contribution in units
        lvl *= mult[k]
    return out, lvl
c, pred = seq(["seasonality", "distribution", "promotion", "price"])
print({k: round(v, 1) for k, v in c.items()}, round(pred, 1))
# {'seasonality': 46.0, 'distribution': 55.2, 'promotion': 334.9, 'price': -49.0} 1482.2
print({k: round(math.log(v), 5) for k, v in mult.items()})
# {'seasonality': 0.04114, 'distribution': 0.04727, 'promotion': 0.24686, 'price': -0.03252}
shap = {k: sum(seq(p)[0][k] for p in permutations(mult)) / 24.0 for k in mult}
print({k: round(v, 1) for k, v in shap.items()})
# price about -42.0, promotion about 329.5, between the two sequential endpoints
```

Expected output is the three printed lines shown as comments: the sequential split, the log split
that sums to `ln(1482.2 / 1095.0) = 0.3028`, and the Shapley split that sits between the sequential
endpoints. The prediction rounds to 1,482.2 against the 1,482.1 in section 6.4, a 0.1-unit
difference from rounding the factors to four decimals.

## 8. Data, standards, and regulatory context

A decomposition is only as good as the join keys underneath it. Four reference systems matter, and
each constrains what the data can support.

### 8.1 Time, calendar, and the retail week

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 8601 | Date and week representation | The week key joining sales, calendar, and distribution files | B |
| National Retail Federation 4-5-4 calendar | Retail fiscal weeks | The week index, including the 53rd week every five to six years | C |
| National statistical office holiday calendars | Public holidays by jurisdiction | The event dates used to build holiday regressors | B |
| IANA time zone database | Local time offsets | Which local date a store's trading day belongs to | B |

The 4-5-4 calendar is a convention with accounting consequences, not a standard. Two retailers can
disagree about which week contains a date in the same year, and a decomposition built on one and
reconciled to the other shows a one-week misalignment that looks exactly like a demand anomaly.
State the calendar in the table footer.

### 8.2 Product, location, and availability data

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identification | That the SKU in the sales file and the SKU in the distribution file are the same object | B |
| GS1 GLN | Party and location identification | Store identity across retailer portals and syndicated data | B |
| GS1 EPCIS 2.0 | Event data | What was where, when: the basis for reconstructing availability from events | B |
| Retailer item master | Retailer-specific item and store keys | The crosswalk from internal SKU to retailer item, which is never stable | C |

The crosswalk is where most decompositions quietly fail. A reset changes the item number without
changing the product, the distribution feed keeps the old number for a quarter, and the
availability series shows a delisting that never happened. Reconcile the crosswalk at every reset.

### 8.3 Price measurement and index numbers

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| UN System of National Accounts, price and volume chapter | Index-number theory | The Laspeyres, Paasche, and Fisher forms, and the exact decomposition identity | B |
| Consumer Price Index Manual (ILO and partner agencies) | Price collection and index construction | Sample design, substitution, and quality adjustment | B |
| Eurostat HICP methodology | Harmonised EU price measurement | Comparability of price indices across member states | B |
| Company price file or contract register | The price actually charged | Whether the series is a list price, a realised price, or an ASP | C |

A **list price** is set before the period and is exogenous to the week's demand; a **realised price**
is an outcome, contaminated by markdowns triggered by weak sell-through; an **ASP** is a ratio that
contains mix. Only the first is safe as a price regressor in a decomposition intended to inform a
pricing decision. Module 07 topic 04 treats the endogeneity of realised price in detail.

### 8.4 Promotion data, and the gap where a standard should be

There is no GS1 or ISO standard for a promotional calendar. Promotion data arrives as retailer
portals, syndicated causal flags, internal trade-plan extracts, or a spreadsheet maintained by a
category manager, and no two retailers deliver the same fields. This is the largest data constraint
on the promotion component, and it is why topic 03 exists.

| Field needed | Typical availability | Consequence if missing |
|---|---|---|
| Promotion start and end date | Usually present | Lift smeared across non-promoted weeks |
| Discount depth versus regular price | Often present, sometimes as a funded rate rather than a shelf price | Price and promotion cannot be separated |
| Feature flag and position | Sometimes, from syndicated causal data | Feature effect absorbed into the bundle |
| Display flag and type | Sometimes, with poor compliance accuracy | Display effect absorbed or understated |
| In-store compliance | Rarely | Planned lift is estimated instead of executed lift |
| Out-of-stock days during the promotion | Rarely at daily grain | Availability loss attributed to weak demand |

The last two rows are the most consequential and the least available. A promotion that ran at
planned depth but was out of stock for two of its seven days produced less lift than planned, and
without the record the decomposition reports a weaker promotion rather than an availability loss.

## 9. Industry practice and case evidence

### 9.1 What commercial systems actually compute

The dominant commercial pattern is a multiplicative pipeline with three estimated parts: base
demand, a seasonal profile, and promotional lift, multiplied together. Oracle's retail causal
forecasting documentation describes exactly this structure — out-of-stock and outlier correction,
promotion estimation by decision tree or generalised linear model, seasonality by additive or
multiplicative Winters exponential smoothing, base demand by competitive selection among simple
exponential smoothing, Holt, moving average, and Poisson methods. Grade D: it describes the method,
not its accuracy.

A second pattern, visible in fresh-goods and availability analytics, decomposes volume into
baseline, promotion, lost sales to availability, lost sales to cannibalisation, and unexplained.
That partition is more useful, because it separates availability explicitly and gives the
unexplained term a name that invites investigation. Grade D for the vendor description; the
partition is sound and is the one this topic recommends.

### 9.2 The academic treatment of the promotion bump

Van Heerde, Leeflang, and Wittink (2004) decompose the sales promotion bump into cross-brand
(switching), cross-period (stockpiling and acceleration), and category-expansion effects,
conditional on discount depth and support type. The methodological point that matters here is
their treatment of support: they build separate price indices for no-support, feature-only,
display-only, and feature-plus-display conditions, and estimate separate response curves for each.
That is the split-calendar design of section 5 Step 13, implemented at scale. The substantive
finding, in general terms, is that feature-supported discounts drive cross-period effects while
display-only promotions drive category expansion. The components behave differently depending on
what accompanied them, which is why a single promotion flag is the wrong regressor. Grade A.

Ando (2008) takes the state-space route for baseline sales on daily scanner data, treating baseline
as an unobserved state with a random-walk trend and estimating by Markov chain Monte Carlo. Two
findings transfer: baseline is time-varying rather than constant, and promotion effects differ
across stores. The second is the argument for rung 5 — a national coefficient is an average over a
heterogeneous population and can be positive while half the stores are negative. Grade A for
method; the category is narrow, so transfer the heterogeneity, not the magnitudes.

Jetta and Rengifo (2011) make the practitioner-facing critique directly. They argue that
industry-standard log-linear baseline models produce "phantom spikes" — baseline estimates that
rise in promotion weeks because the baseline estimator is itself correlated with promotional
activity — and propose a dynamic linear model with a multiple structural change component. Their
reported reduction in weekly baseline volatility is large, above eighty percent in their sample.
Treat the magnitude as sample-specific and the mechanism as the finding. Grade A, effect size as
the authors state it.

### 9.3 The argument about what baseline means

The definitional dispute explains why two correct decompositions disagree. The marketing-mix
convention treats baseline as the non-steerable part of demand — trend, season, holidays, an
intercept — with price, promotion, and distribution as steerable contributions on top. Meta's
open-source Robyn documentation is reported to state that there is no clear definition of baseline,
an unusually candid admission from a vendor. Grade D.

### 9.4 The endogeneity evidence in promotion settings

Jeziorski, Leng, and Seiler study settings in which the treatment — feature advertising — is
randomised but prices adjust endogenously in response. Their result is uncomfortable: regressing
demand on the treatment alone, or on the treatment plus the endogenous price, both give biased
estimates of the direct effect, with the same sign in both specifications. Adding the endogenous
price reduces the bias without eliminating it. They propose a test and a price instrument. Grade
A/B for a working paper, reported as the authors state it.

The implication is narrow and important: if your promotion weeks are also your price-change weeks,
no amount of econometric care recovers a clean split, and the split moves when the calendar
changes. That is a property to disclose, not a problem to solve with a better estimator.

### 9.5 The seasonal-decomposition inheritance

Ratios-to-moving-averages predate computing and were codified in the U.S. Census Bureau's X-11
programme, the ancestor of X-12-ARIMA and X-13ARIMA-SEATS. A parallel model-based tradition,
TRAMO-SEATS, estimates components from an ARIMA model rather than from filters. The two disagree at
the endpoints, which is where the planner stands: a centred moving average cannot be computed for
the most recent observations, and those are where the forecast starts. Grade B.

## 10. Failure modes and red flags

Each row gives the failure, its visible red flag, the mechanism, and the consequence.

| # | Failure | Red flag you can see | Mechanism | Consequence |
|---:|---|---|---|---|
| 1 | Baseline computed as the residual | The residual column is zero or near-zero every period | Baseline absorbs whatever the other components miss | Every component inherits the error; the table sums and is still wrong |
| 2 | Promotion split into depth, feature, and display on one series | Condition number above 30 and a promotion coefficient whose bootstrap spans zero to the whole lift | The four columns move together in execution | Budget is reallocated on a split the data cannot support |
| 3 | Price effect measured by ASP | "Price +25%" reported while every list price rose 5% | Mix shift moves the ratio, not the price | Pricing decisions are justified by a composition change |
| 4 | Distribution omitted or folded into baseline | Residual correlates with the change in ND or WD | Availability lives in a separate feed and is never joined | Launch volume is attributed to demand; the largest driver is invisible |
| 5 | Out-of-stock treated as low demand | Promotions with supply problems report weaker lift than identical promotions with stock | Unmet demand is unobserved | The estate systematically under-invests in promoted items |
| 6 | Pull-forward left inside lift | Residual turns negative for weeks after each promotion | Borrowed volume is counted at the peak and never returned | Repeat promotions look better than they are; the run-rate is over-set |
| 7 | Additive form fitted to multiplicative data | Residual variance rises with the level; the seasonal swing grows with the business | Fixed-unit seasonality assumed | Summer is under-forecast, winter over-forecast, and the error is blamed on weather |
| 8 | Definitions changed mid-year without a version | Baseline share steps at a month boundary with no event to explain it | Two incompatible tables are compared | Trend conclusions are artefacts of a definition change |
| 9 | Calendar misalignment between retailer and manufacturer weeks | A one-week shift in a seasonal peak that no one can explain | 4-5-4 versus ISO week keys | A calendar artefact is chased as a demand problem for a quarter |
| 10 | Table reported without a residual | Every component is "significant" and the errors are folded into the smallest one | The model has no error term to diagnose | Structured error is invisible; the next specification repeats it |
| 11 | Decomposition run at national level only | Store-level variance is large but the national fit is tight | Aggregation cancels the variation that identifies components | Coefficients are stable, unfalsifiable, and useless for local decisions |

A table with three or more of these flags is not a decomposition; it is a restatement of the sales
line in five columns.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks if it fails | How to test it |
|---:|---|---|---|---|
| 1 | Components are separable in the chosen form | The exercise is a partition | A genuine interaction, such as display working only at deep discount, is forced into one component | Test an interaction term; if material, report the bundle |
| 2 | The counterfactual is stable over the window | Contributions must be comparable across periods | A change in competitive structure makes January's baseline meaningless in June | Re-estimate on rolling windows; check parameter stability |
| 3 | Promotion, price, and support vary independently, or the bundle is reported | Otherwise the split is arbitrary | Coefficients are stable, plausible, and jointly unidentified | Condition number; bootstrap the coefficients |
| 4 | Distribution is measured at the same grain as sales | It enters as a driver | Availability is misattributed to baseline or promotion | Check join cardinality; sample a reset week by hand |
| 5 | Price is a list or contract price | A realised price is contaminated by demand | The price coefficient absorbs the demand shock | Compare the price series with the markdown log |
| 6 | Seasonality is stable within the window, or changes slowly | The seasonal profile is applied forward | A retail calendar shift moves the peak and the profile is wrong | Compare seasonal indices across years |
| 7 | The residual is white after the model | Residual structure means a missing component | Structured residual is absorbed as noise and components are biased | ACF; residual by promotion flag; residual by store |
| 8 | Units and value reconcile through a stated bridge | The table must tie to the P&L | The decomposition is dismissed by finance as a parallel universe | Walk three months from scanner gross to ledger net |
| 9 | No structural break inside the window | Parameters are constant | A reset, delisting, or competitor entry is smeared across components | Recursive estimation; a break test at known dates |
| 10 | The hierarchy level matches the decision | Aggregation changes the answer | A store-level decision is informed by a national average | Compare component shares at two levels |

Three limits follow from the ledger.

**The decomposition is not causal unless the design is.** It is a partition of an observed series
into parts that are internally consistent and useful for planning. The promotion component is
causal only to the extent that section 5 Step 13 found independent variation. Everything else is an
allocation of an association.

**It cannot create information the data does not contain.** If the only promotion record is a flag,
the split between depth and support is not recoverable. The correct output is a bundle with a stated
composition, and the correct next action is a data-collection decision, not a modelling one.

**Components are only as stable as the commercial policy that generated them.** A promotion
elasticity estimated during deep discounting does not transfer to shallow discounting, and a
distribution factor estimated during a launch does not transfer to a mature item. Re-estimate when
the policy changes, and say so in the footer.

## 12. Exercises

### Exercise 1 — Write the definitions before the model (design, 30 minutes)

Choose a product series you know or can obtain. Write the five component definitions, each with a
completed counterfactual sentence, a unit, and a named owner.

**Deliverable:** a five-row definition table plus one paragraph on the anchor component.
**Acceptance:** every counterfactual is falsifiable; every component has a unit; at least two
components have different owners; the anchor is justified by the decision, not by convenience.

### Exercise 2 — Choose the form with evidence (calculation, 40 minutes)

A series has annual mean levels of 1,000, 1,120, 1,260, and 1,410 units over four years, with
within-year peak-to-trough seasonal swings of 210, 236, 265, and 297 units. The additive fit has
residuals whose absolute value regressed on the log fitted value gives a slope of 0.91 with a
standard error of 0.14.

**Deliverable:** the amplitude-to-level ratios, the conclusion from each diagnostic, and the
recommended form with its attribution convention.
**Acceptance:** ratios computed and compared rather than judged; the standard error used to
interpret the slope; the convention named and justified in one sentence.

### Exercise 3 — Build the confounding table (application, 45 minutes)

For a store-week dataset with a promotion flag, shelf price, feature flag, display flag, ND, and
weekly units, construct the confounding table of section 5 Step 11 and mark which components are
separately identifiable.

**Deliverable:** the six-row table, a condition-number estimate for the intended design, and a
paragraph stating what will be reported as a bundle.
**Acceptance:** at least four co-movements identified; at least one component declared not
identified rather than estimated anyway; the bundle defined precisely enough that another analyst
would build the same variable.

### Exercise 4 — Distribution, ND, and WD (calculation, 35 minutes)

A brand moves from 480 to 520 carrying stores in a 1,200-store universe, and from 61% to 63% WD.
Category volume per store is not available individually.

**Deliverable:** ND before and after, the ND and WD ratios, the implied size of a new store
relative to the average, and a recommendation on which measure enters the model.
**Acceptance:** all numbers correct to three decimals; the WD/ND interpretation stated; the
recommendation conditioned on a stated assumption about whether volume scales with store size.

### Exercise 5 — The index-number trap (calculation, 45 minutes)

Three items in a category:

| Item | Year 1 price | Year 1 units | Year 2 price | Year 2 units |
|---|---:|---:|---:|---:|
| X | 1.50 | 1,000 | 1.65 | 900 |
| Y | 3.00 | 400 | 3.00 | 520 |
| Z | 6.00 | 100 | 6.60 | 80 |

**Deliverable:** year 1 and year 2 value and ASP, the Laspeyres and Paasche price indices, the price
and volume variances, and the quantity and mix split.
**Acceptance:** both indices computed with correct weights and stated; price plus volume variance
equals the value change exactly; quantity plus mix equals the volume variance exactly; one sentence
on what the ASP would have wrongly attributed.

### Exercise 6 — Read a structured residual (critique, 30 minutes)

A weekly decomposition has residuals with mean zero, no autocorrelation to lag 26, flat variance,
but a correlation of 0.42 with the change in ND, and a mean residual of −180 units in the four weeks
following each promotion.

**Deliverable:** a ranked list of components implicated, the repair for each, and the order of
repairs.
**Acceptance:** both patterns assigned to a named component; the ordering justified by which repair
changes the table most; a white-and-large residual explicitly distinguished from a structured one.

### Exercise 7 — Build the auditable table (application, 60 minutes)

Using any series with a promotion indicator, produce a twelve-period decomposition table in units,
with a residual, that sums to the actual in every row.

**Deliverable:** the table, the footer, and a log of the five audit tests from section 5 Step 21.
**Acceptance:** every row sums within 0.1% of actual; the footer states the form, the convention, the
reference configuration, and the definition version; at least one audit test reported as failed,
with the consequence stated.

### Exercise 8 — Reconcile units to the P&L (design, 45 minutes)

For a decomposition built on gross scanner sales, design the bridge to net revenue in the ledger.

**Deliverable:** a bridge table with one line per adjustment, in currency and as a percentage of
gross, plus the owner of each line.
**Acceptance:** at least five adjustments named, including trade spend and returns; each line has an
owner and a data source; the bridge closes to the ledger within a stated tolerance.

### Exercise 9 — Defend or reject a promotion claim (critique, 40 minutes)

A commercial team claims a promotion delivered a 42% lift, measured as promoted-week sales against
the average of the four preceding weeks. Describe what is wrong with the baseline and rebuild it.

**Deliverable:** a one-page critique, a rebuilt baseline with its stated counterfactual, and a
revised lift with a range rather than a point.
**Acceptance:** at least three separate defects identified; the rebuild states what is switched off;
the range reflects at least the attribution-convention uncertainty from section 6.5.

## 13. Self-check answers

### Answer 1 — definitions

A strong submission gives five definitions that another analyst could execute to the same number.
For a grocery SKU: baseline is the level with no promotion at reference distribution and reference
price, de-seasonalised, owned by demand planning; seasonality is the ratio to the annual mean from
a fit on non-promoted weeks; distribution is achieved WD over reference WD, owned by category
management; promotion is the multiplier for TPR, feature, and display together, owned by the
commercial team; price is the multiplier from the regular price change at the estimated elasticity.

The anchor is normally the baseline, the only component that persists when nothing is executed. A
weak submission lists components without counterfactuals, gives no units, and anchors on total
sales, which makes every other component a difference from a number that already contains them.

### Answer 2 — additive or multiplicative

```text
amplitude / level:
  year 1: 210 / 1,000 = 0.210      year 3: 265 / 1,260 = 0.210
  year 2: 236 / 1,120 = 0.211      year 4: 297 / 1,410 = 0.211

level grew 41.0% (1,410 / 1,000); swing grew 41.4% (297 / 210)
ratio of ratios: 1.414 / 1.410 = 1.003

log-residual slope 0.91, SE 0.14
  test against 0: 0.91 / 0.14 = 6.5   rejected
  test against 1: (0.91 - 1.00) / 0.14 = -0.64   not rejected
```

The amplitude-to-level ratio is constant to three decimals while the level rises by more than
forty percent: the swing is a fixed proportion of the level, not a fixed number of units. The
regression slope is consistent with proportional error and inconsistent with absolute error. Both
diagnostics agree, which is the case where the choice can be made without hedging.

Recommendation: multiplicative form, contributions reported log-additively and converted to units
by sequential substitution in a stated order for the planner-facing table.

### Answer 3 — confounding

For one series with all six columns, the promotion, price, feature, and display block is
near-collinear: the condition number will typically exceed 100, and a bootstrap of the promotion
coefficient will span both "promotion did most of it" and "price did most of it". The defensible
output is a single promotional-event component with a stated composition — TPR depth, feature
present, display present — plus a price component identified only from list-price changes outside
promotional windows. A weak submission reports four coefficients because the software returned
four coefficients.

### Answer 4 — distribution

```text
ND before = 480 / 1,200 = 40.00%
ND after  = 520 / 1,200 = 43.33%
ND ratio  = 520 / 480   = 1.0833    +8.33%
WD ratio  =  63 /  61   = 1.0328    +3.28%

implied size of the average new store, indexed to the existing average:
  (63 - 61) / (43.33 - 40.00) = 2.00 / 3.33 = 0.60
```

The new stores carry about sixty percent of the average store's category volume. The WD/ND ratio
fell from `61 / 40.00 = 1.525` to `63 / 43.33 = 1.454`: the brand is in more stores and in a
smaller share of the volume-weighted estate, which is the same fact stated differently.

Which measure enters depends on the mechanism: volume proportional to store turnover means WD and
1.0328; a similar rate wherever present means ND and 1.0833. The five-point gap exceeds most
promotion effects in a mature category, so this is not a detail.

### Answer 5 — index numbers

```text
year 1 value = 1.50x1,000 + 3.00x400 + 6.00x100 = 1,500 + 1,200 + 600 = 3,300
year 1 units = 1,500;   ASP = 3,300 / 1,500 = 2.200

year 2 value = 1.65x900 + 3.00x520 + 6.60x80 = 1,485 + 1,560 + 528 = 3,573
year 2 units = 1,500;   ASP = 3,573 / 1,500 = 2.382

ASP change = 2.382 / 2.200 - 1 = +8.27%      units unchanged at 1,500

Laspeyres price = (1.65x1,000 + 3.00x400 + 6.60x100) / 3,300
                = (1,650 + 1,200 + 660) / 3,300 = 3,510 / 3,300 = 1.0636  (+6.36%)
Paasche price   = 3,573 / (1.50x900 + 3.00x520 + 6.00x80)
                = 3,573 / (1,350 + 1,560 + 480) = 3,573 / 3,390 = 1.0540  (+5.40%)
Fisher          = sqrt(1.0636 x 1.0540) = 1.0588  (+5.88%)

value change    = 3,573 - 3,300 = +273
price variance  = (1.65-1.50)x900 + 0 + (6.60-6.00)x80 = 135 + 0 + 48 = +183
volume variance = (900-1,000)x1.50 + (520-400)x3.00 + (80-100)x6.00
                = -150 + 360 - 120 = +90
check           = 183 + 90 = 273   exact

quantity and mix:
  units at base mix, base prices: 1,500 + 1,200 + 600 = 3,300
      (unchanged, since total units are unchanged)
  quantity variance = 3,300 - 3,300 = 0
  mix variance      = 3,390 - 3,300 = +90
check              = 0 + 90 = 90   exact
```

The indices differ by about a point, the normal spread when relative prices move unevenly: X and Z
rose 10%, Y was flat, so Laspeyres overweights the flat-priced Y and Paasche overweights the Y that
grew. Unit volume did not change at all, yet the mix effect is +90 because the basket shifted
toward the mid-priced item. ASP rose 8.27% while the Fisher index rose 5.88%; the 2.4-point gap is
the mix effect as a share of base value (`90 / 3,300 = 2.73%`, compounding with 5.88% to about
8.3%). Reporting "ASP +8.27%" as the price effect overstates pricing by two and a half points and
misses the only thing that actually happened.

### Answer 6 — residual reading

Two patterns, two components.

1. **Residual correlated with the change in ND (0.42).** Distribution is omitted or entered at the
   wrong grain. Repair: add ND and WD as separate log drivers, check the join cardinality at a
   reset, re-run. This comes first because a missing driver is a level error that contaminates the
   baseline in every period where distribution moved.
2. **Mean residual of −180 units in the four weeks after each promotion.** Pull-forward is
   unmodelled: borrowed volume is counted as lift and the give-back lands in the residual. Repair:
   a distributed lag on promotion with a give-back window set by the purchase cycle, not by the
   fit. That is topic 05.

What is explicitly not a problem: no autocorrelation and flat variance, so the form and the
seasonal basis are adequate. A large white residual would be a different finding and would call
for a wider interval rather than a repair.

### Answer 7 — the table

Acceptance is mechanical, with one trap. A typical failure computes the baseline as actual minus the
other components, which sums perfectly and has no residual; that table passes the row-sum test and
fails the exercise. Treat a zero residual column as a defect.

```text
test                  result   consequence
row sum               pass     max deviation 0.06 units
definition stability  fail     baseline reference changed in period 7
restatement           pass     max component move 1.8% on re-estimation
sign discipline       pass     distribution positive only where ND rose
residual structure    fail     residual correlated with ND change (0.31)
```

Two failures out of five is a normal and healthy first pass. A log with five passes usually means
the tests were defined loosely enough to pass.

### Answer 8 — the bridge

```text
line                                currency    % of gross   owner
gross scanner sales                 1,000,000      100.0      category
  less consumer returns              -18,000       -1.8      supply chain
  less unsaleables and damages        -6,000       -0.6      supply chain
  less retailer trade spend         -142,000      -14.2      commercial
  less promotional allowances        -58,000       -5.8      commercial
  less off-invoice discounts         -31,000       -3.1      finance
  plus settlement accrual release    +12,000       +1.2      finance
net revenue                          757,000       75.7      finance
```

State the tolerance, typically 0.5% of gross, and name any residual line rather than folding it
into the largest adjustment. A promotion contributing 12% of volume while consuming 14% of value is
a different proposition from one contributing 12% and consuming 8%.

### Answer 9 — rebuilding the lift

Three defects in "promoted weeks versus the four preceding weeks".

1. **The comparison period is not a counterfactual.** The preceding weeks may be low because of a
   seasonal trough, a competitor event, or deliberate pre-promotion destocking.
2. **Pull-forward is inside the lift.** Promoted-week sales include volume borrowed from the
   following weeks, which the comparison period does not net out.
3. **Distribution, price, and display are uncontrolled.** If distribution rose or regular price
   changed between the comparison and the promotion, the difference contains those effects.

A rebuilt baseline is the counterfactual level from a model estimated on non-promoted weeks with
seasonality, trend, distribution, and regular price as drivers, projected into the promotion week.
The lift is actual over that counterfactual, with a distributed lag to capture the give-back, and an
interval reflecting the attribution convention.

A credible revised answer is a range — "lift between 18% and 27% on the week, and a net effect after
give-back between 9% and 16%" — rather than a point. The range is not weakness; it is the honest
width of a quantity that depends on an unobserved counterfactual.

## 14. Interview questions

**1. "Decompose this sales series. Where do you start?"**

**Strong answer:** defines the components and their counterfactuals before touching the data, names
the anchor, states the unit and the decision the table feeds, and only then discusses method.
**Weak answer:** opens with a model choice, usually STL or a regression, without saying what a
component means here.

**2. "How do you decide between additive and multiplicative?"**

**Strong answer:** the amplitude-to-level plot, the log-residual slope, and a Box-Cox interval, plus
the recognition that count data confounds the heteroskedasticity diagnostic.
**Weak answer:** "retail data is always multiplicative", a convention presented as a fact.

**3. "Your promotion weeks always have a display and a price cut. Can you separate them?"**

**Strong answer:** no, not from that data; names the collinearity, states the condition number,
proposes a split-calendar panel or an experiment, and says what will be reported as a bundle.
**Weak answer:** "yes, the regression gives three coefficients", which shows the candidate has never
tested identification.

**4. "What is a baseline?"**

**Strong answer:** a counterfactual with a stated boundary, defined before estimation, whose
contents differ across demand planning, marketing mix, and finance, and whose definition belongs in
the footer.
**Weak answer:** "sales without promotion", stated as unambiguous, with no mention of price or
distribution.

**5. "Why can a baseline be a residual, and why is that bad?"**

**Strong answer:** if fitted last it absorbs specification error and shows phantom spikes correlated
with promotion, the documented failure of industry log-linear models.
**Weak answer:** does not see the problem, because the table still sums.

**6. "ND rose five points and sales rose three percent. Did distribution work?"**

**Strong answer:** computes ROS per carrying store, compares ND with WD, checks whether the new
stores are below average size, and asks whether the sales rise is inside the noise.
**Weak answer:** concludes that distribution drove growth from two aggregates with no denominator
check.

**7. "ASP rose eight percent. What happened to price?"**

**Strong answer:** rebuilds a Laspeyres or Fisher index at constant mix, separates quantity from mix
inside the volume variance, and notes that ASP contains mix by construction.
**Weak answer:** reports ASP as the price effect.

**8. "Your decomposition table sums every month. Is it correct?"**

**Strong answer:** no, additivity is arithmetic and not evidence; lists the audit tests, especially
definition stability and restatement.
**Weak answer:** yes, because it reconciles.

**9. "In a multiplicative model, what is the promotion contribution in units?"**

**Strong answer:** it depends on the attribution convention; logs are order-free, sequential
substitution is order-dependent, Shapley averages over orderings; states the convention and spread.
**Weak answer:** multiplies the baseline by the promotion factor and subtracts, without noticing the
result changes when the order changes.

**10. "What does a structured residual tell you?"**

**Strong answer:** names the component implicated by the pattern — promotion flag, give-back lag,
seasonal lag, ND change, price definition, hierarchy — and gives the repair.
**Weak answer:** calls it unexplained variance and proposes a wider interval, which is right only
when the residual is white.

## 15. Key terms

- **Decomposition** — the partition of an observed series into components with stated definitions
  that sum to it.
- **Component contribution** — the part of the observed value attributed to one component in one
  period.
- **Baseline** — the counterfactual level with named drivers switched off, defined before estimation.
- **Base volume** — the retail-measurement baseline: no temporary promotion, observed distribution,
  regular price.
- **Seasonality** — the recurring calendar pattern at a fixed period, net of trend.
- **Seasonal index** — the factor for one position in the cycle, mean one over a full cycle.
- **Trend** — the slow-moving level net of seasonality.
- **Residual** — the part of the observation no component explains; a diagnostic, not a bin.
- **Promotion lift** — incremental volume attributable to a promotional event after baseline removal.
- **Incremental volume** — volume that would not have occurred without the intervention.
- **TPR** — temporary price reduction from the regular shelf price.
- **Feature** — promotion in a retailer circular, leaflet, or app.
- **Display** — additional in-store placement beyond the normal shelf.
- **ND** — numeric distribution, the share of stores carrying the item.
- **WD** — weighted distribution, the share of category volume covered by carrying stores.
- **ACV and PCV** — all-commodity volume and product category volume, the two common store weights.
- **Availability factor** — the multiplier converting a baseline at reference distribution to
  achieved distribution.
- **ROS** — rate of sale, units per carrying store per period.
- **Price elasticity** — proportional volume change per proportional price change.
- **Mix effect** — the part of a value change caused by a shift in what sold.
- **ASP** — average selling price, revenue divided by units; contains mix by construction.
- **Laspeyres / Paasche / Fisher index** — price indices weighted by base quantities, current
  quantities, and their geometric mean.
- **Log-additive attribution** — reporting multiplicative contributions as logs, where they sum
  exactly and independently of order.
- **Shapley attribution** — averaging a component's marginal contribution over all orderings.
- **Identification** — the property that available variation can distinguish parameters from one
  another.
- **Confounding** — drivers moving together so their separate effects cannot be recovered.
- **Counterfactual** — the unobserved state against which the observed is compared.
- **STL** — Seasonal-Trend decomposition using LOESS, additive and robust.
- **Phantom spike** — a spurious baseline peak correlated with promotion activity.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. (2004). "Decomposing the Sales Promotion
  Bump with Store Data." *Marketing Science*, 23(3). DOI: 10.1287/mksc.1040.0061. Support-type
  specific price indices and the cross-brand, cross-period, category-expansion decomposition.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Terpenning, I. (1990). "STL: A
  Seasonal-Trend Decomposition Procedure Based on Loess." *Journal of Official Statistics*, 6(1),
  3–73.
- Ando, T. (2008). "Measuring the baseline sales and the promotion effect for incense products: a
  Bayesian state-space modeling approach." *Annals of the Institute of Statistical Mathematics*,
  60(4). Narrow category; treat store heterogeneity as the transferable finding, not the magnitudes.
- Jetta, N., and Rengifo, E. W. (2011). "A Model to Improve the Estimation of Baseline Retail
  Sales." *Journal of CENTRUM Cathedra*. DOI: 10.7835/jcc-berj-2011-0048. Phantom spikes in industry
  log-linear baselines; volatility reduction as the authors state it, sample-specific.
- Pollono, F., and Pupkevičs, R. (2023). "Sales variance analysis." *Journal of Revenue and Pricing
  Management*, 22, 57–71. The price, quantity, and mix split used in section 6.6.
- Naik, P. A. (2015). "Marketing Dynamics: A Primer on Estimation and Control." *Foundations and
  Trends in Marketing*. DOI: 10.1561/1700000031. State-space treatment of promotional dynamics.

**Working papers — Grade A/B**

- van Breda, M. F. (1983). "A New Approach to Variance Analysis." SMU Cox School of Business Working
  Paper 83-808. The correspondence between accounting variances and Laspeyres and Paasche indices.
  Grade B: working paper, widely cited, not journal-refereed.
- Jeziorski, P., Leng, Y., and Seiler, S. (2025). "Causal Inference with Endogenous Price Response."
  CEPR Discussion Paper 20252. Bias in direct-effect estimates when prices adjust to a randomised
  treatment. Grade B; reported as the authors state it.
- Gardete, P., and Cotrim, J. (2025). "Dynamic Demand Elasticities in Promotion Cycles." Working
  paper. Spurious-instrument bias in discrete promotion-price settings. Grade B; magnitudes are the
  authors' and are not independently verified here.

**Official statistical practice — Grade B**

- United Nations. *System of National Accounts*, chapter 15, "Price and volume measures." Laspeyres,
  Paasche, and Fisher forms and the exact decomposition identity.
- International Labour Organization, IMF, OECD, Eurostat, and UNECE. *Consumer Price Index Manual:
  Theory and Practice*. Sample design, substitution, and quality adjustment.
- Eurostat. *Harmonised Index of Consumer Prices (HICP) methodology*.
- U.S. Bureau of the Census. *X-11 variant of the Census Method II seasonal adjustment program*,
  Technical Paper 15. The classical ratio-to-moving-average lineage.
- Eurostat. *ESS Guidelines on Seasonal Adjustment*.
- ISO 8601 date and week representation; GS1 General Specifications including GTIN, GLN, and
  EPCIS 2.0. Grade B.

**Company and vendor documentation — Grade C/D**

- Oracle Retail AI Foundation Cloud Service, "Sales & Promo Method" documentation: base demand,
  seasonality, and promotion lift as a multiplicative pipeline. Grade D; method, not accuracy.
- NielsenIQ, "Measuring and improving your business with distribution data": ND, WD, ACV, PCV, and
  total distribution points. Grade C for definitions; company methodology, not independent.
- SKUtrak, "TruDemand" help documentation: volume decomposition into baseline, promotion, lost sales
  to availability, lost sales to cannibalisation, and unexplained. Grade D.
- Meta Robyn documentation, reported to state that there is no clear definition of baseline. Grade D;
  cited as evidence of a definitional dispute, not as a methodological authority.
- National Retail Federation, 4-5-4 retail calendar. Grade C; a convention with accounting
  consequences.

**Practitioner commentary — Grade E**

- Ashok Charan, "Baseline and Incremental Volume," marketing mix modelling notes: baseline including
  advertising, regular price, and distribution. A manufacturer-side convention, useful precisely
  because it differs from the demand-planning one.
- Umbrex, "Promotion Effectiveness Framework": a practitioner 12-step decomposition including pantry
  loading, cannibalisation, and halo.
- POS Potential and SFAsoftware notes on numeric versus weighted distribution and launch thresholds.
  The arithmetic is verifiable; the thresholds are rules of thumb.
- Revista ESPACIOS (2019), "Assessing Promotional Effectiveness in Retail through Estimation of Sales
  Components": an OLS decomposition with a reported lift in the low-to-mid teens percent. Grade E;
  low-tier venue, reported as stated and not relied on.

**Not verified.** No claim here depends on a vendor-reported effect size. Quoted magnitudes are
labelled as the authors report them and marked sample-specific; the numbers in sections 6 and 13
are constructed, not drawn from a dataset.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Next topic: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Baseline and incrementality: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Cannibalization, halo, and pull-forward: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Assortment and distribution effects: [`09_distribution-assortment.md`](09_distribution-assortment.md)
- Seasonal regression, Fourier terms, and distributed lags:
  [`../07_time_series_forecasting/04_regression-lags-fourier.md`](../07_time_series_forecasting/04_regression-lags-fourier.md)
- Hierarchy and partial pooling:
  [`../08_hierarchical_intermittent_probabilistic_forecasting/01_hierarchy-structures.md`](../08_hierarchical_intermittent_probabilistic_forecasting/01_hierarchy-structures.md)
- Experiments and geo tests for identification:
  [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Regression and time-series foundations:
  [`../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Part I index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Part II index: [`../_index.md`](../_index.md)
- Module survey file:
  [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)

