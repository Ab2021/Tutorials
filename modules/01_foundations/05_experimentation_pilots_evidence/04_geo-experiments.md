# Geo Experiments — market-level tests when customer-level randomisation is impossible

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 06–08; Module 05 topics 01–03
> Version 2026.5

---

## 1. Why this topic exists

Some levers cannot be pulled below a market. A television campaign bought on a designated market
area (DMA) cannot be shown to half the households inside it. A delivery promise configured per
catchment polygon cannot be promised to half the postcodes in the polygon. A regional price zone is
a property of a postcode list. A sales-territory incentive is paid to a team that covers a
territory. A carrier contract is signed per lane, and the lane serves every customer in a region.

When the lever is market-level, randomising customers does not merely produce a noisier estimate. It
produces an estimate of something the business cannot do. The experiment has to move up the ladder
to the geography, and the price of climbing is paid in statistical power.

The trade is severe and specific. You exchange thousands, or millions, of independent units for
tens. Between-geo variance is large and does not shrink when you collect more weeks, because the
number of independent coin flips is the number of geos. The design usually produces one draw of the
assignment rather than a sampling distribution, so the familiar standard error from a two-sample
test is an approximation to something that was never sampled. And because geographies are connected
— by delivery routes, commuting, media spillover, and online orders that ignore boundaries — the
control group is a counterfactual only if you work to make it one.

> **Note:** Geo experiments are not a weaker version of customer-level tests. They are a different
> design with a different inferential basis. The pre-period is the asset, the donor pool is the
> instrument, and randomisation or placebo inference replaces the sampling distribution you would
> have had with many independent units.

### 1.1 The levers that force the choice

| Lever | Why it cannot be applied below the market | Typical geo unit |
|---|---|---|
| Television, connected TV, radio, out-of-home | Media is bought and measured on a market footprint | DMA, television region |
| Regional price or promotional price | Price zones are defined by postcode or store list | Postcode district, price zone |
| Delivery fee or free-delivery threshold | Configured per catchment or delivery polygon | Catchment area, postcode cluster |
| Delivery promise (cut-off, next-day coverage) | Set per depot or per routing zone | Depot catchment, drive-time polygon |
| Assortment or planogram reset | Executed store by store, often supplier-funded by region | Store trade area, sales territory |
| Carrier or third-party logistics (3PL) change | Contracted per lane or depot | Lane, depot area |
| Sales compensation or territory design | Paid to a team that covers a territory | Sales territory |
| Service-parts stocking policy | Set per depot network node | Depot catchment |

Each row has the same structure: the decision is made once for a set of customers, and the set is
geographic. A customer-level test of the same lever would measure a world in which the company can
do something it has decided not to do.

### 1.2 Four failures that recur

**One. The twelve-geo test reported with a t-test.** A retailer tests a new delivery fee in 12
postcode districts and 12 matched controls, computes a two-sample t-test, and reports p = 0.03. The
12 treated districts were not randomly assigned; they were chosen because their pre-period looked
like the controls. The reported p-value answers a question about sampling from a population of
identical districts, which is not the population that exists. The correct inference is over the
assignment mechanism, and that mechanism was a judgement.

**Two. The national campaign with an eight-DMA holdout.** A brand runs a national campaign and holds
out eight DMAs to measure lift. The creative runs on national television, on streaming platforms
without geo-fencing, and on social feeds that follow users across DMA boundaries. The holdout is
demographically distinct and receives the campaign anyway. The measured lift is close to zero, and
the campaign is declared ineffective. The design measured the spillover, not the campaign.

**Three. The delivery-fee test that measured orders.** A grocery chain raises the free-delivery
threshold in 18 catchments and sees order volume fall 3.4%. The team reports a failure. Average
order value rose 11%, pick and pack cost per pound fell, and contribution per household rose. The
primary outcome was chosen before the mechanism was understood, and the chosen outcome pointed the
wrong way.

**Four. The matched markets selected after the results.** A team runs a test in four regions, finds
a positive result in two, and then writes the report describing those two as the "matched markets"
and the other two as "operational exceptions". Nothing in the arithmetic is wrong. The inference is
worthless, because the assignment was chosen after the outcome was known.

> **Note:** Failure four is the most common in practice and the hardest to see in a report. The
> defence is procedural, not statistical: the assignment, the primary outcome, and the decision rule
> are fixed and dated before the test period begins.

### 1.3 What this topic claims

The claim is narrow and practical. A geo experiment is interpretable when four things are true:

1. The geo is the level at which the decision is actually made, and the level at which the lever can
   be withheld.
2. The counterfactual is constructed from a pre-period that is long enough and stable enough to
   support it.
3. Inference comes from the assignment mechanism — randomisation, matching plus placebo tests, or a
   donor-pool model validated out of sample — rather than from a formula that assumes independent
   draws.
4. The reported effect is tied to a window, a mechanism, and a decision, and the report states what
   the test cannot see.

Everything else in this topic is elaboration.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decide** whether a proposed test requires geographic assignment, using the deliverability test,
   and **state** what is lost by climbing the ladder from customer to geo.
2. **Define** a geo unit and a normalised outcome for a named lever, and **defend** the definition
   against two alternatives on nesting, contamination, and comparability.
3. **Match or stratify** a pool of geos on pre-period level, trend, and covariates, and **diagnose**
   the resulting balance with a table a reviewer can check.
4. **Distinguish** matched pairs, matched markets, time-based geo pairs, and synthetic control, and
   **choose** among them for a stated number of geos and a stated pre-period length.
5. **Compute** the minimum detectable effect (MDE) for a pair-level geo design from historical
   between-geo variance, and **state** whether the business's hoped-for effect is detectable at all.
6. **Analyse** a geo test with a donor pool, including placebo-in-time and placebo-in-space tests,
   and **interpret** the resulting p-value against its floor.
7. **Diagnose** contamination across geo boundaries from delivery overlap, commuting, media
   spillover, and online ordering, and **propose** a design change or a bound.
8. **Write** a geo test report that states the design, the estimand, the window, the absolute and
   relative effect, the inference method, the compliance evidence, and the generalisation limit.


## 3. The core mental model

Hold three pictures. The first is the ladder and what it costs. The second is the shape of a geo test
in time. The third is the set of machines that build a counterfactual.

### 3.1 The ladder

```text
   THE RANDOMISATION LADDER — WHAT YOU TRADE AS YOU CLIMB

   unit                    typical n        independent draws    counterfactual
   ---------------------------------------------------------------------------
   customer / basket       10^5 - 10^7      many                 randomisation
   household               10^4 - 10^6      many                 randomisation
   store                   10^2 - 10^3      tens to hundreds     randomisation or match
   catchment / postcode    10^1 - 10^2      tens                 matching plus model
   DMA / region            10^1 - 2x10^2    tens                 matching plus model
   national                1                one                  none
   ---------------------------------------------------------------------------
   As n falls, the standard error is driven by between-unit variance, and
   the counterfactual must be constructed rather than assumed.
```

Two properties of the ladder matter more than the exact numbers.

**Property 1 — The effective sample size stops growing when the geos stop growing.** This is the
cluster-design result from Topic 02, applied at the extreme. With 20 geos and an intraclass
correlation (ICC) of 0.10, effective `n` tends to `20 / 0.10 = 200` no matter how many weeks or
orders you add inside each geo. Twenty geos observed for 52 weeks are still twenty independent
facts about the treatment, each measured more precisely. Precision per fact improves; the number of
facts does not.

**Property 2 — The pre-period is where precision is bought.** In a customer-level test, precision
comes from more customers. In a geo test, it comes from a pre-period that predicts the test period
well. A pair whose members moved together for 12 weeks before the test has a small variance for the
difference-in-differences; a pair that merely had similar averages does not. This is why matching
quality, not geo count, is the first thing to inspect in a geo design.

### 3.2 The shape in time

```text
   ONE GEO TEST IN TIME

   PRE-PERIOD (fit)            TEST PERIOD (measure)          WASHOUT
   8-12 weeks                  6-12 weeks                     2-6 weeks
   |---------------------------|-----------------------------|----------|
   Treated geos:   baseline    |  lever applied              | lever off
   Control geos:   baseline    |  no lever                   | no lever
                               ^                             ^
                        lever starts                  lever stops

   The pre-period buys precision. The test period buys effect, and also
   buys decay, stock-up, and competitive response. The washout tells you
   whether what you measured was a level shift or a timing shift.
```

The three windows answer three different questions, and a report that gives one number for the
whole test period has blended them.

- **Pre-period.** Can the counterfactual reproduce the treated geo before treatment?
- **Test period.** What happened while the lever was on?
- **Washout.** Did the effect persist, decay, or reverse? A reversal is evidence of pull-forward.

### 3.3 The four counterfactual machines

```text
   FOUR WAYS TO BUILD THE COUNTERFACTUAL

   1. MATCHED PAIR (randomised)      2. MATCHED MARKET (not randomised)
      T: geo 7   -----------            T: geo 7   -----------
      C: geo 12  -----------            C: geo 12  -----------  chosen for similarity
      effect = (T post - T pre)         effect = (T post - T pre)
             - (C post - C pre)               - (C post - C pre)

   3. SYNTHETIC CONTROL              4. TIME-BASED REGRESSION
      T: geo 7                           T: geo 7   ~~~~ actual
      C: 0.34*A + 0.22*B                 C: 0.6*A + 0.4*B  ~~~~ predicted
         + 0.18*C + 0.14*D + ...             from the pre-period relationship
      effect = T - weighted donor        effect = actual - predicted,
                                                  cumulated over the window
```

All four are descendants of difference-in-differences. They differ in what they assume about how
the untreated world would have moved.

- **Matched pair** assumes parallel movement; strongest when assignment is randomised within pairs.
- **Matched market** assumes the same without randomisation, so the assumption is a judgement.
- **Synthetic control** assumes the treated trajectory lies in the convex hull of the donors.
- **Time-based regression** assumes the treated series can be predicted from control series over time.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Geo | A geographic unit used as the assignment unit in an experiment | Used to mean any location, including a store or a warehouse | A named polygon with an identifier |
| Geo experiment | A test in which non-overlapping geographies are assigned to treatment and control and the lever is delivered at that level | Used for any before-and-after regional comparison | Design with an assignment mechanism |
| Designated market area (DMA) | A Nielsen-defined United States media market, county-based, 210 in total | Treated as a delivery area | Nielsen DMA code |
| Catchment area | The set of households a site or depot can serve within a stated time or distance | Treated as a fixed administrative boundary | Polygon or postcode list plus a rule |
| Sales territory | The accounts and geography assigned to a sales team or representative | Confused with a region on an org chart | Territory identifier, often hierarchical |
| Postcode cluster | A group of adjacent postcode units used as one assignment unit | Treated as a postcode area when it is a district | Postcode district or custom cluster ID |
| Donor pool | The set of untreated geos from which a synthetic control is built | Confused with the control group in a randomised test | List of geo identifiers, untreated |
| Synthetic control | A weighted combination of donor geos constructed to match the treated geo before treatment | Treated as a single matched market | Weights summing to 1 |
| Convex hull | The set of weighted averages of donor geos; the synthetic control must lie inside it | Ignored when the treated geo is an outlier | Geometric property of the donor matrix |
| Matched pair | Two geos paired on pre-period similarity, with one randomised to treatment | Called a matched market when no randomisation occurred | Pair identifier |
| Matched market test (MMT) | A non-randomised comparison of paired markets, one treated and one not | Reported as if it were randomised | Design label, must be disclosed |
| Stratification | Partitioning geos into homogeneous groups before randomising within each | Confused with matching after assignment | Stratum identifier |
| Blocked randomisation | Randomising within blocks of similar geos to guarantee balance on block variables | Called stratification when no blocking occurred | Block identifier plus assignment |
| Constrained randomisation | Restricting the set of admissible assignments to those meeting balance criteria | Ignored in the inference, which then overstates precision | Set of admissible assignments |
| Rerandomisation | Redrawing assignments until balance criteria are met | Treated as free; the redraws change the null distribution | Number of draws and the criterion |
| Balance | Similarity of treated and control geos on pre-period outcomes and covariates | Assessed only on the mean, not the trend | Standardised differences and trend plots |
| Pre-period | The window before treatment used to fit or match | Chosen to be convenient rather than representative | Weeks, with start and end dates |
| Test period | The window during which the lever is applied and the effect is measured | Extended after seeing an unfavourable result | Weeks, fixed in advance |
| Washout | The window after the lever is removed, used to measure persistence or reversal | Omitted, which hides pull-forward | Weeks |
| Carryover | The part of a period's effect that arrives in later periods | Confused with a lag in measurement | Weeks or a decay parameter |
| Spillover | Treatment reaching units assigned to control | Used interchangeably with interference | Direction and magnitude, often unmeasured |
| Contamination | Spillover that changes the control group's outcome | Assumed absent unless proven | Proportion of control units affected |
| Interference | Any dependence of one unit's outcome on another unit's treatment | Treated as a nuisance rather than a design property | Causal structure |
| Stable unit treatment value assumption (SUTVA) | Each unit's outcome depends only on its own treatment | Asserted rather than argued | Assumption, not a test |
| Budget interference | Fixed total spend or inventory reallocated across geos, so treating one changes another | Missed entirely in platform-run tests | Currency reallocated per geo |
| Compliance | The degree to which the lever was delivered as assigned | Assumed to be 100% | Proportion of geos or spend delivered |
| Intent-to-treat (ITT) | The effect of assigning treatment, regardless of delivery | Reported as the effect of the lever itself | Effect per assigned geo |
| Lift | The difference between observed and counterfactual outcome | Used for any positive change | Units, or per cent of baseline |
| Incremental return on ad spend (iROAS) | Incremental revenue or margin divided by incremental spend | Confused with reported ROAS | Currency per currency |
| Placebo test | A test in a period or unit known to be untreated, used to check the method | Run only when the result is inconvenient | Estimate and rank |
| In-space placebo | Applying the method to each donor as if it were treated | Reported without the p-value floor | Rank among donors |
| In-time placebo | Fitting on an earlier window and estimating an effect in a known-untreated window | Run on the same window used for fitting | Estimate and interval |
| Randomisation inference | Inference from the distribution of estimates over the assignment mechanism | Replaced by a t-test on geo-level means | Exact or Monte Carlo p-value |
| Trimmed match | A paired estimator that drops the worst-matched pairs and adapts the trim rate | Used without reporting the trim rate | Trim rate and retained pairs |
| Time-based regression (TBR) | A regression of the treated geo's series on control series over time, used to forecast the counterfactual | Treated as equivalent to a randomised control | Model form and pre-period fit |
| Bayesian structural time series (BSTS) | A state-space model with trend, seasonality, and regression components used for counterfactual forecasting | Treated as assumption-free | Prior and component specification |
| Minimum detectable effect (MDE) | The smallest true effect a design detects with stated power and significance | Computed after the test | Units and per cent of baseline |
| Between-geo variance | Variance of outcomes or changes across geos | Confused with within-geo variance | Variance in outcome units |
| Intraclass correlation (ICC) | The share of total variance that lies between clusters | Assumed to be small in geo tests | Proportion between 0 and 1 |
| Parallel trends | The assumption that treated and control would have moved together absent treatment | Asserted from a graph without a test | Pre-period difference in trends |
| Convex hull violation | A treated geo outside the range of the donor pool, forcing extrapolation | Hidden by reporting weights without fit | Distance from the hull |

Two of these deserve expansion, because the errors they cause are expensive.

**Lift versus incrementality.** Lift is observed minus counterfactual. Incrementality is the part of
lift caused by the intervention rather than by substitution, pull-forward, or competitor response.
A geo test estimates lift well when the counterfactual is good. It estimates incrementality only if
the counterfactual also holds fixed the things the intervention changed elsewhere — which is why
national rollout effects are not the same as test effects.

**Placebo versus control.** A control is a unit that did not receive treatment by design. A placebo
is a unit or period that should show no effect if the method is working. The control can be
contaminated; the placebo is the check that tells you whether it was. A geo test without a placebo
test has a control group but no check on it.

## 5. The mechanics, step by step

This section walks a geo test from the decision that forces it to the report that closes it. Each
step names its output and the branch that changes the design.

### Phase A — Decide whether the geo is the right unit

**Step 1. Name the decision and its unit.**
Write: "if the test shows X, [role] will do Y in [unit]". The unit in that sentence is the unit of
decision, and the design must serve it. If the decision is per store, per item, or per customer, a
geo test answers a different question.

**Step 2. Test whether a finer unit is deliverable.**
Ask whether the lever can be withheld from a subset smaller than a market, whether the measurement
can separate that subset, and whether operations can execute the split without tell. Three yeses
mean the finer unit is available and the geo test costs an order of magnitude in power for nothing.
A software filter is not evidence of deliverability: a delivery-promise change filtered to 30% of
customers still changes depot routing for all of them.

**Step 3. Check whether the decision is national.**
A geo test estimates a partial-equilibrium effect: some geos change, the rest do not. A national
rollout changes every geo and provokes competitor and supplier responses. If the decision is
national and the response is systemic, the geo estimate is an input, not the answer. Say so.

**Step 4. Check the withholding mechanism and the donor pool.**
Advertising can be geo-fenced, a price can be applied to a postcode list, a carrier can be limited
to lanes. A national brand campaign cannot be withheld from a market that receives the media through
a neighbouring feed. You also need untreated geos that are similar and not subject to a simultaneous
shock. If every area changes at once, there is no donor pool and no geo test.

### Phase B — Define the geo and the outcome

**Step 5. Choose the geo definition.**
The geo must nest the delivery mechanism. If the delivery radius is three kilometres, a postcode
district that straddles the radius is a mixture of two regimes. Candidates are media markets
(designated market areas, DMAs), administrative regions, postal units, depot catchments, sales
territories, or custom clusters. Draw the delivery polygon over the boundary: if it crosses
materially, the geo is wrong or the treatment must be redefined.

**Step 6. Normalise the outcome.**
Geos differ in size by orders of magnitude; orders per geo is dominated by population. Use orders
per 1,000 households per week, or per store, or per delivery address. Plot the raw outcome against
the denominator: a strong linear relationship means the unnormalised series will be heteroskedastic
and a few large geos will drive the estimate.

**Step 7. Fix the primary outcome and the metric contract.**
Write down, before the test: metric name, numerator, denominator, population, time bucket, treatment
of returns and cancellations, and averaging window. One primary outcome; everything else is
secondary and labelled. With 20 geos and six outcomes, one will look significant by chance with
probability well above a half.

### Phase C — Build the panel

**Step 8. Set the pre-period length and alignment.**
Weekly data need at least 8 weeks; 12 is common; 52 is better when seasonality is strong. Use
complete Monday-to-Sunday weeks, and check that holidays and promotions fall in the same relative
position across geos. A pre-period that starts mid-week gives one geo a different weekday mix.

**Step 9. Audit for structural breaks.**
Look for store openings, competitor entries, road works, depot changes, and platform changes. A
break inside the pre-period means the matching fits two regimes; a break inside the test period
means the estimate is confounded. Plot every geo, scan for change points, and read the operations
calendar.

**Step 10. Build the balance table.**
Report, by arm: mean level, trend, coefficient of variation, and key covariates (household density,
income, urbanicity, baseline average order value, delivery cost per order, store count). Use
standardised differences rather than p-values, because with 18 geos a balance p-value is
uninformative. The trend difference matters more than the mean difference: the mean is what matching
targets, the trend is what difference-in-differences needs.

### Phase D — Match or stratify

**Step 11. Choose the distance.**
Use the pre-period outcome path as the primary distance (sum of squared differences across weeks),
with covariates as a tie-breaker. Matching on covariates alone can produce pairs that look alike
demographically and diverge in the outcome. Mahalanobis distance accounts for correlation between
features; Euclidean does not.

**Step 12. Pair or block.**
With 20 to 200 geos, pair them: sort by size within region, then match optimally rather than
greedily. With more geos, stratify into four to eight strata on size and region and randomise within
strata. Leave spare geos where possible: a stratum with one treated and one control contributes no
within-stratum variance.

**Step 13. Diagnose balance, and validate on held-out weeks.**
Plot the paired pre-period paths. A good pair tracks week to week; a bad pair has similar averages
and different shapes. Compute the standard deviation of the pair-level pre-period difference,
because that quantity drives the standard error later. Then hold out the last four pre-period weeks:
match on the earlier part and check tracking on the held-out part. Selecting the best-fitting pairs
and testing on the same window produces regression to the mean.

### Phase E — Choose the design

**Step 14. Pick the counterfactual machine.**

| Situation | Design | Why |
|---|---|---|
| 20+ geos, randomisation feasible | Randomised matched pairs | Strongest inference; pairing removes level differences |
| 10–20 geos, randomisation feasible | Randomised pairs or stratified randomisation | Fewer pairs means wider intervals |
| 10–50 geos, randomisation not feasible | Matched markets, reported as observational | Best available; needs placebo checks and a caveat |
| 1–5 treated geos, 10+ donors | Synthetic control | Builds a comparison from many donors; weights are inspectable |
| 1 treated geo, 1–3 controls | Time-based regression | Uses time when the cross-section is too thin |
| Same geos, lever can be switched | Switchback (Topic 05) | Removes between-geo variance entirely |

**Step 15. Decide how assignment will be generated.**
Randomise within pairs and record the seed and date. If you stratify, fix the allocation ratio. If
you constrain or rerandomise, record the criteria and the number of rejected draws, because the
rejection changes the null distribution and ignoring it overstates precision.

**Step 16. Pre-register the analysis.**
Write and date the geo list, pair structure, assignment, primary outcome, test window, estimator,
placebo tests, decision rule, and decision owner. Store it where the analyst cannot quietly revise
it. Every later choice — dropping a pair, changing the window, switching the outcome — is defensible
in isolation and indefensible in aggregate if it was made after seeing the data.

### Phase F — Power before launch

**Step 17. Estimate between-geo variance from history.**
Take the same geos and a pseudo-test window in the pre-period, compute the pair-level
difference-in-differences as if one member had been treated, and take the standard deviation of
those pseudo-effects. Repeat for several windows; one window can be unrepresentative. That standard
deviation, divided by the square root of the pair count, is your standard error.

**Step 18. Compute the MDE and decide.**

```text
SE   = sd_pair / sqrt(P)
MDE  = (t(0.975, P-1) + t(0.80, P-1)) x SE
```

State the MDE in outcome units and as a percentage of the control baseline, and compare it with the
effect the business hopes to see. If the hoped-for effect is below the MDE, the test cannot succeed
even if the lever works. The honest options are more geos, a better counterfactual, a different
design, or not running the test.

### Phase G — Run the test

**Step 19. Instrument compliance.**
Measure what was delivered, per geo per week: spend, price applied, promise shown, assortment
installed, carrier used. Compliance below about 90% in the treated arm turns the test into a diluted
version of itself. Report intent-to-treat, and a per-protocol estimate only with its assumptions
stated.

**Step 20. Monitor contamination.**
Track the routes by which treatment reaches control: delivery from a treated depot into a control
postcode, orders placed in a control geo but fulfilled by treated capacity, media spillover,
commuting, word of mouth. Quantify where possible and write the contamination memo before the
analysis, not after.

### Phase H — Analyse

**Step 21. Estimate the primary effect.**
For randomised pairs, average the pair-level differences and take the standard error from
between-pair variation. For synthetic control, take the post-period gap between treated and
synthetic, with inference from placebo permutations. For time-based regression, take the cumulative
difference between actual and predicted. Report absolute units first, then relative: absolute units
are what the decision affects.

**Step 22. Run the placebo tests.**
In-time: fit on an earlier window and estimate the "effect" in a window known to be untreated. A
method that finds effects where none exist is not ready. In-space: apply the method to each donor as
if it were treated; the rank of the true treated geo gives a p-value with a floor of `1/(J+1)` for
`J` donors. Six donors means the smallest possible p-value is 0.143.

**Step 23. Run the robustness checks.**
Leave one pair or donor out and re-estimate. Change the window: first four weeks, last four weeks,
full period. Change the estimator: pairs, synthetic control, trimmed match. Change the
normalisation: per household, per store. Report the range, and if the sign flips across reasonable
choices, say so.

**Step 24. Decompose the mechanism.**
A geo test gives a total effect; the decision needs the components. For a delivery or pricing test,
decompose into order count, average order value, fulfilment cost per order, returns, and service.
Check that the components add to the total within rounding. This is where a supply chain geo test
differs from an advertising test: the total can be positive while every operating metric the team is
measured on gets worse.

### Phase I — Report

**Step 25. Apply the reporting standard.**
In order: the decision and its owner; the design and assignment mechanism; the geo definition and
count; the primary outcome and metric contract; the pre-period, test period, and washout dates; the
effect, absolute and relative, with an interval; the inference method and assumptions; the placebo
results including the p-value floor; compliance and contamination; the mechanism decomposition; what
the test cannot see; and the decision, with the condition under which it would be reversed.

## 6. Worked example

A grocery delivery business — call it Marlow — tests a free-delivery threshold in 36 postcode
districts in one country. All figures are illustrative and internally consistent; the arithmetic is
reproducible from the tables.

### 6.1 The decision

Marlow currently charges a flat £4.10 delivery fee on every order, with no minimum basket. The
commercial team proposes a free-delivery threshold of £45: orders below £45 keep the £4.10 fee,
orders at or above £45 pay nothing. The hypothesis is that the threshold will push households to
build larger baskets, reduce the number of small unprofitable orders, and raise contribution per
household even if order counts fall.

The decision owner is the commercial director. The decision is whether to apply the threshold
nationally. The lever is configured per catchment in the order-management system, so it cannot be
applied to half the households in a catchment. This is a geo test.

### 6.2 The design

- **Geo:** postcode district, 36 districts in the same country, each containing at least 40,000
  households and each served by a single depot for at least 90% of orders.
- **Pairs:** 18 pairs, matched on the 12-week pre-period path of orders per 1,000 households per
  week, with region and depot as constraints.
- **Assignment:** one member of each pair randomised to treatment. Seed and date recorded.
- **Pre-period:** 12 weeks, complete Monday-to-Sunday weeks.
- **Test period:** 8 weeks.
- **Primary outcome:** orders per 1,000 households per week.
- **Secondary outcomes:** average order value (AOV), contribution per 1,000 households per week,
  delivery cost per order, and the share of orders below £45.

> **Note:** The primary outcome chosen in advance was order rate, not contribution. That choice is
> deliberate in this example, because it produces the most instructive failure: the primary outcome
> moves in the direction the team feared, while the economic outcome moves in the direction they
> wanted. The report has to handle both.

### 6.3 The pair table

Orders per 1,000 households per week, pre-period mean and test-period mean, by pair.

| Pair | Treated pre | Treated post | Control pre | Control post | Difference-in-differences |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.6 | 65.4 | 67.8 | 69.3 | −4.7 |
| 2 | 62.2 | 64.8 | 61.7 | 63.1 | +1.2 |
| 3 | 64.7 | 59.9 | 62.2 | 63.3 | −5.9 |
| 4 | 61.7 | 68.0 | 61.9 | 65.5 | +2.7 |
| 5 | 62.2 | 53.8 | 63.1 | 62.6 | −7.9 |
| 6 | 66.8 | 70.1 | 67.2 | 69.3 | +1.2 |
| 7 | 56.1 | 51.9 | 58.0 | 56.6 | −2.8 |
| 8 | 63.2 | 63.6 | 63.7 | 64.1 | 0.0 |
| 9 | 55.0 | 57.5 | 54.7 | 56.0 | +1.2 |
| 10 | 57.4 | 62.7 | 56.6 | 56.9 | +5.0 |
| 11 | 58.3 | 55.6 | 58.8 | 60.3 | −4.2 |
| 12 | 65.5 | 66.5 | 63.7 | 63.9 | +0.8 |
| 13 | 52.5 | 50.2 | 52.0 | 55.2 | −5.5 |
| 14 | 64.5 | 62.0 | 63.4 | 64.0 | −3.1 |
| 15 | 66.3 | 62.0 | 65.2 | 64.9 | −4.0 |
| 16 | 58.4 | 58.4 | 58.7 | 59.3 | −0.6 |
| 17 | 64.6 | 60.2 | 65.6 | 66.7 | −5.5 |
| 18 | 70.0 | 65.8 | 70.2 | 72.1 | −6.1 |

Group means: treated pre 62.11, treated post 61.02; control pre 61.92, control post 62.95. The
control group grew by 1.03 orders; the treated group fell by 1.09. The difference-in-differences is
`−1.09 − 1.03 = −2.12` orders, which matches the mean of the pair column.

### 6.4 The primary estimate

```text
n pairs          = 18
mean difference  = -2.122 orders per 1,000 households per week
sd of differences= 3.635
SE               = 3.635 / sqrt(18) = 0.857
t(0.975, 17)     = 2.110
95% interval     = -2.122 +/- 2.110 x 0.857
                 = (-3.930, -0.314)
```

The order rate fell by 2.1 orders per 1,000 households per week, a 3.4% decline against a baseline
of 62.1. The interval excludes zero, but only just: the upper bound is a decline of 0.3 orders.

Read the pair column before believing the interval. Four pairs (2, 4, 6, 10) moved in the opposite
direction, one pair did not move at all, and five pairs (3, 5, 13, 17 and 18) fell by more than
five orders. The
distribution is wide relative to the mean. That heterogeneity is the honest feature of the design:
18 pairs is enough to see a 3% effect if the pairs are well matched, and not enough to see a 1%
effect under any circumstances.

### 6.5 Power and MDE

```text
MDE = (t(0.975, 17) + t(0.80, 17)) x SE
    = (2.110 + 0.863) x 0.857
    = 2.973 x 0.857
    = 2.55 orders per 1,000 households per week
    = 4.1% of the 62.1 baseline
```

The design could detect a 4.1% change with 80% power at a two-sided 5% significance level. The
commercial team had hoped to detect a 1.5% change, because a 1.5% order-rate effect at national
scale was worth roughly £1.4m of annual contribution in their model.

What would it take to detect 1.5%? A 1.5% effect is 0.93 orders. Required SE is `0.93 / 2.973 =
0.313`. With `sd_pair = 3.635`, the required pair count is `(3.635 / 0.313)^2 = 135` pairs, or 270
geos. The country has fewer than 200 postcode districts with the required depot structure. The
desired effect was undetectable at any feasible geo count, and no amount of test-period extension
would fix it, because the variance is between geos, not within weeks.

> **Note:** This is the single most useful calculation in a geo design, and it is usually done after
> the test, when it is too late. Run it before the assignment is drawn. If the answer is "not
> detectable", the honest move is to change the design — a longer pre-period with covariates, a
> switchback, or a larger unit change — rather than to run a test that cannot answer the question.

### 6.6 The economic outcome

Order counts are an input, not an objective. The same test measured AOV and contribution.

| Group | Orders per 1,000 households/week | AOV | Contribution per order | Contribution per 1,000 households/week |
|---|---:|---:|---:|---:|
| Control pre | 61.9 | £41.20 | £7.44 | £460.3 |
| Control post | 63.0 | £41.60 | £7.55 | £475.5 |
| Treated pre | 62.1 | £41.10 | £7.41 | £460.0 |
| Treated post | 61.0 | £45.73 | £8.70 | £531.0 |

Contribution per order is `0.28 x AOV − £4.10`, where 28% is the product gross margin and £4.10 is
the variable fulfilment cost per order.

```text
control change      = 475.5 - 460.3 = +15.2
treated change      = 531.0 - 460.0 = +71.0
difference-in-differences = 71.0 - 15.2 = +55.8 per 1,000 households/week
relative            = 55.8 / 460.3 = +12.1%
```

The threshold reduced orders by 3.4% and raised contribution by 12.1%. Both are true. The mechanism
is visible in the AOV column: the average basket rose by £4.63, or 11.3%, and the fulfilment cost
per order did not rise, so the margin per order rose faster than the order count fell.

A sensitivity check is required, because larger baskets cost more to pick. If pick cost rises by 2%
of the basket above the £41.20 baseline, the treated contribution falls from £531.0 to £525.4 and
the effect from £55.8 to £50.2, or 10.9%. The direction survives; the magnitude shrinks by about a
tenth.

### 6.7 Decay, and why the window matters

The eight-week average hides a pattern. Split the test period:

| Weeks | Order effect (orders) | AOV effect (£) | Contribution effect (£) |
|---|---:|---:|---:|
| 1–2 | −3.0 | +6.10 | +71.4 |
| 3–6 | −2.4 | +5.20 | +54.6 |
| 7–8 | −0.6 | +2.00 | +14.8 |

The effect decays. The most plausible mechanism is behavioural: households initially top up baskets
to cross the threshold, then adjust their ordering rhythm so that they place fewer, larger orders
and the basket-size gain shrinks. By weeks 7–8 the contribution effect is £14.8 per 1,000
households per week, or 3.2% — a quarter of the eight-week average.

The terminal-window interval is wider: with a pair-level standard deviation of £52, the standard
error is `52 / sqrt(18) = £12.3`, and the 95% interval is `14.8 +/- 2.110 x 12.3 = (-11.1, +40.7)`.
It includes zero.

This is the central reporting problem of geo tests. The average says "roll out"; the terminal
window says "we do not know". Both are computed from the same data, and choosing one is a
judgement, not an arithmetic step.

### 6.8 Placebo and randomisation inference

**In-time placebo.** The estimator was re-run on four pseudo-test windows inside the pre-period,
with the same pairs and the same 8-week length. The four placebo estimates were +0.4, −1.1, +0.9,
and −0.3 orders, with a standard deviation of 1.2. The observed −2.1 is outside the range of the
placebos, which supports the design.

**Randomisation inference.** Under the sharp null of no effect, each pair difference is equally
likely to carry either sign. Enumerating all `2^18 = 262,144` sign patterns gives a two-sided
p-value of 0.026 for the observed mean. The t-test gives a similar number, which is expected when
the pair differences are roughly symmetric. When they are not — heavy tails, one dominant pair —
the randomisation p-value and the t-test diverge, and the randomisation value is the one to report.

**In-space placebo for the synthetic control check.** One treated district, District 27, was also
analysed with a synthetic control built from six untreated districts with weights 0.34, 0.22, 0.18,
0.12, 0.09, and 0.05. The pre-period root mean squared prediction error (RMSPE) was 1.8 orders. The
post-period gap was −2.9 orders. Running the same procedure on each of the six donors produced
placebo gaps of −0.7, +1.1, −1.4, +0.3, +0.9, and −0.2; District 27 ranked first of seven, giving
`p = 1/7 = 0.143`.

That p-value cannot go below 0.143 with six donors. It is not evidence of no effect; it is the
resolution limit of a six-donor placebo distribution. Reporting "p = 0.14, not significant" without
the floor would be misleading.

## 7. Code, query, or calculation

The calculation below reproduces the primary estimate from the 18 pair differences. It is written
in plain Python with no third-party packages so that it can be run anywhere.

```python
import random, math, statistics as st
# Pair-level differences in orders per 1,000 households per week.
# delta[i] = (treated post - treated pre) - (control post - control pre)
delta = [-4.7, 1.2, -5.9, 2.7, -7.9, 1.2, -2.8, 0.0, 1.2,
         5.0, -4.2, 0.8, -5.5, -3.1, -4.0, -0.6, -5.5, -6.1]
n = len(delta)
est = sum(delta) / n
se = st.stdev(delta) / math.sqrt(n)
t_crit = 2.1098                       # t(0.975, df = 17)
lo, hi = est - t_crit * se, est + t_crit * se

# Randomisation inference: under the sharp null, each pair difference is
# equally likely to have either sign. Sample the 2^18 sign patterns.
random.seed(20260501)
null = []
for _ in range(200_000):
    s = [random.choice((-1, 1)) for _ in range(n)]
    null.append(sum(a * b for a, b in zip(s, delta)) / n)
p_ri = sum(abs(x) >= abs(est) for x in null) / len(null)

print(f"effect={est:.2f} orders  se={se:.2f}  CI=({lo:.2f}, {hi:.2f})")
print(f"t={est/se:.2f}  p_ri~{p_ri:.3f}")
```

Expected output, two lines:

`effect=-2.12 orders  se=0.86  CI=(-3.93, -0.31)`

`t=-2.48  p_ri~0.026`

The Monte Carlo p-value varies in the third decimal between runs with different seeds; the exact
enumeration gives 0.0258.

**Assumptions the code makes, and what each one costs if it is wrong.**

1. **The pair differences are the unit of analysis.** If you pass order-level or household-level
   rows, the standard error will be too small by roughly the square root of the within-pair sample
   size. The code takes 18 numbers, not 18 million.
2. **The pairs are independent.** If two pairs share a depot, a competitor, or a media market, the
   sign-flip null is not the right null. Cluster the pairs or model the dependence.
3. **The sharp null is the hypothesis of interest.** Randomisation inference tests "no effect in any
   pair", not "no average effect". A lever that helps some geos and hurts others equally will not be
   detected. Report the pair-level distribution alongside the mean.
4. **Sign symmetry is a reasonable approximation.** When pair differences are heavily skewed, the
   sign-flip distribution is a poor approximation and a trimmed estimator is preferable.
5. **The random draw is seeded.** The seed is recorded so the p-value is reproducible. A p-value
   that changes when you re-run the script is a defect.

**Hand calculation of the MDE.** With `P = 18`, `t(0.975, 17) = 2.110` and `t(0.80, 17) = 0.863`.
`MDE = (2.110 + 0.863) x 0.857 = 2.55` orders, or 4.1% of the baseline. If you use the common
approximation `2.8 x SE`, you get 2.40 orders; the approximation understates the MDE at small
degrees of freedom, which is exactly where geo tests live.

## 8. Data, standards, and regulatory context

A geo test is a data-join problem before it is a statistical one. The assignment unit, the outcome,
the covariate set, and the delivery footprint must all be expressed in the same geographic coding
system, and that system is usually not the one the business uses internally.

### 8.1 Geographic coding systems

| System | Granularity | Maintainer | Why it matters in a geo test | Grade |
|---|---|---|---|---|
| ISO 3166-1 | Country | International Organization for Standardization | Joins cross-border reporting; does not resolve sub-national units | B |
| ISO 3166-2 | Country subdivision | International Organization for Standardization | Stable codes for regions and states; names change, codes do not | B |
| NUTS | EU territorial units, three levels | Eurostat | The default for EU regional tests; level choice changes the geo count by an order of magnitude | B |
| ONS Postcode Directory | Postcode to geography lookup, United Kingdom | Office for National Statistics | Maps postcodes to districts, councils, and output areas; updated quarterly | B |
| ZIP Code Tabulation Area (ZCTA) | Census approximation of ZIP codes, United States | United States Census Bureau | Not identical to United States Postal Service ZIP codes; delivery and census geographies diverge | B |
| Nielsen DMA | Media market, county-based, 210 in the United States | Nielsen | The unit on which national television is bought and measured | C |
| FIPS county code | County, United States | National Institute of Standards and Technology | Joins public covariates; not a media or delivery unit | B |
| Sales territory | Internal commercial unit | The company | Often the only unit on which incentives can be changed | None, internal |
| Depot catchment or drive-time polygon | Operational service area | The company or a routing vendor | The unit at which delivery promises are actually configured | None, modelled |

Three consequences follow.

**The units do not nest cleanly.** A DMA is a set of whole counties; a depot catchment is a
drive-time polygon; a postcode district is a postal unit. A household can live in one DMA, one
county, one postcode district, and one depot catchment, and those four geographies can overlap in
ways that no single map shows. Before you assign, build a crosswalk and check how many households
fall into more than one candidate unit.

**Codes change.** Postcode districts are split, counties are redrawn, DMAs are occasionally
revised, and NUTS is revised periodically. A test that spans a boundary change has two geographies
in one dataset. Record the vintage of every lookup table, and freeze it for the duration of the
test.

**Vendor geographies are proprietary.** DMA definitions belong to Nielsen and are licensed. A
platform's "geo" may be a postal code, a DMA, or a custom polygon, and the platform's definition is
what its delivery and measurement use, not yours. Ask for the definition in writing.

### 8.2 Price and promotion tests

If the geo test varies price, several legal regimes may apply, and this section is educational
rather than legal advice.

- In the European Union, Directive (EU) 2019/2161 (the "Omnibus Directive") amended the Price
  Indication Directive to regulate, among other things, the announcement of price reductions. A
  test that displays a reference price must comply with the rules on how that reference is
  established. The rules apply to the displayed price, not to the internal test design.
- In the United States, the Robinson-Patman Act of 1936 prohibits certain forms of price
  discrimination between competing purchasers of commodities of like grade and quality, subject to
  defences including cost justification and meeting competition. Trade promotions that vary by
  region can raise questions under it; consumer price tests generally do not, because consumers are
  not competing purchasers.
- Most-favoured-nation and price-parity clauses in platform or distributor contracts can prohibit
  differential pricing by geography even where competition law permits it.

The design consequence: price and promotion geo tests need legal review before launch, and the
review should happen while the design is still changeable.

## 9. Industry practice and case evidence

The public evidence base for geo experiments is overwhelmingly from digital advertising. That is a
limitation of this section, and it should shape how much you transfer to delivery, pricing, and
assortment decisions.

### 9.1 The Google lineage

| Work | Year | Contribution | Grade |
|---|---:|---|---|
| Vaver and Koehler, "Measuring Ad Effectiveness Using Geo Experiments" | 2011 | Formalised the randomised geo design with a pre-period, a regression estimator, and a variance formula for power; used the 210 United States DMAs as the unit | C |
| Vaver and Koehler, "Periodic Measurement of Advertising Effectiveness Using Multiple-Test-Period Geo Experiments" | 2012 | Extended the design to rotating assignments across periods, tightening intervals for ongoing measurement | C |
| Kerman, Wang, and Vaver, "Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework" | 2017 | Added a time-based regression counterfactual for cases with few geos or a single matched market | C |
| Chen and Au, "Robust Causal Inference for Incremental Return on Ad Spend with Randomized Paired Geo Experiments" | 2019 preprint; 2022 journal | Introduced trimmed match for heavy-tailed pair differences and budget-constraint interference | A |
| Chen, Longfils, and Remy, "Trimmed Match Design for Randomized Paired Geo Experiments" | 2021 preprint | Extended trimmed match to the design phase: pairing, power analysis, rerandomisation | B |

Two substantive points from this line of work are worth carrying into supply chain settings.

**Constrained randomisation helps, and its benefit is modest.** Vaver and Koehler report that
ranking geos on the pre-period outcome and restricting assignment to balanced draws reduced
confidence-interval width by roughly a tenth in their example. That is a real gain and a small one
relative to the gain from better matching or more geos. Do not treat constrained randomisation as a
substitute for a donor pool.

**Budget interference is a first-class problem.** When a campaign has a fixed budget, treating some
geos changes spend in the others. Chen and Au model this explicitly and use trimming to handle the
resulting heavy tails. The supply chain analogue is inventory: if a treated region draws down
shared stock, control regions face different availability, and the control is no longer untreated.

### 9.2 The synthetic control tradition

Synthetic control did not originate in advertising. Abadie, Diamond, and Hainmueller's 2010 study of
California's tobacco control programme is the canonical application, and its inference logic is the
one geo tests borrow.

- The treated unit is California; the donor pool is 38 states, with several excluded because they
  had their own large tobacco programmes.
- The synthetic control is a weighted average: Colorado 0.164, Connecticut 0.069, Montana 0.199,
  Nevada 0.234, Utah 0.334.
- The estimated effect by 2000 was roughly 26 packs per capita lower than the synthetic
  counterfactual, with an average reduction of about 20 packs per capita over 1989–2000.
- Inference is by placebo: each donor state is treated as if it were the treated unit, and
  California's post-to-pre mean squared prediction error ratio is compared with the placebo
  distribution. California ranked first of 39, giving `p = 1/39 = 0.026`.

The transferable lesson is the inference, not the application. With one treated unit, the p-value
comes from the rank among placebos, and its floor is `1/(J+1)`. That floor is a property of the
design, not of the data, and it must be reported.

### 9.3 What platform vendors document

Meta's open-source GeoLift documentation describes a synthetic control approach with augmented and
generalised variants, and provides power calculators for test length, investment, and the number of
test markets. Google's Meridian GeoX documentation describes stratified sampling plus clustering of
geos on key performance indicator features, a design search that maximises power, and an MDE
formula.

Both are vendor documentation. They are useful for understanding what the tools do and are not
independent evidence that the tools work. Grade D.

A vendor blog argues that synthetic control produces roughly four times more precise results than
matched-market tests. No benchmark, dataset, or interval is disclosed in the material summarised
here, so the claim cannot be checked. Grade D, and the number should not be repeated as fact.

### 9.4 What is not publicly evidenced

There is very little peer-reviewed public evidence on the performance of geo experiments for
supply chain levers: delivery promises, delivery fees, assortment resets, carrier changes, and
service-parts stocking. The methods transfer because the statistical structure is the same. The
variance magnitudes do not transfer, because supply chain outcomes are driven by different
mechanisms and different data quality.

Two gaps matter. There is no public benchmark of geo-test bias for operational levers: advertising
tests have been studied with known ground truth, delivery-fee tests have not, so you cannot cite a
bias correction and can only run placebos. And there is almost no published evidence on long-run
effects, which is why the worked example in section 6 is labelled as a constructed illustration.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| One treated geo | The report says "the pilot region" and the geo count is one | No between-unit variance; the comparison is before-and-after | Confidence interval cannot be computed honestly; effect is confounded with any local shock | Use a donor pool or a switchback; if neither is possible, call it a case study |
| Assignment chosen after results | The "matched markets" list has a date later than the test start date | Post-hoc selection of favourable units | Interval is meaningless; effect is selected | Pre-register the geo list and the assignment |
| Control contaminated by delivery | Control-geo orders are fulfilled from a treated depot | Operational footprint ignores administrative boundaries | Control is partly treated; effect biased towards zero | Measure the share; exclude or bound; redefine the geo |
| Budget or inventory interference | Treated arm spend rises while total budget is fixed | Reallocation inside the platform or the network | Control is not untreated; effect is a relative, not absolute, estimate | Model the interference or randomise the budget too |
| National campaign with a nominal holdout | Holdout DMAs receive the creative through national media or social feeds | Media does not respect the geo boundary | Measured lift is close to zero; the campaign is wrongly judged | Use a media plan that can actually withhold; measure spillover |
| Order-level analysis | The dataset has millions of rows and the geo count is 36 | Rows treated as independent observations | Standard error too small by roughly the square root of rows per geo | Aggregate to the geo; report the geo count |
| Relative lift without absolute | The headline is "+12%" with no baseline | Percentages travel better in slides than currency | Decision-makers cannot size the effect; small bases look large | Report absolute units first, then relative |
| Average reported, terminal window omitted | The report shows only the full-period mean | Decay or pull-forward is averaged away | A temporary effect is booked as durable | Report the window-by-window path and the terminal estimate |
| Donor pool contains treated or affected geos | A donor's spend or orders moved during the test | Contamination of the counterfactual | Counterfactual absorbs part of the treatment effect | Exclude affected geos; run leave-one-out |
| Pre-period too short or broken | The pre-period is four weeks, or spans a lockdown | Matching fits noise or two regimes | Poor pre-period fit; biased post-period gap | Minimum 8–12 weeks; change-point audit |
| Primary outcome switched | The report's "primary" metric is not the one in the pre-registration | Multiplicity and motivated reasoning | Inflated false-positive rate; result not reproducible | Freeze the metric contract; label secondary outcomes |
| t-test used where assignment was not random | The methods section says "we compared treated and control markets" | The sampling model assumes draws that never happened | p-values too small; precision overstated | Randomisation or placebo inference; disclose the design |
| Geography definition changed mid-test | The geo list has two vintages, or a district split | Boundary or lookup revision | Two geographies in one series; assignment ambiguous | Freeze the lookup vintage; document any change |
| Compliance unmeasured | Nobody can say what share of treated geos received the lever | Delivery failure or partial rollout | Effect is diluted; a null result is uninterpretable | Instrument delivery per geo per week |

Two of these deserve emphasis because they are invisible in a well-formatted report.

**Assignment chosen after results** leaves no trace in the arithmetic. The table is correct, the
interval is correct, and the only evidence is a date. This is why the pre-registration needs a
timestamp that the analyst cannot edit, and why a reviewer should ask for the geo list as it existed
before launch.

**Budget or inventory interference** is the failure most likely to be present in a platform-run
test and least likely to be disclosed. If the platform holds total spend constant, the control geos
are not untreated: they are receiving the spend that the treated geos did not consume. The estimate
is then a relative effect under a fixed budget, which is a legitimate quantity and a different one
from the absolute effect of the campaign.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| No interference between geos (SUTVA at geo level) | The control group must be untreated | Control outcomes respond to treatment; effect biased towards zero or away from it | Measure cross-boundary delivery, commuting, and media spillover |
| Parallel trends (for difference-in-differences) | The counterfactual is the control's change | Treated and control diverge for reasons unrelated to treatment | Plot pre-period trends; placebo-in-time tests |
| Donor pool spans the treated geo (for synthetic control) | The synthetic control must be a convex combination | Extrapolation; weights become unstable and fit is poor | Check pre-period RMSPE; inspect weights; check hull distance |
| No contemporaneous shock correlated with assignment | The comparison must isolate the lever | A competitor entry, road closure, or weather event drives the result | Event audit; in-time placebos; sensitivity to excluding the period |
| Treatment delivered as assigned | The estimate must be of the lever, not of the intent | Dilution; null result with full compliance would have been positive | Per-geo compliance instrumentation; ITT versus per-protocol |
| Pre-period is representative of the test period | Matching and forecasting depend on stationarity | Seasonal or trend breaks make the fitted relationship wrong | Compare pre-period and test-period covariates; use a holdout pre-period |
| Effect is stable over the measurement window, or the window captures the relevant horizon | A single number must summarise the effect | Decay or pull-forward makes the average misleading | Window-by-window analysis; washout period |
| The geo is the unit at which the decision is made | The estimate must inform the decision | The test answers a question nobody acts on | Restate the decision rule with the decision owner |
| Randomisation is credible, or matching is ignorable | Inference depends on the assignment mechanism | Selection bias; optimistic intervals | Pre-registration; balance table; placebo tests |
| No anticipation before the test period | Pre-period behaviour must be untreated | Treated geos change behaviour before launch, contaminating the baseline | Check the last two pre-period weeks for divergence |

The most important limit is not in the table. **A geo test estimates a partial-equilibrium effect in
a small set of markets over a short window.** National rollout changes prices, availability, and
competitor behaviour everywhere at once. The test tells you what happened when 18 markets changed
and the rest did not. That is useful, and it is not the same as what will happen when all markets
change. No statistical correction turns one into the other; only a phased rollout with a holdout
does.

A second limit: **geo tests are usually underpowered for the effects that matter commercially.**
The worked example could detect 4.1% and the business wanted 1.5%. This is not a defect of that
particular test; it is the normal condition. Designs that detect small effects need either many
geos, a much better counterfactual, or a unit change large enough to produce a large effect.

## 12. Exercises

### Exercise 1 — Feasibility triage (recall and application, 30 minutes)

For each lever, state whether a geo test is required, and why: (a) a new delivery cut-off of 21:00
configured per depot; (b) a loyalty coupon mailed to individual households; (c) a national
television campaign; (d) a change to the pick-path algorithm inside a warehouse; (e) a regional
price zone.

**Deliverable:** a five-row table with lever, deliverable unit, and a one-sentence justification.
**Acceptance criteria:** each answer names the smallest unit at which the lever can be withheld;
at least one answer identifies a lever that is not a geo test at all; no answer simply says
"because it is regional".
**Time:** 30 minutes.

### Exercise 2 — MDE from history (calculation, 45 minutes)

Using the 18 pair differences in section 6, compute the MDE at 80% power and a two-sided 5% level.
Then compute the pair count required to detect a 1.5% effect, and state whether it is feasible for
a country with 190 eligible districts.

**Deliverable:** the arithmetic, the MDE in orders and per cent, and the required pair count.
**Acceptance criteria:** correct degrees of freedom; the percentage is computed against the stated
baseline; the feasibility statement names the binding constraint.
**Time:** 45 minutes.

### Exercise 3 — Matching plan (design, 60 minutes)

You have 44 postcode districts, 12 weeks of weekly order data, and four covariates: household
density, median income, distance to the nearest depot, and baseline AOV. Write a matching plan.

**Deliverable:** the distance metric, the pairing or blocking algorithm, the balance table you would
produce, and the held-out validation you would run.
**Acceptance criteria:** the plan states why the chosen distance is preferred to one alternative;
the balance table includes trend as well as level; the validation uses pre-period data that was not
used for matching.
**Time:** 60 minutes.

### Exercise 4 — Placebo interpretation (critique, 30 minutes)

A synthetic control with eight donors reports a post-period gap of −4.1 units and a placebo rank of
first of nine, described as "p = 0.11, not significant". Critique the statement.

**Deliverable:** a short paragraph correcting the interpretation, plus the correct way to report the
result.
**Acceptance criteria:** identifies the p-value floor; distinguishes "no evidence of an effect" from
"evidence of no effect"; states what would raise the resolution.
**Time:** 30 minutes.

### Exercise 5 — Contamination map (application, 45 minutes)

A grocery chain changes the free-delivery threshold in 20 of 40 postcode districts. Depots serve
both treated and control districts; 30% of control-district orders are fulfilled by a depot that
also serves a treated district.

**Deliverable:** a contamination diagram, an estimate of the share of control orders affected, and
two design options with their costs.
**Acceptance criteria:** the diagram shows the flow of orders, not just the geography; the estimate
states its assumptions; one option changes the geo definition and one changes the analysis.
**Time:** 45 minutes.

### Exercise 6 — Vendor report critique (critique, 45 minutes)

A platform reports a 9% lift from a "geo-matched market study" in 14 markets over four weeks, with
no pre-period fit statistic, no compliance data, and a confidence interval of ±1.8 percentage
points. Write a critique.

**Deliverable:** a one-page review with at least six questions and a recommendation on whether to
accept the number.
**Acceptance criteria:** at least one question concerns the assignment mechanism; at least one
concerns the interval's width relative to the geo count; the recommendation distinguishes "usable
as directional evidence" from "usable as a decision input".
**Time:** 45 minutes.

### Exercise 7 — Pre-analysis plan (design, 60 minutes)

Write the pre-analysis plan for a geo test of a next-day delivery promise in 24 catchments.

**Deliverable:** a one-to-two-page plan covering the geo list, assignment, primary outcome, window,
estimator, placebo tests, decision rule, and owner.
**Acceptance criteria:** the plan is specific enough that a different analyst could execute it
without asking a question; the decision rule states a threshold and a direction; the plan names
what would count as a failed test.
**Time:** 60 minutes.

### Exercise 8 — From result to decision (design, 45 minutes)

Using the worked example, write the two-paragraph decision memo to the commercial director. The
first paragraph states what the test found; the second states what you recommend and why.

**Deliverable:** the memo, with the numbers a director needs and no more than one caveat per
paragraph.
**Acceptance criteria:** absolute effect stated before relative; the terminal-window result is
included; the recommendation is conditional on a named next step; no claim exceeds the design.
**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 1 — feasibility triage.**

| Lever | Unit | Geo test required? |
|---|---|---|
| Delivery cut-off of 21:00 per depot | Depot catchment | Yes. The promise is configured per depot and cannot be withheld from part of a catchment without changing routing for all of it |
| Loyalty coupon mailed to households | Household | No. The lever is deliverable at household level, and a geo test would waste power |
| National television campaign | Media market | Yes, and the withholding mechanism is the binding question. If the media cannot be geo-fenced, there is no control |
| Pick-path algorithm inside a warehouse | Zone or pick face | No. It is a warehouse-level operational test, best run as a switchback or a zone-level cluster design |
| Regional price zone | Postcode list | Yes. Price zones are defined by geography and usually applied to all customers in the zone |

The fourth row is the one most often misclassified. A warehouse change is not a geo test even when
the warehouse serves a region, because the lever is applied inside the building and the outcome is
measured per pick, not per household.

**Exercise 2 — MDE and required pairs.**

`SE = 3.635 / sqrt(18) = 0.857`. `MDE = (2.110 + 0.863) x 0.857 = 2.55` orders per 1,000
households per week. Against a baseline of 62.1, that is `2.55 / 62.1 = 4.1%`.

A 1.5% effect is `0.015 x 62.1 = 0.93` orders. Required `SE = 0.93 / 2.973 = 0.313`. Required pairs
`= (3.635 / 0.313)^2 = 134.9`, so 135 pairs, or 270 geos. The country has 190 eligible districts,
so the design is infeasible by a factor of about 1.4 in geo count. Extending the test period does
not help, because the variance that matters is between pairs, not within weeks. The binding
constraint is the number of comparable geos.

**Exercise 3 — matching plan, model answer.**

Use the pre-period outcome path as the primary distance: standardise each district's 12 weekly
values, then compute the sum of squared differences across the 12 weeks for every candidate pair.
Use the four covariates as a tie-breaker and as a hard constraint on region (do not pair across
regions with different competitive conditions).

Pair with an optimal matching algorithm rather than greedily, because greedy pairing can leave the
last few pairs badly matched. Within each pair, randomise which member is treated.

The balance table should report, for treated and control: pre-period mean, pre-period linear trend,
coefficient of variation, and each covariate, with standardised differences. Report the trend
difference prominently, because difference-in-differences depends on it.

Validate by holding out the last four weeks of the pre-period: match on weeks 1–8, then check
whether the pairs still track in weeks 9–12. A pair that tracks in the fitting window and diverges
in the holdout is a bad pair that got lucky.

**Exercise 4 — placebo interpretation.**

The statement is wrong in two ways. First, with eight donors there are nine placebo units including
the treated one, so the smallest attainable two-sided p-value is `1/9 = 0.111`. The reported 0.11 is
not a failure to reach significance; it is the maximum evidence the design can produce. Second,
"not significant" implies evidence of no effect, which the test cannot provide. The correct
statement is: "the treated unit ranked first of nine, the strongest result this donor pool can
produce; the design cannot resolve p-values below 0.11, so the result is consistent with an effect
and the resolution is limited by the pool size."

To raise the resolution, enlarge the donor pool, use a different inference method (for example a
randomisation test over a larger set of admissible assignments), or run the test in more geos.

**Exercise 5 — contamination map.**

The diagram should show orders flowing from households in control districts to depots, and depots
serving both arms. The key number is not the geographic overlap but the order share: 30% of control
orders are fulfilled by a depot that also serves a treated district. If the treatment changes depot
routing or capacity, those orders are partially treated.

The bias is towards zero. If the true effect is a 4% reduction in orders and 30% of control orders
are exposed to the same operational change, the measured effect is roughly `4% x (1 − 0.30) = 2.8%`,
assuming exposure is partial and the effect is proportional. State the assumption; it is a bound,
not a correction.

Two design options:

1. **Redefine the geo to the depot catchment.** Cost: the number of geos falls, possibly below the
   power threshold.
2. **Model exposure.** Estimate a dose-response with the exposed share as a covariate, or restrict
   the primary analysis to unexposed control districts. Cost: the restricted sample is smaller and
   may be selected on depot geography.

**Exercise 6 — vendor report critique.**

Questions to ask:

1. How were the 14 markets assigned — randomisation, matching, or selection?
2. What was the pre-period, and what was the pre-period fit statistic?
3. What was the outcome, and was it measured by the platform or by the advertiser?
4. What share of treated markets received the intended dose, and what share of control markets were
   exposed?
5. Is the ±1.8 point interval a randomisation interval, a model interval, or a reporting convention?
6. What would the interval be if the analysis were at market level with 14 units?
7. Was any placebo test run, and what was its p-value floor?
8. Was the four-week window pre-specified, and does the effect decay within it?

Recommendation: the number is usable as directional evidence and not as a decision input, because
the assignment mechanism is undisclosed, the interval is implausibly narrow for 14 markets, and
there is no compliance or contamination evidence. Ask for the market-level data and re-analyse.

**Exercise 7 — pre-analysis plan, rubric.**

A strong plan contains: the 24 catchments with identifiers and a boundary vintage; the pair or
stratum structure and the assignment seed; the primary outcome with numerator, denominator, and
window; the pre-period and test-period dates; the estimator and the inference method; the placebo
tests; the decision rule with a threshold; the owner; and the date. It also states what would count
as a failed test — for example, "if the pre-period fit RMSPE exceeds 2.5 orders, the design will be
abandoned before launch".

A weak plan lists the outcome and the dates and says "we will compare treated and control". It
cannot be executed by a second analyst without further decisions, which means the decisions will be
made after the data arrives.

**Exercise 8 — decision memo, model.**

Paragraph one: "The test reduced orders by 2.1 per 1,000 households per week (95% CI 0.3 to 3.9,
or 3.4%) and raised contribution by £55.8 per 1,000 households per week (about 12%) over eight
weeks. The contribution gain decayed to £14.8 in weeks 7–8, with an interval that includes zero."

Paragraph two: "I recommend extending the test by eight weeks in the same districts with a washout
period, rather than rolling out nationally. The eight-week average is real but is not yet a
steady-state estimate, and the decay pattern is consistent with households adjusting their ordering
rhythm rather than permanently increasing basket size. If the terminal four-week effect remains
above £15 per 1,000 households per week after the extension, the national case is worth building."

The memo should not say "the test proved" anything. It should say what was measured, over what
window, and what remains unknown.

## 14. Interview questions

**1. "When would you run a geo experiment rather than an A/B test?"**

A strong answer starts from deliverability: the lever cannot be withheld below a market, so the
assignment unit has to be the market. It then states the cost — tens of independent units instead
of thousands, between-geo variance dominating, and the need for a constructed counterfactual. A
weak answer says "when you cannot randomise customers", which describes a constraint without
describing the design consequence.

**2. "How many geos do you need?"**

A strong answer refuses the question as posed and asks for the between-geo standard deviation, the
effect size, and the design. It then gives the formula `MDE = (t_alpha + t_beta) x sd / sqrt(P)` and
notes that the answer is often "more than are available". A weak answer gives a number such as 20
or 50 without conditions.

**3. "What is the difference between a matched market test and a matched pair test?"**

A strong answer says the pair test randomises within pairs, so inference comes from the assignment;
the matched market test does not randomise, so the comparison is observational and the interval is
optimistic. A weak answer treats them as synonyms because both involve matching.

**4. "Why is a t-test usually wrong for a geo experiment?"**

A strong answer explains that the t-test assumes independent draws from a population, while the geo
test observes one assignment from a known set of possible assignments; the correct null distribution
is over assignments (randomisation inference) or over placebo units (synthetic control). A weak
answer says "because the sample is small", which is not the issue.

**5. "What does a placebo test do?"**

A strong answer distinguishes in-time and in-space placebos and explains what each falsifies: the
in-time test checks whether the method finds effects where none exist, the in-space test builds a
null distribution from donor units and gives a p-value with a floor of `1/(J+1)`. A weak answer says
"it checks the result is real", which is not a mechanism.

**6. "Your control geos are next to treated geos. What do you do?"**

A strong answer names the channels — delivery overlap, commuting, media spillover, online orders —
quantifies the exposed share, and offers three responses in order: redefine the geo, measure and
bound the contamination, or change the design. A weak answer says "assume no spillover" or "add a
buffer", without saying how wide or why.

**7. "What is the difference between lift and incrementality?"**

A strong answer says lift is observed minus counterfactual, while incrementality is the part caused
by the intervention rather than by substitution or pull-forward; a geo test estimates the first and
only estimates the second if the counterfactual is right and the mechanism is decomposed. A weak
answer uses the words interchangeably.

**8. "How would you analyse a test with one treated city and 20 candidate donors?"**

A strong answer describes a synthetic control: weights on donors chosen to match the pre-period,
pre-period fit checked with RMSPE, post-period gap as the estimate, and placebo-in-space inference
with its floor. It mentions the convex hull constraint and leave-one-out sensitivity. A weak answer
picks the single most similar city and compares before and after.

**9. "The result decayed after six weeks. What does that mean?"**

A strong answer separates three mechanisms — pull-forward, behavioural adjustment, and competitive
response — and says which data would distinguish them: a washout that reverses suggests
pull-forward; a new steady state at a lower level suggests adjustment; a competitor response
suggests checking competitor activity in treated geos. A weak answer calls the result "not durable"
and stops.

**10. "How would you report a geo test to a commercial director?"**

A strong answer gives the decision, the absolute effect with its interval, the relative effect, the
window, the terminal estimate, and the one thing the test cannot see. It avoids p-values in the
headline and puts them in the appendix. A weak answer leads with a percentage and a p-value.

## 15. Key terms

- **Geo** — the geographic unit used as the assignment unit.
- **Geo experiment** — a test in which whole geographies are assigned to treatment and control.
- **DMA** — designated market area; a Nielsen media market, county-based.
- **Catchment area** — the households a site or depot can serve within a stated time or distance.
- **Donor pool** — the untreated geos from which a synthetic control is built.
- **Synthetic control** — a weighted combination of donors matched to the treated geo's pre-period.
- **Matched pair** — two geos paired on pre-period similarity, one randomised to treatment.
- **Matched market test** — a non-randomised paired comparison, reported as observational.
- **Constrained randomisation** — restricting assignment to draws that meet balance criteria.
- **Balance** — similarity of treated and control on pre-period level, trend, and covariates.
- **Pre-period** — the window used to fit or match before treatment.
- **Test period** — the window during which the lever is applied.
- **Washout** — the window after the lever is removed, used to measure persistence.
- **Spillover** — treatment reaching control units.
- **Contamination** — spillover that changes control outcomes.
- **Interference** — dependence of one unit's outcome on another unit's treatment.
- **SUTVA** — the assumption that a unit's outcome depends only on its own treatment.
- **Budget interference** — fixed spend or inventory reallocated across geos.
- **Compliance** — the share of the lever actually delivered as assigned.
- **ITT** — intent-to-treat; the effect of assignment regardless of delivery.
- **Lift** — observed minus counterfactual.
- **iROAS** — incremental return on ad spend.
- **Placebo test** — a test in a period or unit known to be untreated.
- **In-space placebo** — applying the method to each donor as if treated.
- **In-time placebo** — estimating a "effect" in a known-untreated window.
- **Randomisation inference** — inference from the distribution over admissible assignments.
- **Trimmed match** — a paired estimator that drops the worst-matched pairs.
- **TBR** — time-based regression; forecasting the treated series from control series.
- **MDE** — minimum detectable effect at stated power and significance.
- **ICC** — intraclass correlation; the share of variance between clusters.

## 16. Sources

**Peer-reviewed and academic**

- Abadie, A., Diamond, A., and Hainmueller, J. (2010). "Synthetic Control Methods for Comparative
  Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the
  American Statistical Association*, 105(490), 493–505. DOI: 10.1198/jasa.2009.ap08746. Grade A.
- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., and Scott, S. L. (2015). "Inferring
  Causal Impact Using Bayesian Structural Time-Series Models." *Annals of Applied Statistics*,
  9(1), 247–274. DOI: 10.1214/14-AOAS788. Grade A.
- Chen, A., and Au, T. C. (2022). "Robust Causal Inference for Incremental Return on Ad Spend with
  Randomized Paired Geo Experiments." *Annals of Applied Statistics*. DOI: 10.1214/21-AOAS1493.
  Grade A.

**Preprints and technical reports**

- Chen, A., Longfils, M., and Remy, N. (2021). "Trimmed Match Design for Randomized Paired Geo
  Experiments." arXiv:2105.07060. Grade B; preprint, not peer-reviewed at the time of access.
- Vaver, J., and Koehler, J. (2011). "Measuring Ad Effectiveness Using Geo Experiments." Google
  Research technical report. Grade C; company research, rigorous and widely cited, not
  peer-reviewed.
- Vaver, J., and Koehler, J. (2012). "Periodic Measurement of Advertising Effectiveness Using
  Multiple-Test-Period Geo Experiments." Google Research technical report. Grade C.
- Kerman, J., Wang, P., and Vaver, J. (2017). "Estimating Ad Effectiveness Using Geo Experiments in
  a Time-Based Regression Framework." Google Research technical report. Grade C.
**Vendor and platform documentation**

- Meta, GeoLift Methodology documentation, facebookincubator.github.io/GeoLift. Discusses synthetic
  control, augmented synthetic control, generalised synthetic control, and power calculators. Grade
  D.
- Google, Meridian GeoX Design Methodology documentation, developers.google.com/meridian/geox.
  Discusses stratified sampling, clustering of geos, design search, and MDE. Grade D.
- Haus, "Matched Market Tests Don't Cut It: Why Haus Uses Synthetic Control in Incrementality
  Experiments." Vendor blog claiming roughly four times the precision of matched-market tests with
  no disclosed benchmark. Grade D; the multiple is not used in this text.
- Li Tan Analytics, "Geo Test Power Calculator", litananalytics.com. Source of the 50-market floor
  and 100-market target heuristic; undated, accessed 2026. Grade D.

**Standards and official**

- ISO 3166-1 and ISO 3166-2, country and country-subdivision codes, International Organization for
  Standardization. Grade B.
- NUTS classification, Eurostat. Grade B.
- ONS Postcode Directory, Office for National Statistics, United Kingdom. Grade B.
- ZIP Code Tabulation Areas, United States Census Bureau. Grade B.
- FIPS county codes, National Institute of Standards and Technology. Grade B.
- Nielsen DMA definitions and market counts. Proprietary; the 210-market figure is as reported in
  the geo-experiment literature. Grade C.
- Regulation (EU) 2016/679 (General Data Protection Regulation), including Article 4(1) and Recital
  26. Grade B.
- Directive (EU) 2019/2161 amending Directive 98/6/EC on price indication. Grade B.
- Robinson-Patman Act, 15 U.S.C. §13 (1936). Grade B.
- Media Rating Council accreditation programme. Grade B for the programme; individual accreditation
  status must be checked at the time of use.

**Note on citations.** Every source above was returned by the research pass for this topic. Grades
reflect provenance. No statistic is attributed to a source that did not report it, and the worked
example is constructed: none of its numbers describes a real company.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_randomisation-units.md`](02_randomisation-units.md)
- Previous topic: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- Previous topic: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Next topic: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Power and MDE for few-cluster designs: [`06_power-mde.md`](06_power-mde.md)
- Interference treated formally, including shared inventory: [`07_interference-spillover.md`](07_interference-spillover.md)
- Related: [`09_mmm-incrementality.md`](09_mmm-incrementality.md)
- Causal foundations: [`../04_statistics_causal_inference/07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Difference-in-differences: [`../04_statistics_causal_inference/08_difference-in-differences.md`](../04_statistics_causal_inference/08_difference-in-differences.md)
- Directed acyclic graphs: [`../04_statistics_causal_inference/06_dags.md`](../04_statistics_causal_inference/06_dags.md)
- Survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Pricing and promotion context: [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Case library: [`../../../course/28_domain_case_library.md`](../../../course/28_domain_case_library.md)
- Part map: [`../_index.md`](../_index.md)

