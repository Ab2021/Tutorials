> Module 04 — Statistics and Causal Inference
> Topic 08 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.05, 04.07

---

# Worked Example: Difference-in-Differences

## 1. Why this topic exists

Most supply chain changes are not randomised. A distributor switches a region to three deliveries
per week because a customer asked. A retailer installs a new receiving process at the DCs that
volunteered. A manufacturer changes a carrier contract in one lane and leaves the others alone.
By the time an analyst is asked "did it work?", the change has already happened, the manager who
approved it has an opinion, and the only defensible question left is what would have happened
without it.

Difference-in-differences (DiD) is the workhorse answer to that question. It compares the change
in a treated group with the change in a group that did not receive the treatment. The design is
old, intuitive, and easy to run badly. Its identifying assumption — parallel trends — is about a
future that did not occur, so it can never be verified, only argued for and stress-tested. Its
standard errors are routinely too small. And since roughly 2020, a literature has shown that the
most common way practitioners implement it, a two-way fixed effects regression on staggered
rollout data, can return a number that is not any group's effect at all.

That last point is why this topic sits in the curriculum at full length rather than as a paragraph
in Topic 07. Between about 2018 and 2024, the econometrics of DiD changed materially. Goodman-Bacon
(2021) decomposed the two-way fixed effects estimator into a weighted average of all possible 2x2
comparisons and showed that some of those comparisons use already-treated units as controls and
enter with negative weights. Callaway and Sant'Anna (2021) and Sun and Abraham (2021) proposed
estimators that avoid the problem. Roth (2022) showed that the pre-trend test practitioners rely on
has low power and that conditioning on passing it makes inference worse, not better. An analyst
who learned DiD from a 2015 textbook and applies it to a 2026 staggered rollout is using a method
whose failure mode is now well documented and whose repair is standard in software.

The supply chain setting sharpens every one of these problems:

- **Treatment is assigned to a cluster.** A DC, a region, a lane, or a supplier is treated, and
  every store or shipment inside it inherits the treatment. The number of independent units is the
  number of clusters, often between 8 and 40.
- **Rollout is staggered.** Nobody converts 40 DCs in one week. Waves are chosen by region,
  contract expiry, or willingness, which is exactly the setting where the negative-weighting
  problem bites.
- **The outcome is serially correlated.** Weekly store sales, on-time percentages, and inventory
  turns all carry strong persistence, which is the Bertrand, Duflo, and Mullainathan (2004)
  problem: conventional standard errors can be three to four times too small.
- **The comparison group is chosen by convenience.** The neighbouring region, the DC that did not
  volunteer, the stores that were not in the pilot. Each choice embeds a different claim about
  what would have happened.

This file works one retail-distribution case from raw weekly data to a defensible number with its
uncertainty. It shows the arithmetic by hand, because the arithmetic is where most of the
misunderstanding lives. It then shows what changes when the rollout is staggered, and how to
repair it without pretending the repair is free.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a 2x2 table from panel data, compute the four cell means, the two within-group
   changes, and the DiD estimate, and state the units at every step.
2. **Write** the regression form of the 2x2 design and **interpret** each coefficient — intercept,
   group indicator, time indicator, interaction — as a specific cell or difference of cells.
3. **State** the parallel-trends assumption as a claim about the unobserved counterfactual, and
   **distinguish** it from the claim that the groups were similar before treatment.
4. **Explain** why a non-significant pre-trend test is not evidence that trends are parallel, and
   **quantify** the trend magnitude the test could have detected.
5. **Specify** an event-study regression, **choose** a reference period deliberately, and **read**
   the coefficient path as an effect trajectory rather than a single average.
6. **Diagnose** the staggered-adoption problem in a two-way fixed effects regression, and
   **describe** in plain terms what the Goodman-Bacon decomposition shows and what the
   Callaway–Sant'Anna and Sun–Abraham estimators change.
7. **Cluster** standard errors at the level of treatment assignment, **justify** the level, and
   **report** the cluster count alongside the estimate.
8. **Produce** a decision-grade sentence for a DiD result: the estimate, its interval, the
   assumption that carries it, the test that could have refuted it, and the decision it does or
   does not support.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures in your head. The first is the four-cell table, because every DiD estimate is
built from four numbers and nothing else. The second is the pair of lines, because the whole
argument of the method is about the line you cannot see.

### 3.1 The four cells

```text
                    THE 2 x 2, WITH THE CELL MEANS FILLED IN

                             pre-period             post-period
                           (weeks 15-26)          (weeks 27-38)
                        +----------------------+----------------------+
      treated           |       1,240          |       1,318          |
      (6 DCs)           |   cases per          |   cases per          |
                        |   store-week         |   store-week         |
                        +----------------------+----------------------+
                        |      +78 = within-treated change             |
                        +----------------------------------------------+

      comparison        |       1,180          |       1,222          |
      (6 DCs)           |   cases per          |   cases per          |
                        |   store-week         |   store-week         |
                        +----------------------+----------------------+
                        |      +42 = within-comparison change          |
                        +----------------------------------------------+

      DiD = 78 - 42 = +36 cases per store-week
```

Read the table in three steps. The first difference is horizontal: each group is compared with
itself before and after, which removes any fixed difference between the groups. The second
difference is vertical: the two changes are subtracted, which removes any shock that hit both
groups equally. What remains is the part of the treated group's change that the comparison group's
change does not explain.

Neither step requires the groups to be alike in levels. A treated DC may ship 60 more cases per
store-week than a comparison DC in every week of the panel, and the design is untroubled. This is
the property that makes DiD attractive in operations, where treated units are almost never
randomly selected and almost always differ in level.

### 3.2 The line you cannot see

```text
   cases per store-week
         |
   1,320 |                                  *  1,318  treated, observed
         |                              *
         |                          *
   1,280 |                      ·  ·  ·  ·  ·  ·  ·  1,282  counterfactual
         |                  ·  ·                    (1,240 + 42)
   1,240 |*  *  *  *  *  *  ·                       treated pre-trend,
         |               ·                           extended unchanged
         |
   1,220 |            ·                    o  1,222  comparison, observed
         |         ·                  o
   1,180 |o  o  o  o  o  o  o  o
         |
         +--------------------------------------------> week
           15          26  |  27          38
                        adoption
                        <-- 12 wk --><-- 12 wk -->
```

The dashed extension is the assumption. It says: had the programme never run, the treated DCs
would have continued on a path parallel to the comparison DCs, so their post-period mean would
have been 1,282 rather than the 1,318 actually observed. The estimate of +36 is the vertical gap
between the star and the dot-dash line at week 38.

Three properties of that sentence matter.

**It is about a counterfactual, not about the past.** "The groups had similar trends before" is
evidence for the assumption, not the assumption. The assumption concerns weeks 27 to 38, which
were never observed without treatment. A perfect pre-period history is consistent with a
post-period divergence, and a noisy pre-period with parallel trends.

**It is a statement about trends, not levels.** DiD tolerates any fixed difference between groups.
It does not tolerate a difference in how the groups were moving.

**A statistical test does not make it more plausible.** A test can fail to reject because trends
are parallel or because the pre-period is short and noisy. Section 6.7 gives the arithmetic that
separates those explanations.

### 3.3 What the two differences remove, and what they leave

DiD is strong against fixed heterogeneity and common time shocks. It is weak against anything
that hits the treated and comparison groups differently at the moment of treatment. In supply
chain work that weakness usually has a name: the treated DCs were chosen because they were
improving, or because a customer demanded the change, or because their volumes were growing.

### 3.4 One design, four numbers, two assumptions

The two assumptions are usually stated as one. Separate them and the diagnostics become obvious.

1. **Parallel trends.** Absent treatment, the treated group's outcome would have moved in parallel
   with the comparison group's. This is about the counterfactual trajectory.
2. **No composition change.** The units inside each group, and the way the outcome is measured,
   are the same before and after. If three stores close during the post period, the treated mean
   moves for a reason that has nothing to do with the programme.

The second assumption is the one that kills more supply chain analyses than the first, and it is
the one almost never tested. A DC that opens, closes, or is renumbered during the rollout changes
the denominator of every mean in the table. Section 10 lists the red flags.

> **Note:** The 2x2 is not a simplified version of the regression. With two periods and two
> groups, the regression with group and period indicators and their interaction reproduces the
> four cell means exactly, and the interaction coefficient equals the DiD estimate exactly. Section
> 6.4 shows the algebra with the case numbers. If your regression and your hand calculation
> disagree in a balanced 2x2, one of them is wrong.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A DiD claim without units is not a claim;
it is a direction.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Difference-in-differences (DiD) | The change in a treated group minus the change in a comparison group over the same window | Used for any before-after comparison with no comparison group | Outcome units per unit of time |
| 2x2 DiD | The two-group, two-period version, computed from four cell means | Treated as a toy version rather than the identity the regression reproduces | Four means, outcome units |
| Treated group | Units that received the intervention | Defined by intention rather than by actual exposure | Binary indicator, at the assignment unit |
| Comparison group | Units that did not receive the intervention, used to estimate the counterfactual change | Called a "control group" when it is a convenience sample | Binary indicator |
| Pre-period | Time before the treatment date, used for the first difference and for trend evidence | Defined by data availability rather than by the decision | Weeks, dates, or periods |
| Post-period | Time after the treatment date | Started at the announcement rather than at implementation | Weeks, dates, or periods |
| First difference | The within-unit or within-group change between periods | Confused with the DiD estimate itself | Outcome units |
| Second difference | The subtraction of the comparison group's change from the treated group's change | Reported without stating the window | Outcome units |
| Parallel trends | Absent treatment, the treated group's outcome would have moved in parallel with the comparison group's | Restated as "the groups were similar before" | A counterfactual statement, not a number |
| Counterfactual trend | The trajectory the treated group would have followed without treatment | Replaced by the comparison group's observed trajectory | Outcome units per week |
| Common shock | A change affecting both groups equally, removed by the second difference | Assumed away when the groups operate in different markets | Outcome units |
| Differential shock | A change affecting the treated group and comparison group differently at the treatment date | The main threat the design cannot remove by differencing | Outcome units |
| Composition change | A change in which units or stores make up a group between periods | Ignored when the panel is unbalanced | Count of units, or share |
| Ashenfelter dip | A pre-treatment decline in outcomes among units selected into a programme | Read as evidence that the programme worked | Outcome units, by period before treatment |
| Anticipation | Behaviour changing before the formal treatment date because the change was announced | Absorbed silently into the reference period | Weeks before the event |
| Event time | Time measured relative to each unit's own adoption date | Confused with calendar time | Integer periods, `k = t − E_i` |
| Event study | A regression of the outcome on event-time indicators, tracing the effect path | Reported as one average effect | Coefficients by event time |
| Lead coefficient | An event-study coefficient for a period before adoption | Interpreted as a treatment effect | Outcome units, relative to reference |
| Lag coefficient | An event-study coefficient for a period at or after adoption | Averaged without checking the path's shape | Outcome units, relative to reference |
| Reference period | The event-time period normalised to zero and omitted from the regression | Chosen by software default without comment | One event-time value, usually −1 |
| Endpoint binning | Pooling the most distant leads or lags into a single indicator | Used to hide a divergent tail | One coefficient covering several periods |
| Pre-trend test | A joint test that the lead coefficients are zero | Read as proof that trends are parallel | F statistic, degrees of freedom, p-value |
| Placebo treatment date | A fake adoption date in the pre-period, used to check for a spurious "effect" | Placed where the pre-period is shortest | Date, plus the window used |
| Negative control outcome | An outcome the treatment cannot plausibly affect | Chosen because data were available, not because it is unaffected | Outcome units |
| Two-way fixed effects (TWFE) | A regression with unit and time fixed effects and a treatment indicator | Treated as the definition of DiD rather than one estimator | Indicator coefficient |
| Staggered adoption | Units adopt treatment at different calendar dates | Analysed with a single post indicator | Adoption date per unit |
| Treatment cohort | The set of units that adopt in the same period | Ignored in the regression specification | Cohort label, e.g. `g = 27` |
| Group-time ATT | The average treatment effect for cohort `g` at time `t` | Averaged into one number without saying which average | `ATT(g,t)`, outcome units |
| Goodman-Bacon decomposition | The algebra showing TWFE equals a weighted average of all 2x2 comparisons | Quoted as a criticism of DiD rather than of one estimator | Weights summing to 1, some negative |
| Forbidden comparison | A 2x2 that uses already-treated units as the comparison group | Assumed absent because the regression "controls for unit effects" | One 2x2, with a signed weight |
| Negative weight | A decomposition weight below zero, so a comparison's estimate enters with the wrong sign | Mistaken for a coding error | Signed share of the TWFE estimate |
| Variance-weighted ATT | The estimand TWFE actually targets under heterogeneity | Reported as "the ATT" | Weighted average, outcome units |
| Not-yet-treated control | Units that will be treated later but are untreated during the comparison window | Used without checking that the window ends before their adoption | Binary, time-varying |
| Never-treated control | Units that are never treated in the sample | Preferred, but often unavailable in operations | Binary indicator |
| Callaway–Sant'Anna estimator | A group-time ATT estimator using never-treated or not-yet-treated controls, with explicit aggregation | Described as "the new DiD" without the aggregation choice | `ATT(g,t)` plus an aggregation rule |
| Sun–Abraham estimator | An interaction-weighted event study that estimates cohort-specific paths and weights them by cohort shares | Confused with a simple reweighting of TWFE | Cohort-specific event-time coefficients |
| Imputation estimator | Fits the untreated model on untreated observations only, imputes `Y(0)`, then averages the gaps | Described as a robustness check rather than an estimator | Imputed potential outcomes |
| Clustering at the treatment level | Computing standard errors allowing correlation within the unit that was assigned treatment | Clustering on the row identifier | Number of clusters, `G` |
| Wild cluster bootstrap | A bootstrap that resamples cluster-level residuals, used when `G` is small | Applied without stating the null imposed | p-value, bootstrap replications |
| Dose or continuous treatment | Treatment measured in units of intensity rather than as a binary switch | Dichotomised at an arbitrary threshold | Natural units, e.g. deliveries per week |
| Triple difference (DDD) | A third difference across a group that should be unaffected, removing a group-specific shock | Added without a third dimension that varies independently | Outcome units, three-way contrast |

Four of these deserve expansion, because the damage they cause is disproportionate.

**Parallel trends versus "similar before".** The assumption is not that the groups had the same
level, or even the same pre-period average. It is that their *changes* would have matched from the
treatment date, absent treatment. Different levels with identical trends is the case DiD is built
for; identical levels with diverging trends is the case it cannot survive.

**Forbidden comparison.** The phrase sounds moralistic; it is algebraic. When a later-treated
cohort is compared with an earlier-treated one, the earlier cohort's post-period serves as a
"control" even though its own effect is still evolving. The 2x2 contains a difference between two
treatment effects. Unit fixed effects absorb the level gap and time effects the common trend; the
contamination lives in the interaction.

**Negative weight.** A weight below zero subtracts the comparison's estimate from the weighted
average. It arises when treatment-indicator variance weights the comparisons and the timing
structure makes some move opposite to others. One bad comparison with a large negative weight can
move the estimate further than all the clean ones combined.

**Pre-trend test.** Failure to reject is consistent with parallel trends and with a test too weak
to see a trend that matters. Roth (2022) shows that conditioning on passing biases the estimate
and under-covers the interval, because the estimates that pass are selected.

## 5. The mechanics, step by step

The sequence below is ordered so that each step constrains the next. Most bad DiD work is not a
computational error; it is a step skipped, usually Step 1 or Step 2, and discovered only when
somebody asks what the number means.

### Phase A — Fix the question before touching data

**Step 1. Name the decision the estimate will inform.**

- Write the decision as a sentence with a threshold: "Extend the three-delivery schedule to the
  remaining nine DCs if the effect exceeds 25 cases per store-week."
- Key branch: **is the decision about a policy or about a population?** A decision about extending
  a programme needs the effect of the programme as implemented. A decision about a theory needs an
  estimand that may weight cohorts differently.

**Step 2. Fix the assignment unit and the outcome unit.**

- Assignment happens where the intervention is applied: the DC, the region, the lane, the
  supplier. The outcome may be measured lower down, at the store or the shipment.
- Key branch: **do the two levels differ?** If treatment is assigned to 12 DCs but the outcome is
  measured on 4,800 store-weeks, you have 12 independent units, not 4,800. Topic 04.05 gives the
  variance arithmetic; this topic gives the design consequences.

**Step 3. Fix the time origin and the window.**

- The origin is the implementation date, not the announcement date and not the fiscal quarter in
  which the budget was approved. If anticipation is plausible, model it as a lead rather than
  pretending it did not happen.
- Choose pre and post windows of equal length where possible, and state why the length was
  chosen. A four-week pre-period cannot support a trend claim.
- Key branch: **does the post window extend past a known regime change?** A carrier renegotiation,
  a holiday peak, or a system migration inside the post period contaminates the comparison.

### Phase B — Build the four cells

**Step 4. Collapse the raw rows to cell means.**

- Average the outcome over units within group and period. Keep the unit count and the observation
  count beside every mean.
- Key branch: **is the panel balanced?** If stores enter or leave, a mean can move because the mix
  changed. Report the count per cell; a change of more than a few percent needs a fix.

**Step 5. Compute the two first differences and the second difference.**

- `treated change = post mean − pre mean`; `comparison change = post mean − pre mean`;
  `DiD = treated change − comparison change`.
- Key branch: **are the units consistent across the subtraction?** Cases per store-week minus
  cases per store-week is fine. A total minus a per-store average is not.

**Step 6. Check composition and measurement.**

- Confirm the same stores, the same product scope, and the same definition of the outcome in both
  periods. Confirm that the data extract was not changed at the same time as the intervention.
- Key branch: **did the outcome definition change?** A switch from "cases shipped" to "cases
  delivered" mid-panel produces a level shift that looks exactly like an effect.

### Phase C — Move to the regression form

**Step 7. Write the 2x2 regression.**

- `Y_it = alpha + beta·Treated_i + gamma·Post_t + delta·(Treated_i × Post_t) + e_it`.
- The coefficients are cell combinations, not abstractions: `alpha` is the comparison pre mean,
  `beta` the baseline gap, `gamma` the comparison group's change, `delta` the DiD estimate.
- Key branch: **is the interaction coded 1 only in the treated post cell?** A common error is to
  use a post indicator that turns on at different dates for different units without a
  cohort-specific structure.

**Step 8. Add covariates only where they earn their place.**

- Time-invariant covariates are absorbed by the group indicator and cannot be estimated. Time-
  varying covariates can improve precision if they predict the outcome and are unaffected by
  treatment.
- Key branch: **is the covariate a post-treatment outcome?** Controlling for a variable that the
  treatment changed — a mediator such as delivery frequency or inventory level — removes part of
  the effect and biases the estimate toward zero.

**Step 9. Replace the group indicator with unit fixed effects for panel data.**

- With more than two periods, `Y_it = alpha_i + gamma_t + delta·D_it + e_it` is the two-way fixed
  effects form. With two periods it is numerically identical to Step 7.
- Key branch: **does `D_it` switch on once and stay on?** If treatment is absorbing, the
  coefficient is a DiD. If it toggles, the design is closer to a switchback and the timing
  assumptions differ.

### Phase D — Look at the path, not only the average

**Step 10. Estimate an event study.**

- Replace the single post indicator with indicators for event time: `Y_it = alpha_i + gamma_t +
  sum_k tau_k·1[t − E_i = k] + e_it`, omitting one period as the reference.
- The coefficients trace the effect's build-up, peak, and decay. An average over a path that rises
  and then falls can be zero while the effect is real and large at both ends.
- Key branch: **are the leads flat?** Flat leads are supportive. A visible slope is a warning, not
  a verdict; Step 12 quantifies how much slope the data could have detected.

**Step 11. Choose the reference period and handle the endpoints deliberately.**

- The usual choice is `k = −1`, the period immediately before adoption. Choosing `k = −2` leaves
  `−1` free and lets you see anticipation. Either is defensible; the default is not a reason.
- Binning distant leads or lags into one coefficient changes the estimate and can create negative
  implicit weights. Report whether binning was used and at what distance.
- Key branch: **is there a never-treated group?** Without one, the full lead-and-lag path is not
  separately identified from a linear trend, and the choice of reference period materially changes
  the answer.

**Step 12. Test pre-trends, then quantify what the test could have seen.**

- Run the joint test on the leads. Report the F statistic, the degrees of freedom, and the p-value.
- Then compute the smallest linear pre-trend the test could detect at 80% power, and express it as
  a share of the estimated effect. This is the step almost everyone skips.
- Key branch: **is the detectable trend smaller than the effect?** If the test could only detect a
  pre-trend that would explain most of the estimate, passing tells you almost nothing.
  estimate."

### Phase E — Handle staggered adoption

**Step 13. Diagnose whether the rollout is staggered.**

- Plot adoption dates by unit. If more than one date exists and treatment is absorbing, the TWFE
  coefficient is a weighted average of many 2x2 comparisons, not a single effect.
- Key branch: **are the effects homogeneous and constant?** If they are, TWFE is fine. If they grow
  with exposure, vary by cohort, or vary by calendar time, it is not.

**Step 14. Decompose the TWFE estimate.**

- Report the weights and the 2x2 estimates. Identify any comparison that uses already-treated
  units as controls and any weight below zero.
- Key branch: **how much of the estimate comes from the forbidden comparisons?** If the answer is
  "most of it", the TWFE number is not an average of effects you would defend.

**Step 15. Re-estimate with group-time effects.**

- Estimate `ATT(g,t)` for each cohort and period, using never-treated units where available and
  not-yet-treated units otherwise. The Callaway–Sant'Anna and Sun–Abraham estimators are the two
  standard implementations; an imputation estimator is a third.
- Key branch: **is the control group clean for every `(g,t)`?** A not-yet-treated control must be
  untreated throughout the window used for that comparison, not merely at its start.

**Step 16. Aggregate on purpose.**

- Choose the aggregation that matches the decision: event-time average for a build-up question,
  cohort average for a rollout question, calendar-time average for a budget question, overall
  average for a programme-level question. They are different numbers and all can be correct.
- Key branch: **does the aggregation weight cohorts by size or equally?** State it, because a
  small early cohort with a large effect can dominate an equal-weighted average.

### Phase F — Infer and report

**Step 17. Cluster at the level of treatment assignment.**

- If treatment is assigned to a DC, cluster at the DC. Clustering at the store is not enough,
  because stores inside a DC share the treatment and the DC-level shock.
- Key branch: **is the cluster level coarser than the treatment level?** Clustering at region when
  treatment is assigned at DC is conservative but loses power. Clustering at store when treatment
  is assigned at DC is anti-conservative and common.

**Step 18. Deal with few clusters.**

- With fewer than about 30 to 50 clusters, asymptotic cluster-robust tests over-reject. Report the
  wild cluster bootstrap p-value, or a randomisation inference p-value, alongside the asymptotic
  one.
- Key branch: **how many clusters are treated?** A design with 100 clusters and 4 treated is a
  4-cluster design for the treatment effect, whatever the software prints.

**Step 19. Write the sentence the design supports.**

- Format: estimate, interval, units, assumption, the test that could have refuted it, and the
  decision it does or does not support.
- Key branch: **does the executive summary use a stronger verb than the methods section?** "The
  programme raised sales by 34 cases" is a causal claim. "We estimate that the programme raised
  sales by 34 cases, under parallel trends" is the same number with its warrant attached.

> **Note:** Steps 13 to 16 are not robustness checks to be run after the main result. If the
> rollout is staggered and effects plausibly grow with exposure, the TWFE estimate is the wrong
> estimate, and the group-time estimator is the main result. Running both and reporting the one
> that supports the business case is the failure this sequence exists to prevent.

## 6. Worked example

Meridian Distribution is a grocery distributor serving 15 regional distribution centres (DCs).
Each DC supplies about 20 stores. In week 27 of a 52-week year, Meridian began a three-delivery-
per-week schedule at six DCs, up from two. The commercial hypothesis is that more frequent
delivery raises shelf availability, and therefore sales, without enough extra cost to offset the
gain. All figures are illustrative and internally consistent.

### 6.1 The decision and the data

**The decision.** Extend the schedule to the remaining DCs if the effect exceeds 28 cases per
store-week. The threshold comes from the cost side, computed in section 6.9; it is fixed before
the analysis, which is the point.

**The data.** A weekly extract with one row per store-week:

| Column | Type | Example | Note |
|---|---|---|---|
| `week` | integer | 27 | ISO week of the year |
| `dc_id` | string | `DC-04` | Treatment assigned here |
| `store_id` | string | `S-0412` | 20 stores per DC, unbalanced in practice |
| `cases_sold` | integer | 1,284 | Cases sold in that store-week |
| `deliveries` | integer | 3 | Scheduled deliveries that week |
| `open_flag` | boolean | true | False if the store was closed part of the week |

The raw extract has 15 DCs × 20 stores × 52 weeks, about 15,600 rows before exclusions. A
five-row sample looks like this:

| week | dc_id | store_id | cases_sold | deliveries | open_flag |
|---:|---|---|---:|---:|---|
| 26 | DC-04 | S-0412 | 1,262 | 2 | true |
| 27 | DC-04 | S-0412 | 1,301 | 3 | true |
| 27 | DC-04 | S-0413 | 1,188 | 3 | true |
| 27 | DC-07 | S-0701 | 1,204 | 2 | true |
| 28 | DC-07 | S-0701 | 1,241 | 2 | true |

**The exclusions, stated before the estimate.** Drop partial weeks, stores with fewer than 40 of
52 weeks, and the four holiday weeks. Each changes the cell means, so each is a design decision
belonging in the methods note, not a footnote.

> **Note:** The extract must be pulled with the same outcome definition in both periods. A switch
> from "cases shipped" to "cases delivered" during the year would create a level shift roughly the
> size of the effect being measured, and no amount of regression would reveal it.

### 6.2 From weekly rows to cell means

Take the six wave-1 DCs and the six never-treated DCs. Use weeks 15 to 26 as the pre-period and
weeks 27 to 38 as the post-period, twelve weeks each. Average across stores and weeks within each
group and period.

| Cell | DCs | Stores | Store-weeks | Mean cases per store-week |
|---|---:|---:|---:|---:|
| Treated, pre | 6 | 118 | 1,392 | 1,240 |
| Treated, post | 6 | 118 | 1,384 | 1,318 |
| Comparison, pre | 6 | 118 | 1,390 | 1,180 |
| Comparison, post | 6 | 118 | 1,391 | 1,222 |

Two details in that table are easy to skip and both matter. The store count is 118, not 120,
because two stores closed during the year; the panel is unbalanced at the margin but the same
stores appear in both periods, so the composition is stable. The store-week counts differ by a
handful because of the holiday exclusions, which is why the means are not computed on identical
row counts.

### 6.3 The 2x2 arithmetic by hand

```text
treated change     = 1,318 - 1,240 = +78.0 cases per store-week
comparison change  = 1,222 - 1,180 = +42.0 cases per store-week

DiD                = 78.0 - 42.0   = +36.0 cases per store-week

as a share of the comparison pre mean:
    36.0 / 1,180 = 3.05%

as a share of the treated pre mean:
    36.0 / 1,240 = 2.90%
```

The first line is the number a manager would quote if asked "how much did sales go up?" It is
+78, and it is wrong as an estimate of the programme, because 42 of those cases were coming
anyway. The comparison group is not a formality; it is doing 54% of the work in this example.

**Why the comparison group is the six never-treated DCs.** They are in the same retail market,
face the same consumer demand, and buy through the same supplier base. They differ in store
format and in average volume, which is a level difference and is differenced away. What matters
is whether they were on the same trajectory, which section 6.6 examines.

### 6.4 The regression and what each coefficient means

With two periods and two groups, the regression is:

```text
Y_it = alpha + beta·Treated_i + gamma·Post_t + delta·(Treated_i × Post_t) + e_it

where  Treated_i = 1 for the six wave-1 DCs, 0 for the three comparison DCs
       Post_t    = 1 for weeks 27-38, 0 for weeks 15-26
       Y_it      = cases sold per store-week
```

The four coefficients are not abstract. Each is a cell or a difference of cells:

| Coefficient | Meaning | Value | Units |
|---|---|---:|---|
| `alpha` | Comparison group, pre-period mean | 1,180 | cases per store-week |
| `beta` | Baseline gap: treated pre minus comparison pre | 60 | cases per store-week |
| `gamma` | Comparison group's change from pre to post | 42 | cases per store-week |
| `delta` | DiD: treated change minus comparison change | 36 | cases per store-week |

Reconstruct the four cells from the coefficients to check the fit:

```text
comparison, pre   = alpha                 = 1,180
comparison, post  = alpha + gamma         = 1,180 + 42  = 1,222
treated, pre      = alpha + beta          = 1,180 + 60  = 1,240
treated, post     = alpha + beta + gamma + delta
                  = 1,180 + 60 + 42 + 36  = 1,318
```

All four match the table in 6.2. That is the check to run every time: if the regression does not
reproduce the cell means, the coding is wrong, not the data.

**What `beta` is not.** It is not "the effect of being a wave-1 DC". It is the pre-period level
difference, which may reflect store format, catchment income, or the manager's willingness to
volunteer. The design does not need it to be zero. It needs the *change* to be comparable.

**What `delta` assumes.** That `gamma`, the comparison group's change, is what the treated group
would also have experienced. That is the parallel-trends assumption in regression clothing.

### 6.5 Standard errors: why 12, not 5,557

The regression above is estimated on 12 DCs observed twice, or 24 DC-period rows. It could also be
estimated on 1,392 + 1,384 + 1,390 + 1,391 = 5,557 store-weeks, and the software will report a
standard error based on that row count. That standard error is wrong, because treatment was assigned to
DCs. Stores inside a DC share the DC's management, its route plan, its labour pool, and its
shocks.

The clean way to see it is to collapse to one change per DC and run the difference in means. The
per-DC changes are:

| Wave-1 DC | Change | Never-treated DC | Change |
|---|---:|---|---:|
| DC-01 | +84 | DC-10 | +54 |
| DC-02 | +58 | DC-11 | +24 |
| DC-03 | +96 | DC-12 | +50 |
| DC-04 | +70 | — | — |
| DC-05 | +82 | — | — |
| DC-06 | +78 | — | — |
| **Mean** | **+78.0** | **Mean** | **+42.0** |

```text
treated changes:   mean = 468 / 6 = 78.0
                   deviations = 6, -20, 18, -8, 4, 0
                   sum of squares = 36 + 400 + 324 + 64 + 16 + 0 = 840
                   variance = 840 / 5 = 168.0
                   sd = 12.96

comparison changes: mean = 252 / 6 = 42.0
                   deviations = 12, -18, 8, -8, 2, 4
                   sum of squares = 144 + 324 + 64 + 64 + 4 + 16 = 616
                   variance = 616 / 5 = 123.2
                   sd = 11.10

SE of the DiD = sqrt(168.0/6 + 123.2/6)
              = sqrt(28.00 + 20.53)
              = sqrt(48.53)
              = 6.97 cases per store-week

t = 36.0 / 6.97 = 5.17, df = 11
95% CI = 36.0 ± 2.201 × 6.97 = 36.0 ± 15.3 = [20.7, 51.3]
```

The design effect makes the cost of pretending visible. With 24 weeks per DC and an
intra-cluster correlation of 0.62 in the DC-week residuals:

```text
DEFF = 1 + (m - 1) × ICC = 1 + 23 × 0.62 = 15.26
sqrt(DEFF) = 3.91

naive SE if 24 DC-weeks were independent = 6.97 / 3.91 = 1.78
naive 95% CI                            = 36.0 ± 3.9
```

An analyst who ignores the clustering reports an interval of roughly 32 to 40 and concludes the
programme clears the 28-case hurdle with room to spare. The clustered interval, 21 to 51, does
not. Same data, same point estimate, opposite decision. This is the Bertrand, Duflo, and
Mullainathan (2004) problem in miniature, and it is the most common way a defensible DiD estimate
is turned into an indefensible recommendation.

### 6.6 The event study: the path behind the average

The 2x2 gives one number for a twelve-week post period. The event study gives twelve, plus eight
leads. Estimate it with the same fixed effects, omitting event time `k = −1`.

| Event time `k` | Weeks from adoption | Coefficient | SE | Reading |
|---:|---|---:|---:|---|
| −8 | 8 weeks before | +4.0 | 4.1 | noise |
| −7 | 7 weeks before | +1.0 | 4.0 | noise |
| −6 | 6 weeks before | +3.0 | 4.0 | noise |
| −5 | 5 weeks before | −2.0 | 4.0 | noise |
| −4 | 4 weeks before | +2.0 | 4.0 | noise |
| −3 | 3 weeks before | −1.0 | 4.1 | noise |
| −2 | 2 weeks before | +1.0 | 4.0 | noise |
| −1 | 1 week before | 0 | — | reference period |
| 0 | adoption week | +10.0 | 4.0 | build-up begins |
| 1 | 1 week after | +19.0 | 4.1 | |
| 2 | 2 weeks after | +28.0 | 4.2 | |
| 3 | 3 weeks after | +35.0 | 4.2 | |
| 4 | 4 weeks after | +39.0 | 4.3 | |
| 5 | 5 weeks after | +41.0 | 4.3 | |
| 6 | 6 weeks after | +43.0 | 4.3 | plateau approaching |
| 7 | 7 weeks after | +45.0 | 4.4 | plateau |
| 8–11 | 8 to 11 weeks after | +43.0 | 4.4 | binned, four periods |

**Consistency check.** The mean of the twelve post coefficients must equal the 2x2 estimate if
the post window is the same. Sum the unbinned and binned values:

```text
10 + 19 + 28 + 35 + 39 + 41 + 43 + 45 = 260
four binned periods at 43          = 172
total                              = 432
mean = 432 / 12 = 36.0 cases per store-week
```

It matches. If it did not, the event-study window and the 2x2 window would differ, and the
appendix would need to say which is the headline.

**What the path adds.** The average of +36 conceals a ramp. The effect is roughly zero at
adoption, reaches about +40 by week 4, and plateaus near +43. For an operating decision this is
more useful than the average: the benefit is not available in the first two weeks, so a business
case that assumes the full effect from week 1 overstates first-quarter value by roughly a third.
The plateau also suggests the mechanism is a stocking and shelf-availability adjustment that takes
about a month to complete, which is a testable claim about stores, not a statistical artefact.

**What the path does not show.** Persistence beyond week 11: the sample ends at week 52, and the
appendix should show the later weeks marked as outside the balanced window.

### 6.7 What the pre-trend test could and could not see

The joint test on the seven free leads gives `F(7, 11) = 1.12`, `p = 0.42`. The usual sentence is
"the pre-trends are flat, supporting the parallel-trends assumption." That sentence is not
supported by the test, and the arithmetic shows why.

```text
assume each lead has SE = 4.0 cases per store-week
leads at k = -8 ... -1, mean k = -4.5
sum of squared deviations = 12.25 + 6.25 + 2.25 + 0.25
                          + 0.25 + 2.25 + 6.25 + 12.25 = 42.0

SE of a linear slope through the leads = 4.0 / sqrt(42.0) = 0.617 cases/week
MDE at 80% power, 5% two-sided        = 2.80 × 0.617 = 1.73 cases/week
over a 12-week post window             = 1.73 × 12 = 20.7 cases
as a share of the 36.0-case estimate   = 20.7 / 36.0 = 58%
```

The test had 80% power to detect a differential pre-trend only if that trend would have produced
about 21 cases over the post window, which is 58% of the estimated effect. A trend half that
size — enough to explain 10 of the 36 cases — would have been missed more than one time in five.
So `p = 0.42` is consistent with parallel trends and with a bias worth 10 cases. It rules out a
gross divergence. It does not certify the counterfactual.

This is Roth's (2022) point with numbers attached. He shows that pre-trend tests in published
event studies frequently have power only against violations larger than the effect being
estimated, and that conditioning on passing the test makes the reported interval too narrow,
because the estimates that survive are a selected sample.

> **Note:** Do not respond to a weak pre-trend test by lengthening the pre-period until the test
> passes. That is specification search on the diagnostic. The defensible responses are to state
> the detectable trend, to add a placebo date, and to report a sensitivity analysis for departures
> from parallel trends (Topic 04.10).

### 6.8 The rollout was staggered, and the TWFE number is not the effect

Meridian did not stop at six DCs. Three more — wave 2 — adopted in week 40. The remaining six
never adopted. An analyst who runs the obvious regression on the full panel,

```text
Y_it = alpha_i + gamma_t + delta·D_it + e_it
```

with `D_it = 1` once a DC has adopted, gets `delta = 26.2`. That number is 7.8 below the
cohort-weighted average of 34.0, and the reason is the composition of the comparisons.

The Goodman-Bacon (2021) decomposition writes `delta` as a weighted average of every 2x2
comparison in the data. For this panel the decomposition looks like this:

| Comparison | Control used | 2x2 estimate | Signed weight | Contribution |
|---|---|---:|---:|---:|
| Wave 1 vs never-treated | 6 clean DCs | +36.0 | +0.30 | +10.80 |
| Wave 2 vs never-treated | 6 clean DCs | +30.0 | +0.12 | +3.60 |
| Wave 1 vs wave 2, before week 40 | Wave 2, not yet treated | +18.0 | +0.72 | +12.96 |
| Wave 2 vs wave 1, after week 40 | Wave 1, **already treated** | +8.0 | −0.14 | −1.12 |
| **TWFE** | | | **1.00** | **+26.24** |

Two rows carry the lesson.

**The third row is why the estimate is low.** It gets 72% of the weight because wave 2's adoption
in week 40 makes its treatment indicator vary over a long stretch of the panel, and the variance
of the treatment indicator is what drives the weights. But its window is weeks 27 to 39, when the
wave-1 effect is still ramping — the event study says the effect at week 3 is about +35, not the
plateau of +43. So this comparison measures an early-window effect and hands it most of the
weight. Nothing is wrong with the comparison; the weighting is what misleads.

**The fourth row is the forbidden comparison.** After week 40, wave 1 is already treated, and its
outcome is still rising as its effect matures. Using it as the control for wave 2 means subtracting
a change that contains wave 1's treatment effect. The resulting 2x2, +8.0, is not the effect of
the programme on wave 2; it is roughly wave 2's effect minus the part of wave 1's effect that
accrued in the same window. Its weight is negative, −0.14, so it subtracts 1.1 cases from the
total. A negative weight is not a software fault. It is the algebra telling you that this
comparison moves the estimate in the opposite direction to the effect it is supposed to measure.

**What the decomposition does not say.** It does not say DiD is invalid. It says the two-way
fixed effects estimator with a single treatment indicator is a variance-weighted average, and
that the weights are not the ones a decision-maker would choose. With homogeneous, constant
effects, all the 2x2s would agree and the problem would vanish. The problem is real precisely
because effects grow with exposure, which is the normal case in operations.

### 6.9 The corrected estimate, and the decision

The repair is to estimate cohort-by-period effects and aggregate them deliberately. Using the
Callaway–Sant'Anna (2021) group-time estimator with never-treated DCs as controls where available
and not-yet-treated DCs otherwise:

| Cohort | DCs | Post weeks used | Average effect | SE |
|---|---:|---:|---:|---:|
| Wave 1 (week 27) | 6 | 12 | +36.0 | 7.0 |
| Wave 2 (week 40) | 3 | 12 | +30.0 | 9.8 |
| **Overall, weighted by cohort size** | **9** | | **+34.0** | **7.4** |

```text
overall = (6 × 36.0 + 3 × 30.0) / 9
        = (216.0 + 90.0) / 9
        = 306.0 / 9
        = 34.0 cases per store-week

95% CI, t with 11 df = 34.0 ± 2.201 × 7.4
                     = 34.0 ± 16.3
                     = [17.7, 50.3]
```

**The decision arithmetic.** A DC serves about 20 stores. The contribution margin is €2.10 per
case. The programme costs about €60,000 per DC per year in slot booking, extra runs, and picking
labour.

```text
annual benefit at the point estimate
    = 20 stores × 34.0 cases × EUR 2.10 × 52 weeks
    = EUR 74,256

net = 74,256 - 60,000 = EUR +14,256 per DC-year

break-even effect
    = 60,000 / (20 × 2.10 × 52)
    = 60,000 / 2,184
    = 27.5 cases per store-week

annual benefit at the lower confidence bound
    = 20 × 17.7 × 2.10 × 52
    = EUR 38,657
net = 38,657 - 60,000 = EUR -21,343
```

**The defensible sentence.** "We estimate that three deliveries per week raised sales by 34 cases
per store-week (95% CI 18 to 50), using a difference-in-differences design across 12 DCs with
cohort-by-period effects and standard errors clustered at the DC. The estimate is conditional on
parallel trends, which the pre-period supports only weakly: the pre-trend test had power to detect
a violation explaining 58% of the effect, not a smaller one. The point estimate clears the
27.5-case break-even threshold, but the lower bound does not."

**The decision this supports.** Not a full rollout. A staged extension with a pre-registered
stopping rule, or a negotiation on the €60,000 cost, or a search for why wave 2 is weaker. That is
a normal place for a quasi-experiment to land.

### 6.10 What an analyst should have asked

1. Why are the six wave-1 DCs the treated group rather than a random sample of willing DCs? If
   they volunteered because they were already improving, the pre-trend test was too weak to rule
   it out. The level gap of 60 cases is differenced away; the trend gap is not.
2. What happened in week 40 besides wave 2's adoption? A calendar-time shock hitting wave 2 and
   the never-treated group differently would contaminate the wave-2 estimate, which carries one
   third of the overall weight. And does the effect persist past week 11, or is it a one-quarter
   stocking adjustment with a different business case?

## 7. Code, query, or calculation

One block, and it reproduces section 6.5 exactly: the difference in mean changes, and a
cluster-robust variance with the small-sample correction that turns the HC0 number into the
6.97 reported above.

```python
import numpy as np

# One change per DC: six wave-1 DCs, then six never-treated DCs
chg   = np.array([84, 58, 96, 70, 82, 78, 54, 24, 50, 34, 44, 46], float)
treat = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)

X = np.column_stack([np.ones(12), treat])          # intercept, then DiD
b  = np.linalg.lstsq(X, chg, rcond=None)[0]
e  = chg - X @ b
G, N, K = len(chg), len(chg), X.shape[1]           # one row per cluster here
XtX_inv = np.linalg.inv(X.T @ X)
V = XtX_inv @ (X.T * e**2) @ X @ XtX_inv           # HC0 cluster-robust
V *= (G / (G - 1)) * ((N - 1) / (N - K))            # small-sample correction
se = np.sqrt(np.diag(V))
print(f"comparison change = {b[0]:.1f}")
print(f"DiD               = {b[1]:.1f} (SE {se[1]:.2f})")
# comparison change = 42.0 ; DiD = 36.0 (SE 6.97)
```

Three checks before you trust a variant of this. The point estimate must equal the mean-of-changes
difference, which it does by construction here. The standard error must equal
`sqrt(s_t^2/6 + s_c^2/6)`, which it does because with one row per cluster the CRVE reduces to the
two-sample formula. And the correction factor must be present; without it the printed SE is 6.36,
about 9% too small, and that is the smallest error in the sequence.

## 8. Data, standards, and regulatory context

DiD is a data-hungry design disguised as a simple one. It needs a panel that is stable in
definition, balanced at the unit level, and aligned in time. Most of the work is provenance, not
estimation.

**The identifiers that make the design possible.**

| Element | What it must be | Why DiD fails without it |
|---|---|---|
| Unit identifier | Stable across the whole window | A renumbered DC becomes two units and breaks the within-unit difference |
| Adoption date | Recorded per unit, at implementation | A single company-wide date hides staggering and triggers the wrong estimator |
| Treatment flag | Derived from the adoption date, not typed by hand | Hand flags drift from the rollout record |
| Outcome definition | Identical in both periods | A definition change is indistinguishable from an effect |
| Store roster | Dated, with openings and closures | Composition change moves means without any behaviour changing |

**Time conventions.** Weeks must be numbered by one rule across all source systems. ISO 8601 weeks
start on Monday; retail 4-4-5 calendars do not align with them, so a fiscal week 27 can contain
parts of two ISO weeks. If the DC schedule runs on a fiscal calendar and the sales extract on ISO
weeks, the adoption date lands in different weeks in the two files. Topic 01A.04 treats this in
detail; for DiD the rule is one calendar, documented, never recomputed downstream.

**The regulatory frame is indirect but real.** DiD itself is not regulated. The data it consumes
are:

| Requirement | Where it bites | Grade |
|---|---|---|
| GDPR and comparable privacy law | Store-level sales are usually aggregate, but worker-level productivity outcomes may be personal data | B |
| Works council and employee consultation | A productivity DiD at DC level may require consultation before the data are analysed, not only before the change | B |
| Accounting standards for the cost side | Contribution margin must match the management accounts used in the business case | B |
| Internal audit and model risk policy | A rollout decision based on a quasi-experiment is a model output and may need documentation under model risk rules | C |

> **Note:** The most common data failure in DiD is not missing data. It is a silent change in the
> extract. A new warehouse table, a renamed column, or a switch from shipped to delivered cases
> will produce a level shift at the changeover date that the model will attribute to the
> treatment. Record the extract version and the query hash for each period; if you cannot say
> which code produced the pre-period mean, you cannot defend the difference.

**What is convention rather than requirement.** The twelve-week pre and post windows, the choice
of `k = −1` as the reference, and the 5% significance level are all conventions. None is mandated
by a standard, and each should be stated as a choice with a reason.

## 9. Industry practice and case evidence

### 9.1 The four studies that define the method's reputation

**Ashenfelter and Card (1985)** is the origin of the diagnostic that bears Ashenfelter's name. Using
Social Security earnings histories for the 1976 cohort of Comprehensive Employment and Training
Act (CETA) trainees and a comparison group drawn from the Current Population Survey, they
documented that trainees' real earnings fell sharply in the year before training — from $2,785 in
1974 to $1,898 in 1975 — while the comparison group's earnings rose. The paper's enduring
contribution is not the training effect, which it reports as highly sensitive to the choice of
selection year and to the inclusion of individual-specific trends, with estimates ranging roughly
from $200 to $2,000 per year. It is the demonstration that **the pre-period is part of the
identification, not a formality**, and that a dip at the moment of selection can mimic or mask an
effect. Grade A.

**Card and Krueger (1994)** is the most cited DiD in applied economics. New Jersey raised its
minimum wage from $4.25 to $5.05 per hour on 1 April 1992 while neighbouring eastern Pennsylvania
did not. A two-wave survey of 410 fast-food restaurants gave a DiD estimate of +2.76 full-time
equivalent employees per store (t = 2.03), and a regression estimate of +2.30 (standard error
1.19) — a relative gain, not the loss the competitive model predicted. Prices rose about 3.2% to
3.7% faster in New Jersey. The design lesson is the specification check: within New Jersey,
initially low-wage stores expanded while high-wage stores moved with Pennsylvania, which is the
pattern a valid comparison group should produce. Grade A for the study; the interpretation
remains contested in the literature, which is a separate matter from the design.

**Bertrand, Duflo, and Mullainathan (2004)** changed how the method is taught. They surveyed 92
DiD papers in six journals from 1990 to 2000, found an average of 16.5 periods per unit, and found
that almost none corrected for serial correlation. Using placebo laws in state-level wage data,
they showed that a spurious effect appeared significant at the 5% level in up to 45% of placebo
interventions. Estimated residual autocorrelations were 0.51, 0.44, and 0.33 at lags one to three.
Grade A for the method and the simulation.

**Goodman-Bacon (2021)** supplied the algebra behind a suspicion that had been growing since the
mid-2010s. His decomposition shows that the two-way fixed effects DiD estimator is a weighted
average of all 2x2 comparisons, that comparisons using already-treated units as controls receive
negative weights, and that the estimand is a variance-weighted average requiring both parallel
trends and constant treatment effects over time. In his replication of the unilateral-divorce
literature, TWFE gives −3.08 while the event-study average is about −5. Grade A.

### 9.2 Why the correction has not diffused

The components are standard commands; what lags is their use. A 2026 analyst who runs
`Y ~ treat*post + unit + time` on a staggered rollout and reports the coefficient is using an
estimator that the field has documented as biased in exactly that setting.

| Observed practice | Why it persists | Consequence |
|---|---|---|
| One treatment indicator for a staggered rollout | It is the default in every textbook and course | The estimate is a variance-weighted mix, possibly with negative weights |
| Pre-trend p-value reported as a pass | It is the only diagnostic the software prints | Low-power tests certify designs they cannot support |
| Clustering at the row level | The row identifier is the only ID in the extract | Standard errors too small by a factor of two to four |
| Business case built on the point estimate | The interval is in an appendix nobody reads | Rollout proceeds when the lower bound is below break-even |
| Control group chosen after seeing the result | The analysis is iterative and undocumented | The reported interval ignores the selection |

The last row is the most damaging and the least discussed. A DiD with a comparison group chosen
because it produced a plausible answer is not a DiD; it is a specification search with a causal
verb attached. The defence is procedural: fix the comparison group in the design memo, before the
outcome data are joined.

### 9.3 Vendor and analyst claims

Analytics vendors now sell "causal impact" modules that run a DiD or a synthetic control behind a
dashboard. The claims are typically of the form "measure the true incremental lift of your
promotions" without stating the comparison group, the parallel-trends argument, the clustering
level, or the estimator's behaviour under staggered rollout. Grade D, dated 2026, based on the
general form of vendor material rather than a specific audited product test. The correct reading
is that the software computes a DiD; whether the design supports a causal claim is a property of
the deployment, not of the licence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No comparison group | "We compared before and after" | Common trend attributed to treatment | Effect overstated by the secular change | Add an untreated group or label the result descriptive |
| Comparison group chosen post hoc | Several comparisons were tried | Selection on the estimate | Interval too narrow, effect inflated | Pre-register the comparison in a design memo |
| Divergent pre-trends | Leads slope visibly | Differential trend continues | Bias equal to slope times window | Find another comparison; model trends; bound the bias |
| Pre-trend test treated as proof | "p > 0.05, so parallel trends hold" | Low power | Unsupported assumption | Report detectable trend as a share of the effect |
| Staggered rollout with one indicator | Adoption dates differ across units | Negative weights, forbidden comparisons | Estimate is not any cohort's effect | Decompose; use group-time estimators |
| Reference period chosen by default | Nobody can say which `k` was omitted | Normalisation changes all coefficients | Results not comparable across papers | State the reference and test the alternative |
| Endpoint binning | A single "8+" coefficient | Implicit homogeneity, negative implicit weights | Path distorted at the tail | Show unbinned or justify the bin |
| Clustering at the wrong level | `G` equals the row count | Treatment assigned to clusters | SE too small by `sqrt(DEFF)` | Cluster at assignment; report `G` |
| Few treated clusters | 40 clusters, 3 treated | Asymptotics fail | Over-rejection | Wild cluster bootstrap or randomisation inference |
| Composition change | Store count drifts between cells | Mean moves without behaviour changing | Spurious level shift | Balanced panel or store fixed effects with a roster check |
| Definition change mid-panel | A step at a date unrelated to rollout | Measurement shift | Effect contaminated | Version the extract; compare definitions |

**Divergent pre-trends** is the failure the method is named for: the estimate contains the
continuation of a gap that was already opening. In the worked example a differential trend of
about 3 cases per week would explain the whole effect, and the test could detect only about 1.7,
which is why the pre-trend result is reported with its power rather than as a verdict.

**Staggered rollout with one indicator** is invisible in the regression output: every coefficient
looks reasonable, `R^2` is fine, and the standard error is small. But the coefficient is a weighted
average whose weights can be negative, so a programme positive in every cohort can produce a
negative TWFE estimate. Plot adoption dates before estimating anything.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Parallel trends | Supplies the counterfactual change for the treated group | A differential trend that starts before or at treatment | Event-study leads; placebo date; sensitivity bounds |
| No anticipation | Keeps the pre-period uncontaminated | Announcement effects, pre-positioning, contract renegotiation | Free the `k = −1` coefficient; check the announcement date |
| Stable composition | Makes the cell means comparable across periods | Store openings, closures, reclassification | Count units per cell; balance the panel |
| Stable outcome definition | Makes the two periods the same measurement | Extract or process change | Version the query; compare definitions |
| No differential shock at treatment | The second difference must remove only common shocks | A simultaneous promotion, price change, or competitor exit | Negative control outcome; check the calendar |
| SUTVA within the comparison group | Control units must not be affected by treated units | Shared fleet, shared stock, demand diversion | Map sharing channels; use distant controls |
| Correct treatment date | Event time must be right | Date recorded at approval, not implementation | Reconcile against the rollout log |
| Homogeneous or correctly modelled effects | TWFE only targets a simple average under homogeneity | Effects that grow with exposure or vary by cohort | Decompose; estimate group-time effects |
| Correct clustering level | Standard errors must reflect assignment | Clustering below the treatment level | Compare SEs at two levels; report `G` |

**DiD cannot rescue a bad comparison group.** No estimator or robustness table fixes a control
group that is a different business. If the never-treated DCs are the ones that could not be
converted because their volumes are collapsing, the honest output is a descriptive comparison.

**The modern corrections fix weighting, not identification.** They remove the negative-weight
contamination; they do not make parallel trends true. A group-time estimate from a bad comparison
group is a well-weighted bad number.

## 12. Exercises

### Exercise 1 — Build the 2x2 from raw rows (calculation, 40 minutes)

From 12 weeks of store-week data for four treated and four comparison DCs, compute the four cell
means, the two changes, and the DiD in cases per store-week.
**Deliverable:** the 2x2 table with `n` per cell, three arithmetic lines, and one operational
sentence. **Accepted when:** units at every step; store counts reported; DiD equals the difference of
the two changes exactly.

### Exercise 2 — The regression and the standard error (calculation, 45 minutes)

Run the 2x2 regression and the DC-level difference in means on the Exercise 1 data, confirm the
interaction equals your hand calculation, and compute the cluster-robust standard error by hand.
**Deliverable:** coefficient table with each coefficient mapped to a cell, plus the SE arithmetic. **Accepted when:** all four cells reconstructed; `G` reported; small-sample correction
applied.

### Exercise 3 — State the assumption (writing, 25 minutes)

Write the parallel-trends assumption for Exercise 1 as a counterfactual sentence, then three
sentences that would make it credible to a sceptical operations director.
**Deliverable:** one assumption sentence and three supporting sentences. **Accepted when:** the assumption refers to the post period and to what would have happened;
no support restates it; at least one names a falsification test.

### Exercise 4 — Power of the pre-trend test (calculation, 35 minutes)

From supplied lead coefficients and standard errors, compute the smallest linear pre-trend the
joint test could detect at 80% power, as a share of the estimated effect.
**Deliverable:** slope SE, MDE per week, implied bias over the post window, and the share. **Accepted when:** formula shown; units per week; the share interpreted in one sentence.

### Exercise 5 — Read an event-study path (interpretation, 30 minutes)

A path is flat for six leads, jumps to +12 at `k = 0`, rises to +30 by `k = 4`, and decays to +8 by
`k = 12`. Write the operational reading.
**Deliverable:** one paragraph on build-up, peak, decay, and what the average hides. **Accepted when:** no causal verb without the assumption; one operational implication;
reference period identified.

### Exercise 6 — Diagnose a staggered rollout (diagnosis, 50 minutes)

Adoption dates for 20 regions are 8 in month 3, 5 in month 9, 7 never. A colleague reports a single
TWFE coefficient of +4.1 points.
**Deliverable:** adoption-date table, the 2x2 comparisons embedded in the estimate, a
recommendation. **Accepted when:** at least one forbidden comparison identified; estimator and aggregation
named; cohort sizes used.

### Exercise 7 — Rebuild with group-time effects (analysis, 60 minutes)

Estimate cohort-by-period effects on the Exercise 6 data with never-treated controls, then
aggregate by event time, by cohort, and overall.
**Deliverable:** three aggregates with intervals and a sentence on why they differ. **Accepted when:** control group stated per `(g,t)`; weights stated; the three numbers are
not averaged together.

### Exercise 8 — Cluster-level inference (analysis, 40 minutes)

Recompute the Exercise 2 standard error at three levels — store-week, DC, and region (8 DCs, 2
regions) — and report `G` and the implied design effect at each.
**Deliverable:** the SE table with the recommended level and its justification. **Accepted when:** the level is justified by assignment, not by which SE is smallest; the
few-cluster problem is addressed.

### Exercise 9 — The decision memo (writing, 45 minutes)

Write a one-page rollout recommendation using the Exercise 7 estimate, a stated cost, and a
break-even calculation.
**Deliverable:** estimate with interval, break-even effect, net benefit at point estimate and
lower bound, recommendation. **Accepted when:** recommendation consistent with the lower bound; main threat named;
decision rule stated before the result.

## 13. Self-check answers

### 13.1 Exercise 2 arithmetic, in full

The four cell means are treated pre 820, treated post 902, comparison pre 760, comparison post 781,
all in cases per store-week.

```text
first differences
    treated change    = 902 - 820 = +82.0 cases per store-week
    comparison change = 781 - 760 = +21.0 cases per store-week

second difference
    DiD = 82.0 - 21.0 = +61.0 cases per store-week

share of the comparison pre mean = 61.0 / 760 = 8.0%
```

The per-DC changes, four per group:

```text
treated:    86, 74, 92, 76      mean = 328/4 = 82.0
            deviations 4, -8, 10, -6
            sum of squares = 16 + 64 + 100 + 36 = 216
            variance = 216/3 = 72.00 ; sd = 8.485

comparison: 30, 12, 26, 16      mean = 84/4 = 21.0
            deviations 9, -9, 5, -5
            sum of squares = 81 + 81 + 25 + 25 = 212
            variance = 212/3 = 70.67 ; sd = 8.407

SE = sqrt(72.00/4 + 70.67/4)
   = sqrt(18.00 + 17.67)
   = sqrt(35.67)
   = 5.97 cases per store-week

t = 61.0 / 5.97 = 10.21, df = G - 1 = 7
95% CI = 61.0 ± 2.365 × 5.97 = 61.0 ± 14.1 = [46.9, 75.1]
```

A common wrong answer omits the comparison change and reports +82. The error is 21 cases, or 34%
of the correct estimate, and it comes from attributing the secular growth of the whole market to
the programme.

### 13.2 The regression, coefficient by coefficient

`Y_it = alpha + beta·Treated + gamma·Post + delta·(Treated × Post) + e_it` gives:

| Coefficient | Value | Derivation | Units |
|---|---:|---|---|
| `alpha` | 760 | comparison pre mean | cases per store-week |
| `beta` | 60 | 820 − 760, the baseline gap | cases per store-week |
| `gamma` | 21 | 781 − 760, the common change | cases per store-week |
| `delta` | 61 | 82 − 21, the DiD | cases per store-week |

Reconstruction, which must reproduce all four cells:

```text
comparison pre   = 760
comparison post  = 760 + 21            = 781
treated pre      = 760 + 60            = 820
treated post     = 760 + 60 + 21 + 61  = 902
```

Interpretation in units: a treated DC sells 61 more cases per store-week after the change than it
would have sold if its pre-period growth of 82 cases had continued at the comparison group's rate
of 21. `beta` is not an effect; it is the pre-existing level gap of 60 cases, which the design
diffs away.

### 13.3 Design rubrics

**Exercise 3 (assumption statement).** Full marks require "without the programme, the treated DCs'
sales would have grown in parallel with the comparison DCs' over weeks 27 to 38", plus three
supports that are not restatements: a lead test, a placebo adoption date, and a negative control
outcome. Deduct heavily for "the groups were similar before", which states the evidence.

**Exercise 5 (event-study reading).** A strong answer identifies the reference period, reads the
build-up as an implementation lag, notes that the peak at `k = 4` is not captured by the average,
and names whether the decay is depletion or measurement. A weak answer describes the shape without
saying what it implies for the decision.

**Exercise 6 (staggered diagnosis).** Full marks identify the three comparison types, name the
comparison that uses already-treated regions as controls, and note that its weight depends on
cohort size and treatment variance, not on how sensible the comparison is. The recommendation must
name an estimator, a control group, and an aggregation rule. Deduct for "use a newer method"
without the aggregation choice, because the aggregation is where most of the remaining judgement
lives.

**Exercise 9 (decision memo).** Full marks compute break-even, evaluate net benefit at both the
point estimate and the lower bound, and recommend the action consistent with the lower bound. A
memo that recommends full rollout on the point estimate while reporting a lower bound below
break-even fails, however good the econometrics.

## 14. Interview questions

**1. "Explain DiD to an operations director."** Strong: compares the treated sites' improvement
with similar untreated sites' improvement, so the market-wide trend cancels. Weak: "compare before
and after."

**2. "What is the parallel-trends assumption?"** Strong: a claim about what would have happened
absent treatment, distinct from "similar before". Weak: "the two groups are the same", which is
neither necessary nor sufficient.

**3. "Your pre-trend test has p = 0.4. Are you done?"** Strong: no; asks what trend the test could
detect and compares it with the effect. Weak: yes, the assumption holds.

**4. "Why not just use store fixed effects?"** Strong: fixed effects remove levels, not
differential trends, and with staggered adoption TWFE can carry negative weights. Weak: "fixed
effects solve confounding."

**5. "What is a forbidden comparison?"** Strong: a 2x2 using already-treated units as controls, so
the estimate is a difference of two treatment effects entering with a negative weight. Weak: "a
comparison you shouldn't make."

**6. "How would you handle a rollout in three waves?"** Strong: cohort-by-period effects, never- or
not-yet-treated controls, deliberate aggregation, plus the naive weights for comparison. Weak: add
a wave dummy.

**7. "Where do you cluster?"** Strong: at the level of assignment, reporting total and treated
cluster counts, with a wild bootstrap if few. Weak: "at the store, because that's the row."

**8. "The DiD is significant but the business case is marginal."** Strong: report the interval,
break-even effect, and net benefit at the lower bound; recommend staging if the lower bound is
below break-even. Weak: quote the point estimate.

**9. "How would you test the design?"** Strong: leads, placebo date, negative control outcome, and
sensitivity to departures from parallel trends. Weak: "check the p-value."

**10. "When would you not use DiD?"** Strong: no credible comparison group, a simultaneous
differential shock, or a changed outcome definition. Weak: "when the data are bad."

## 15. Key terms

- **DiD** — treated change minus comparison change over the same window.
- **2x2** — four cell means, two first differences, one second difference.
- **First difference** — within-group change between periods.
- **Second difference** — the subtraction that removes the common trend.
- **Parallel trends** — absent treatment, the treated group would have moved parallel to the comparison group.
- **Counterfactual trend** — the unobserved path the treated group would have followed.
- **Common shock** — a change hitting both groups equally, removed by differencing.
- **Differential shock** — a change hitting the groups differently at treatment, not removed.
- **Ashenfelter dip** — a pre-treatment decline among units selected into a programme.
- **Anticipation** — behaviour changing before the formal treatment date.
- **Event time** — periods relative to a unit's own adoption date.
- **Event study** — regression on event-time indicators, tracing the effect path.
- **Lead / lag** — event-time coefficient before / at or after adoption.
- **Reference period** — the omitted event time, usually `k = −1`.
- **Pre-trend test** — joint test that the leads are zero; low power by default.
- **Placebo date** — a fake adoption date used to detect spurious effects.
- **TWFE** — two-way fixed effects regression with unit and time effects.
- **Staggered adoption** — units adopting at different calendar dates.
- **Cohort** — units adopting in the same period.
- **ATT(g,t)** — average effect for cohort `g` at time `t`.
- **Goodman-Bacon decomposition** — TWFE as a weighted average of all 2x2 comparisons.
- **Forbidden comparison** — a 2x2 using already-treated units as controls.
- **Negative weight** — a signed decomposition weight below zero.
- **VWATT** — the variance-weighted estimand TWFE targets under heterogeneity.
- **Callaway–Sant'Anna** — group-time ATT estimator with explicit aggregation.
- **Sun–Abraham** — interaction-weighted cohort-specific event study.
- **Imputation estimator** — fit on untreated observations, impute `Y(0)`, average the gaps.
- **Wild cluster bootstrap** — cluster-level bootstrap for few-cluster inference.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Ashenfelter, O. and Card, D. (1985). "Using the Longitudinal Structure of Earnings to Estimate
  the Effect of Training Programs." *Review of Economics and Statistics*, 67(4), 648–660.
  DOI: 10.2307/1924810.
- Card, D. and Krueger, A. B. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food
  Industry in New Jersey and Pennsylvania." *American Economic Review*, 84(4), 772–793.
- Bertrand, M., Duflo, E., and Mullainathan, S. (2004). "How Much Should We Trust
  Differences-in-Differences Estimates?" *Quarterly Journal of Economics*, 119(1), 249–275.
  DOI: 10.1162/003355304772839588.
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing."
  *Journal of Econometrics*, 225(2), 254–277. DOI: 10.1016/j.jeconom.2021.03.014.
- Callaway, B. and Sant'Anna, P. H. C. (2021). "Difference-in-Differences with Multiple Time
  Periods." *Journal of Econometrics*, 225(2), 200–230. DOI: 10.1016/j.jeconom.2020.12.001.
- Sun, L. and Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with
  Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175–199.
  DOI: 10.1016/j.jeconom.2020.09.006.
- Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel
  Trends." *American Economic Review: Insights*, 4(3), 305–322. DOI: 10.1257/aeri.20210236.
- Borusyak, K., Jaravel, X., and Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and
  Efficient Estimation." *Review of Economic Studies*, 91(6), 3253.

**Methodological guidance — Grade B**

- Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel Trends." Sensitivity
  bounds for departures from parallel trends; treated in Topic 04.10.
- Cameron, A. C. and Miller, D. L. (2015). "A Practitioner's Guide to Cluster-Robust Inference."
  *Journal of Human Resources*, 50(2).
- Schmidheiny, K. and Siegloch, S. (2023). "On Event Studies and Distributed-Lags in Two-Way Fixed
  Effects Models." *Journal of Applied Econometrics*, 38(5).

**Software and documentation — Grade B**

- `did` / `csdid` (Callaway–Sant'Anna); `eventstudyinteract` (Sun–Abraham);
  `did_imputation` (Borusyak–Jaravel–Spiess); `bacondecomp` (Goodman-Bacon decomposition).

**Industry and vendor — Grade D**

- Vendor "causal impact" and "incrementality" modules in commercial analytics platforms, 2026.
  Claims of automated causal measurement typically omit the comparison group, the clustering
  level, and the staggered-rollout problem. Grade D; dated 2026.

**Note on citations.** Bibliographic details were checked against publisher records. Numerical
results are those the authors report and are not independently re-estimated here. The Meridian
Distribution case is constructed for teaching and is not a real company.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite: [`05_autocorrelation-robust-errors.md`](05_autocorrelation-robust-errors.md)
- Prerequisite: [`07_experiments-quasi.md`](07_experiments-quasi.md)
- Confounding and bad controls: [`06_dags.md`](06_dags.md)
- Next topic: [`09_elasticity-endogeneity.md`](09_elasticity-endogeneity.md)
- Sensitivity to omitted bias: `10_sensitivity-omitted-bias.md` (planned)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Calendars, week numbering, and time zones: [`../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
