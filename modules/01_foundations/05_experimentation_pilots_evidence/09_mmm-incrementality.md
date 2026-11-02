# Marketing Mix Modelling and Incrementality — what an MMM can and cannot identify without an experiment

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 04, 06, 09; Module 05 topics 03–04, 07
> Version 2026.5

---

## 1. Why this topic exists

A marketing mix model (MMM) is a regression of a business key performance indicator (KPI) on marketing inputs and
control variables, with two transformations — adstock, which spreads an input over later periods, and saturation,
which bends the response curve. The output is a decomposition: how much of last year's revenue each channel
"drove", and what each channel's return on investment (ROI) was.

That output is used to move money. It is one of the few analytics artefacts in a commercial organisation that
routinely results in a nine-figure reallocation, and it is very often the only quantitative input to that
decision. This topic exists because the artefact is systematically over-read, and because the correction is not
statistical sophistication but a design decision: the model has to be anchored to something exogenous, and the
anchor is an experiment.

The supply chain reader may wonder why this belongs in a supply chain course. In most consumer businesses the
marketing budget and the trade-promotion budget are the same commercial pot, and trade spend — listings, feature,
display, promotional allowances — is increasingly modelled with the same machinery. Retail media is now a line in
the procurement budget, bought by the same teams that buy freight. And the identification problem is identical to
the one in demand forecasting with promotional drivers: an endogenous driver, correlated with everything else,
estimated from a short history.

Four failures recur, and they are the reason this is a topic rather than a paragraph.

**One. The decomposition reported as causation.** A model fitted to 104 weeks of data produces a slide saying "TV
drove 34% of sales". What the model supports is narrower: under stated structural assumptions, and conditional on
the controls, the fitted surface attributes 34% of the modelled KPI to transformed TV. If TV rose while
distribution widened and price fell, the model cannot tell them apart, and the error has no sign anyone can
determine.

**Two. The two models that disagree and both "fit well".** The same team fits the model twice with different
transform parameters and gets television ROI of 2.9 in one run and 1.4 in the other. Both have a coefficient of
determination (R²) near 0.91. Both are defensible, and they imply opposite budget decisions. Nothing in the fit
statistics separates them, because the distinction is not about fit but about which parameter combinations the
data can separate.

**Three. The prior that did all the work.** A vendor delivers a "Bayesian MMM" whose channel priors were set by
the vendor's category benchmarks. The posterior intervals are narrow and the headline ROI sits close to the prior
mean. The analysis has measured nothing about this business; it has restated a benchmark with a confident-looking
interval. Improved methodology has made this failure more common, because strong priors always produce a tight
posterior.

**Four. The holdout that could not fail.** A brand holds out 6% of its media budget for four weeks and measures a
1.2% revenue difference with an interval spanning zero, reported as "no lift" and read as "the channel does not
work". A true 4% effect would have been missed most of the time. The test had no power to answer the question it
was asked, which is a design failure, not a finding.

### 1.1 What this topic claims

The claim is narrow and it is the spine of everything that follows.

1. An observational MMM can always produce an **associational decomposition**: a description of
   how the KPI co-moved with the transformed inputs, conditional on the controls.
2. An MMM can produce a **budget response** — the marginal effect of moving £1 between channels —
   only under structural assumptions that the data cannot test, and only within the range of
   spend that was observed.
3. An MMM **cannot** manufacture a counterfactual. The counterfactual comes from variation in
   spend that was generated for a reason other than the KPI. An experiment generates that
   variation deliberately; a natural shock generates it accidentally; a smooth budget line does
   not generate it at all.
4. The honest architecture is therefore **calibrated**: the experiment sets the prior on the
   channel effect, the MMM distributes that evidence across time, markets, and channels that were
   not tested, and the report states which numbers are anchored and which are not.

Everything else in this topic elaborates those four sentences.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** attribution, lift, and incrementality, and **classify** a stated claim into the
   correct category with a one-sentence justification.
2. **Write** the estimand an MMM is being asked to produce, in the form "the change in [KPI] per
   unit change in [spend], at [spend level], over [horizon], holding [controls] fixed", and
   **identify** which parts the data can and cannot support.
3. **Demonstrate algebraically** why a constant or smoothly trending spend path identifies only a
   ratio of the effect and carryover parameters, and **explain** what variation breaks the degeneracy.
4. **Diagnose** collinearity using correlation, variance inflation factor (VIF), and the auxiliary
   regression, and **compute** the resulting inflation of a coefficient's standard error.
5. **Compare** two specifications with near-identical fit and opposite budget implications, and
   **state** what evidence would separate them.
6. **Translate** a geo or holdout lift estimate into a prior on channel ROI, **naming** each
   judgement in the translation and its direction of bias.
7. **Compute** average and marginal ROI from a fitted response curve, **convert** both to a
   contribution basis using a stated margin, and **locate** the breakeven spend level.
8. **Write** a two-sentence statement of an MMM result to a chief financial officer (CFO) that
   states the anchor, the interval, the spend range, and the decision it supports.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three pictures. The first is the pipeline and the place where it goes flat. The second is the shape of the
likelihood when the inputs move together. The third is the containment relationship between attribution, lift, and
incrementality.

### 3.1 The pipeline, and where identification is lost

```text
   THE MMM PIPELINE AND ITS FOUR UNKNOWNS

   spend_t          adstock(θ)          saturation(K, α)         regression
   £k per week ---> x_t = Σ_j θ^j s_(t-j) ---> h(x_t)      --->  KPI_t = β·h(x_t) + γ·Z_t + e_t
                       |                        |                      |
                  θ: how fast the          K, α: where the        β: how much the
                  effect decays            curve bends            channel contributes
                       |                        |                      |
                       v                        v                      v
              +----------------------------------------------------------------+
              |  All four are identified only if the spend path contains        |
              |  variation that is NOT predictable from the control variables.   |
              |  A budget that follows the brand calendar, the season, and the   |
              |  prior year's plan contains almost none.                          |
              +----------------------------------------------------------------+
```

Three properties of this picture explain most of what follows.

**Property 1 — The transformations are not innocent.** Adstock and saturation are the model's substantive claims
about how marketing works, and their parameters enter the answer. A model with a long carryover attributes revenue
to last quarter's spend; a short one attributes it to this week's. Both can fit. The choice between them is a
causal claim dressed as a hyperparameter.

**Property 2 — The baseline is a residual, not a measurement.** Whatever the model does not explain with marketing
and controls is called the baseline, so its size is a function of the control set and the transforms, not a
property of the business. Two analysts with the same data and different control sets will report baselines ten
points apart, and both will call it "base sales".

**Property 3 — Fit measures the wrong thing.** Every parameter combination on a ridge fits approximately equally
well, so model selection on R², root mean squared error, or any predictive criterion selects on noise along the
ridge. This is the single most important sentence in the topic, and section 9 cites the published statement of it.

### 3.2 The ridge

```text
   LIKELIHOOD OVER (β, θ) WHEN SPEND MOVES SMOOTHLY

        θ  (carryover)
        ^
   0.90 |   .  .  .  .  .  .  .  .  .  .     each dot is a parameter pair
   0.75 | .  .  .  .  .  .  .  .  .  .  .    with R^2 within 0.01 of the best
   0.60 |.  .  .  .  .  .  .  .  .  .  .  .
   0.45 |  .  .  .  .  .  .  .  .  .  .  .
        |
        +--------------------------------------> β  (effect per unit of stock)

        left end:  small β, long carryover      right end: large β, short carryover
        Both reproduce the same KPI series. They imply different marginal ROIs,
        different optimal budgets, and different answers to "what if we cut TV".
```

The ridge is the geometric form of the identification problem. A likelihood that is flat along a direction
contains no information about where on that direction the truth lies. Adding data does not help if the new data
lies in the same direction; it sharpens the ridge across its width and leaves it flat along its length.

What breaks the ridge is variation in spend that is not collinear with the ridge: a step change, an on/off
pattern, a market where the channel ran and another where it did not. That is what an experiment is for.

### 3.3 Attribution, lift, and incrementality

```text
   THREE QUANTITIES, NESTED, OFTEN CONFLATED

   +-------------------------------------------------------------------+
   |  ATTRIBUTION                                                      |
   |  a rule that assigns credit for an observed outcome to touchpoints |
   |  last click, linear, time decay, data-driven, or model-based       |
   |  needs: a rule.  does not need: a counterfactual.                  |
   |                                                                    |
   |   +-------------------------------------------------------------+  |
   |   |  LIFT = observed outcome - counterfactual outcome           |  |
   |   |  needs: a counterfactual, however constructed               |  |
   |   |                                                             |  |
   |   |   +-----------------------------------------------------+   |  |
   |   |   |  INCREMENTALITY = the part of lift caused by the    |   |  |
   |   |   |  intervention, net of substitution, pull-forward,   |   |  |
   |   |   |  and competitor response                            |   |  |
   |   |   |  needs: a counterfactual AND a mechanism argument   |   |  |
   |   |   +-----------------------------------------------------+   |  |
   |   +-------------------------------------------------------------+  |
   +-------------------------------------------------------------------+
```

Read the nesting carefully, because it is the source of most vendor confusion.

- **Attribution** is bookkeeping: it answers "which touchpoint gets the credit under this rule", and
  a change of rule changes the answer with no change in the world. It is useful for pacing, creative
  diagnostics, and fraud detection. It is not an estimate of effect.
- **Lift** is a comparison against a constructed counterfactual. A geo test estimates lift; an MMM
  estimates a model-implied lift, conditional on the model being right. Lift can be positive because
  demand was pulled forward from next month, in which case it is real and temporary.
- **Incrementality** is lift attributed to the intervention rather than to the response it provoked
  elsewhere. A promotion can produce lift that is entirely pull-forward; a search ad can produce
  lift that would have happened through the organic listing anyway. Incrementality needs the
  counterfactual plus a mechanism argument, usually supplied by a second measurement: a washout
  period, a holdout of the organic path, or a competitor check.

A useful discipline: whenever you say "ROI", say which of the three you mean, and over what window. Unqualified
ROI in a deck is almost always attribution divided by spend.

The architecture this topic recommends combines the two: spend varies because a test made it vary, and the prior
carries that evidence into the model. Everything else is either an associational decomposition or a benchmark
restated with a confident interval.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Marketing mix model (MMM) | A regression of a business KPI on marketing inputs and controls, with carryover and saturation transforms | Used to mean any marketing dashboard | Model specification, not a dataset |
| Key performance indicator (KPI) | The modelled outcome: revenue, gross profit, units, or a normalised index | Switched between models without restating results | Currency per period, or units per period |
| Media variable | The measured marketing input: spend, impressions, gross rating points, or exposures | Treated as interchangeable across the three | Currency, impressions, or rating points |
| Control variable | A non-marketing driver included to absorb confounding: price, distribution, promotion, seasonality, macro | Omitted, then called "baseline" | Varies by control |
| Adstock (carryover) | The transformation that spreads an input's effect over later periods | Used to mean "lag" | Unitless decay parameter θ, or a kernel |
| Decay rate (θ) | The fraction of this period's stock that persists into the next | Reported without the period length it applies to | Proportion per week, 0 to 1 |
| Half-life | The number of periods for the stock to fall to half its value | Confused with the decay rate | Weeks |
| Saturation | The transformation that makes the response concave in the stock | Confused with diminishing returns in the data | Functional form plus parameters |
| Hill function | A saturating form `h(x) = x^α / (x^α + K^α)` | Called "the logistic curve" | Parameters α (shape) and K (scale) |
| Shape parameter (α) | Controls whether the Hill curve is S-shaped (α > 1) or concave throughout (α ≤ 1) | Left at a default and never checked against the data | Unitless, typically 0.5 to 3 |
| Inflection point (K) | The stock level at which the Hill curve reaches half its maximum | Read as "the saturation point" | Same units as the stock |
| Average ROI | Total attributed KPI divided by total spend over the same window | Used to justify the next pound of spend | Currency of KPI per currency of spend |
| Marginal ROI | The derivative of the response curve with respect to spend, at a stated spend level | Assumed equal to average ROI | Currency of KPI per currency of spend |
| Response curve | The fitted mapping from spend to KPI, including transforms | Presented without its valid spend range | Function plus domain |
| Return on ad spend (ROAS) | Revenue per unit of ad spend, usually as reported by a platform | Used as if it were incremental | Revenue per currency |
| Incremental ROAS (iROAS) | Incremental revenue per incremental unit of spend, from a designed comparison | Applied to a total-spend denominator | Revenue per currency, experiment-derived |
| Lift | Observed outcome minus counterfactual outcome | Used for any positive change | KPI units, absolute and per cent |
| Counterfactual | The outcome that would have occurred without the intervention | Assumed rather than constructed | A series, not a number |
| Incrementality | The part of lift caused by the intervention, net of substitution and pull-forward | Equated with lift | KPI units |
| Attribution | A rule assigning credit for an outcome to touchpoints | Presented as a causal decomposition | Share of outcome, summing to 1 |
| Endogenous | A regressor correlated with unobserved determinants of the outcome | Used to mean "biased" generally | Property of a variable in a model |
| Exogenous variation | Variation in a regressor generated independently of the outcome | Confused with "outside the company's control" | Property of the variation, not the variable |
| Identification | Whether the data can distinguish parameter values that imply different answers | Confused with statistical significance | A property of the design |
| Collinearity | Near-linear dependence among regressors | Diagnosed from a correlation matrix alone | Correlation, VIF, condition number |
| Variance inflation factor (VIF) | `1 / (1 - R²_j)` from regressing input j on the others | Reported without the auxiliary regression | Ratio, 1 and above |
| Prior | A distribution over a parameter before seeing the model's data | Set by the vendor and not disclosed | Distribution with mean and scale |
| Posterior | The distribution over a parameter after combining prior and likelihood | Read as "the truth" | Distribution |
| Calibrated MMM | An MMM whose channel priors come from experiments or other exogenous evidence | Used for any Bayesian MMM | Design label |
| Lift test | A designed experiment producing an incremental estimate for a channel or market | Used for a pre/post comparison | Design plus estimate |
| Holdout | A set of units or a share of budget deliberately withheld from the intervention | Confused with a control group that is also exposed | Units or budget share |
| Geo lift test | A lift test in which markets are assigned to treatment and control | Called a "matched market study" when not randomised | Design label |
| Baseline | The modelled component not attributed to marketing | Treated as a measured quantity | KPI units per period |
| Elasticity | Percentage change in KPI per percentage change in an input | Used for budget allocation without a curve | Unitless |
| Budget optimisation | Reallocating spend across channels to maximise the fitted KPI subject to constraints | Run on a curve outside its valid range | Currency reallocated |
| Triangulation | Comparing estimates from designs with different assumptions | Used to mean "two models agreed" | Evidence set |

Three of these deserve expansion, because the errors they cause are expensive. Attribution, lift, and
incrementality are treated in section 3.3; the quantitative confusions are these.

**Average ROI versus marginal ROI.** Average ROI is total attributed outcome over total spend. It is the number an
MMM reports and the number a CFO remembers. Marginal ROI is the slope of the response curve at the current spend
level. Only the marginal number is relevant to the next pound. On any concave curve the marginal value is below
the average, and the gap grows as spend approaches saturation. A channel with an average ROI of 2.1 and a marginal
ROI of 1.4 is profitable in aggregate and value-destroying at the margin. Section 6 works this arithmetic.

**ROI on revenue versus ROI on contribution.** "ROI" in an MMM is almost always incremental revenue per unit of
spend. Value creation depends on the contribution margin. If the margin is 55%, the breakeven revenue-ROI is `1 /
0.55 = 1.82`. An ROI of 1.6 is a loss. A deck that reports ROI without the margin has not told the reader whether
the spend was profitable.

**Identification versus precision.** Precision is the width of an interval given a design; identification is
whether the design can distinguish the parameter values at all. A wide interval is an honest statement of
ignorance. A narrow interval around an unidentified parameter is a false statement of knowledge, and it is the
characteristic failure of over-regularised models.

## 5. The mechanics, step by step

This section walks an MMM engagement from the decision that commissions it to the sentence that reports it. Each
step names its output and the branch that changes the design.

### Phase A — Frame the estimand

**Step 1. Name the decision and the counterfactual it needs.** Write: "if the model shows X, [role] will move
[amount] from [channel] to [channel] over [period]". The counterfactual is the spend path not taken. If nobody
will act, the model is a reporting exercise and should be costed as one.

**Step 2. Separate the four questions the model is asked.** Decomposition, attribution, response, and
optimisation. The first two need a model; the last two need identification. Most engagements are sold as the first
and used as the fourth, which is where the overclaiming enters.

**Step 3. Fix the KPI and its deflator.** Revenue, gross profit, or units; gross or net of trade spend; nominal or
deflated by a price index. A model on nominal revenue in an inflationary period will attribute price growth to
whatever moved with price. State the deflator and the margin used.

**Step 4. Fix the time bucket and the horizon.** Weekly is the working standard for consumer goods: it matches the
media buying cycle and gives 104 observations in two years. Daily data cannot identify carryover below the
purchase cycle; monthly data cannot separate adstock from seasonality.

### Phase B — Assemble the inputs

**Step 5. Choose spend, not impressions, as the media variable.** Spend is the decision variable. Impressions
measure delivery and are themselves bought, so an impressions model answers a question about media buying
efficiency rather than budget allocation. If impressions are used, state the implied cost-per-impression
assumption.

**Step 6. Build the control set and defend it.** Price, distribution, promotion, seasonality, competitor activity,
category trend, and any known structural event. The control set determines what the marketing coefficients can
mean. A missing price variable is not a small omission; it is a decision to attribute price-driven volume to
whatever else moved.

**Step 7. Understand that the baseline is a residual.** Adding a control shrinks the baseline and changes every
channel's share. Adding a competitor price index can move baseline by several points. Report the baseline with the
model specification, never as a standalone business fact.

**Step 8. Reconcile spend to the ledger.** Agency invoices, the accounts payable ledger, and platform-reported
spend will disagree, sometimes by 5–15%, through timing, rebates, value-added tax treatment, and
principal-versus-agent accounting. Choose one source, state it, and reconcile the others to it in an appendix.

### Phase C — Transform

**Step 9. Estimate adstock, and understand that it may not be estimable.** The geometric form is `x_t = s_t +
θ·x_(t-1)`, with total effect of a one-period burst equal to `s / (1 - θ)`. If spend is roughly constant at `s`
over a long stretch, the stock converges to `s / (1 - θ)` and the model contains only the product `β / (1 - θ)`.
The data then identifies the ratio, not the two parameters. Everything downstream inherits that.

**Step 10. Choose a saturation form and justify it.** The Hill form `h(x) = x^α / (x^α + K^α)` is the common
choice. With `α ≤ 1` the curve is concave throughout; with `α > 1` it is S-shaped, implying a region of increasing
returns at low spend. That shape is a strong claim, and it is often chosen by default rather than by evidence.

**Step 11. Fix the order of the transforms.** Adstock then saturation is the common convention; saturation then
adstock is a different model with different behaviour under a spend pulse. State which you used. The two are not
interchangeable and the difference is not visible in the fit.

**Step 12. Check the units.** Spend in £ thousands per week, KPI in £ thousands per week, coefficient β in KPI
units per stock unit. The coefficient is not an ROI until it is multiplied by the derivative of the saturation
function at the relevant spend level. Most reporting errors in this area are unit errors wearing a statistical
costume.

### Phase D — Estimate

**Step 13. Choose the estimator and know what it does.** Ordinary least squares (OLS) on collinear inputs produces
unstable coefficients. Ridge regression shrinks them toward zero, which is a prior in disguise and is usually
desirable. Bayesian estimation with explicit priors makes the same shrinkage inspectable. Neither fixes
identification; both make the ridge navigable.

**Step 14. Look at the whole Pareto front, not the best model.** Multi-objective fitting produces a set of models
trading off fit against decomposition plausibility. Selecting the single best-fitting member of that set is
selecting on noise along the ridge. Report the range of channel estimates across the retained set; that range is a
better uncertainty statement than the model's own interval.

**Step 15. Run the diagnostics, and read them for what they are.** Convergence diagnostics (R-hat, divergences),
posterior predictive checks, residual autocorrelation, residual seasonality. These validate the computation and
the specification's adequacy. They do not validate the causal claim, and a clean diagnostic panel is routinely
mistaken for one.

### Phase E — Diagnose identification

**Step 16. Measure collinearity properly.** Compute the correlation matrix, then the VIF for each input via the
auxiliary regression of that input on all others and the controls: `VIF_j = 1 / (1 - R²_j)`. A VIF of 14 inflates
that coefficient's standard error by `sqrt(14) = 3.8`. Report the VIFs alongside the ROIs; a channel with a wide
interval and a VIF of 15 is telling you something specific.

**Step 17. Map the spend range.** Plot the histogram of weekly spend per channel. Report the 10th and 90th
percentiles. Any budget question outside that range is extrapolation, and the fitted curve's behaviour there is
determined by the functional form, not by the data.

**Step 18. Run a separation test if you can.** A designed contrast — a high/low alternation, a temporary pause, a
market holdout — breaks the ridge directly. The published separation tests are cheap relative to a full experiment
and expensive relative to doing nothing. If the organisation will not run one, say in the report that the ridge
was not broken.

**Step 19. Look for a negative control.** A channel that was paused for a period, or a market where it did not
run, should show a near-zero fitted contribution for that period. A model that assigns material contribution to a
channel that was off is misspecified, and the misspecification is visible without any experiment.

### Phase F — Calibrate

**Step 20. Obtain an experiment whose estimand matches.** A geo lift test, a switchback, a budget holdout, or a
platform-run lift test with a genuine control. The design should match the model's unit, KPI definition, and
horizon.

**Step 21. Translate the estimate into a prior, naming every judgement.** Six judgements recur: recency, duration
(whether the window captured the carryover), population, granularity (channel versus sub-channel), spend level,
and estimand (zero-spend versus reduced-spend). Each has a direction of bias; write them down.

**Step 22. Set the prior with a defensible scale.** A mean equal to the lift estimate and a scale derived from its
interval is the minimum. A prior narrow enough to dominate the likelihood is not a prior; it is an assumption with
a distribution.

**Step 23. Read the prior-to-posterior shift.** If the posterior is close to the prior, the model's data
contributed almost nothing. If it is far from the prior, check for a translation error before treating it as a
discovery.

### Phase G — Report

**Step 24. Report shares with intervals, and label them associational.** "TV accounts for 34% of modelled revenue
(90% interval 22–44%), conditional on this specification" is defensible. "TV drives 34% of sales" is not.

**Step 25. Report average and marginal ROI separately, at a stated spend level.** Never one number. The gap
between them is the finding.

**Step 26. State the valid range.** "These curves are fitted on weekly retail media spend between £310k and £540k;
the recommendation stays inside that band."

**Step 27. State what would change the answer.** Name the experiment that would move the number most, and roughly
how much. It is more useful than a p-value.

**Step 28. Write the CFO sentence.** Section 6.8 gives the template.

## 6. Worked example

A household-goods manufacturer — call it Halden Home — sells through grocery retail and its own e-commerce site.
It has 104 weeks of weekly data and commissions an MMM to decide how to split £61.8m of annual marketing spend.
All figures are illustrative and reproducible from the tables.

### 6.1 The setup

| Item | Value | Unit |
|---|---:|---|
| Mean weekly net revenue | 9,200 | £ thousands per week |
| Annualised revenue | 478.4 | £ millions per year |
| Television spend | 420 | £ thousands per week (21.8 £m per year) |
| Paid search spend | 346 | £ thousands per week (18.0 £m per year) |
| Retail media spend | 423 | £ thousands per week (22.0 £m per year) |
| Total media spend | 1,189 | £ thousands per week (61.8 £m per year) |
| Contribution margin | 55 | per cent of net revenue |
| Breakeven revenue-ROI | 1.82 | £ of revenue per £ of spend (`1 / 0.55`) |

> **Note:** The breakeven ROI is `1 / margin`. At a 55% margin every £1 of spend must return £1.82
> of revenue to pay for itself, so a reported ROI of 1.6 is a loss.

### 6.2 The collinearity

All three channels follow the brand calendar. Television flights launch with retail media boosts, and paid search
rises when both are on air because consumers search for the brand after seeing it.

| Pair | Correlation of weekly spend |
|---|---:|
| Television and retail media | 0.86 |
| Television and paid search | 0.71 |
| Retail media and paid search | 0.64 |

Regressing television spend on the other two channels plus trend, seasonality, and the price index gives `R² =
0.93`. So `VIF = 1 / (1 - 0.93) = 14.3`, and the standard error of the television coefficient is inflated by
`sqrt(14.3) = 3.78`. A coefficient whose standard error would be 0.09 in an orthogonal design has one of `0.09 ×
3.78 = 0.34` here: on £21.8m of spend, a 68% interval of about ±£7.4m and a 95% interval of about ±£14.5m of
attributed revenue. The model is not lying when it reports a wide interval for television. It is reporting the
consequence of the design.

### 6.3 The two models

The team fits the model twice with different hyperparameter settings. Both are retained on the Pareto front;
neither is a mistake.

| Quantity | Model A | Model B |
|---|---:|---:|
| Television ROI | 2.9 | 1.4 |
| Paid search ROI | 5.4 | 3.1 |
| Retail media ROI | 1.6 | 3.6 |
| R² | 0.910 | 0.905 |
| Normalised root mean squared error | 5.8% | 6.0% |
| Attributed revenue, television | £63.2m | £30.5m |
| Attributed revenue, paid search | £97.2m | £55.8m |
| Attributed revenue, retail media | £35.2m | £79.2m |
| Total attributed | £195.6m (40.9%) | £165.5m (34.6%) |
| Implied baseline | £282.8m (59.1%) | £312.9m (65.4%) |

The fit difference is 0.005 in R²; the difference in attributed revenue is £30m. Model A says television (2.9)
beats retail media (1.6), and both sit above the 1.82 breakeven. Model B says retail media (3.6) beats television
(1.4), and the conclusion reverses. Both are consistent with the data: they sit at different points along the
ridge in section 3.2, Model A with a larger television β and shorter carryover, Model B the reverse.

### 6.4 The geo test for retail media

Because the decision is a reallocation and the models disagree, the team runs a geo lift test on retail media
only. Twenty-four designated market areas (DMAs) are paired into 12 pairs on the pre-period path; one member of
each pair is randomised to a 40% retail media spend increase for eight weeks, funded from central budget so that
control markets are untouched.

| Quantity | Value | Unit |
|---|---:|---|
| Mean treated-DMA baseline revenue | 383.3 | £ thousands per week |
| Baseline retail media spend per DMA | 17.63 | £ thousands per week |
| Planned increase | 40 | per cent |
| Incremental spend per DMA | 7.05 | £ thousands per week |
| Measured lift | 4.6 | per cent of baseline |
| Lift in currency | 17.63 | £ thousands per week |
| 95% interval on the lift | 2.2 to 7.0 | per cent |
| Implied iROAS | 2.50 | £ revenue per £ spend |
| 95% interval on iROAS | 1.19 to 3.81 | £ revenue per £ spend |

The arithmetic for the point estimate:

```text
lift (£k/week)      = 0.046 x 383.3 = 17.63
incremental spend   = 0.40 x 17.63  = 7.05
iROAS               = 17.63 / 7.05  = 2.50
interval bounds     = 8.43 / 7.05 = 1.19  and  26.83 / 7.05 = 3.81
```

The interval is wide because 12 pairs is a small design. Its width is the honest statement of what a 12-pair test
can say, and it is still far more informative than the model's own interval, which was ±0.25.

### 6.5 Why the model's own interval was too narrow

Model A reported retail media ROI as `1.60` with a 95% interval of `(1.35, 1.85)`, which conditions on the
hyperparameters and ignores the uncertainty over them. Across the retained Pareto set the figure ranged from 1.2
to 3.6. The reported interval is not the interval the decision needs.

### 6.6 Building the prior

The lift estimate becomes a normal prior on retail media ROI with mean 2.50 and a standard deviation taken from
the interval: `(3.81 - 1.19) / 3.92 = 0.67`.

The likelihood comes from Model A's retail media estimate, widened to reflect the model-selection spread: mean
1.60, standard deviation 0.75. The posterior for a normal-normal update is:

```text
prior precision      = 1 / 0.67^2 = 2.228
likelihood precision = 1 / 0.75^2 = 1.778
posterior precision  = 2.228 + 1.778 = 4.006
posterior sd         = sqrt(1 / 4.006) = 0.500

posterior mean = (2.228 x 2.50 + 1.778 x 1.60) / 4.006
               = (5.570 + 2.845) / 4.006
               = 8.415 / 4.006 = 2.10
```

The calibrated estimate is **2.10 with a 90% interval of 1.28 to 2.92** and a 95% interval of 1.12 to 3.08. Three
things changed. The point estimate moved from 1.60 to 2.10, from below the 1.82 breakeven to above it. The
interval widened from ±0.25 to ±0.98 at 95%, because the experimental uncertainty is larger than the conditional
model uncertainty. And the number is now attributable to a design: "anchored on a 12-pair geo test in 24 markets
over eight weeks".

> **Note:** The prior is not a truth serum. It is an estimate from one test, in one set of markets,
> over one window, with its own interval. Its function is to move the model's answer into a region
> the data supports and make the remaining uncertainty visible. Had the test run in peak season
> only, the prior would be biased upward and the posterior with it.

### 6.7 Average and marginal ROI

The fitted retail media response curve is a Hill function in spend, with the shape parameter `α = 2.0` and the
inflection point `K = 300` (£ thousands per week). The maximum response is calibrated so that the curve reproduces
the posterior average ROI at current spend, giving `V_max = 1,335.1` £ thousands per week.

| Quantity | At current spend (423) | Unit |
|---|---:|---|
| Fitted retail media revenue | 888.3 | £ thousands per week |
| Average ROI | 888.3 / 423 = 2.10 | £ per £ |
| Marginal ROI (derivative) | 1.41 | £ per £ |
| Ratio of marginal to average | 0.67 | unitless |

The marginal ROI is 1.41, below the 1.82 breakeven. Retail media is profitable on average and value-destroying at
the margin. Both statements are true, and only the second is relevant to the next pound.

The spend level at which the marginal pound breaks even solves `dy/dx = 1.82`:

| Trial spend (£k/week) | Marginal ROI (£ per £) |
|---:|---:|
| 423 (current) | 1.41 |
| 380 | 1.62 |
| 356 | 1.82 |
| 340 | 1.93 |

At about £356k per week — a 16% reduction from £423k — the marginal pound returns exactly the breakeven. The
annual arithmetic:

```text
spend reduction      = 423 - 356 = 67 £k/week = 3.48 £m/year
revenue reduction    = 888.3 - 780.8 = 107.5 £k/week = 5.59 £m/year
contribution lost    = 5.59 x 0.55 = 3.07 £m/year
spend saved          = 3.48 £m/year
net contribution gain = 3.48 - 3.07 = +0.41 £m/year
```

A 16% cut in retail media raises annual contribution by roughly £0.41m, because the revenue lost at the margin is
worth less than the spend saved.

### 6.8 The CFO paragraph

The report's recommendation, in the form a finance audience can act on:

> "Every additional pound of retail media currently returns £1.41 of revenue against a breakeven
> of £1.82, so we recommend reducing the budget by about 16%, from £21.8m to £18.3m a year, which
> adds roughly £0.4m of annual contribution. That marginal figure is anchored on a randomised
> 12-pair geo test in 24 markets over eight weeks, which put the channel's average return at £2.10
> per pound (90% range £1.28 to £2.92). Television we cannot yet size: our two defensible model
> forms put it anywhere between 1.4 and 2.9, and we have not earned the right to pick a point
> inside that range. A television holdout would cost about £1.1m of foregone media and would take
> twelve weeks."

Four properties make this paragraph honest. It leads with the margin, not the model. It gives the anchor and its
interval. It states the spend range implicitly through the amounts. And it says which number is not yet known,
with the cost of finding out.

## 7. Code, query, or calculation

The calculation reproduces the marginal-ROI and breakeven results from section 6.7. It is deliberately small: the
point is that the decision number is a derivative, and a derivative of a fitted curve is arithmetic, not a
platform feature.

```python
def hill(x, v_max, k, alpha):
    return v_max * x**alpha / (x**alpha + k**alpha)

def marginal(x, v_max, k, alpha):
    return v_max * alpha * k**alpha * x**(alpha-1) / (x**alpha + k**alpha)**2

v_max, k, alpha = 1335.1, 300.0, 2.0    # £k/week, £k/week, unitless
spend, target = 423.0, 1.82             # £k/week, breakeven revenue-ROI

# With alpha > 1 the marginal curve is hump-shaped, so scan outward from the
# peak rather than bisecting: there are two spend levels at which it equals
# the target and only the higher one is relevant.
peak = k * ((alpha - 1) / (alpha + 1)) ** (1 / alpha)
x = peak
while marginal(x, v_max, k, alpha) > target:
    x += 1.0

print(f"avg at {spend:.0f}: {hill(spend, v_max, k, alpha)/spend:.2f}")
print(f"marginal at {spend:.0f}: {marginal(spend, v_max, k, alpha):.2f}")
print(f"breakeven spend: {x:.0f} £k/week")
```

Expected output, three lines: `avg at 423: 2.10`; `marginal at 423: 1.41`; and `breakeven spend: 356 £k/week`.

**Assumptions the calculation makes, and what each costs if it is wrong.**

1. **The curve form is the fitted form.** If the true response is S-shaped and the fitted curve is
   concave, the marginal ROI at low spend is overstated. The breakeven is inside the observed
   range, which limits but does not eliminate this risk.
2. **`α = 2.0` and `K = 300` are estimated, not known.** Both are on the ridge, and a model with
   `α = 1.4` would put the breakeven elsewhere. This is why the experimental anchor matters and why
   the recommendation is a band rather than a point.
3. **The maximum response is calibrated to the posterior average.** That makes the average ROI
   correct at current spend by construction and pushes all the error into the shape, which is the
   least identified part of the model: the right place for error, and the wrong place to forget it.
4. **Contribution margin is constant at 55%.** If incremental volume carries a lower margin, the
   breakeven rises and the recommended cut grows.
5. **No cross-channel interaction.** The calculation treats the retail media curve as fixed when
   television spend changes. If the two are complements, cutting one lowers the return on the
   other, and the arithmetic is optimistic.

**A hand calculation for a meeting.** `(888.3 - 817.0) / (423 - 380.7) = 1.69` per pound, which sits between the
marginal at 423 (1.41) and at 381 (1.62), as a secant between two points must.

## 8. Data, standards, and regulatory context

An MMM is a data-join problem before it is a statistical one. Four kinds of source have to be reconciled — media
cost, media delivery, business outcome, and business drivers — and each has its own conventions, its own calendar,
and its own definition of the same word.

### 8.1 The four sources

| Source | Typical owner | Grain | What it answers | Common defect |
|---|---|---|---|---|
| Media cost (invoices, ad server, platform billing) | Marketing, finance | Campaign, week | What was spent | Rebates, value-added tax, principal versus agent |
| Media delivery (impressions, rating points, clicks) | Agency, platform | Campaign, day | What was bought and delivered | Measurement accreditation, invalid traffic |
| Business outcome (sales, shipments, scan) | Finance, retail partners | Stock-keeping unit (SKU), store, week | What happened | Gross versus net, returns, promotional deductions |
| Business drivers (price, distribution, promotion) | Category management, revenue management | SKU, account, week | What else moved | Inconsistent promotion calendars across accounts |

**The unit and calendar problem.** Media data is typically in campaign weeks, sales data in fiscal weeks, and
retail scan data in the retailer's own week numbering, often a 4-4-5 calendar. A model built on mismatched
calendars smears one week of media across two weeks of sales, attenuating the effect and lengthening the apparent
carryover. Fix the calendar to ISO 8601 weeks or to a single named retail calendar, and document the mapping.

**The gross-versus-net problem.** Gross sales, net sales after promotional deductions, and shipments to trade are
three different series. Trade spend is frequently a deduction from revenue under IFRS 15 (variable consideration,
International Financial Reporting Standards) rather than a cost of sales. That accounting choice changes the
denominator of every ROI, and it is the most common reason two functions disagree about the same campaign's
return. Agree the definition in writing before the model is fitted.

### 8.2 Measurement standards

| Standard or body | Governs | What it constrains in an MMM | Grade |
|---|---|---|---|
| Media Rating Council (MRC) accreditation | Audience measurement services | Whether a rating point or impression figure has been independently audited | B |
| MRC viewable impression guidelines | Display and video viewability | The definition of a "viewable" impression; changes the delivery denominator | B |
| Interactive Advertising Bureau (IAB) measurement guidelines | Digital ad impression counting | Counting rules that vary by platform and must be normalised | C |
| Gross rating point (GRP) and target rating point (TRP) conventions | Television and radio audience delivery | The currency of television buying; not comparable across media without a cost conversion | C |
| ISO 8601-1:2019 | Date and time representation | Week numbering and period boundaries in the panel | B |
| 4-4-5 retail calendar convention | Retail fiscal periods | Week alignment between media and scan data | C, convention |

Two consequences follow. First, **accreditation status is a property of a specific measurement service at a
specific time** and must be checked rather than assumed; a platform's own dashboard is not an accredited
measurement. Second, **rating points and impressions are not comparable currencies**, so a model that mixes them
across channels is fitting a scale difference and calling it a media effect. Convert to a common currency — spend
is the usual choice — or fit separate saturation parameters with an explicit conversion.

### 8.3 Privacy regulation and its effect on the data

| Regime | Jurisdiction | Effect on MMM inputs | Grade |
|---|---|---|---|
| Regulation (EU) 2016/679 (General Data Protection Regulation, GDPR) | European Union | Lawful basis for any user-level data; aggregate modelling is less exposed than user-level attribution | B |
| Directive 2002/58/EC (ePrivacy Directive) | European Union | Consent for storage and access to terminal equipment; constrains cookie and identifier-based measurement | B |
| ePrivacy Regulation | European Union | Proposed; status must be checked against the current text before relying on it | Not graded, status uncertain |
| Platform identifier restrictions | Industry-wide | Reduced user-level tracking; aggregate and modelled measurement becomes more important, not less | C |

The practical effect is that the measurement base has shifted toward the MMM and away from user-level attribution,
at exactly the moment when the MMM's identification problem has become better understood. That is the central
tension of the discipline in 2026, and it is worth stating plainly to any stakeholder who believes that privacy
change has made MMMs more trustworthy. It has made them more necessary and no more identified.

### 8.4 What is convention rather than requirement

Two years of weekly data, an R² above 0.8, an ROI above 1.0, and a weekly refresh are conventions, not
requirements. The R² threshold is meaningless, because seasonality and price alone can reach 0.95 on weekly
revenue; the ROI threshold is wrong, because the criterion is `1 / margin`; and the refresh cadence is inherited
from reporting cycles, with nobody owning the drift between refreshes.

## 9. Industry practice and case evidence

### 9.1 The measurement-economics literature

Two peer-reviewed papers frame the problem better than any vendor document.

**Lewis and Rao (2015)**, "The Unfavorable Economics of Measuring the Returns to Advertising", *Quarterly Journal
of Economics* 130(4), reports 25 large field experiments run with major United States retailers and brokerages,
most reaching millions of customers, totalling about $2.8m of digital ad spend. Their central finding is about
precision rather than bias: the median confidence interval on ROI was more than 100 percentage points wide,
because individual-level sales are extremely volatile relative to per-capita advertising cost, and informative
experiments can require more than ten million person-weeks. Their conclusion is not that measurement is hopeless
but that randomised designs are the only ones that inject unbiased information, and that selection effects in
observational methods are a first-order concern (Grade A).

**Gordon, Zettelmeyer, Bhargava, and Chapsky (2019)**, "A Comparison of Approaches to Advertising Measurement:
Evidence from Big Field Experiments at Facebook", *Marketing Science* 38(2), compares randomised experiments with
a battery of observational estimators — exact matching, propensity score matching, stratification, regression
adjustment, and combinations — on 15 United States advertising experiments covering roughly 500 million
user-experiment observations. Observational methods often fail to reproduce the experimental effects, even after
conditioning on extensive demographic and behavioural covariates. The working-paper version reports that in
roughly half the studies the observational estimate was off by a factor of about three (Grade A for the published
finding; the factor-of-three detail is from the working paper and should be cited as such).

Neither paper is about MMMs specifically. Both are about the same structural problem: a marketing input chosen by
the same people who observe the outcome, measured over a short window, with an effect that is small relative to
the noise in the outcome.

### 9.2 The MMM identification literature

**Dew, Padilla, and Shchetkina (2024)**, "Your MMM is Broken: Identification of Nonlinear and Time-varying Effects
in Marketing Mix Models", argues that nonlinear effects and time-varying effects are frequently not separately
identifiable from typical MMM data: patterns that look like saturation can be produced by a model with dynamics in
effectiveness, and the two imply different optimal budget allocations. They show the conflation is especially
likely when marketing variables are autocorrelated, which is exactly what adstock produces, and they propose
separation tests — including a high/low alternation that breaks the degeneracy in one or two periods. This is a
preprint and had not been peer-reviewed at the time of access (Grade B).

The practical reading is uncomfortable and correct: **the saturation curve, on which budget optimisation depends,
is the least identified part of the model.**

**Chan and Perry (2017)**, a Google technical report on media mix modelling, organises the problem into three
groups: data limitations (too few observations, correlated inputs, a narrow spend range), selection bias and
endogeneity (targeting, seasonality, and funnel effects such as television driving branded search), and model
selection and uncertainty (many specifications fitting similarly well). It is a company technical report whose
framing has been widely adopted (Grade C).

### 9.3 The vendor tooling, and how to read it

| Tool | Owner | Method as documented | Calibration support | Grade |
|---|---|---|---|---|
| Robyn | Meta, open source | Ridge regression with evolutionary hyperparameter search over adstock (geometric or Weibull) and Hill saturation; Pareto front of models trading fit against decomposition | Accepts lift and geo test results as calibration input | D |
| Meridian | Google, open source | Bayesian MMM with Markov chain Monte Carlo; reparameterised so ROI is a model parameter, allowing a prior directly on ROI | Documents calibration from experiments, including translation adjustments | D |
| Meridian GeoX | Google | Design and analysis of geo experiments producing lift estimates intended as MMM priors | Designed as the upstream experiment layer | D |

Both tools are technically serious and both are vendor documentation; the grade reflects provenance, not quality.
Both now foreground calibration from experiments, which is the industry converging on the architecture this topic
recommends. A tool supporting calibration does not mean the calibration was done: a `calibration_input` argument
in a repository is not evidence that any test was run.

Google's public documentation on calibrating treatment priors is unusually candid. It states that there is no
single formula for converting an experiment into a prior, and lists the dimensions on which an experiment and a
model can differ: recency, duration, media granularity, population, model complexity, and estimand — noting that
an MMM typically estimates a zero-spend counterfactual while some experiments estimate a reduced-spend one. It
recommends plotting priors and examining prior-to-posterior shifts rather than treating priors as constraints
(Grade D, accessed 2026).

**Zhang, Wurm, Li, Wakim, Kelly, Price, and Liu (2024)**, "Media Mix Model Calibration With Bayesian Priors", a
Google Research publication, describes the reparameterisation that makes ROAS a model parameter so a prior can be
placed on it directly, and reports simulations in which this reduces bias and uncertainty in posterior ROAS
estimates. It is company research with a simulation-based evaluation, not an independent benchmark (Grade C).

### 9.4 Vendor claims and how to appraise them

| Claim as usually made | What it would require to be true | Question that exposes it |
|---|---|---|
| "Our MMM explains 94% of sales variation" | That R² measures media effect | "What is the R² of a model with seasonality and price but no media?" |
| "Channel ROI is 3.4x" | An identified response at current spend | "Is that average or marginal, and what is the breakeven given the client's margin?" |
| "We use Bayesian priors, so the estimates are robust" | That priors are informative and disclosed | "Show the prior, the posterior, and the shift between them." |
| "Calibrated with industry benchmarks" | That the benchmarks apply to this business | "Which benchmark, from which markets, in which year, and how was it translated?" |
| "Validated against a holdout" | That the holdout had power to fail | "What was the minimum detectable effect (MDE) of that holdout, and what was the pre-registered decision rule?" |
| "We optimise your budget across channels" | That the response curves are identified over the proposed range | "What is the observed spend range per channel, and does the recommendation stay inside it?" |
| "AI-driven, no assumptions" | Nothing; the claim is empty | "Which functional form, and what does the curve do outside the observed range?" |

The pattern is that every claim can be tested by asking for a number the claimant would have had to compute. A
vendor who has done the work produces it in a minute; one who has not describes the methodology again.

### 9.5 What is not publicly evidenced

There is no large public benchmark of MMM budget-allocation accuracy against known ground truth. The published
comparisons are of individual estimators against individual experiments (section 9.1), not of production MMMs
against a ledger of true effects, so claims of the form "our model is X% more accurate" are unfalsifiable as
stated.

There is also very little public evidence on MMM performance for trade promotion and retail media, the levers most
relevant to a supply chain audience. The methods transfer from brand media; the variance magnitudes and the
confounders do not. Retail media carries a specific problem: spend is often a condition of distribution, so it
moves with the listing decision, and the listing decision moves sales. That is a textbook confounder and it is
rarely modelled.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Decomposition reported as causation | The slide says "drove 34% of sales" and the method note says "regression" | A fitted surface is described as a causal decomposition | Budget moved on a claim the design cannot support | Use associational language; state the structural assumptions; anchor the largest channels |
| Two models, opposite decisions | Two ROI tables in the appendix differ by a factor of two and both are called "the model" | The Pareto front was collapsed without reporting the spread | The recommendation depends on an arbitrary selection | Report the range across retained models as the uncertainty |
| Collinear channels never diagnosed | No VIF, correlation matrix, or auxiliary regression in the appendix | Brand calendar moves all channels together | Coefficients unstable; intervals understated | Compute and report VIF; widen intervals accordingly |
| Model interval reported as decision interval | A 95% interval of ±0.25 on a channel with VIF 14 | Interval conditions on hyperparameters and ignores model selection | Overconfidence by an order of magnitude | Report model-selection spread alongside the conditional interval |
| Extrapolation beyond observed spend | The recommendation doubles a channel whose 90th percentile spend is 1.4x current | Functional form determines behaviour outside the range | Confident nonsense at the extremes | Report 10th–90th percentile per channel; bound the recommendation |
| Adstock absorbing seasonality | Carryover of 0.85 and a December spike attributed to November media | Smooth transforms can mimic seasonal structure | Effect mis-dated and mis-sized | Compare carryover against the purchase cycle; test without the December weeks |
| Prior does all the work | The posterior mean is within 0.05 of the prior mean and the interval is narrow | Priors set from benchmarks, not from this business | The analysis restates a benchmark as a measurement | Compare prior and posterior; report the shift; require an experiment for the largest channel |
| Fit used as validation | "R² of 0.94, therefore the model is good" | Seasonality and price alone can produce a high R² | A model with no media effect passes the test | Report the no-media benchmark R² and the lift over it |
| Holdout with no power | The holdout was 4% of budget for three weeks; the result was "no effect" | The design could not have detected a plausible effect | A working channel is retired | Compute the MDE before the holdout; publish it with the result |
| Estimand mismatch between test and model | The experiment measured a 40% spend increase; the prior is applied to a zero-spend counterfactual | Non-linear response means the two are different quantities | The prior is biased, usually upward | Translate explicitly; widen the prior for the mismatch |
| Trade spend treated as media | The model's largest "channel" is a promotional allowance booked as a revenue deduction | Promotional spend and brand media have different accounting and different mechanisms | ROI denominators are inconsistent across channels | Reconcile to the profit and loss account; state whether each input is a cost or a deduction |
| Optimiser run on a flat ridge | The recommended allocation moves 40% of budget between two channels with overlapping intervals | The optimiser maximises a point estimate on an unidentified surface | A large reallocation supported by noise | Optimise only over channels with experimental anchors; constrain moves |

The first row is the most common failure because it is a language habit rather than a modelling choice: the
analyst knows the number is conditional, the slide does not say so, and the reader has no way to recover the
conditionality. The fix is one clause on the slide. The prior-dominance row is the one to watch as Bayesian
tooling spreads, because a narrow, stable, plausible-looking posterior looks like quality; the diagnostic is to
print the prior and posterior means side by side.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Marketing spend is exogenous conditional on the controls | Without it, the coefficient absorbs unobserved determinants of the KPI | Reverse causality and omitted variables bias the effect, usually upward | Instrument or experiment; test for lead-lag patterns between spend and KPI |
| The transforms are correctly specified | Adstock and saturation define what the coefficient means | Misspecification biases the effect and mis-dates it | Compare functional forms; run separation tests; check against the purchase cycle |
| Response parameters are stable over the estimation window | One coefficient is applied to 104 weeks | A structural break makes the estimate an average of two regimes | Rolling estimation; change-point tests; residual inspection |
| The control set captures the main confounders | The residual is called baseline and treated as non-marketing | Omitted price or distribution inflates media effects | Add controls and check coefficient stability; directed acyclic graph (DAG) review |
| Spend variation is sufficient to identify the parameters | The ridge must be broken | Only a ratio of effect to carryover is identified | Plot the spend path; compute the VIF; run a separation test |
| The observed spend range contains the decision | Response outside the range is model-driven | Extrapolation error grows with distance from the range | Report 10th–90th percentiles; bound recommendations |
| No interference across channels or markets | Each channel's effect is estimated holding the others fixed | Halo effects (television driving search) are misattributed | Model cross-channel terms; test with a channel pause |
| The KPI measures the outcome the decision affects | Revenue is not contribution | Spend is optimised toward revenue at the expense of margin | Re-estimate on gross profit; convert with a stated margin |
| The experiment used for calibration is representative | The prior is transported to other markets, periods, and spend levels | Prior bias enters the posterior directly | Compare tested and modelled populations; widen the prior for distance |
| The experiment's estimand matches the model's | A zero-spend counterfactual is not a 40% increase | The prior is on a different quantity than the likelihood | State both estimands; translate; widen |

Three limits are worth stating separately, because they are not fixable within the method.

**The method cannot create exogenous variation.** No estimator, prior, or diagnostic substitutes for spend moving
for a reason unrelated to the outcome.

**The saturation curve is the least identified object and the most decision-relevant.** The marginal ROI is a
derivative of that curve, so every budget recommendation rests on the weakest part of the estimate. The response
is to anchor the level with an experiment and bound the recommendation, not to pretend the shape is known.

**The partial-equilibrium problem.** An MMM estimated on historical variation tells you what happened when one
channel moved and the others did not. A large reallocation moves several at once, provokes competitor response,
and may change the production cost base.

## 12. Exercises

### Exercise 1 — Classify the claim (recall and application, 25 minutes)

Classify each statement as attribution, lift, or incrementality, and say in one sentence what evidence would be
needed to support it as written.

(a) "Paid search received credit for 38% of online conversions last quarter." (b) "The geo test showed 4.6% more
revenue in treated markets than in control markets." (c) "The promotion created 12,000 incremental units over the
quarter, net of pull-forward." (d) "Retail media delivers a 3.4x return." (e) "Television added 2.1 points of
market share in the four weeks after launch."

**Deliverable:** a five-row table with claim, category, and required evidence. **Acceptance criteria:** every row
names the counterfactual (or its absence); at least one row is correctly identified as a claim that mixes
categories; no row is answered with a synonym. **Time:** 25 minutes.

### Exercise 2 — The identification algebra (calculation, 35 minutes)

A channel has been held at a constant £400k per week for 60 weeks. The model is `KPI_t = β · x_t + controls`, with
`x_t = s_t + θ·x_(t-1)`.

(a) Show that the model identifies only `β / (1 - θ)`, not `β` and `θ` separately. (b) Compute the identified
quantity for `β = 0.4` and `θ = 0.7`, and for `β = 0.1` and `θ = 0.925`. (c) State two spend patterns that would
separate the parameters, and say what each costs to run.

**Deliverable:** the algebra, both computations, and the two designs with costs. **Acceptance criteria:** the
algebra shows the steady state explicitly; the two parameter pairs produce the same identified quantity; the
designs are operational rather than abstract. **Time:** 35 minutes.

### Exercise 3 — Collinearity and interval inflation (calculation, 30 minutes)

An auxiliary regression of television spend on the other channels and controls gives `R² = 0.93`. In an orthogonal
design the television ROI standard error would be 0.11.

(a) Compute the VIF and the inflated standard error. (b) Convert both to an interval on attributed revenue for
£21.8m of spend. (c) State what you would report to a client who wants a single number.

**Deliverable:** the arithmetic and a two-sentence reporting recommendation. **Acceptance criteria:** correct VIF
and square-root relationship; intervals in currency, not just in ROI units; the recommendation does not simply
widen the interval without explanation. **Time:** 30 minutes.

### Exercise 4 — Two models, one decision (critique, 45 minutes)

Two specifications fit the same 104 weeks of data. Model A gives television 2.9 and retail media 1.6; Model B
gives television 1.4 and retail media 3.6. R² is 0.910 and 0.905 respectively. The client wants to move £5m from
television to retail media.

Write a two-page note explaining why the models differ, what the client can and cannot conclude, and what you
would do next.

**Deliverable:** the note, with a recommended next step and its cost. **Acceptance criteria:** the explanation
references carryover and saturation rather than "noise"; the note does not pick a model; the next step is a
design, not more modelling. **Time:** 45 minutes.

### Exercise 5 — Translate the test into a prior (design, 50 minutes)

You have a geo lift test result for retail media: iROAS 2.50, 95% interval 1.19 to 3.81, measured in 24 markets
over 8 weeks, with a 40% spend increase, ending four months ago. The MMM covers 36 months nationally and models a
zero-spend counterfactual.

Write the calibration memo: the prior you would set, the six translation judgements, the direction of bias each
introduces, and the scale you would use.

**Deliverable:** a one-page memo with a prior specification (distribution, mean, standard deviation) and a
justification for each adjustment. **Acceptance criteria:** all six judgements addressed; each has a stated
direction of bias; the prior scale is wider than the raw interval and the reason is given. **Time:** 50 minutes.

### Exercise 6 — Marginal versus average (calculation, 45 minutes)

Using the Hill curve in section 6.7 (`V_max = 1,335.1`, `K = 300`, `α = 2.0`, spend in £ thousands per week), and
a contribution margin of 55%:

(a) Compute the marginal ROI at £300k, £356k, and £423k per week. (b) Find the spend at which marginal ROI equals
the breakeven. (c) Compute the annual contribution effect of moving from £423k to that level. (d) State what
changes if the margin is 45% instead of 55%.

**Deliverable:** the calculations and a one-paragraph recommendation. **Acceptance criteria:** marginal and
average are clearly distinguished; the breakeven is computed from the margin; part (d) states the direction of the
change before computing it. **Time:** 45 minutes.

### Exercise 7 — Rewrite the claim (critique, 30 minutes)

Rewrite each sentence so that it is defensible given an observational model with no experiment:

(a) "Television drove 31% of sales and returned £2.90 per pound." (b) "Our model proves that retail media is
saturated." (c) "Reallocating £10m to search will generate £54m of revenue." (d) "The baseline is £283m of organic
demand."

**Deliverable:** four rewritten sentences plus a note on what each original overclaimed. **Acceptance criteria:**
each rewrite names the condition, the range, or the anchor; no rewrite introduces a new unsupported claim; at
least one rewrite states that the number is not identified. **Time:** 30 minutes.

### Exercise 8 — Design the television holdout (design, 60 minutes)

Television is the one channel the client cannot size (models say 1.4 to 2.9). Design a holdout that could actually
constrain the prior: units, duration, allocation, primary outcome, expected MDE, and the decision rule.

**Deliverable:** a one-to-two-page design note with a power calculation. **Acceptance criteria:** the design
states the MDE in ROI units and compares it with the width of the current disagreement; the allocation is feasible
for a national television buy; the decision rule is written before the test. **Time:** 60 minutes.

### Exercise 9 — Vendor deck appraisal (critique, 50 minutes)

A vendor presents: "Our AI-powered MMM explains 94% of sales variation, uses Bayesian priors calibrated to your
category, and identifies a 4.1x return on retail media. We recommend a 30% budget shift into retail media."

Write a one-page appraisal with at least eight questions and a recommendation.

**Deliverable:** the appraisal, with a clear position on whether the number is usable. **Acceptance criteria:** at
least two questions concern identification rather than data quality; at least one concerns the calibration source;
the recommendation distinguishes "usable as a hypothesis" from "usable as a decision input". **Time:** 50 minutes.

## 13. Self-check answers

**Exercise 1 — classification.**

| Claim | Category | Evidence required |
|---|---|---|
| (a) Search received credit for 38% of conversions | Attribution | Only a stated rule; no counterfactual is claimed, so none is needed |
| (b) Treated markets had 4.6% more revenue than control markets | Lift | A valid counterfactual; the design is adequate as stated |
| (c) Promotion created 12,000 incremental units net of pull-forward | Incrementality | A washout window or a long enough window to observe the reversal |
| (d) Retail media delivers 3.4x | Ambiguous; usually attribution dressed as effect | The estimand, the window, the denominator, and whether it is average or marginal |
| (e) Television added 2.1 points of share after launch | Lift, weakly | A counterfactual; a before-and-after comparison is not one |

Row (d) is the one that matters. A bare ROI multiple almost always means attributed revenue over spend, and the
word "delivers" imports the causal claim that the arithmetic does not contain. Row (e) is the second trap: "after
launch" is a timing statement, and a pre/post comparison is a counterfactual only if nothing else changed, which
is exactly what nobody checked.

**Exercise 2 — the identification algebra.**

(a) With constant spend `s`, the stock converges to `x* = s + θ·x*`, so `x* = s / (1 - θ)`. The marketing term is
therefore `β · s / (1 - θ)`, which contains only the product. Any pair `(β, θ)` with the same `β / (1 - θ)`
produces an identical fit.

(b) For `β = 0.4`, `θ = 0.7`: `0.4 / 0.3 = 1.333`; with `s = 400`, that is £533.3k per week of attributed KPI. For
`β = 0.1`, `θ = 0.925`: `0.1 / 0.075 = 1.333`, and the same £533.3k. Two very different parameter pairs, the same
fit, and very different implications: the first says the effect is mostly immediate, the second that it persists
for months.

(c) Two patterns break it. **On/off flights**: alternate four weeks at £600k and four at zero. The cost is lost
reach, and a possible change in the response regime, because media at zero is not media at low. **Market
holdout**: run the channel in 20 markets and not in 20 matched markets for eight weeks. The cost is the foregone
media in control markets plus the measurement.

**Exercise 3 — collinearity.**

`VIF = 1 / (1 - 0.93) = 14.3`, and `sqrt(14.3) = 3.78`. The inflated standard error is `0.11 × 3.78 = 0.416`. On
spend of £21.8m, the 95% interval on attributed revenue is `±1.96 × 0.416 × 21.8 = ±£17.8m`: a £63.2m point
estimate runs from £45.4m to £81.0m.

The reporting recommendation: give the point estimate with the interval, and state the VIF that produced it. Do
not present the interval as a defect of the model. It is a property of the design, and the design is the thing the
client can change — by running a test, not a better model.

**Exercise 4 — two models.**

The models differ along the ridge: Model A carries a larger television coefficient and a shorter carryover, Model
B the reverse. Because television and retail media spend are correlated at 0.86, the model cannot separate them,
and the two combinations are observationally equivalent to within 0.005 in R². This is not a software problem, a
data-quality problem, or an analyst error.

The client can conclude that the direction of the reallocation is not determined by the model. It cannot conclude
that the reallocation is wrong: it might be right, and the model cannot say.

The next step is a design, not a model. A retail media geo test of the kind in section 6.4 moves one number from
unidentified to anchored, and a television holdout does the same for the other. Run the retail media test first:
it is cheaper, and it carries the live commercial pressure.

**Exercise 5 — calibration memo.**

The six judgements, with direction of bias for each:

| Judgement | Fact | Direction of bias on the prior |
|---|---|---|
| Recency | Test ended four months ago | Widen; media effectiveness drifts, so the point estimate is less certain, not systematically wrong |
| Duration | 8 weeks, retail media carryover estimated at 3–4 weeks | Slight downward: an 8-week window may truncate the tail, so the true return is a little higher |
| Population | 24 markets, selected for comparability | Widen: tested markets are not a random sample of the national base |
| Granularity | Channel level, no sub-channel split | Widen: the prior applies to a composite whose mix may differ |
| Spend level | 40% increase from baseline, not a zero-spend counterfactual | Downward: the test measures a return over an interval that starts from a positive stock, so the zero-spend average is higher |
| Estimand | Reduced-spend versus zero-spend | Downward for the same reason, and this is the largest single adjustment |

A defensible specification is normal with mean 2.20 (below the raw 2.50 because of the last two rows) and standard
deviation 0.90 (wider than the raw 0.67 because of recency, population, and granularity). That prior is weaker
than the raw test and stronger than no prior, which is the correct relationship between an experiment and a model
that must transport it.

> **Note:** Every adjustment above is a judgement. Another analyst could defend mean 2.40 or 2.00.
> The point of the memo is not the exact number; it is that the number is written down, justified,
> and reviewable before the posterior is computed. A calibration nobody can audit is a prior with
> unknown provenance.

**Exercise 6 — marginal versus average.**

(a) Marginal ROI at £300k: `2.23`; at £356k: `1.82`; at £423k: `1.41`. (b) The breakeven is `1 / 0.55 = 1.82`,
reached at approximately **£356k per week**.

(c) Moving from £423k to £356k reduces spend by £67k per week, or £3.48m a year. Fitted revenue falls from £888.3k
to £780.8k per week, a loss of £107.5k per week or £5.59m a year. Contribution lost is `5.59 × 0.55 = £3.07m`, so
the net effect is `3.48 − 3.07 = +£0.41m` a year.

(d) At a 45% margin the breakeven rises to `1 / 0.45 = 2.22`, reached at about £300k per week. Revenue there is
£667.6k per week, a loss of £11.5m a year; contribution lost is `11.5 × 0.45 = £5.17m` against spend saved of
£6.40m, a net gain of about **£1.23m**. A lower margin raises the breakeven, so more of the spend fails it and the
cut grows.

**Exercise 7 — rewrites.**

(a) "Under this specification, television accounts for 31% of modelled revenue (90% interval 22–39%) and an
average return of £2.90 per pound over 2024–25, at weekly spend between £340k and £500k. That figure is not
experimentally anchored."

(b) "The fitted retail media curve is flat over the upper half of the observed spend range, which is consistent
with saturation; the model cannot distinguish saturation from a time-varying effect with the available data."

(c) "At current spend levels, the fitted search curve implies about £1.4 of revenue per additional pound over the
first £2m of reallocation; beyond that the curve flattens and the estimate is outside the observed range."

(d) "The residual not attributed to media or the included controls is £283m; its size depends on the control set
and it is not a measure of organic demand."

Each rewrite adds a condition, a range, or an anchor. None adds a new claim, which is the test of whether the
original was overclaiming or merely under-describing.

**Exercise 8 — television holdout, model answer.**

The design has to answer one question: is television ROI nearer 1.4 or 2.9? The disagreement spans 1.5 ROI units,
so the design needs an MDE of roughly 0.7 units to halve it.

Feasible design: 40 designated market areas, 20 pairs matched on the pre-period path, one member of each pair held
out from addressable television for 12 weeks, 12-week pre-period, 4-week washout. With a between-pair standard
deviation of 2.5% and 20 pairs, the MDE on a 12-week cumulative basis is about `2.97 × (0.025 / sqrt(20)) = 1.7%`
of baseline. If television is attributed at £63m on a £478m base (13%), that is about 1.9 ROI units — too coarse.

The honest conclusion is often that the design is infeasible at the precision required: accept the range and
manage within it, run the holdout to rule out the extremes, or find a lever that produces a larger contrast. A
design note concluding "this cannot be measured at the precision the decision needs" is a successful design note.

**Exercise 9 — vendor appraisal.**

1. Is 4.1x average or marginal, and over what spend range?
2. What is the client's contribution margin, and therefore the breakeven ROI?
3. Which channels were calibrated to experiments, and what were those experiments?
4. Where did the "category prior" come from — which markets, which years, which spend levels?
5. What is the R² of a model with seasonality and price but no media variables?
6. What was the VIF for retail media, and the interval after accounting for model selection?
7. What share of retail media spend is a condition of distribution, and is the listing decision
   modelled?
8. Does the recommended 30% shift stay inside the observed spend range for both channels?
9. What is the MDE of the evidence behind the 4.1x?
10. Who owns the model after handover, and at what cadence is it re-estimated?

Recommendation: usable as a hypothesis and not as a decision input. The R² claim is not evidence of media effect,
the calibration source is undisclosed, and a 30% shift is a large move justified by a point estimate on a curve
whose shape is the least identified part of the model. Ask for the model object, the priors, and the Pareto front,
and re-analyse before committing the budget.

## 14. Interview questions

**1. "What can an MMM tell you without an experiment?"**

A strong answer separates the three outputs: an associational decomposition, a conditional response curve within
the observed spend range, and a budget recommendation that depends on structural assumptions the data cannot test.
It says the last is a claim about the world, not a reading of the data. A weak answer says "it tells you channel
ROI", which conflates the three.

**2. "Why is R² a bad way to validate an MMM?"**

A strong answer notes that seasonality and price alone often explain most of the variance in weekly revenue, so a
model with no media effect can reach a high R²; the relevant comparison is the lift over a no-media benchmark, and
even that does not test identification. A weak answer says "R² does not imply causation", which is true and too
general to be useful.

**3. "Two models fit equally well and give opposite answers. What do you do?"**

A strong answer identifies the ridge — different carryover and effect combinations — and says the next step is a
design that varies spend in a way the controls cannot predict. It reports the range rather than choosing a model.
A weak answer picks the one with the better fit or the more plausible ROI, which is selecting on noise.

**4. "Explain adstock and saturation, and which one you trust less."**

A strong answer gives the geometric form and the Hill form, explains that adstock converts spend into a stock and
saturation converts stock into response, and says the saturation shape is less identified because it is inferred
from variation over a narrow range. A weak answer describes both correctly and expresses no view on
identifiability.

**5. "What is the difference between attribution and incrementality?"**

A strong answer says attribution is a rule that sums to the outcome and needs no counterfactual, while
incrementality is the part of the outcome that would not have happened otherwise and needs a counterfactual and a
mechanism argument. It gives an example where the two diverge sharply, such as branded search. A weak answer says
incrementality is "more accurate attribution".

**6. "How would you set a prior on a channel's ROI?"**

A strong answer describes the source (a geo or holdout test), the six translation judgements, the widening for
transport, and the prior-to-posterior diagnostic. It says the prior should be disclosed and auditable. A weak
answer says "use the experiment result as the prior", which ignores that the experiment and the model estimate
different quantities.

**7. "A vendor says their model explains 94% of sales. What do you ask?"**

A strong answer asks for the no-media benchmark R², the calibration source, the observed spend range, the VIFs,
and the spread across the Pareto front. A weak answer asks about data quality and sample size, which are the wrong
first questions.

**8. "Why is average ROI the wrong number for a budget decision?"**

A strong answer says the decision is about the next pound, which is the derivative of the response curve, and that
on a concave curve the marginal return is below the average and falls with spend. It gives the breakeven
relationship `1 / margin`. A weak answer says "you should look at marginal ROI" without explaining why the gap
grows.

**9. "Your client wants to double television spend. What do you say?"**

A strong answer asks what the observed spend range is, notes that doubling is extrapolation beyond it, and says
the model's behaviour out there is determined by the functional form rather than by data. It proposes a staged
increase with measurement. A weak answer checks whether the ROI is above one.

**10. "What is the smallest experiment that would improve this model?"**

A strong answer proposes a targeted separation test — a pause, a high/low alternation, or a single-market holdout
on the channel with the widest interval — and states what it would and would not identify. A weak answer proposes
"a bigger geo test" without connecting the design to the specific parameter that is unidentified.

## 15. Key terms

- **Marketing mix model (MMM)** — a regression of a KPI on marketing inputs and controls, with
  carryover and saturation transforms.
- **KPI** — key performance indicator; the modelled outcome, with its currency and period stated.
- **Adstock** — the transformation that spreads an input's effect over later periods; parameter θ.
- **Saturation** — the concave transformation from stock to response.
- **Hill function** — `x^α / (x^α + K^α)`; α sets the shape, K the inflection point.
- **Response curve** — the fitted mapping from spend to KPI, valid only over its observed range.
- **Average ROI** — attributed KPI divided by spend over the same window.
- **Marginal ROI** — the derivative of the response curve at a stated spend level.
- **Breakeven ROI** — `1 / contribution margin`; the return below which spend destroys value.
- **ROAS** — return on ad spend; revenue per unit of spend, usually platform-reported.
- **iROAS** — incremental ROAS; incremental revenue per incremental spend, from a design.
- **Lift** — observed outcome minus counterfactual outcome.
- **Counterfactual** — the outcome that would have occurred without the intervention.
- **Incrementality** — the part of lift caused by the intervention, net of substitution and
  pull-forward.
- **Attribution** — a rule assigning credit for an outcome to touchpoints.
- **Endogeneity** — correlation between a regressor and unobserved determinants of the outcome.
- **Exogenous variation** — variation in a regressor generated independently of the outcome.
- **Identification** — whether the data can distinguish parameter values that imply different
  answers.
- **Ridge** — the flat direction in the likelihood along which parameters are not separately
  identified.
- **Collinearity** — near-linear dependence among regressors.
- **VIF** — variance inflation factor, `1 / (1 - R²_j)` from the auxiliary regression.
- **Prior and posterior** — distributions over a parameter before and after the likelihood.
- **Calibrated MMM** — an MMM whose channel priors come from experiments or other exogenous
  evidence.
- **Holdout** — units or budget deliberately withheld from an intervention.
- **Geo lift test** — a lift test assigning whole markets to treatment and control.
- **Separation test** — a designed spend contrast intended to break the ridge.
- **Baseline** — the modelled component not attributed to marketing; a residual.
- **Triangulation** — comparing estimates from designs with different assumptions.

## 16. Sources

**Peer-reviewed and academic**

- Lewis, R. A., and Rao, J. M. (2015). "The Unfavorable Economics of Measuring the Returns to
  Advertising." *Quarterly Journal of Economics*, 130(4), 1941–1973. DOI: 10.1093/qje/qjv023.
  Reports 25 field experiments, a median ROI interval exceeding 100 percentage points, and the
  person-week requirement for informative tests. Grade A.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. (2019). "A Comparison of
  Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook."
  *Marketing Science*, 38(2), 193–225. DOI: 10.1287/mksc.2018.1135. Compares randomised and
  observational estimators on 15 experiments; the factor-of-three error figure is from the
  working-paper version. Grade A.

**Preprints and technical reports**

- Dew, R., Padilla, N., and Shchetkina, A. (2024). "Your MMM is Broken: Identification of
  Nonlinear and Time-varying Effects in Marketing Mix Models." arXiv:2408.07678. Grade B;
  preprint, not peer-reviewed at the time of access.
- Chan, D., and Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling." Google
  technical report. Grade C; company research, widely cited, not peer-reviewed.
- Zhang, Y., Wurm, M., Li, E., Wakim, A., Kelly, J., Price, B., and Liu, Y. (2024). "Media Mix
  Model Calibration With Bayesian Priors." Google Research publication. Grade C; simulation-based
  evaluation by the tool's owner.

**Vendor and platform documentation**

- Google, Meridian documentation, "Calibrate treatment priors",
  developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration. Discusses
  translation judgements, including the zero-spend versus reduced-spend estimand mismatch.
  Accessed 2026. Grade D. Companion pages, "ROI, mROI, and Contribution parameterizations" and
  GeoX "Intro to analysis", are Grade D.
- Meta, Robyn repository (facebookexperimental/Robyn) and CRAN reference manual, version 3.12.1.
  Documents ridge regression with evolutionary hyperparameter search, geometric and Weibull
  adstock, Hill saturation, the Pareto front, and calibration input from lift tests. Grade D.

**Standards, official, and regulatory**

- Regulation (EU) 2016/679 (General Data Protection Regulation). Grade B.
- Directive 2002/58/EC (ePrivacy Directive). Grade B.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (Financial Accounting Standards Board). Grade B.
- ISO 8601-1:2019, date and time representation. Grade B.
- Media Rating Council accreditation programme and viewable impression guidelines. Grade B for the
  programme; individual accreditation status must be checked at the time of use.
- Interactive Advertising Bureau measurement guidelines. Grade C; industry self-regulation.

**Note on citations.** Every source above was returned by the research pass for this topic. Grades reflect
provenance, not quality, and no statistic is attributed to a source that did not report it. Numbers constructed in
this text — the Halden Home case, the Hill parameters, the posterior update — are labelled as illustrative. The
claim that no large public benchmark of MMM budget-allocation accuracy exists describes the material reviewed here
and is not a systematic review.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Geo experiments, the design that anchors a channel prior: [`04_geo-experiments.md`](04_geo-experiments.md)
- Switchback and operational pilot designs: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Power, MDE, and the reason a small holdout cannot fail: [`06_power-mde.md`](06_power-mde.md)
- Interference, including shared budget and shared inventory: [`07_interference-spillover.md`](07_interference-spillover.md)
- Sequential testing and the cost of peeking at an MMM refresh: [`08_sequential-testing.md`](08_sequential-testing.md)
- Turning a claim into a testable estimand: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- Randomisation units, and why the market is the unit for media: [`02_randomisation-units.md`](02_randomisation-units.md)
- Cluster designs and design effects: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Appraising vendor and analyst evidence: [`10_vendor-evidence-appraisal.md`](10_vendor-evidence-appraisal.md)
- Regression with time-series structure, the estimation base for an MMM:
  [`../04_statistics_causal_inference/04_regression-time-series.md`](../04_statistics_causal_inference/04_regression-time-series.md)
- Directed acyclic graphs, for choosing the control set: [`../04_statistics_causal_inference/06_dags.md`](../04_statistics_causal_inference/06_dags.md)
- Elasticity and endogeneity, the same problem in demand models: [`../04_statistics_causal_inference/09_elasticity-endogeneity.md`](../04_statistics_causal_inference/09_elasticity-endogeneity.md)
- Omitted-variable sensitivity, for bounding what the controls cannot fix: [`../04_statistics_causal_inference/10_sensitivity-omitted-bias.md`](../04_statistics_causal_inference/10_sensitivity-omitted-bias.md)
- Survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Pricing, promotions, and revenue management: [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Product economics and margin conversion: [`../../../course/25_product_economics_leadership.md`](../../../course/25_product_economics_leadership.md)
- Return to the part map: [`../_index.md`](../_index.md)

