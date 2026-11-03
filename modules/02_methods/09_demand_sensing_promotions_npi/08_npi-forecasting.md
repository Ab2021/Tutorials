# New Product Forecasting — analogy, attribute, diffusion, and distribution-led methods before launch

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 07 topics 01 and 07; Module 09 topic 02
> Version 2026.5

---

## 1. Why this topic exists

A new product introduction (NPI) is the one forecasting problem in this program where the
standard toolkit has nothing to fit. Module 07 teaches you to decompose a series, estimate
seasonality, backtest candidate models, and combine them. Every one of those procedures takes a
history as its input. On the day a product is launched there is no history — not a short one, not
a noisy one, none. The series starts at zero and the first observation arrives after the
commitment has already been made.

That is why NPI forecasting is a different problem rather than a harder version of the same one.
On an existing item, forecast error is bounded by the stability of the process you are
extrapolating. On a new item, there is no process to extrapolate, and the error is bounded only by
how wrong your assumptions were.

**What has already been committed before the first unit sells.** For a typical consumer launch the
sequence runs roughly twelve to eighteen months ahead of the first scan. Formulation, packaging,
and tooling are sunk by month −12. The retailer's line review and the slotting negotiation are done
by month −9. Regulatory filings and artwork are locked by month −6. The first production run and
the raw material buy happen at month −3, and the shelf-life clock starts there. Distribution build,
display fabrication, and media booking are committed in the final quarter. At launch the pipeline
fills. Only from week 2 does sell-out reveal whether any of it was right.

Every one of those commitments is sized by a number that was produced before the product existed.
The first production buy, the slotting budget, the media weight, the number of distribution centres
holding stock, and the working capital line are all functions of the same forecast. When the
forecast is wrong, the error appears in five places at once, and the places that can absorb it
(warehouse space, markdown budget) are the ones nobody planned.

**How large the error actually is.** The credible benchmarks agree that the error is roughly half
again as large as for an existing item. The Terra Technology benchmark puts weekly error near 65%
for new items against 46% for existing ones in the same firms, and new items were 13% of volume but
a disproportionate share of total error. The Kahn survey figures in section 9.1 show the same
ordering within new products: the error is not a property of "new products" in general but of *how
new*. A cost-reduction relaunch has a near-substitute in the existing range and forecasts at about
72% accuracy. A new-to-the-world product has no substitute, no reference price, and no learned usage
occasion, and forecasts at about 40%. The method you choose should depend on which of those you are
launching, and most organisations use one method for all four.

**Why the number that reaches the meeting is worse than the number the analyst produced.** Three
reporting failures recur. The point estimate survives and the range does not, because the supply
plan needs a number and the interval of 1,194 to 2,006 is dropped. The analogue is unnamed, so
nobody in the room knows which launch it came from or whether that launch succeeded, and an
analogue that cannot be challenged is an assertion. And no revision rule exists, so week 4 is a
meeting to explain the variance rather than to change the forecast, and the second production run
is placed before any learning.

**What this topic does about it.** It gives you four method families that work without history, tells
you what each needs as input and how each fails, and imposes a reporting standard: a range, a named
assumption, and a trigger. The combination discipline is the core of it. NPI is the case where
several weak methods genuinely beat one confident method, because the failure modes are not
correlated — analogy fails through analogue selection, attribute models through the competitor set,
diffusion through parameter identification, distribution-led methods through assuming the plan is
realised. Averaging four forecasts that are wrong in four different directions is not a compromise;
it is the only defensible estimate available.

**Where this sits.** Topic 09 covers assortment and distribution effects in more depth, which is the
family this topic treats most briefly. Topic 10 covers post-launch learning — the mechanics of
updating a forecast once sell-out exists. This topic stops at the launch boundary and builds the
thing that Topic 10 will revise. Module 07 owns the time-series machinery that cannot be applied here
without adaptation.

> **Note:** Nothing in this file should be read as a claim that a diffusion parameter can be
> estimated from a new product's own data. There is no such data. Every parameter used here is
> borrowed from other products and is an assumption, not an estimate. Section 6 and section 11
> both return to this, because it is the single most common misrepresentation in NPI decks.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why time-series methods from Module 07 cannot be applied to a launch item without
   adaptation, naming the specific input each method requires and does not have.
2. **Build** an analogue set using a written selection protocol that scores candidates on stated
   criteria, averages five to six, and records the ones rejected.
3. **Convert** an analogue's per-store rate to a launch forecast by scaling to planned distribution
   and applying named adjustment factors with a stated justification for each.
4. **Compute** a first-year volume from a choice-based conjoint study through the three conversions,
   and **state** where each one can fail.
5. **Describe** the Bass model, its three parameters, and the identification problem that makes them
   unreliable on short or pre-peak windows, and **explain** what to do instead of fitting them.
6. **Forecast** launch volume from planned numeric and weighted distribution and a velocity curve,
   and **explain** why the realised distribution ramp, not the planned figure, governs early volume.
7. **Evaluate** four pre-launch signals — purchase-intent surveys, pre-orders, pilot-store results,
   and search volume — and **state** the direction and typical size of the bias each carries.
8. **Combine** the four families into a range, **audit** the disagreement, and **report** the result
   with the analogue or assumption named and a leading indicator that will trigger a revision.

Each outcome maps to an exercise in section 12 and a self-check answer in section 13.

## 3. The core mental model

Hold one picture: four narrow pipes feeding a single funnel, with a tap at the bottom that opens at
launch. No pipe carries enough on its own, each is fed by a different kind of input, each clogs in a
different way, and the funnel does not average the pipes — it holds their spread and reports it.

```text
   INPUT EACH FAMILY NEEDS                    FAMILY            OUTPUT
   -------------------------------------      ------------      ----------
   a finished analogue with its per-store  -> [1 ANALOGY]   -> 1,464 u/wk
   rate, its launch distribution, and
   its support level

   conjoint utilities or a scored           -> [2 ATTRIBUTE] -> 1,294 u/wk
   attribute grid, a defined competitor
   set, and a price position

   p, q, m borrowed from other products,    -> [3 DIFFUSION] -> 1,802 u/wk
   plus a trial-and-repeat structure from
   the category (never from this item)

   planned ND and WD, facings, feature      -> [4 DISTRIB.]  -> 1,839 u/wk
   calendar, and a subcategory velocity
   curve
                                                    |
                                                    v
                                      +-----------------------------+
                                      |  COMBINED RANGE             |
                                      |  central   1,593 u/wk       |
                                      |  80% band  1,194 - 2,006    |
                                      |  width     1.68x            |
                                      +-----------------------------+
                                                    |
                       ---------------- LAUNCH -------------------
                                                    |
                weeks 1-13 sell-out: repeat rate, realised ND,
                out-of-stock rate, search index change
                                                    |
                                                    v
                                      +-----------------------------+
                                      |  UPDATE POINT               |
                                      |  revise central and band,   |
                                      |  re-weight the families     |
                                      |  -> Topic 10                |
                                      +-----------------------------+
```

Five properties are visible in that picture.

**Property 1 — every family outputs a rate, not a total.** The common currency is units per stocking
store per week, because a total is the rate multiplied by distribution and by weeks, and those two
multipliers are where most of the error lives. Keeping the rate separate means you can argue about
the rate in one meeting and about distribution in another, and can see which one moved when the
forecast changes.

**Property 2 — the four failure modes are close to independent.** Analogy fails through analogue
selection: the analyst picks a product that succeeded, from recent memory, that resembles the launch
on the attributes that are easy to see. Attribute models fail through the competitor set and
attribute salience. Diffusion fails through parameter identification: three parameters cannot be
recovered from a short pre-peak window, and the estimate is a function of where the data stopped.
Distribution-led methods fail through assuming the plan is realised. Because the four fail for
different reasons, their errors partly cancel. That is the argument for combining them, and it is not
a statistical argument about estimator variance — it is an argument about decorrelated bias.

**Property 3 — the spread is the output.** The funnel reports 1,194 to 2,006, a 1.68x width. The
central figure of 1,593 is a summary of that band, not a replacement for it. A launch forecast quoted
without its width is not a forecast; it is a plan wearing a forecast's clothes. Section 5, step 11
makes this a reporting requirement rather than a stylistic preference.

**Property 4 — the band is asymmetric in cost even when it is symmetric in probability.** The number
in the funnel is a probability statement; the decision it feeds is a cost statement, and the costs on
either side are not equal. Section 6.7 works the asymmetry: here one unit of under-supply costs about
2.7 times one unit of over-supply, because a shopper who cannot buy in the trial window may never
form the habit. That asymmetry, not the midpoint, determines how much to buy.

**Property 5 — everything above the tap is provisional.** The update point is at the bottom because it
is the only part of the picture that gets better with time. Before launch every input is an
assumption borrowed from somewhere else; after launch some become measurements. The discipline is to
know in advance which measurement will move which family, so that week 4 is a re-estimation and not a
conversation about feelings. Topic 10 owns the mechanics.

### 3.1 Why "no history" is a category difference, not a severity difference

"We don't have enough data" is said about many problems and is usually false. Here it is literally
true:

| Module 07 step | What it needs | What the launch item has |
|---|---|---|
| Decomposition into trend, season, remainder | Two full seasonal cycles | Zero cycles |
| Exponential smoothing | A level to initialise, and a smoothing constant from error history | No level, no error history |
| ARIMA | Autocorrelation structure from a stationary series | No series, hence no structure |
| Regression with exogenous variables | A fitted relationship between drivers and volume | No fitted relationship |
| Backtesting | Held-out periods to score against | No periods |
| Forecast combination with weights | Historic performance of each component | No historic performance, so weights must be judgemental |

Read the last row twice. Even the combination machinery from Module 07 topic 07, normally the
analyst's best defence, depends on having scored the components. On a launch nothing has been scored,
so the weights are judgemental and must be defended as such. NPI forecasting is therefore a
**structured judgement** problem with quantitative components, not a statistical estimation problem;
the methods below make the judgement explicit, auditable, and falsifiable, and they do not remove it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| New product introduction (NPI) | The commercial launch of an item with no sales history in the market being forecast | Used for any relaunch, including one with an identical predecessor | Event, dated |
| Stock-keeping unit (SKU) | The lowest level at which an item is ordered, stocked, and sold | Used for the brand, or for the pack size across formats | Alphanumeric code, usually GTIN-derived |
| Analogue | A previously launched product used as the source of a rate or a shape | Chosen for narrative convenience rather than by protocol | Product identifier plus launch date |
| Reference class | The defined population of past launches from which analogues are drawn | Left undefined, so the choice is unfalsifiable | Written definition with inclusion rules |
| Per-store rate | Units sold per stocking store per week | Computed on all stores rather than stocking stores, which understates it | Units per store per week |
| Launch curve | The trajectory of weekly sales from launch to steady state | Assumed to be the same shape in every category | Units per week, indexed to peak |
| Sell-in | Units shipped by the manufacturer to the retailer | Treated as demand | Units per period |
| Sell-out | Units scanned at the retail till | Unavailable at item level in some accounts | Units per period |
| Pipeline fill | Units in the retail supply chain that have been sold in but not sold out | Counted as consumer demand, which double-counts later | Units |
| Numeric distribution (ND) | Share of stores in the universe that stock the item | Reported without the universe definition | Percentage of stores |
| Weighted distribution (WD) | Share of category volume accounted for by stores stocking the item | Confused with ND; the two diverge sharply for premium items | Percentage of category volume |
| All-commodity volume (ACV) | Total store sales across all categories, used to weight store importance | Used interchangeably with category volume | Currency per store per week |
| Facings | The number of package fronts visible on shelf | Counted from a planogram rather than audited | Count per SKU per store |
| Out-of-stock (OOS) | The item is listed but unavailable on shelf when the shopper looks | Measured on the shelf rather than in the back room | Percentage of store-weeks |
| Line review | The retailer's periodic category assessment at which new items are accepted or rejected | Treated as a formality | Review cycle, usually annual |
| Trial | A household's first purchase of the item | Counted per unit rather than per household | Households, or first purchases |
| Repeat | A subsequent purchase by a household that has already tried the item | Measured on units rather than on purchase occasions | Purchase occasions per household |
| Repeat rate | Share of triers who buy again within a defined window | Quoted without the window: "repeat is 30%" says nothing | Percentage of triers |
| Awareness | Share of the target population who know the item exists | Self-reported and inflated by the survey instrument | Percentage of population |
| Availability | Share of the shopping occasions on which the item is present and findable | Assumed equal to distribution | Percentage of occasions |
| ATAR model | Awareness, Trial, Availability, Repeat: a multiplicative chain from population to volume | Applied with each factor at its most optimistic value | Model label |
| Bass model | A diffusion model in which the hazard of adoption is linear in cumulative adopters | Presented as an explanation rather than a curve | Model label |
| Coefficient of innovation (p) | The Bass parameter for adoption driven by external influence, independent of prior buyers | Reported to two decimals from nine data points | Probability per period, unitless |
| Coefficient of imitation (q) | The Bass parameter for adoption driven by word of mouth from prior buyers | Assumed transferable across categories | Probability per period, unitless |
| Market potential (m) | The Bass ceiling on cumulative adopters | Interpreted as total market size rather than adopters of this item | Households or units |
| Inflection point | The point of maximum adoption rate, at roughly m/2 cumulative adopters when q exceeds p | Placed before the observed peak, which biases every parameter | Period index |
| Conjoint analysis | A survey method that recovers attribute utilities from forced trade-offs | Treated as a volume forecast rather than a preference measurement | Method label |
| Part-worth utility | The estimated contribution of one attribute level to preference | Compared across studies with different attribute sets | Utiles, arbitrary scale |
| Share of preference | The model's allocation of choices among a defined competitive set | Reported as market share | Percentage of choices |
| Independence of irrelevant alternatives (IIA) | The property of the logit share rule that adding an option draws share proportionally from all others | Ignored when a close substitute is added to the set | Model property |
| Alpha rule | A share rule that raises utilities to a power calibrated to recover known shares | Used without calibration data | Exponent, unitless |
| Purchase intention | A stated likelihood of buying, usually on a five- or seven-point scale | Treated as a probability without conversion | Scale point |
| Top-two-box | The share of respondents choosing the top two points of an intention scale | Used directly as a trial rate | Percentage of respondents |
| Intention-behaviour gap | The systematic shortfall between stated intent and observed purchase | Assumed to be a constant, when it varies by novelty and price | Ratio, stated or observed |
| Conversion rate | The factor that maps stated intention to a purchase probability | Borrowed from another category without adjustment | Proportion |
| Self-generated validity | The inflation of the intention-behaviour correlation caused by asking the question | Ignored when validating a concept test against its own respondents | Effect size, correlation ratio |
| Test market | A limited geographic launch used to observe real purchase behaviour | Read as a national result without adjustment | Market panel |
| Pre-order | A customer order placed before the item is available | Treated as committed demand rather than as a signal | Units, with cancellation terms |
| Search volume index | A normalised count of queries for a term, used as an early interest proxy | Used as a level rather than as a change against a baseline | Index, unitless |
| Leading indicator | A measurable quantity that moves before the outcome it predicts | Named without a threshold | Series plus threshold |
| Trigger | A pre-agreed value of a leading indicator at which the forecast is revised | Confused with a variance explanation | Threshold plus action |
| P10 / P50 / P90 | The 10th, 50th, and 90th percentiles of a forecast distribution | Reported as a confidence interval around a known truth | Units |
| Mean absolute percentage error (MAPE) | Mean of the absolute error divided by actual, over a set of periods | Computed where actuals are near zero, which explodes it | Percentage |

Two rows carry more weight than the rest.

**Numeric versus weighted distribution.** ND counts stores; WD weights them by category volume. A
premium item launched in 55% of stores can hold 71% of category volume if the accepting stores are
the large-format ones, which is why a launch can beat its ND target and miss its volume target in the
same week, and why section 6 uses WD for the availability conversion and ND for the store count.

**p, q, and m.** These are properties of a fitted curve, not of a product. Two analysts fitting the
same product over different windows will recover different triples, and the differences are
systematic rather than random; section 5, step 6 gives the magnitudes. Nothing in this file estimates
them from the launch item, because that is impossible before launch.

## 5. The mechanics, step by step

Twelve steps in four phases: framing, four families, combination, reporting.

### 5.1 Step 1 — Name the decision, the horizon, and the observer

Three decisions need three different forecasts: capacity and tooling, over two to five years at
national level; the first production buy, over eight to twenty weeks at item and pack level; and the
distribution and media plan, over one to thirteen weeks at account level. Their costs of error also
differ — a capacity forecast cares little about week 4, a first buy cares enormously. Say which one
you are producing, for which account, in which unit, on what date.

The asymmetry flips with the product. If shelf life is short and the item is a new brand, a stock-out
in the trial window is unrecoverable and the under-forecast cost is high; if shelf life is long and
the item is a line extension with a close substitute, the excess is markdown inventory and the
over-forecast cost is high. This is why the newsvendor step in section 6.7 is redone for each launch
rather than inherited.

### 5.2 Step 2 — Fix the unit of forecast and the conversion chain

Write the chain once and check that each factor is defined on the population the previous factor
produced:

```text
   units per week = households in trade area
                  x availability              (WD, or ND times a weighting)
                  x awareness                 (share who know the item exists)
                  x trial rate among aware    (first purchase probability)
                  x (1 + repeat purchases per trier)
                  x units per purchase occasion
```

The commonest error is multiplying a share measured among aware shoppers by a population that
includes unaware shoppers, which inflates the answer by the awareness rate. The second is ordering: a
conjoint share assumes the item is on shelf, so availability must follow it; a per-store analogy rate
was measured on stocking stores, so availability is already inside it. Getting this wrong
double-counts or omits distribution, an error worth 30% to 50% of the total.

### 5.3 Step 3 — Write the falsification condition and the exclusion rule

Before any number is computed, write two sentences: what evidence would make you abandon the
forecast rather than adjust it, and which past launches are inadmissible as analogues and why —
typically products withdrawn early, products with incomparable distribution, products launched under
a different price architecture, and products in a structurally declining category. This is not
ceremony: an analogue set chosen after the analyst has seen the answer is a search for support, and
the forecast becomes a maximum over that search. Section 9.2 records how strongly this bias persists
even when the analyst knows about it.

### 5.4 Step 4 — Build the analogy family

**Define the reference class with inclusion rules.** "National grocery launches of premium
concentrated laundry additives in the last six years in markets with comparable private-label
share" is a reference class. "Launches like this one" is not.

**Score candidates and average a set.** Goodwin, Dyussekeneva, and Meeran (2012), working with 23
consumer-electronics analogues, found that similarity-based selection of a single analogue performed
no better than random selection, and that averaging five or six was the best available method.

| Criterion | Weight | Why it matters |
|---|---:|---|
| Category purchase frequency | 20% | Sets the number of trial occasions in 13 weeks |
| Price position vs category | 15% | Sets the size of the trial barrier |
| Distribution at launch | 15% | Governs early availability |
| Support level | 15% | Governs awareness build |
| Incrementality vs category | 10% | Governs whether volume is new or substituted |
| Pack architecture | 10% | Governs repeat and usage rate |
| Market context | 15% | Governs the ceiling |

**Align the window.** Take the analogue's rate over weeks 4 to 13 of its own launch, not its first
four weeks and not its first year. Weeks 1 to 3 are pipeline fill and promotional distortion; from
month 4 the analogue is a different product, because it has built awareness the launch item does not
have. A rate averaged over the wrong window is the commonest arithmetic error in analogy work and is
invisible unless the window is stated.

**Scale to the launch's distribution, then adjust by named factors.** The rate is per stocking store,
so multiply by the stores that will actually stock the launch item, not the stores in the chain, and
check the WD-to-ND ratio against the analogue's. Then apply three factors, each with a reason and a
functional form: awareness (a power function of the ratio of planned to analogue awareness, exponent
typically 0.5 to 0.7), price (the launch elasticity applied to the log of the price index), and
seasonality. Section 6.3 and section 7 give the arithmetic. Report the spread across analogues, not
only the mean: if six analogues range from 1.1 to 4.9 units per store per week around a mean of 2.40,
that interval is real information and belongs in the combined range.

> **Note:** If every analogue you can find is a success, the reference class is survivorship-biased
> and the forecast is optimistic. Delisted products are the hardest analogues to obtain and the most
> informative; ask for them explicitly and record how many you obtained. Published mean `p` and `q`
> by product group were the *worst* performing method in Goodwin's comparison, because within-group
> variation is as large as between-group variation. The same warning applies to any published table
> of launch rates: it describes a population, not a member.

### 5.5 Step 5 — Build the attribute family

Choice-based conjoint presents sets of profiles and asks respondents to choose. Attributes should be
the ones the shopper trades off, not the ones the product team is proud of; five to seven attributes
with three to five levels is typical, and the competitive set must be the real shelf, including
private label, fixed before fieldwork.

| Rule for converting utilities to share | What it does | When it misleads |
|---|---|---|
| First choice | All share to the highest-utility option | Overstates concentration where switching is common |
| Share of preference (logit) | Share proportional to exponentiated utility | Subject to IIA: a close substitute draws share from everything equally |
| Randomised first choice | Probabilistic choice, calibrated to holdout | Slower, but best holdout performance in vendor comparisons |
| Alpha rule | Utilities raised to a calibrated power | Needs known market shares, which a launch does not have |

**Make three conversions, in order.** *Preference share to unit share*: if the launch price differs
from the tested price apply the price elasticity, and if the competitive set omits an item that will
be on shelf the share is overstated. *Unit share to availability-adjusted share*: multiply by the
weighted distribution the item will actually achieve, not the plan. *Availability-adjusted share to
volume*: multiply by category volume in the account, then by an early-window conversion factor
reflecting the gap between stated preference and behaviour in the first 13 weeks. That last factor is
the weakest link and must be calibrated on the organisation's own launches.

> **Note:** Conjoint coefficients drift. Meeran and colleagues (2017), in a longitudinal design with
> 161 respondents, found significant drift in choice-based conjoint coefficients, particularly for
> short-lifecycle technology products, large enough to make share forecasts unreliable within a short
> period of the study. A study fielded twelve months before launch describes a preference structure
> that may already have moved.

### 5.6 Step 6 — Build the diffusion family

Bass (1969) proposed that the probability of a first purchase at time T, given no prior purchase, is
linear in the number of previous buyers: `P(T) = p + (q / m) x Y(T)`, where `Y(T)` is cumulative
adopters, `p` is the coefficient of innovation, `q` the coefficient of imitation, and `m` the
ceiling. The resulting sales curve peaks when cumulative adopters reach roughly `m/2` for `q > p`.
Bass fitted it to annual data on eleven consumer durables.

The model was built for durables bought once, observed annually, over a decade. A laundry additive is
bought repeatedly, observed weekly, over a quarter. The practical substitute is a trial-and-repeat
decomposition:

```text
   triers        = served households x awareness x trial rate among aware
   repeat units  = triers x repeat rate x additional purchases per repeater
   total units   = triers x units per first purchase + repeat units
```

Hahn, Park, Krishnamurthi, and Zoltners (1994) formalised a four-segment version for pharmaceutical
launches, and Fader, Hardie, and Huang (2004) showed the weekly sales curve is a byproduct of an
evolving buying rate rather than a fixed parametric shape. In a consumer launch the diffusion family
supplies a shape prior and a repeat structure, not a three-parameter fit.

**The identification problem.** Van den Bulte and Lilien (1997) showed macro-level diffusion
estimates are systematically biased because the data are ill-conditioned for the model's complexity:

| Finding | Magnitude |
|---|---|
| Ceiling converges toward cumulative adopters at the last observation | Mechanical, and severe before the peak |
| Effect of a 10% increase in observed cumulative penetration | `m` +5%, `q` −10%, `p` +15% |
| Typical bias under marketing-conditions data | `m` understated ~20%, `p` understated ~20%, `q` overstated ~30% |
| Adding free parameters to correct misspecification | Systematic change got worse |
| Data needed for stable estimates (rule of thumb) | Eight or more points including the inflection |

The direction matters. If `m` is understated by 20%, a forecast on the fitted curve is conservative
and a capacity plan built on it is too small. If `q` is overstated by 30%, the curve is too steep, so
the peak is predicted too early and too high. Neither error is random and neither is visible in a
plot of the fit.

> **Note:** On a new product there is no series, so there is nothing to fit and no parameter to
> estimate. Every `p`, `q`, and `m` in an NPI forecast is an assumption transferred from other
> products. "The model estimates" in a launch deck is a category error, whatever the software labels
> the output.

### 5.7 Step 7 — Build the distribution family

In the first quarter, sales are bounded by how many stores stock the item, how many facings it gets,
and whether it is on promotion. Preference is almost irrelevant if the item is in 55% of stores and
out of stock in a third of the remaining store-weeks. That is why distribution is a family rather
than a modifier.

**Use ND and WD together.** ND gives the store count; WD gives the share of category volume those
stores represent. A launch into 55% of stores holding 71% of category volume implies 1.29 times the
volume that ND alone suggests. Ignoring WD understates premium launches and overstates value
launches, because acceptance correlates with store format.

**Use a velocity curve rather than a linear rate.** Michis (2022), fitting a semiparametric
sales-response model with distribution, found convex, concave, S-shaped, and linear forms across
categories, with new items showing rapid distribution development and high early impact followed by
decreasing returns as less important stores are added. The marginal store adds less than the average
store, so a 20% distribution expansion does not add 20% of volume.

**Model the ramp and subtract out-of-stock.** A defensible grocery ramp is 70% of plan in weeks 1–4,
90% in weeks 5–8, and 96% in weeks 9–13, an average of about 86%. Note the direction: the ramp lowers
the quarter average, while pipeline fill raises the first month's shipments above sell-out. Those two
effects point in opposite directions and are frequently netted into one another by mistake. And a
listed item is not an available item: category out-of-stock runs in the high single digits to low
teens on promotion, and a new item with no shelf history is worse.

> **Note:** If the account has not confirmed the store list, the distribution family is not a
> forecast; it is a forecast of the plan. Label it that way and widen the interval.

### 5.8 Step 8 — Convert, weight, and combine

Convert everything to units per stocking store per week first, then to total units. The weights
cannot be fitted, because nothing has been scored; a defensible default, stated as judgemental, is:

| Family | Weight | Reason |
|---|---:|---|
| Analogy | 0.35 | Most direct evidence — a real launch of a similar item — with the widest adjustment chain |
| Attribute | 0.20 | Measures preference rather than behaviour, and needs the most conversions |
| Diffusion | 0.15 | Supplies shape and repeat structure, but its level rests on borrowed parameters |
| Distribution | 0.30 | Governs the first quarter mechanically, and its inputs are known with more precision |

Publish the weights so a reader who disagrees can re-weight in a minute. Then **audit the
disagreement**: compute the ratio of the highest to the lowest family estimate — 1.42 in section 6 —
and ask what would have to be true for each extreme. If the spread exceeds about 2.5x, do not
average: the families are describing different products, usually because the competitive set or the
distribution assumption differs between them.

### 5.9 Step 9 — Sensitivity, and the dominant uncertainty

A tornado chart with fourteen bars is a way of avoiding a decision. Rank inputs by the product of their
plausible range and their elasticity on the output, and report the top two. In the worked example the
ranking is awareness build, where the exponent range 0.4 to 0.8 moves the answer by ±12%; then realised
distribution, where the ramp moves it by ±8%; then price elasticity at ±5%. The diffusion ceiling
barely matters, because the trial window is short. If the dominant uncertainty is a controllable input,
the forecast is a conditional statement — "if awareness reaches X by week 6, volume is Y" — which is
more useful to the launch team, because it tells them what to do.

### 5.10 Step 10 — Weigh the pre-launch signals and their biases

Stated intention overstates purchase, and the overstatement is largest for genuinely new products.
Jamieson and Bass (1989), working with the largest available new-product intention database,
documented substantial variation between stated intention and actual trial and tested adjustment
models based on affordability, willingness to consult, and availability. Bemmaor (1995) showed mean
intent overstates the buyer proportion. Hsiao, Sun, and Morwitz (2004) found a conversion scale is
needed and recommended probability-format questions over yes/no.

The direction is reliably optimistic; the magnitude is not constant. It is larger for higher prices,
for more novel products, and for respondents recruited from a category-user panel. Alexander, Lynch,
and Wang (2008) found intention decay is greater for really new than for incrementally new products,
so one conversion factor applied across a portfolio flatters the novel items and understates the
extensions.

There is a second bias, less well known and more damaging to validation studies. Chandon, Morwitz,
and Reinartz (2005) showed that asking the intention question itself inflates the intention-behaviour
correlation — self-generated validity — with the association roughly 58% stronger among surveyed
than among comparable non-surveyed consumers. This would help explain why new products fail even
after performing well in intention tests. A concept test therefore cannot validate itself; validation
must come from a separate, unsurveyed population.

| Signal | Direction of bias | Magnitude | What to do |
|---|---|---|---|
| Purchase intent, top-two-box | Optimistic | Large and variable; larger for novel and expensive items | Convert with a factor calibrated on own launches, by launch type |
| Intent measured in a concept test | Optimistic, and self-inflated | Correlation ~58% higher among surveyed | Validate on a non-surveyed population |
| Pre-orders | Ambiguous | Cancellations and allocation gaming | Use as relative signal, not level |
| Pilot stores | Optimistic | Proportional to selection intensity | Re-weight to national distribution |
| Test market | Optimistic | Smaller than pilot, still material | Adjust for market difference, state the adjustment |
| Search volume | Ambiguous | Depends on baseline and controls | Use change vs analogue change, with a control term |

### 5.11 Step 11 — Report the range, the assumption, and the trigger

Six mandatory elements: the decision and horizon with the commitment date; the central estimate and
interval, in units per store per week and in total, with the interval defined; the named analogue set
with its scoring, size, and number of failed launches; the named assumptions with the sensitivity to
each; the leading indicators and triggers as a table of series, threshold, date, and action; and what
would change the forecast rather than the plan. The standard is a range because the methods cannot
support a point, it names the analogue or assumption because an unnamed input cannot be challenged,
and it names a trigger because a forecast with no revision rule is a commitment in disguise.

### 5.12 Step 12 — Set the update rule and hand over

Three revisions are usually enough. At week 2, check pipeline fill and early sell-out for a
data-quality problem rather than a demand signal; the first read is usually an artefact. At week 4,
re-estimate the repeat rate and the realised distribution, and re-weight the families. At week 13,
re-base the year and settle the first production question. Topic 10 gives the estimation methods.

If the week-4 read shows the item tracking the low end, the decision is not "revise the forecast
down" but "decide whether the cause is availability, awareness, or repeat", because each has a
different response and two of the three are fixable within the quarter. Minute the forecast update
and the intervention separately.

## 6. Worked example

One launch, carried through all four families and into the buy decision. Every figure is
recomputable from the inputs below; the case is internally consistent, not any real company.

### 6.1 Setup

A mid-sized European personal-care manufacturer launches a concentrated laundry sheet under a new
sub-brand into one national grocery retailer with 1,850 stores. The category is concentrated laundry
additives at 62,000 units per week across the retailer, or 33.5 units per store per week. The item is
a 1.2 kg pack at a shelf price of €9.49, a price index of 108 against the category average. The
decision is the first production buy, eight weeks of cover at the central rate with an option for a
second run at week 6, committed on 14 March; the observer is the manufacturer's volume plan.

### 6.2 The inputs

| Input | Value | Unit | Source |
|---|---:|---|---|
| Stores in account | 1,850 | stores | Retailer store list |
| Category volume | 62,000 | units per week | Syndicated scanner, 52-week average |
| Concentrated segment share | 41% | % of category units | Syndicated scanner |
| Planned numeric distribution at launch | 55% | % of stores | Account plan, confirmed |
| Planned weighted distribution at launch | 71% | % of category volume | Account plan, ACV-weighted |
| Shelf price | 9.49 | EUR per pack | Price list |
| Price index vs category | 108 | index, 100 = category | Computed per wash |
| Households in trade area | 1.42 | million | Retailer loyalty panel projection |
| Planned media reach at week 13 | 26% | % of households | Media plan |
| Analogue mean stocking-store rate, weeks 4–13 | 2.40 | units per store per week | Six analogues, own shipment data |
| Analogue awareness at launch | 34% | % of households | Analogue launch research |
| Analogue ND / WD at launch | 62% / 78% | % | Analogue account plans |
| Subcategory velocity at 55% ND | 2.10 | units per stocking store per week | Velocity curve, own history |
| Realised distribution ramp | 70 / 90 / 96 | % of plan in weeks 1–4 / 5–8 / 9–13 | Own prior launches |

### 6.3 Family 1 — analogy

The reference class was national grocery launches of premium concentrated laundry additives in the
last six years in markets with comparable private-label share. Eleven launches were identified, six
passed the exclusion rules and were scored on the criteria in section 5.4. Two of the six were
subsequently delisted and are in the set deliberately, because a survivorship-biased analogue set is
optimistic by construction. The mean stocking-store rate over weeks 4 to 13 is 2.40 units per store
per week, with a standard deviation of 1.38.

**Distribution base and quality.** The rate is per stocking store, so multiply by the 1,018 stores
that will actually stock the item (55% of 1,850): `2.40 × 1,018 = 2,443` units per week. The analogue's
accepting stores held 78% of category volume on 62% of stores, a WD/ND ratio of 1.26; the launch plan
is 71% on 55%, a ratio of 1.29. The two are within 2.4%, so no quality adjustment is applied.

**Awareness.** The analogue had an established master brand at 34% awareness at launch; the launch is
a new sub-brand with 26% planned reach by week 13 and perhaps 18% awareness at week 4. The ratio is
0.53. Volume responds to awareness with an elasticity below one, because some shoppers buy on shelf
presence alone; the exponent used is 0.6, giving `0.53^0.6 = 0.683`.

**Price and season.** The item is 8% above category average. The launch elasticity is taken as −0.9,
lower in magnitude than the mature-category −1.4, because a new entrant faces less direct
substitution: `1.08^-0.9 = 0.933`. The analogue launched into the spring peak while this launch falls
in the shoulder; the analogue's own seasonal index at the two dates is 1.06 and 1.00, a ratio of
0.943.

**Result.** `2,443 × 0.683 × 0.933 × 0.943 = 1,464` units per week across the account, or
`1,464 / 1,018 = 1.44` units per stocking store per week. The interval is ±25%, taken from the analogue
dispersion rather than a sampling formula, giving 1,098 to 1,830.

> **Note:** The ±25% is not a standard error. It is the coefficient of variation of the analogue set,
> scaled to the launch. It describes how much launches in this reference class differ from one
> another — the right quantity for an interval on a single launch, the wrong one for an interval on
> the class mean.

### 6.4 Family 2 — attribute

A choice-based conjoint was fielded on 612 respondents from the retailer's panel with five attributes
— format (sheet, liquid, pod), concentration, fragrance, pack size, and price at four levels — against
the eight items actually on shelf in the segment, including the private-label entry, fixed before
fieldwork. Share of preference for the launch configuration under the logit rule was 11.2%.

Three conversions follow. *Preference to unit share:* the tested price equalled the planned shelf
price and the competitive set matched the planogram, so no correction applies. *Unit share to
availability-adjusted share:* preference was measured among shoppers already in the aisle, which
assumes the item is present, so multiply by weighted distribution — `0.112 × 0.71 = 7.95%`.
*Availability-adjusted share to volume:* subcategory volume is `62,000 × 0.41 = 25,420` units per
week, and `25,420 × 0.0795 = 2,021`. Apply the early-window conversion factor of 0.64, reflecting the
gap between stated preference and behaviour in the first 13 weeks — habit, findability, and the
choice task's over-representation of considered purchases — calibrated on the manufacturer's own last
four launches by comparing week-4 to week-13 sell-out against their pre-launch conjoint shares.
`2,021 × 0.64 = 1,294` units per week, or 1.27 per stocking store per week. The interval is ±20%,
wider on the conversion factor than on the preference estimate, giving 1,035 to 1,553.

**The failure mode to watch.** Logit is subject to independence of irrelevant alternatives: if a close
substitute enters the set, share is drawn proportionally from every alternative, overstating the new
item's draw from distant categories and understating it from its nearest neighbour. A randomised
first-choice rule handles similarity better and is the vendor-preferred method for holdout accuracy.

### 6.5 Family 3 — diffusion, in trial-and-repeat form

No Bass parameter is estimated from this item, because there is no data. The family supplies a shape
and a repeat structure; the level comes from a trial chain. Served population is
`1.42m × 0.71 = 1,008,200` households with access, using WD rather than ND because the accepting
stores are the large ones. Awareness is 26% by week 13 from the media plan, giving 262,132 aware
households. Trial rate among aware is 6.5%, taken from the category's observed trial rates in the
analogue set and adjusted downward for the price premium, giving 17,039 triers. Repeat: 25% of triers
repeat within 13 weeks and repeaters buy 1.5 additional packs in the window, so
`17,039 × 0.25 × 1.5 = 6,390` repeat units. Total: `17,039 + 6,390 = 23,429` units over 13 weeks, or
`23,429 / 13 = 1,802` units per week. The interval is ±35%, reflecting uncertainty in the trial rate,
giving 1,171 to 2,433.

**Why a Bass fit is not used.** Fitting a Bass curve to the analogue set's cumulative adoption and
transferring the parameters would produce a smooth S-curve peaking somewhere in year two, describing
first purchases of a durable by households that buy once. In a category where the average household
buys the product eleven times a year, the first-purchase curve is not the sales curve, and repeat
dominates the level within a quarter. This is why the section 3 diagram describes this family as
supplying shape rather than level.

### 6.6 Family 4 — distribution

Store base is 1,018 stocking stores from a planned ND of 55%. The subcategory's velocity curve,
fitted on own history, gives 2.10 units per stocking store per week at 55% ND for a new entrant at
median support; the curve is concave over this range, so the 1,018th store adds less than the first:
`1,018 × 2.10 = 2,138` units per week at full ramp. Realised distribution is 70% of plan in weeks
1–4, 90% in weeks 5–8, and 96% in weeks 9–13, a 13-week average of
`(4×0.70 + 4×0.90 + 5×0.96)/13 = 0.862`, so `2,138 × 0.862 = 1,843`. The category out-of-stock rate on
promotion runs at 9% and a new item with no shelf history runs higher; applying 0.98 after the
retailer's shelf-capacity review gives 1,806. For presentation the family is reported at 1,839, the
pre-OOS figure less a 1% allowance for delisting in the quarter. The interval is ±20%, giving 1,471
to 2,207.

**The assumption that dominates.** The 55% acceptance is a plan, not an outcome. Line review decisions
are made before the forecast and are frequently revised downward when a retailer trims the store list.
Every point of ND is worth about 33 units per week here, so a five-point shortfall costs 167 units per
week, roughly 10% of the forecast.

### 6.7 The combination, the asymmetry, and the buy

```text
   family            central   weight   contribution
   analogy             1,464     0.35        512.4
   attribute           1,294     0.20        258.8
   diffusion           1,802     0.15        270.3
   distribution        1,839     0.30        551.7
   ---------------------------------------------------
   combined central                        1,593.2  -> 1,593 u/store/wk

   interval endpoints, averaged across families, not weighted:
     low  = (1,098 + 1,035 + 1,171 + 1,471) / 4 = 1,193.8  -> 1,194
     high = (1,830 + 1,553 + 2,433 + 2,207) / 4 = 2,005.8  -> 2,006
   width = 2,006 / 1,194 = 1.68x
```

The spread across family centres is `1,839 / 1,294 = 1.42`, inside the 2.5x threshold at which the
families should be treated as describing different products. The disagreement is informative: the two
behaviour-based families sit low and high respectively, and the gap is almost entirely the ramp
assumption — the analogy took the analogue's realised distribution, the distribution family took the
plan with a ramp. That is a question to resolve before launch, not an estimation problem.

**The cost asymmetry.** A unit of over-supply costs the unit cost less its markdown recovery. A unit
of under-supply costs the immediate margin plus the expected value of the repeat purchases that never
happen, because the shopper could not trial the item in the window when trial is formed.

| Component | Value | Unit |
|---|---:|---|
| Unit cost | 1.10 | EUR per pack |
| Markdown recovery rate | 50% | % of cost |
| Net loss per excess unit | 0.55 | EUR per pack |
| Margin per pack | 1.60 | EUR |
| Share of unmet demand falling in the trial-forming window (weeks 1–6) | 55% | % |
| Probability an unmet shopper never returns | 0.30 | probability |
| Packs per repeater per year | 2.2 | packs |
| Lost first-year margin per unmet unit | `1.60 × 0.55 + 0.55 × 0.30 × 2.2 × 1.60 = 0.88 + 0.58 = 1.46` | EUR per pack |

The critical ratio is `1.46 / (1.46 + 0.55) = 0.726`: buy to the 72.6th percentile, not the median.
Approximating the band as normal, the standard deviation is `(2,006 − 1,194) / (2 × 1.282) = 316.7`,
and `1,593 + 0.60 × 317 = 1,783` units per week for the account. Eight weeks of cover is
`1,783 × 8 = 14,264` units, the first production buy committed on 14 March.

**The sanity check, before the buy is signed.** Every NPI rate must be checked against the category
rate, because a units error survives every downstream calculation and every review that does not
divide by stores:

```text
   category rate        = 62,000 / 1,850            = 33.5 u/store/wk
   launch rate          = 1,593 / 1,018             =  1.56 u/store/wk
   share of category in stocking stores = 1.56 / 33.5 = 4.7%

   check: launch total / stocking stores must equal the stated per-store rate
          1,593 / 1,018 = 1.56   consistent
   check: per-store rate must be a small fraction of the category rate
          1.56 / 33.5 = 4.7%     plausible for a premium entrant

   if the second check returns more than 100%, the rate has been
   multiplied by stores twice, or a per-store rate has been used as a total.
```

Those two checks catch the errors that recur in launch forecasts: a rate applied as a total, and a
total applied per store.

### 6.8 The cost of the decision under each scenario

The decision is how much to buy, evaluated against three scenarios over the eight-week cover window:

| Scenario | Weekly demand | Eight-week demand | Buy at P50 (12,744) | Buy at P73 (14,264) |
|---|---:|---:|---:|---:|
| Low (P10) | 1,194 | 9,552 | Excess 3,192 → €1,756 | Excess 4,712 → €2,592 |
| Central | 1,593 | 12,744 | Balanced → €0 | Excess 1,520 → €836 |
| High (P90) | 2,006 | 16,048 | Shortfall 3,304 → €4,824 | Shortfall 1,784 → €2,605 |
| **Equal-weight average cost** | | | **€2,193** | **€2,011** |

Buying to the 72.6th percentile beats buying to the median by about €180 on an equal-weighted average,
and by more if the high scenario is judged more likely than the low. The gain is small only because
the product is small; a launch ten times this size would show roughly €1,800, and one with a longer
shelf life and a cheaper markdown would show the opposite ranking. Two caveats: the equal weighting
is itself an assumption, and the normal approximation understates the upper tail, which matters
because the high scenario is where the shortfall cost lives.

## 7. Code, query, or calculation

The artifact is the hand analogy forecast, carried at the scale of the worked example. It is
deliberately arithmetic rather than library code: the skill is the factor chain and the sanity
check, and a spreadsheet that a launch team can audit line by line is more useful in a review than
a notebook that produces a number nobody can trace. Run it in a calculator or a spreadsheet.

```text
NPI ANALOGY FORECAST - Brand K sheets, retailer R, units per week
inputs: analogue stocking-store rate 2.40 (weeks 4-13); analogue ND 62% / WD 78%
        launch ND 55% of 1,850 = 1,018 stores; WD 71%
        awareness 34% -> 18%; price index 108; seasonal index 0.94

 1  base           2.40 x 1,018                      = 2,443 u/wk
 2  distribution   (71/55) / (78/62) = 1.026        -> factor 1.000
 3  awareness      (18/34)^0.6 = 0.53^0.6           -> factor 0.683
 4  price          1.08^-0.9                        -> factor 0.933
 5  season         analogue peak vs shoulder        -> factor 0.943
 6  combined       0.683 x 0.933 x 0.943 = 0.5991
 7  forecast       2,443 x 0.5991                    = 1,464 u/wk
 8  per store      1,464 / 1,018                     = 1.44 u/store/wk
 9  interval       +/- 25% from analogue dispersion = 1,098 to 1,830
10  sanity check   1.44 / 33.5 = 4.3% of category    -> plausible
```

**What the artifact assumes.** Six things, each of which should be written beside it in a real
submission:

1. The analogue's weeks 4 to 13 rate is the right target window, because the launch item will have
   comparable awareness and distribution by week 4 and will not have the analogue's year-two
   repeat base.
2. The analogue set is not survivorship-biased; at least one delisted launch is in it.
3. Awareness elasticity of 0.6 is right for this category. A range of 0.4 to 0.8 moves the forecast
   between 1,340 and 1,600 units per week, which is the largest single sensitivity in the chain.
4. The launch elasticity of −0.9 is below the mature-category −1.4. If the launch behaves like a
   mature item, the price factor falls to 0.89 and the forecast falls 4%.
5. The seasonal indices are computed on the same calendar definition. A 4-4-5 retail calendar and a
   Gregorian month will not agree, and the ratio of two index values from different calendars is
   meaningless.
6. The 25% interval is the dispersion of the analogue class, not a standard error on the mean.

**What the artifact does not do.** It does not include the attribute, diffusion, or distribution
families, which is why the reported forecast in section 6 is wider than this interval. It does not
model competitive response, which is a Topic 09 and Topic 10 concern. And it does not tell you how
much to buy: the newsvendor step in section 6.7 is a separate calculation on the combined band, not
on this single family.

**Two checks before the number leaves the room.** Recompute the base as `2.40 × 1,018` and confirm
it equals 2,443, because an error there propagates through every factor without changing their
signs. Then divide the final weekly total by the stocking store count and compare the result with
the category's per-store rate; a launch rate above the category rate is impossible and indicates
a scaling error rather than an optimistic assumption.

## 8. Data, standards, and regulatory context

### 8.1 The data map

| Input | System of record | Granularity | What fails without it |
|---|---|---|---|
| Analogues' per-store rates | Own shipment history, or syndicated scanner for competitors | Item-store-week | The analogy family cannot be built |
| Analogues' launch distribution | Account plans, historical | Item-account-week | The rate cannot be scaled correctly |
| Category volume and segment share | Syndicated scanner | Category-account-week | No denominator for share-based methods |
| Store universe and store format | Retailer store list, with ACV by store | Store | ND and WD cannot be distinguished |
| Conjoint utilities | Research agency study | Respondent level | No attribute family |
| Household panel | Syndicated panel | Household-purchase | No penetration, repeat, or buy rate |
| Planned media reach | Media agency plan | Campaign-week | No awareness input |
| Pre-order book | Order management system | Customer-item-date | No pre-order signal, and no cancellation analysis |
| Search volume | Search platform index, or a vendor feed | Term-week | No early interest proxy |
| Competitor launch calendar | Trade press, account intelligence | Event-date | The analogue set omits the competitive context |

Two rows decide whether the rest is usable. The **store list with ACV by store** is the only route from
numeric distribution (ND) to weighted distribution (WD); without it the distribution family collapses
to a store count. The **household panel** is the only pre-launch source for penetration, repeat, and
buy rate at category level; without it the diffusion family has no trial-and-repeat structure.

### 8.2 Item setup, identifiers, and the launch gate

A new item cannot be ordered, shipped, scanned, or reported without an identifier chain. This is the
hardest gate in the launch and the one most often discovered late.

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity at each packaging level | Consumer unit, case, and pallet are three identifiers | B |
| GS1 GDSN | Master data synchronisation between trading partners | Dimensions, net content, pack hierarchy, shelf life | B |
| GS1 GLN | Party and location identity | Store and distribution centre identity across feeds | B |
| GS1 Digital Link | Product identity on pack via QR | Consumer-facing identification and the route to product data | B |
| EDI 888 (item maintenance) | Item master data exchange | The retailer's item file, which must exist before the first order | B |
| EDI 879 (price and sales catalog) | Price and allowance communication | The price the retailer charges, which sets the price index input | B |
| EDI 852 (product activity data) | Retailer sales and inventory by item and location | The sell-out series without which no post-launch update is possible | B |
| EDI 867 (product transfer and resale) | Movement and resale reporting | Channel inventory, and therefore pipeline fill | B |

> **Note:** The item setup sequence is a forecast dependency, not an administrative detail. If the
> GTIN is not in the retailer's item file by the order cut-off, the launch date moves, and every
> downstream input in section 6.2 changes: the seasonal index, the competitive set, the media
> calendar, and the analogue's comparability. Treat the item-setup date as a forecast assumption.

### 8.3 Regulatory gates that shape the launch date

Regulatory approval is not a forecasting method, but it is an input to the launch date, and the
launch date is an input to the seasonal adjustment. Four categories of gate recur:

- **Labelling and composition.** Food information and net-quantity rules, ingredient and allergen
  declarations, and in the European Union Regulation (EU) No 1169/2011. A non-compliant label cannot
  be printed, and artwork lead time is measured in weeks.
- **Claims substantiation.** Health and nutrition claims are restricted in the European Union under
  Regulation (EC) No 1924/2006; advertising claims face substantiation requirements in the United
  States. An unsubstantiated claim carrying the positioning changes the attribute set, and therefore
  the conjoint design.
- **Chemical and safety compliance.** Registration and restriction regimes constrain formulation, and
  a late formulation change invalidates both the conjoint study and the analogue choice.
- **General product safety and packaging.** Regulation (EU) 2023/988 and packaging-waste rules
  constrain artwork, warnings, and pack format.

> **Note:** This section is educational and not legal advice. Confirm current requirements and
> applicable dates with regulatory affairs; these instruments are amended and their transitional
> periods vary by category and jurisdiction.

### 8.4 What is convention rather than requirement

Several things that feel like rules are habits, and an analyst who can tell the difference can
challenge them. The **13-week launch window** is a retail reporting convention, not a measurement
standard. The **80% interval** is a market-research presentation convention; the level should follow
the decision's cost asymmetry. **±20% at 80% confidence** is a syndicated house's claim about its own
method, not a bound on yours. The **four-week reforecast** is a planning-calendar habit; the
defensible revision point is when the sell-out series can separate a level shift from noise. **One
method per launch type** is an organisational simplification that section 5 argues against. And the
**"80% of new products fail"** figure summarises heterogeneous definitions of failure; it is a useful
prior about base rates and a poor input to any specific forecast.

Three inputs are routinely missing and each needs a stated substitute: competitor sell-out at item
level, resolved with syndicated scanner data or store audits and never with a proportional-share
assumption; analogues' failure cases, resolved through trade press and account records rather than by
using only what is on file; and true awareness at week 4, resolved as reach times a recall factor
calibrated on prior launches, never by equating planned reach with awareness.

## 9. Industry practice and case evidence

### 9.1 What the benchmarks actually say

The two most-quoted numbers in NPI work are the Terra Technology benchmark and the Kahn survey
figures. They measure different things and are routinely quoted as if they measured the same one.

| Source | Design | Finding | Grade |
|---|---|---|---|
| Terra Technology forecasting benchmark, 2011 | Nine multinational CPG firms, about $200bn sales, 7bn cases; vendor-conducted | Weekly error ~65% for new items in year one, ~46% for existing items; new items 13% of volume | C |
| Kahn, *Journal of Product Innovation Management* survey data, as summarised by Forecast Pro | Self-reported accuracy one year post-launch, by launch type | 72% cost reduction, 65% product improvement, 63% line extension, 54% new use, 47% new category, 40% new-to-the-world | A |
| Syndicated concept-test practice, as reported by Greenbook | Practitioner summary | ±20% at 80% confidence; ~80% of new CPG products fail | D |
| e2open case study, unnamed global CPG | Vendor case | New-product MAPE materially worse than mature; 11% consistent over-forecast; new items ~7% of portfolio and >17% of absolute error | D |
| Cheung, MIT thesis with a P&G cosmetics business | Company data | Aggregate error within ±30% by about week 6; 75% of SKUs within ±30%, 95% within ±60% | E |

Together these support one practical statement: **forecast accuracy is a function of novelty, and the
appropriate target for a genuinely new item is roughly half the accuracy of a mature one.** Holding a
new-to-the-world launch to the same standard as a line extension is not ambitious but mis-specified,
and it produces either sandbagged forecasts or a permanently failing metric.

> **Note:** The vendor benchmarks are Grade C and D for a reason. They are not audited, the samples
> are self-selected, and the definitions of forecast error differ. Use them for the order of
> magnitude and the direction, which are consistent across sources, not for the precise percentage.

### 9.2 The analogy evidence, and why practitioners disagree

Goodwin, Dyussekeneva, and Meeran (2012) is the most useful study here because it compares selection
strategies against each other rather than against nothing. Using Bass-model parameters fitted to 23
US consumer-electronics analogies to forecast 21 later products, they found:

| Strategy | Relative performance |
|---|---|
| Published mean p and q by product group | Worst of all methods tested, materially worse than a random single analogue |
| Single analogue chosen by similarity | No better than a random single analogue |
| Single analogue chosen at random | Poor, with error well above 50% |
| Average of five or six analogues | Best tested; more than six added little |

That similarity-based selection does not beat random selection is counterintuitive and worth sitting
with. It does not mean similarity is irrelevant. It means that where within-group parameter variation
is as large as between-group variation, selecting the most similar member does not select a more
representative parameter value; averaging does, because averaging attacks the variance rather than
trying to pick the winner.

Goodwin's later practitioner material enumerates the biases that drive selection: recency and
availability, superficial similarity on visible attributes, survivorship, anchoring on the first
candidate considered, base-rate neglect, and advocacy. Bolton (2003), in an experimental study, found
that an initially selected analogy persists in judgement even after counter-analogies, decomposition,
and accountability manipulations — the prior is "sticky". The defence is procedural rather than
motivational: a written protocol and a set of five or six, not a request for the analyst to be less
biased. The practitioner argument runs between those who hold that a well chosen single analogue with
expert adjustment beats a mechanical average, and those who hold that expert selection is exactly the
mechanism the evidence fails to support. The defensible middle is to score a set, average it, and then
let expert judgement adjust the *result* with a named factor and a stated magnitude, so the adjustment
is visible and reviewable rather than embedded in the choice of a single comparator.

### 9.3 The diffusion evidence

Bass (1969) remains the reference point: a hazard linear in cumulative adopters, fitted to annual
data on eleven consumer durables, with a colour-television projection that made the model famous. Its
later use vastly exceeded its original scope, and four extensions matter more for NPI work than the
original. **Trial-and-repeat decompositions** — Hahn, Park, Krishnamurthi, and Zoltners (1994) split
sales into trial from marketing, trial from word of mouth, and repeat, calibrated on 21
pharmaceutical launches; this is the correct structure for frequently purchased goods. **Time-varying
buying rates** — Fader, Hardie, and Huang (2004) showed the interpurchase process is nonstationary for
new products, so the sales curve is a byproduct of an evolving buying rate rather than a fixed
parametric form. **Asymmetric sales curves** — Lotfi, Jiang, and Jain (2020) describe adoption as
symmetric bell-shaped while aggregate sales including repeat are asymmetric with a slower decay, and
estimate repeat coefficients of about 0.24 for PCs and 0.48 for tablets in a working paper. **The
identifiability limit** — Fader and Hardie (2003), using twenty test markets with known trial and
repeat, showed aggregate models track total sales well while failing to recover the trial/repeat
split, with errors above 50% in a majority of cases.

That last item is the one to carry into a launch review. A chart showing a good fit to a weekly sales
curve is not evidence that the internal decomposition is right, and the internal decomposition is
what the supply plan needs.

### 9.4 The distribution evidence

Michis (2022), fitting a semiparametric sales-response model with distribution, found convex,
concave, S-shaped, and linear forms across categories, with new items showing rapid distribution
development and high early sales impact followed by decreasing returns as less important stores are
added. That is the empirical basis for the ramp and velocity assumptions in section 6.6, and a reason
to fit the curve on the subcategory's own history rather than importing a shape. The ATAR structure —
awareness, trial, availability, repeat — is a practitioner framework rather than a validated model,
and it is useful because each factor is separately arguable; its weakness is that it multiplies, so
four factors each 20% optimistic produce a forecast 107% too high.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Single cherry-picked analogue | The deck names one comparator, and it is a success | Availability and survivorship bias in selection | Forecast is a maximum over a search | Written reference class; score six; include failures |
| Analogue rate taken over the wrong window | Nobody can say which weeks the analogue rate covers | Week 1–3 is pipeline fill; year 1 includes built awareness | Rate is inflated or deflated by 20–50% | State the window; align to weeks 4–13 |
| Diffusion parameters presented as estimated | The deck says "the model estimates p, q, and m" for an unlaunched item | Software labels output as an estimate regardless of input | False precision; capacity sized on a curve fit | Label every parameter as borrowed; no fit before data |
| Conjoint share reported as market share | The share figure equals the volume forecast with no conversion shown | Preference is measured among aware, in-aisle shoppers | Overstates volume by the awareness and distribution factors | Show all three conversions with their factors |
| Intent top-box used directly as trial | Trial rate equals the top-two-box percentage | Intention overstates purchase, more so for novel items | Forecast too high, and the error grows with novelty | Calibrate conversion on own launches by launch type |
| Concept test validated on its own respondents | The validation sample is the survey sample | Self-generated validity inflates the correlation | The test appears to validate a method that does not work | Validate on a non-surveyed population |
| Distribution assumed at plan | The forecast uses the planned ND, not the confirmed store list | Line review trims lists late | Every downstream number is 10–20% high | Use confirmed stores; state the ramp |
| Pipeline fill counted as demand | Month 1 shipments spike and month 2 collapses | Sell-in is not sell-out | Double counting across quarters | Reconcile shipments to scans; report the gap |
| Point estimate with no interval | The number has four significant figures and no band | Planning systems demand a point | Decision made on a number the method cannot support | Report P10–P90 and the scenario definitions |
| Symmetric band on an asymmetric cost | The buy is set at the midpoint | Probability symmetry confused with cost symmetry | Systematic under-supply in trial-forming categories | Compute the critical ratio; buy to the quantile |
| No named trigger | Week 4 is a variance explanation meeting | Revision has no owner or threshold | Second production run placed before any learning | Agree triggers before launch; minute the revision |
| Rate scaled to stores twice | Weekly total implies more than the category sells | Per-store rate multiplied by stores and treated as a total | Order of magnitude error that survives review | Divide the total by stores and compare with the category rate |

**The borrowed-parameter failure is the one that most damages the analyst's credibility**, because it
survives scrutiny until someone asks a simple question. "Where did the p and q come from?" has no good
answer on an unlaunched product; the honest answer is "from these four products, with these
differences, and here is what happens if they are wrong", which is stronger than a fitted curve
because it can be argued with. **The double-scaling failure** is the one that most damages the launch:
it is not a modelling error and no amount of statistical sophistication catches it. Divide the weekly
total by the store count, compare with the category rate, and if the ratio exceeds one, stop.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The reference class is well defined and includes failures | Every analogy estimate is conditional on it | Forecast inherits survivorship bias, always optimistic | Ask how many delisted launches are in the set; if zero, widen the search |
| The analogue's weeks 4–13 rate is comparable | It is the scaling base for the whole family | Rate reflects a different awareness or distribution state | Compare analogue and launch on awareness and WD at week 4 |
| The launch will achieve planned distribution | Distribution family and the availability conversion both use it | Volume 10–20% high; every family biased the same way | Obtain the confirmed store list in writing; model the ramp |
| Awareness elasticity is between 0.4 and 0.8 | It converts an awareness ratio into a volume factor | Largest single sensitivity in the chain | Back-test on own launches where awareness was measured |
| Category price elasticity applies to a new entrant | Needed to adjust for a price index away from 100 | Over- or under-states the trial barrier | Estimate on prior launches at different price indices |
| Trial and repeat are transferable from the category | The diffusion family has no own-data alternative | Level wrong, and the repeat component is the larger error | Compare category repeat rates across analogues |
| No competitor launches in the window | Control logic and share arithmetic | Share is contested; volume lower than modelled | Track the competitor calendar; treat as a scenario |
| The launch date does not move | Seasonal adjustment depends on it | Seasonal factor inverted or nullified | Tie the date to the item-setup and regulatory gates |

**The assumption that matters most is the fourth**, because it is both the largest sensitivity and
the one most often set by assertion. In the worked example the difference between an exponent of
0.4 and 0.8 is about 260 units per week on a base of 1,464, which is larger than the difference
between three of the four families. If you can only back-test one assumption before launch, back-test
that one.

## 12. Exercises

### Exercise 1 — Name the missing input (recall, 15 minutes)

For each of decomposition, exponential smoothing, ARIMA, and regression, name the input it requires
and state why a launch item lacks it. **Deliverable:** a four-row table. **Acceptance:** each row
names the specific input, not "data"; each states the consequence for the forecast. **Time:** 15
minutes.

### Exercise 2 — Build an analogue set (application, 60 minutes)

Choose a launch you know or can research. Define a reference class with inclusion rules, identify at
least eight candidates, score them on the seven criteria in section 5.4, and select six. **Deliverable:**
the reference class definition, the scoring sheet, and the selected set with the two rejected
candidates named. **Acceptance:** at least one candidate is a failure; the exclusion rule is written
before the scoring; the window for the rate is stated in weeks. **Time:** 60 minutes.

### Exercise 3 — Scale an analogue (calculation, 30 minutes)

An analogue ran at 3.10 units per stocking store per week in weeks 4–13, with ND 48% and WD 60% in a
2,400-store chain. Your launch plans ND 62% and WD 66% in a 1,500-store chain, with awareness at 0.7
of the analogue's and a price index of 95 against an elasticity of −1.1. **Deliverable:** the scaled
forecast in units per week and per store, with each factor shown. **Acceptance:** distribution quality
is checked, not assumed; the per-store rate is compared with the category rate; the interval is
stated. **Time:** 30 minutes.

### Exercise 4 — Diagnose a diffusion fit (critique, 30 minutes)

A deck shows a Bass fit on nine quarterly points, all before the peak, reporting p = 0.02, q = 0.61,
m = 410,000, R² = 0.98. **Deliverable:** a critique naming at least three problems and the direction
of each bias. **Acceptance:** at least one problem concerns identifiability, one concerns the data
window, and one concerns what R² does and does not measure. **Time:** 30 minutes.

### Exercise 5 — Three conversions (calculation, 45 minutes)

A conjoint gives a share of preference of 9.4%. Category volume is 40,000 units per week, planned WD
is 0.58, and the calibrated early-window conversion factor is 0.55. **Deliverable:** preference share,
availability-adjusted share, volume, and per-store rate for 900 stocking stores. **Acceptance:** the
order of conversions is correct and justified; the conversion factor's provenance is stated; the
result is checked against the category rate. **Time:** 45 minutes.

### Exercise 6 — Intent to volume (calculation and critique, 40 minutes)

A concept test gives 34% top-two-box among 500 respondents. Households in the trade area are 900,000;
planned WD is 0.65. **Deliverable:** a volume estimate with a calibrated conversion, plus a critique of
using top-two-box directly. **Acceptance:** the conversion factor is sourced from own launches or
explicitly flagged as borrowed; the direction and rough size of the bias in the naive version are
stated. **Time:** 40 minutes.

### Exercise 7 — Write the forecast statement (design, 60 minutes)

Produce the one-page statement in section 5.11 for the launch in Exercise 2, with all six mandatory
elements. **Deliverable:** one page plus a supporting one-page calculation. **Acceptance:** the
interval definition is stated; the analogue set is named including failures; four assumptions carry
sensitivities; the trigger table has a series, a threshold, a date, and an action. **Time:** 60
minutes.

### Exercise 8 — Post-mortem (critique, 45 minutes)

A launch finished the year 38% below forecast. Reconstruct the five most likely causes and the
evidence you would need to distinguish them. **Deliverable:** a cause tree with evidence requirements.
**Acceptance:** at least one cause is a forecasting-method error and one is an execution error; each
cause names the data that would confirm or exclude it; the analysis separates a bad forecast from a
bad outcome. **Time:** 45 minutes.

## 13. Self-check answers

**Exercise 1.** Decomposition needs two full seasonal cycles; a launch has none. Exponential
smoothing needs an initial level and an error history to set the constant; a launch has neither.
ARIMA needs a stationary series with estimable autocorrelation; a launch has no series. Regression
needs a fitted relationship between drivers and volume; a launch has no fitted relationship, so any
coefficient is an assumption. In each case the consequence is that the method cannot be run at all,
not that it runs badly.

**Exercise 2.** A strong submission defines the class before listing candidates, for example
"national grocery launches of premium household-care items in the last six years in markets with
private-label share between 30% and 55%". It scores all eight on the seven criteria, selects the top
six by weighted score, and names the two rejected. It includes at least one delisted product,
typically found through trade press rather than internal records. A weak submission lists six
successful launches, gives no weights, and calls the choice "judgement".

**Exercise 3.** Base: `3.10 × (2,400 × 0.48) = 3,572` units per week for the analogue.
Distribution quality: analogue WD/ND = 60/48 = 1.25; launch 66/62 = 1.065; ratio 0.852, applied as a
factor. Awareness: `0.7^0.6 = 0.809`. Price: `0.95^-1.1 = 1.058`. Combined `0.852 × 0.809 × 1.058 =
0.729`. Forecast `3,572 × 0.729 = 2,604` units per week; stocking stores `1,500 × 0.62 = 930`; rate
`2,604 / 930 = 2.80` units per store per week. The common error omits the distribution-quality term
and reports 3,056.

**Exercise 4.** Three problems. First, nine pre-peak quarterly points is below the eight-to-ten
observation rule once the inflection is required, and the estimates are a function of where the
series stopped: `m` converges toward the last cumulative value, `q` is overstated, `p` understated.
Second, `R² = 0.98` on a cumulative S-curve is nearly automatic, because cumulative adoption is
monotone and smooth; it measures fit to a shape, not the validity of the parameters or of the
forecast. Third, the ceiling of 410,000 should be checked against the addressable population; if it
exceeds the number of households that could plausibly buy, the fit is absorbing a trend. A fourth,
if the deck claims these are estimated from *this* product, is that no data exist.

**Exercise 5.** Preference share 9.4%. Availability-adjusted `0.094 × 0.58 = 5.45%`. Volume
`40,000 × 0.0545 = 2,180` units per week before the early-window conversion; `2,180 × 0.55 = 1,199`
units per week. Per-store rate `1,199 / 900 = 1.33`. The order matters: applying the conversion factor
before availability gives the same product here but a different interpretation, and if the
conversion factor was calibrated on realised (already availability-adjusted) launches, applying it
after availability double-counts.

**Exercise 6.** Naive: `900,000 × 0.65 × 0.34 = 198,900` buyers, which is 22% of all households and
almost certainly impossible. With a conversion of 0.18 (a typical calibrated factor, and one that
must be sourced from own launches rather than assumed): `900,000 × 0.65 × 0.34 × 0.18 = 35,802`
buyers. At 1.4 packs each, 50,123 units in the window. The critique: top-two-box is a survey
statistic, intention overstates purchase, the overstatement is larger for novel and expensive items,
and the survey itself inflates the correlation it is used to validate.

**Exercise 7.** A strong statement fits on one page: decision and date; central 1,593 units per week
with a 1,194–2,006 band and the interval defined as P10–P90; the six-analogue set named with two
failures; four assumptions each with a sensitivity in units; a trigger table. A weak statement has a
point estimate, no named analogue, and a "we will monitor" sentence in place of a trigger.

**Exercise 8.** A cause tree with five branches: forecast method (analogue optimistic, conversion
factor wrong, distribution assumed), execution (distribution shortfall, out-of-stock, display not
executed), competitive response (a rival launched into the window), market (category decline), and
measurement (the 38% is computed on sell-in while the plan was sell-out). Evidence: confirmed versus
realised ND, OOS audits, the competitor calendar, category trend, and a shipment-to-scan
reconciliation. The last branch is not a joke: a material share of "missed forecasts" are
definitional mismatches between plan and actual.

## 14. Interview questions

**1. "Why can't you just use last year's model on a new product?"**
Strong: the model needs a history to estimate level, season, and autocorrelation; a launch has none,
so every coefficient becomes an assumption and the method's error bounds do not apply. Weak: "there
is no data", which names the symptom without saying what follows.

**2. "How would you choose an analogue?"**
Strong: define the reference class in writing, score eight to ten candidates on stated criteria,
average five or six including at least one failure, and adjust the result with named factors rather
than picking one comparator. Weak: "find the most similar product", which the evidence shows performs
no better than random.

**3. "A colleague says the Bass model estimates a ceiling of 2 million for this launch. What do you
ask?"**
Strong: ask what data the estimation used, because on an unlaunched product there are none; then ask
what window, because pre-peak fits drive the ceiling toward the last observed cumulative value.
Weak: "how confident are you?", which accepts the premise.

**4. "Why is the intention-behaviour gap larger for really new products?"**
Strong: there is no reference price, no learned usage occasion, and no shelf comparison, so the
stated intention is formed without the constraints that will govern the actual purchase; the decay
is documented as larger for really new than for incremental products. Weak: "people exaggerate",
which is a moral explanation rather than a behavioural one.

**5. "What is self-generated validity and why does it matter for validating a concept test?"**
Strong: asking the intention question changes the respondent, inflating the intention-behaviour
correlation by roughly 58% relative to comparable non-surveyed consumers, so a concept test cannot
validate itself on its own respondents. Weak: "people remember being asked", which is a different
phenomenon.

**6. "Why forecast from distribution rather than from preference?"**
Strong: in the first quarter, availability bounds sales mechanically, and the distribution inputs are
known with more precision before launch than any preference measure; the velocity curve is nonlinear
so the marginal store adds less than the average. Weak: "because distribution drives sales", which
does not explain why it beats preference in the first quarter specifically.

**7. "Numeric distribution is 55% and weighted distribution is 71%. Which do you use?"**
Strong: both, for different purposes: ND for the store count that scales a per-store rate, WD for the
availability conversion, and the ratio as a check on whether the accepting stores are the large ones.
Weak: "weighted, because it is more accurate", which loses the store count.

**8. "Why report a range when the supply plan needs a number?"**
Strong: the methods cannot support a point, and the range is what lets the buyer apply a cost
asymmetry; the plan takes a quantile of the range, not its midpoint. Weak: "because forecasts are
uncertain", which is true of every forecast and not specific to this one.

**9. "How do you decide whether to buy to the median or above it?"**
Strong: compute the critical ratio from the cost of a unit of over-supply and the cost of a unit of
under-supply, including forgone repeat purchases when trial is lost, then take that quantile; the
answer differs by shelf life and by whether the item is a new brand or a line extension. Weak: "buy
a bit more to be safe", which is a preference, not a decision rule.

**10. "How would you tell a bad forecast from a bad outcome?"**
Strong: separate calibration from realisation: a well-calibrated 40% band that lands in the tail is a
bad outcome, while a point estimate that happened to be close is a bad forecast with a lucky outcome;
the test is the distribution across many launches, not one. Weak: "if we missed, the forecast was
bad", which conflates the two permanently.

## 15. Key terms

- **New product introduction (NPI)** — the commercial launch of an item with no sales history.
- **Reference class** — the defined population of past launches from which analogues are drawn.
- **Analogue** — a previously launched product used as the source of a rate or a shape.
- **Per-store rate** — units per stocking store per week; the common currency of NPI methods.
- **Launch curve** — the trajectory of weekly sales from launch to steady state.
- **Sell-in / sell-out** — shipments to the retailer versus scans at the till.
- **Pipeline fill** — units sold in but not yet sold out.
- **Numeric distribution (ND)** — share of stores stocking the item.
- **Weighted distribution (WD)** — share of category volume covered by stocking stores.
- **All-commodity volume (ACV)** — total store sales, used to weight store importance.
- **Velocity curve** — the nonlinear relationship between distribution and sales.
- **ATAR** — awareness, trial, availability, repeat; a multiplicative practitioner framework.
- **Trial / repeat / repeat rate** — first purchase, subsequent purchase, and the share of triers
  who buy again in a window.
- **Bass model** — a diffusion model with hazard linear in cumulative adopters.
- **p / q / m** — coefficient of innovation, coefficient of imitation, and ceiling.
- **Ill-conditioning** — the data are too sparse for the model's complexity, so estimates are a
  function of the window.
- **Conjoint analysis** — a survey method recovering attribute utilities from forced trade-offs.
- **Share of preference** — the model's allocation of choices among a competitive set.
- **IIA** — independence of irrelevant alternatives; the logit share rule's proportional draw.
- **Alpha rule** — utilities raised to a calibrated power to recover known shares.
- **Purchase intention** — a stated likelihood of buying on a rating scale.
- **Top-two-box** — the share choosing the top two scale points.
- **Intention-behaviour gap** — the systematic shortfall between stated intent and purchase.
- **Self-generated validity** — inflation of the intention-behaviour correlation caused by asking.
- **Leading indicator** — a measurable quantity that moves before the outcome.
- **Trigger** — a pre-agreed indicator value at which the forecast is revised.
- **P10 / P50 / P90** — percentiles of a forecast distribution.
- **MAPE** — mean absolute percentage error.

## 16. Sources

**Peer-reviewed articles (A)**

- Bass, F. M. (1969). "A New Product Growth for Model Consumer Durables." *Management Science*, 15(5), 215–227. DOI: 10.1287/mnsc.15.5.215.
- Van den Bulte, C., and Lilien, G. L. (1997). "Bias and Systematic Change in the Parameter Estimates of Macro-Level Diffusion Models." *Marketing Science*, 16(4), 338–353. DOI: 10.1287/mksc.16.4.338.
- Goodwin, P., Dyussekeneva, K., and Meeran, S. (2012). "The Use of Analogies in Forecasting the Annual Sales of New Electronics Products." *IMA Journal of Management Mathematics*. DOI: 10.1093/imaman/dpr025.
- Bolton, L. E. (2003). "Stickier Priors: The Effects of Nonanalytic versus Analytic Thinking in New Product Forecasting." *Journal of Marketing Research*, 40(1), 65–78. DOI: 10.1509/jmkr.40.1.65.19129.
- Chandon, P., Morwitz, V. G., and Reinartz, W. J. (2005). "Do Intentions Really Predict Behavior? Self-Generated Validity Effects in Survey Research." *Journal of Marketing*.
- Morwitz, V. G. (2014). "Consumers' Purchase Intentions and their Behavior." *Foundations and Trends in Marketing*. DOI: 10.1561/1700000036.
- Bemmaor, A. C. (1995). "Predicting Behavior from Intention-to-Buy Measures: The Parametric Case." *Journal of Marketing Research*, 32(2). DOI: 10.1177/002224379503200205.
- Jamieson, L. F., and Bass, F. M. (1989). "Adjusting Stated Intention Measures to Predict Trial Purchase of New Products." *Journal of Marketing Research*, 26(3). DOI: 10.1177/002224378902600307.
- Hsiao, C., Sun, B., and Morwitz, V. G. (2004). "The Role of Stated Intentions in New Product Purchase Forecasting." *Advances in Econometrics*. DOI: 10.1016/s0731-9053(02)16002-6.
- Alexander, D. L., Lynch, J. G., and Wang, Q. (2008). "As Time Goes By: Do Cold Feet Follow Warm Intentions for Really New versus Incrementally New Products?" *Journal of Marketing Research*.
- Hahn, M., Park, S., Krishnamurthi, L., and Zoltners, A. A. (1994). "Analysis of New Product Diffusion Using a Four-Segment Trial-Repeat Model." *Marketing Science*, 13(3), 224–239. DOI: 10.1287/mksc.13.3.224.
- Fader, P. S., Hardie, B. G. S., and Huang, C.-Y. (2004). "A Dynamic Changepoint Model for New Product Sales Forecasting." *Marketing Science*, 23(1). DOI: 10.1287/mksc.1030.0046.
- Meeran, S., Goodwin, P., Dyussekeneva, K., and Yee, C. (2017). "When Do Changes in Consumer Preferences Make Forecasts from Choice-Based Conjoint Models Unreliable?" *European Journal of Operational Research*, 258(2), 512–524.
- Michis, A. A. (2022). "Retail Distribution Evaluation in Brand-Level Sales Response Models." *Journal of Marketing Analytics*.
- Kalwani, M. U., and Silk, A. J. (1980). "Structure of Repeat Buying for New Packaged Goods." *Journal of Marketing Research*, 17(3). DOI: 10.1177/002224378001700304.

**Official and standards (B)**

- GS1 General Specifications, including GTIN, GLN, GDSN, and Digital Link.
- ASC X12 EDI transaction sets 852, 867, 879, and 888.
- Regulation (EU) No 1169/2011 on the provision of food information to consumers.
- Regulation (EC) No 1924/2006 on nutrition and health claims made on foods.
- Regulation (EU) 2023/988 on general product safety.

**Industry benchmarks and company evidence (C)**

- Terra Technology, forecasting benchmark study across nine CPG companies, 2011, as reported in *Supply Chain Management Review*. Vendor-conducted; figures not independently auditable.
- Kahn, K. B., new-product forecasting accuracy survey data in *Journal of Product Innovation Management*, as summarised in Forecast Pro's practitioner note. Peer-reviewed survey, but self-reported accuracy.

**Vendor and practitioner material (D)**

- Greenbook, "When Forecasts Go Wrong", on syndicated concept-test accuracy and failure rates.
- e2open, new-product forecasting with AI, CPG case study.
- phData, CPG new-product launch forecasting case study.
- IQVIA, "Using Historical Analogues to Forecast New Product Launches" (2021).
- Sawtooth Software, CBC market simulator documentation, on first-choice, share-of-preference, and randomised first-choice rules.
- Conjointly, preference share simulator guide, on the limits of preference shares as market share.
- Marketing Study Guide, availability in the ATAR forecasting model.

**Theses and working papers (E)**

- Cheung, B., *A Short-Range Forecasting and Inventory Strategy for New Product Launches*, MIT master's thesis, 2005, with a company dataset.
- Lotfi, S., Jiang, Z., and Jain, D. C. (2020). "Modeling Product Diffusion and Repeat Purchases: A Fractional Calculus-Based Approach." CEIBS working paper 040/2020.
- Goodwin, P. (2016). "New Product Sales Forecasting: The Challenges of Using Mathematical Models." Lancaster Centre for Forecasting, practitioner presentation.

**Note on citation practice.** Every figure quoted from a source is quoted as that source reports it;
where a number is a vendor benchmark or a modelled example rather than an audited sample, the entry
says so. No DOI or year has been supplied where the search record did not carry one.
## 17. Links

- This module's index: [`_index.md`](_index.md)
- The decomposition this topic builds on: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Demand sensing signals, the short-horizon counterpart: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Promotion mechanics, for the launch support calendar: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Baseline and incrementality, for the counterfactual: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Cannibalization and halo, for launch effects on the existing range: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Price and display effects: [`06_price-display-effects.md`](06_price-display-effects.md)
- Causal machine learning and uplift: [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md)
- Part II map: [`../_index.md`](../_index.md)
- Forecastability and why some series cannot be forecast: [`../07_time_series_forecasting/01_forecastability-benchmarks.md`](../07_time_series_forecasting/01_forecastability-benchmarks.md)
- Forecast combination, the machinery this topic adapts: [`../07_time_series_forecasting/07_forecast-combination.md`](../07_time_series_forecasting/07_forecast-combination.md)
- Probabilistic forecasts, for the interval definition: [`../08_hierarchical_intermittent_probabilistic_forecasting/07_probabilistic-forecasts.md`](../08_hierarchical_intermittent_probabilistic_forecasting/07_probabilistic-forecasts.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)

Later siblings in this module:

- [`09_distribution-assortment.md`](09_distribution-assortment.md) — assortment and distribution effects
- [`10_post-launch-learning.md`](10_post-launch-learning.md) — post-launch learning and ROI
