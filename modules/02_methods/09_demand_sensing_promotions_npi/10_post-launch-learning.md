# Post-Launch Learning and ROI — updating the forecast with early sales and settling the ROI question

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topic 10; Module 09 topics 04–09
> Version 2026.5

---

## 1. Why this topic exists

Two questions arrive in the same meeting, usually within ten minutes, and both are answered with
arithmetic that does not state what it is conditional on.

The first is: **the launch is four weeks old and the numbers are in. What do we now believe?**
Sales are running 20% above plan. Someone else says the display ended and the shelf is empty in
one store in twenty. A third says the plan was sandbagged anyway. The meeting produces a number
that is a negotiation, and it is loaded into the supply plan that afternoon.

The second is: **did it make money?** A different spreadsheet multiplies the lift by the unit
margin, ignores the margin given away on units that would have sold anyway, counts the display fee
but not the deductions that arrive in three months, and reports a return on investment (ROI) of
4.0. The promotion was approved on that number. The next one will be too.

Both failures have the same shape: a number is produced without saying what it was conditional
on. The revised forecast does not say what it knew; the ROI does not say which lift it used.
Neither artefact is dated, so neither can be evaluated later, which means the organisation cannot
learn from either.

**Three consequences that reach the balance sheet.**

1. **The re-forecast damages the launch it is trying to save.** A launch opens 20% above plan on
   two weeks of display data, is re-forecast upward, the factory commits to a second tool and a
   larger order, the display ends, the rate halves, and the excess is written down next quarter.
   The revision was not wrong because the data were wrong. It was wrong because the data were not
   adjusted for the thing that produced them.
2. **A profitable promotion is killed, or an unprofitable one is repeated.** Gross-lift ROI says
   four euros returned per euro spent; adjusted for forward buying and cannibalisation it destroyed
   money. The company repeats it, or cancels a genuinely defensive event and loses share to a
   competitor who did not.
3. **Nobody learns, because nothing was recorded.** The post-mortem happens eight months later,
   attended by whoever is available, and produces a slide saying "launch execution could have been
   stronger." The next launch uses the same analogue, the same optimism, the same ramp curve. The
   prior is unchanged.

**Why this is the module's closing topic rather than an appendix.** Topics 04 to 09 are
measurement machinery — the counterfactual, the adjustments, price and display, heterogeneity, the
prior, and availability — and all of it produces estimates. This topic is where estimates meet an
actual number and are either revised or defended.

**What makes it hard rather than merely neglected.** The update problem has two opposite failure
modes — anchoring on the plan, and over-reacting to launch noise — and the same behaviour can look
like either one depending on what happens next. Early sales are censored by availability, inflated
by novelty and display, and contaminated by pipeline fill, so the raw number is not an observation
of the thing you want to estimate. The sample is smaller than it looks, and ROI has at least three
defensible definitions whose difference can flip the sign.
- A launch is judged over a lifecycle and a promotion over a season. Using one horizon's rule for
  the other produces wrong accept/reject decisions in both directions.
- The learning loop is an information-systems problem. The model is the easy part; the frozen
  prior, the as-of dataset, the reason code, and the register owner are the hard part.

This topic assumes the apparatus from topics 04 to 09 and the evaluation discipline from Module 07
topic 10. It does not re-teach forecast value added (FVA). It applies the same logic — a claim is
meaningful only against a stated counterfactual — to a different object: the information set
rather than the human intervention.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** three update policies — hold the plan, replace it with the observed rate, and
   re-forecast the remainder with a shrunk estimate — and **state** what each assumes.
2. **Adjust** an early observation for censoring, display, novelty, and pipeline fill before use,
   and **quantify** how much of a headline variance the adjustment removes.
3. **Compute** a revised total from elapsed actuals plus a re-estimated rate over the remaining
   periods, with an explicit shrinkage weight, and **compare** it with the original plan.
4. **State** the uncertainty on a revision using an effective sample size that accounts for
   clustering, and **explain** why a nominal count of store-weeks overstates precision.
5. **Select** leading indicators for a revision trigger, **name** the lag on each, and **defend**
   the threshold.
6. **Evaluate** a forecast against the information available at issue, and **separate** a forecast
   that was wrong from one that was wrong because the world changed.
7. **Compute** ROI against gross lift, correctly costed gross lift, and adjusted lift, with a full
   cost base, and **explain** why the three can differ in sign.
8. **Choose** a payback horizon for a launch or a promotion, and **state** what the wrong one costs.
9. **Design** the record that lets an organisation learn from one launch and apply it to the next.

The exercises in section 12 and the self-check answers in section 13 test these outcomes.

## 3. The core mental model

Hold one picture: a loop with six stations and one gate. The gate is where information is frozen
for evaluation. Almost every organisation has the loop and has lost the gate.

```text
   THE POST-LAUNCH UPDATE LOOP

   (1) PRIOR                 (2) EARLY SIGNAL            (3) REVISION
   pre-launch forecast       weeks 1-8 of sales          re-estimate the rate
   analogue set, attributes  distribution, availability  shrink toward the prior
   stated assumptions        price, display              apply to remaining weeks
        |                         |                           |
        v                         v                           v
   +----------+             +--------------+            +--------------+
   |   P0     |------------>|  observed r  |----------->| r' = (1-w)P0 |
   |  3.333   |             |  3.25 raw    |            |      + w*r   |
   +----------+             +--------------+            +--------------+
        |                         |                           |
        |                   [ADJUST FIRST]                     |
        |                   de-censor, strip display,          |
        |                   remove pipeline fill               |
        |                                                     v
        |                                            (4) DECIDE AND RECORD
        |                                             order, allocate, commit
        |                                                     |
        |                                               [FREEZE HERE]
        |                                        prior + as-of data + decision
        |                                                     |
        v                                                     v
   +------------------------------------------------------------------+
   | (5) EVALUATE at the horizon against the FROZEN information set    |
   |     attribute error to a mechanism, not to a person               |
   +------------------------------------------------------------------+
                                  |
                                  v
                      (6) PORTFOLIO LEARNING
                      launch register -> next prior
```

Read it in one pass: a prior enters, an observation enters, the observation is cleaned before it
is allowed to speak, the two are combined with a weight, the combination is applied only to
periods that have not happened, and the whole thing is recorded so it can be scored later. The
freeze gate sits where it does because everything to its left is an input to a decision and
everything to its right is evidence about that decision.

### 3.1 Three update policies, and what each believes

Every revision is one of three policies in different clothes. Naming them makes the argument
explicit.

| Policy | Rule | What it assumes | What it produces |
|---|---|---|---|
| **Hold the plan** | Revised total = plan; actuals absorbed as variance | Early data are noise, or the plan is a commitment | Stability, and a growing unmanaged variance |
| **Replace** | Revised total = observed rate × full period | The observed rate is an unbiased steady-state estimate | Maximum responsiveness, maximum variance |
| **Re-forecast the remainder** | Elapsed actuals + re-estimated rate × remaining periods | Elapsed periods are known; future ones are estimated | Error confined to the periods not yet observed |

The third is the only internally coherent policy, and the one this topic teaches. The others are
not always wrong — hold-the-plan is correct when the plan is contractual — but they are usually
chosen by default rather than argued. The re-forecast rule has a further property worth memorising:
**it cannot revise history.** A launch that sold 9,600 units in four weeks sold 9,600 units. A
revised annual total implying a different first four weeks is a fiction, and it is the fiction that
produces the year-end reconciliation surprise.

### 3.2 The shrinkage weight

The re-estimated rate is a weighted average of the prior rate `P0` and the adjusted observed rate
`r`: `r' = (1 - w) x P0 + w x r`, with `0 <= w <= 1`.

`w` is not a constant a library returns. It is a judgement about relative credibility, and the
professional obligation is to state it, justify it, and show the answer at two other values.
Three things move it: the precision of the observation (stores, weeks, dispersion); the stability
of the phenomenon (a ramping launch supports a lower `w` than a mature category); and the cost of
being wrong in each direction.

A defensible default for a consumer launch on four weeks is `w` between 0.2 and 0.4; for a mature
promotion on eight weeks, 0.7 or higher. A scheme derived purely from sampling error will usually
be far too confident: the dominant uncertainty in a launch is model uncertainty — the analogue, the
distribution curve, the repeat assumption — not the standard error of a store mean.

### 3.3 Two errors pointing in opposite directions

The revision literature is often summarised as "people anchor and adjust insufficiently." That
is true and is half the story. A revision process can commit either error.

**Insufficient adjustment.** Lim and O'Connor (1995) documented anchoring in a controlled task;
Eroglu and Croxton (2010) tied it to individual differences among planners; Meub and Proeger (2016)
reproduced it experimentally. The signature is a revision that lands a small fraction of the way
toward the data, and a plan still being defended in week twelve.

**Excessive adjustment.** The opposite error treats the most recent weeks as the new truth.
Mathews and Diamantopoulos (1990) found that managers disproportionately revise forecasts that were
initially low, producing upward drift unrelated to the data. Van den Broeke and colleagues (2019),
across eight companies and more than 300,000 adjustments, found that adjustments grow larger and
more positive as the horizon shortens — closest to the sales point, where the noisiest information
lives.

Both are usually present at once on different series, which is why "our forecasters anchor" and
"our forecasters over-react" can both be true in one meeting. The remedies differ. Anchoring is
treated by pre-committing to a revision rule and scoring the revision against the prior.
Over-reaction is treated by adjusting the observation before use and shrinking toward the prior. A
process that does only one has chosen which error it prefers, usually without saying so.

### 3.4 The information set is the unit of evaluation

A forecast is conditioned on what was known when it was made. Evaluating it without reconstructing
that condition confuses two failures:

- **A forecast that was wrong.** The information at issue was sufficient to do better, and the
  method or the judgement failed to use it.
- **A forecast that was wrong for a good reason.** Information arrived afterwards that could not
  have been known, and the forecast was correct given what existed.

These demand different responses. The first is a method, data, or competence problem. The second
is a portfolio problem: the organisation may be taking the right risks and losing some, which is
not a defect. An organisation that cannot tell them apart punishes the second and fails to fix
the first, consistently, because the loudest misses are usually caused by events.

The operational rule: **freeze the information set at the moment of decision.** Not the forecast
alone — the prior, the data vintage, the assumptions, the analogue used, the competitive and
distribution facts on the table, and the decision taken. The evaluation then asks two questions:
was the forecast good given what was known, and was the revision good given what was known when it
was made.

### 3.5 Information flows forward, blame flows backward

Blame flows backward, and it flows to whichever role is least able to defend itself: a demand
planner is blamed for a launch that was late because manufacturing slipped, which slipped because
a tool was late, which was late because capital approval took a quarter. The reason codes in
section 5 exist to force the backward flow along the forward path — each names a mechanism, and
each mechanism has an owner who is not the person being scored. That is the difference between a
post-mortem and a tribunal.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Post-launch revision | A dated change to a launch forecast after the first observations | Used for any plan change | Units per period, with a revision date |
| Re-forecast | Rebuilding the unelapsed portion from a re-estimated rate | Used to mean "restating the whole year" | Units per period, elapsed and remaining separated |
| Shrinkage weight (`w`) | Weight on new information when combining an observation with a prior | Treated as an estimated parameter | Proportion, 0 to 1 |
| Prior | The pre-launch forecast and its supporting assumptions | Confused with the plan or the target | Units per period, plus assumptions |
| Observed rate | Sales per store per week over the observed window | Treated as demand without adjustment | Units per store-week |
| Store-week | One store observed for one week | Counted as an independent observation | Count, with clustering stated |
| Effective sample size | The independent-observation equivalent of a clustered sample | Assumed equal to the raw count | Count, dimensionless |
| Design effect | Variance under clustering divided by variance under independence | Ignored in precision calculations | Ratio, typically 1 to 20 |
| Pipeline fill | Stock shipped to fill the channel before consumer sales | Counted as demand | Units, or weeks of cover |
| Sell-in / sell-through | Manufacturer shipments to the trade / trade sales to consumers | Used interchangeably | Units per period |
| ACV and weighted distribution | Store weight by total store sales, and the share of it stocking the item | Confused with store count | Proportion of category volume |
| Velocity | Rate of sale per stocked store per week | Computed across out-of-stock stores | Units per store-week |
| Repeat rate | Share of first buyers who buy again in a stated window | Measured before a repurchase cycle elapses | Proportion per cohort |
| Censoring | Truncation of observed demand by unavailability | Called weak demand | Units suppressed, or a rate |
| Launch spike | Early peak from novelty, display, and trial | Extrapolated as a run rate | Units per store-week |
| Stabilisation point | Week at which the rate stops trending and becomes forecastable | Declared by calendar rather than by test | Week number |
| Analogue | A past launch used to construct the prior | Chosen for being memorable rather than comparable | Product identifier, plus similarity |
| Anchoring | Insufficient movement from an initial number when evidence arrives | Used loosely for any conservatism | Revision ratio |
| As-of vintage | A dataset as it stood at a stated timestamp | Replaced by current values | Timestamped snapshot |
| Information set | Everything knowable when a forecast was issued | Assumed to equal today's data | Documented list with a date |
| Frozen prior | The pre-launch forecast stored immutably at issue | Overwritten by the planning system | Units per period, with a version |
| Root-cause code | A controlled label for the mechanism behind a variance | Written in free text, never aggregated | Controlled vocabulary |
| Gross lift | Incremental units against a baseline, before adjustment | Called incremental profit | Units |
| Adjusted lift | Gross lift corrected for cannibalisation, halo, and pull-forward | Confused with gross lift in ROI reporting | Units, or contribution currency |
| Baseline | Counterfactual sales without the intervention | Taken from the plan, which already contains the expected effect | Units per period |
| Trade spend | Total manufacturer funding, including deductions and fees | Equated with the price reduction | Currency |
| Contribution margin | Price minus variable cost, per unit | Confused with gross margin percentage | Currency per unit |
| ROI | Net incremental profit divided by total intervention cost | Computed on revenue rather than contribution | Ratio, dimensionless |
| Payback period | Time until cumulative contribution equals the investment | Computed without discounting, and called a return | Months |
| Lifecycle horizon | Period over which a launch investment is judged | Set to the budget year by default | Months or years |
| Launch register | Portfolio record of priors, revisions, and outcomes | A folder of presentations | Database object |
| Attribution error | Explaining an outcome by a person's competence when a mechanism explains it | The default mode of most post-mortems | Categorical |

Three carry most of the damage. **Observed rate versus demand:** the rate computed from scanner
data is sales conditional on being on shelf, so where the shelf was empty demand is unobserved,
not zero. **Gross versus adjusted lift:** the gap is not a rounding error; in section 6 it is
larger than the entire gross increment and it changes the sign. **Payback versus return:** payback
is a liquidity statement, return is a profitability statement, and a payback-only rule rejects
long-payback high-return projects while accepting short-payback low-return ones.

## 5. The mechanics, step by step

Eight phases. A to F are the revision; G and H are settlement and learning. Each step names its
input, its output, and the branch point where a competent analyst goes wrong.

### Phase A — Before launch: freeze what you will later wish you had

**Step 1. Store the prior immutably, with its assumptions.** Input: the pre-launch forecast from
topic 08. Output: weekly forecast per store group, analogue set, assumed distribution curve,
assumed repeat rate, and issue date. Branch: **is the prior stored as a number or as an
argument?** A number can be compared with an outcome; an argument can be diagnosed. Most systems
keep the number and discard the argument, which is why post-mortems degenerate into "the forecast
was wrong".

**Step 2. Write the plan and the failure threshold separately.** Output: the plan (what the
company will do), the target (what it hopes for), and the threshold at which the launch is
re-scoped or stopped. Branch: **was the threshold written before or after the first read?** A
threshold set after the result is a rationalisation. Its purpose is to make the stopping decision
cheap by making it pre-agreed.

**Step 3. Define the observed metric and its baseline.** Output: sell-in or sell-through, weekly
bucket, store universe, unit of measure, and the treatment of out-of-stock store-days. Branch:
**is the store universe fixed at launch?** Stores added in week six were not in the prior, and
mixing them in produces a step change that looks like demand and is distribution growth (topic
09).

### Phase B — Weeks 0 to 8: read the signal without believing it

**Step 4. Choose leading indicators and name their lags.**

| Indicator | What it tells you | Typical lag | Distortion |
|---|---|---|---|
| Sell-in | Channel fill and reorder behaviour | 0–1 week | Counts pipeline as demand |
| Scanner sell-through | Consumer offtake | 1–7 days after the sale | Censored by availability; inflated by display |
| Store count and ACV | Distribution build | 1–2 weeks, often manual | Announced before executed |
| On-shelf availability | Whether the rate is measurable at all | Same day to 1 week | Recorded stock is not physical stock |
| Repeat rate | Whether trial converts | One repurchase cycle, 3–12 weeks | Measured too early, on a biased cohort |
| Trade orders net of cancellations | Trade confidence | 2–4 weeks | Over-orders during allocation |
| Competitor response | Whether the environment changed | 4–12 weeks | Invisible until it is priced |

The rule: **do not revise on an indicator whose lag exceeds the window you have observed.** A
repeat rate measured at week three, on a ten-week repurchase cycle, measures heavy users, not the
market.

**Step 5. Fix the reading window and hold it.** Output: a stated window, typically weeks 3 to 8
for a consumer launch, excluding the first two weeks of display and pipeline. Branch: **was the
window chosen to include or exclude the good weeks?** A window that moves with the result is a
filter, not a measurement. Fix it in the launch plan, beside the threshold.

**Step 6. Adjust the observation before using it.** Four adjustments, in order, each with a stated
magnitude: remove pipeline fill if the metric is sell-in; de-censor by dividing the rate by the
in-stock share, `r_adj = r_obs / (1 - oos_share)`, where stock-outs are plausibly random with
respect to velocity; strip weeks carrying display, feature, or sampling, or model and subtract
their effect; and net out novelty using a decay factor from the analogue set. Branch: **are the
adjustments applied before or after the shrinkage?** Before. Shrinking a contaminated observation
toward a clean prior produces a number with no interpretation at all.

### Phase C — Re-estimate, with a stated weight

**Step 7. Compute the adjusted rate and its uncertainty.** Branch: **how many independent
observations do you actually have?** Store-weeks are correlated within store over time and across
stores within week. A design effect of 5 to 15 is normal for weekly store data; ignoring it
understates the standard error by a factor of two to four, which is exactly the factor separating
"significant" from "not".

**Step 8. Choose and state the weight.** Output: `w`, with a one-sentence justification and a
two-value sensitivity. Branch: **would you have chosen the same `w` if the observation had come in
below plan?** If not, it is not a weight; it is a preference.

**Step 9. Apply it to the remaining periods.** Output: `r' = (1-w) x P0 + w x r`, then `revised =
actual to date + r' x remaining periods`. Branch: **are the remaining periods uniform?** They are
not: seasonality, a second support wave, and competitor entry all change them. Apply the rate to a
remaining-period profile where one exists; otherwise say that the flat application is an
approximation and state its likely direction of bias.

### Phase D — State the uncertainty in a usable form

**Step 10. Convert the standard error into an interval on the total being decided, and say what
would change the answer.** Output: an interval on units, not on the rate, because decision-makers
act on totals, plus one sentence naming the observation that would move the revision materially.
Branch: **is the interval a probability or a scenario range?** A scenario range is a communication
device with no coverage guarantee. If you say "80% interval", it must be computed as one.

### Phase E — Decide, and record the decision

**Step 11. Separate the forecast from the plan change.** Output: the revised forecast, and
separately the decision it implies. Branch: **did the forecast move because the plan moved?** A
planner who lowers the forecast because supply cannot cover demand has changed the plan.
Recording it as a forecast change corrupts both the evaluation and the learning.

**Step 12. Freeze the information set.** Output: a snapshot of the frozen prior, the as-of data
used, the adjusted rate, `w`, the revised total, the interval, the decision, and the date. Branch:
**before or after the decision meeting?** Before. A snapshot taken afterwards records the
consensus, not the evidence.

### Phase F — At the horizon: evaluate correctly

**Step 13. Score the prior against the information at issue.** Branch: **is it scored against
today's restated history?** Restatements, redefined store universes, and re-based panels change
the denominator. Score against the as-of data that existed, and report the restatement separately.

**Step 14. Score the revision against the information at revision.** Branch: **whole horizon or
remainder?** Remainder only. The elapsed weeks were not forecast by the revision; they were
observed.

**Step 15. Attribute the residual to a mechanism.** Output: a root-cause code per material variance
from a small controlled vocabulary — distribution shortfall, competitor entry, price or promotion
execution, supply-driven unavailability, demand-model error, forecast-process error, and genuine
market surprise. Branch: **does every code have an owner who is not the forecaster?** If all codes
point at one role, the taxonomy is a blame device and will be gamed within two cycles.

### Phase G — Settle the ROI

**Step 16. Fix the lift definition before computing any ratio.** Three definitions coexist, and
they answer different questions: lift at full margin (what the sales report shows; too high),
gross lift correctly costed (promoted units at promoted contribution less baseline units at full
contribution), and adjusted lift (gross, less pull-forward displacement, less cannibalisation at
the donor margin, plus halo at the halo margin). Only the third answers whether the business
earned; section 6.4 shows all three on one event.

**Step 17. Build the full cost base.** The subsidy on **all** promoted units, not only the
incremental ones; fixed fees — feature, display, slotting, listing, sampling; incremental
logistics; returns, markdown, and disposal of unsold promotional stock; the administrative cost of
deductions, chargebacks, and claims, including those arriving a quarter later; and, where
material, the working-capital cost of building stock ahead of the event.

**Step 18. Compute the ratio against the stated denominator and state the horizon.** Branch: **is
the denominator total cost or incremental cost?** ROI on total cost answers "was the event worth
running"; ROI on incremental cost answers "was the marginal euro well spent". Both are legitimate;
mixing them across a portfolio is not.

**Step 19. Apply the right horizon to the right investment.** A promotion is judged in-season,
including the weeks in which pull-forward reverses: the event plus four to six weeks. A launch is
judged over its lifecycle, discounted, with a stated terminal assumption: two to five years.
Branch: **is the launch being judged on its first quarter?** Almost every launch looks bad in its
first quarter, because the investment is front-loaded and the volume is not. That is a category
error producing both false kills and false survivals.

### Phase H — Close the loop into the portfolio

**Steps 20–21. File the result, then feed the base rate back.** Output: one row per launch — prior,
revision history, outcome, root-cause codes — and an updated analogue set, outcome distribution,
and ramp curve for the next prior. Branch: **a database or a folder?** A folder is not searchable,
and an unsearchable register is the failure documented in section 9.5. Branch: **does the analogue
set survive its own failures?** Organisations routinely drop failed launches from the library
because they are embarrassing, which guarantees the next prior is optimistic.

### 5.1 How this closes Module 09

The module is one chain, and this topic is the last link. Topic 01 established which signals carry
lead time and which are echoes. Topic 02 separated the components of a sales series so each could
be attributed. Topic 03 described the mechanics of a promotion and the data each generates. Topic
04 built the counterfactual that turns promoted sales into a candidate increment. Topic 05 applied
the three adjustments that turn a candidate increment into an economic one. Topic 06 handled price
and display, without which the increment is contaminated by the treatment being measured. Topic 07
replaced the average effect with a distribution across stores and customers. Topic 08 constructed a
prior for a product with no history. Topic 09 made availability explicit, because early demand is
mostly distribution.

This topic consumes all of them at once. The revision needs the prior from 08, the availability
correction from 09, the baseline and adjustments from 04 and 05, and the effect heterogeneity from
07 if it is to be made at store-cluster level. The ROI needs 04 and 05 in particular, because the
difference between gross and adjusted lift decides whether the answer means anything.

What the module does not do is decide what to hold. It produces a probability — a distribution of
demand over a horizon, revised in the light of evidence, with its uncertainty stated. Turning that
probability into an order quantity, a safety stock, and a service commitment is the subject of
Module 10: a different discipline with a different cost structure. This module ends with a
probability that becomes a quantity. That is the handover.

## 6. Worked example

A constructed case, internally consistent. A company launches a new 750 ml variant into 600
grocery stores with a 26-week plan. The numbers are chosen to make each step visible; they are not
a claim about any real launch.

### 6.1 The prior and the first four weeks

The prior was built by analogue, the method of topic 08. Two comparable variants launched in the
previous two years averaged 3.33 units per store per week at steady state, so the plan rate is
3.333 units per store-week and the 26-week plan is `3.333 x 600 x 26 = 52,000` units.

| Week | Units | Units per store-week | What was happening |
|---|---:|---:|---|
| 1 | 3,000 | 5.00 | Launch display, feature in the retailer circular |
| 2 | 2,700 | 4.50 | Display continued; pipeline reorders arriving |
| 3 | 2,000 | 3.33 | Display ended; shelf reset mid-week |
| 4 | 1,900 | 3.17 | Normal conditions; 4.0% of store-days out of stock |
| **Total** | **9,600** | **4.00 average** | |

The launch review reads: 9,600 units against a four-week plan of 8,000, which is **+20.0%**. The
proposal is to re-forecast the year at the observed rate. That proposal is wrong, and the next
subsection shows by how much.

### 6.2 The revision ladder

Strip the temporary support. Weeks 1 and 2 carried display and feature; they are evidence about
what display does (topic 06), not about the run rate. The clean window is weeks 3 and 4:
`(2,000 + 1,900) / (2 x 600) = 3,900 / 1,200 = 3.250`.

De-censor for availability. Four percent of store-days in the clean window were out of stock.
Assuming stock-outs are approximately random with respect to velocity — a strong and usually
optimistic assumption — `3.250 / (1 - 0.040) = 3.385`.

Shrink toward the prior, with `w = 0.30` fixed in the launch plan before any data arrived:
`0.70 x 3.333 + 0.30 x 3.385 = 2.333 + 1.016 = 3.349`.

Apply to the remaining periods only. Twenty-two weeks remain:
`3.349 x 600 x 22 = 44,206`, so `revised = 9,600 + 44,206 = 53,806`, which is `+1,806` units or
`+3.5%` against plan. The ladder:

| Path | Rate for weeks 5–26 | Remaining | Revised total | vs plan |
|---|---:|---:|---:|---:|
| Hold the plan, absorb actuals | 3.333 | 44,000 | 53,600 | +1,600 (+3.1%) |
| Raw rate, `w = 1` | 4.000 | 52,800 | 62,400 | +10,400 (+20.0%) |
| Raw rate, `w = 0.30` | 3.533 | 46,640 | 56,240 | +4,240 (+8.2%) |
| Censoring-adjusted, `w = 1` | 3.385 | 44,687 | 54,287 | +2,287 (+4.4%) |
| **Censoring-adjusted, `w = 0.30`** | **3.349** | **44,206** | **53,806** | **+1,806 (+3.5%)** |
| Censoring-adjusted, `w = 0.50` | 3.359 | 44,344 | 53,944 | +1,944 (+3.7%) |

Two readings matter. **The adjustment does most of the work:** moving from the raw rate at
`w = 1` (+20.0%) to the adjusted rate at `w = 1` (+4.4%) removes 15.6 percentage points of apparent
upside; moving from `w = 1` to `w = 0.30` removes a further 0.9. An organisation arguing about the
shrinkage weight while skipping the availability correction is arguing about the small term.
**Hold and re-forecast are close here** — +3.1% against +3.5% — because the launch is genuinely
near plan once the noise is removed. That is the common case, and why a disciplined process often
concludes "no material change", a conclusion a review meeting finds unsatisfying and a supply
planner finds valuable.

### 6.3 What a dozen store-weeks can and cannot support

Suppose the same launch had been piloted in 12 stores for four weeks. Forty-eight store-weeks
sounds like a sample. It is not. Store-level weekly standard deviation in this category is about
1.0 unit against a mean of 3.3, a coefficient of variation (CV) of 0.30. With a design effect of 3
— modest for twelve stores — the effective sample size is 16 and the standard error is
`1.0 / sqrt(16) = 0.250`, or 7.5% of the rate. At 600 stores the arithmetic differs in nominal terms
and only moderately in practice: 1,200 store-weeks, design effect 12, effective sample size 100,
standard error `1.0 / sqrt(100) = 0.100`, or 2.5%.

```text
adjusted rate 3.385, SE 0.100
95% interval: 3.385 +/- 1.96 x 0.100 = [3.19, 3.58]
low  case: 3.19 x 600 x 22 + 9,600 = 42,108 + 9,600 = 51,708  (-0.6% vs plan)
high case: 3.58 x 600 x 22 + 9,600 = 47,256 + 9,600 = 56,856  (+9.3% vs plan)
```

The interval runs from roughly the plan to nine per cent above it. The point estimate is a
convenience; the order decision should be made against the interval and the cost of being wrong in
each direction. With twelve stores the interval is roughly twice as wide, spanning a shortfall of
several per cent to an excess of about twenty. A dozen store-weeks can tell you the product sells.
It cannot tell you how well, and a revision built on it is a statement about the prior dressed as
a statement about the data.

### 6.4 The promotion ROI, three ways

A separate question on the same portfolio: an existing item is promoted for four weeks at a 15%
discount. Normal price 4.00 per unit, variable cost 2.20, contribution 1.80. Promoted price 3.40,
contribution 1.20. Baseline 9,600 units, promoted 15,360, gross lift 5,760 (+60.0%).

| Cost line | Value (EUR) |
|---|---:|
| Feature and display fees | 1,400 |
| In-store materials | 400 |
| Incremental logistics at 0.04 per unit on 15,360 | 614 |
| Markdown on 400 leftover units at 0.375 | 150 |
| **Total** | **2,564** |

**Level 1 — the sales report.** Lift valued at full contribution: `5,760 x 1.80 = 10,368`, and
`ROI = 10,368 / 2,564 = 4.04`. Wrong in a specific way: the discount was paid on all 15,360
units, including the 9,600 that would have sold at full price, so the lift is valued at a margin
not earned on any of it.

**Level 2 — gross lift, correctly costed.** Earned contribution minus contribution forgone on the
baseline: `15,360 x 1.20 - 9,600 x 1.80 = 18,432 - 17,280 = 1,152`, and `ROI = 1,152 / 2,564 =
0.45`. The event did not cover its costs before any adjustment for demand shifting.

**Level 3 — adjusted lift, using topic 05.** Pull-forward: 1,900 of the lift units were
forward-bought by households that would have bought within five weeks anyway, a displacement
charge of `1,900 x (1.80 - 1.20) = 1,140`. Cannibalisation: 900 units from a sibling at a
contribution of 1.50, a charge of 1,350. Halo: 400 units of a complementary item at 0.80, a gain of
320.

```text
adjusted increment = 1,152 - 1,140 - 1,350 + 320 = -1,018
net of costs       = -1,018 - 2,564 = -3,582
ROI = -1,018 / 2,564 = -0.40
```

Three correct computations, three questions, one of which is decision-relevant. The lesson is not
that promotions are bad. A promotion with a negative adjusted ROI can still be worth running — as
a defensive move, as the price of a distribution gain, as trial generation booked against a later
launch. What it cannot honestly be is a profit centre, and an organisation reporting 4.04 will keep
funding it as one.

### 6.5 The payback horizon: a launch is not a promotion

The launch investment is committed before the first sale: tooling 90,000; launch marketing 40,000;
slotting and free fill 30,000; artwork and regulatory 20,000; total 180,000. Using the revised
forecast rather than the plan — the point of section 6.2 — year one is twice the 26-week revision,
at a contribution of 1.45 reflecting introductory pricing and launch-period trade support:

```text
year 1 volume       = 2 x 53,806 = 107,612 units
year 1 contribution = 107,612 x 1.45 = 156,037 EUR
```

Year two at 120,000 units and 1.60 gives 192,000; year three at 130,000 units gives 208,000.

```text
simple payback:
  cumulative after year 1 = 156,037; shortfall = 23,963
  monthly in year 2       = 192,000 / 12 = 16,000
  payback = 12 + 23,963 / 16,000 = month 13.5

discounted at 10%, end-of-year convention:
  year 1 PV = 156,037 / 1.10 = 141,852
  year 2 PV = 192,000 / 1.21 = 158,678
  payback = 12 + (180,000 - 141,852) / (158,678 / 12) = month 14.9
```

Applying the promotion rule to the launch — "it must pay back in the season" — kills a launch that
returns its capital in fourteen months and earns for two more years. Applying the launch rule to
the promotion funds a permanently loss-making event because a different event succeeded. The
horizon is part of the definition of the metric, not a presentation choice.

### 6.6 Wrong, and wrong for a good reason

At week 26 the launch lands at 49,400 units against the plan of 52,000, a shortfall of 5.0%. The
post-mortem must answer two questions, and they are different questions.

**Was the prior good, given what was known at issue?** The analogue set was the two most similar
variants, distribution was met within two points, and price was as planned. On that information
the prior was reasonable, and the 5% miss sits inside the interval the analogue dispersion implies.

**Was the revision good, given what was known at week 5?** It said 53,806; the outcome was 49,400,
so it was 8.9% high. The reason is visible in data that existed at week 5 only in embryonic form:
a competitor entered in week 9 with a comparable product 10% cheaper, and the rate in competitive
stores fell by about a fifth from week 10. The week-5 revision could not have known. A week-12
revision could have.

The conclusion the record supports: the prior was acceptable, the analogue method is not the
problem, the week-5 weight is not the problem, and the process lacked a **revision trigger for
competitor entry** — a design defect with a named owner and a fixable remedy. Without the frozen
record, the meeting concludes "the forecast was 8.9% too high" and the action is "forecast better",
which is not an action.

> **Note:** The numbers are constructed to make each step visible, not to benchmark launch
> performance. The weight `w = 0.30` is a policy choice, not an estimate, which is why section 7
> reports three values; the design effect of 12 determines the interval more than the point estimate
> does; and the ROI adjustments are point estimates on a baseline with its own error, so the honest
> presentation is a range.

## 7. Code, query, or calculation

The calculation below is the section 6.2 arithmetic written for replication rather than for
reading. Every input is stated, every intermediate value is named, and the last three lines report
the answer against two reference paths so a reader can see how much of the revision came from
adjustment and how much from the weight.

Assumptions: the observation is scanner sell-through over a fixed universe of 600 stores; weeks 1
and 2 carried display and are excluded from the run rate; out-of-stock store-days are 4.0% in the
clean window and are assumed random with respect to store velocity; the weight is a stated policy
value, not an estimate.

```text
GIVEN
  p0   = 52,000 / (600 x 26) = 3.3333 units per store-week   (prior rate)
  A    = 9,600 units over weeks 1-4                          (elapsed actuals)
  clean window weeks 3-4: 2,000 + 1,900 = 3,900 units
  oos  = 0.040 of store-days, weeks 3-4
  S    = 600 stores;  n = 22 remaining weeks;  w = 0.30

STEP 1  strip the temporary support
  r_clean = 3,900 / (2 x 600)                        = 3.2500
STEP 2  de-censor for availability
  r_obs   = 3.2500 / (1 - 0.040)                     = 3.3854
STEP 3  shrink toward the prior
  r_prime = (1 - 0.30) x 3.3333 + 0.30 x 3.3854
          = 2.3333 + 1.0156                         = 3.3489
STEP 4  re-forecast the remaining periods only
  remaining = 3.3489 x 600 x 22                      = 44,205
  revised   = 9,600 + 44,205                         = 53,805
STEP 5  report against the references
  plan                                 52,000   +1,805  (+3.5%)
  hold plan, absorb actuals            53,600   +1,600  (+3.1%)
  raw rate, w = 1                      62,400  +10,400  (+20.0%)
```

Expected output: 53,805 units, 1,805 units and 3.5% above plan, against a headline of 20.0% above
plan. Two integrity checks before trusting any such number. First, the revised total must equal
elapsed actuals plus the remaining period, and must be at least the 9,600 already banked. Second,
the total must move monotonically in `w` between the `w = 0` and `w = 1` paths; if it does not, the
adjustments are being applied after the shrinkage.

## 8. Data, standards, and regulatory context

### 8.1 No standard defines a post-launch revision

Nothing in ISO, in the accounting standards, or in any regulator's guidance prescribes how to
revise a forecast after launch, what weight to give early data, or how to compute a promotional
ROI. The Association for Supply Chain Management's Supply Chain Operations Reference (SCOR) model
includes forecast accuracy among its reliability metrics, but measures the process output at a
period end and says nothing about the revision path (grade C: a reference model, not a binding
standard). The consequence is the same as in Module 07 topic 10 — every organisation invents its
own definition, acceptable only if it is written down and stable.

### 8.2 The data objects, and why each one fails

| Object | Required content | Why it fails in practice |
|---|---|---|
| Frozen prior | Weekly forecast per store group, analogue set, assumptions, issue date | Planning systems overwrite in place; the number is gone by week two |
| As-of sales vintage | Scanner or shipment data with the extraction timestamp | Feeds are revised retrospectively; the panel is re-based; the universe changes |
| Availability series | Out-of-stock share by store-week | Rarely captured at store level; often a manual audit two weeks late |
| Distribution series | Store count and ACV by week, with effective dates | Maintained by another team, in another system, on another calendar |
| Trade calendar | Feature, display, price, coupon by store-week | Lives in a promotion system the demand team cannot query |
| Cost ledger | Fees, deductions, chargebacks, logistics, markdown | Deductions arrive 60 to 120 days later; the ROI is computed before they land |
| Root-cause codes | A controlled vocabulary applied at review | Free text, applied retrospectively, never aggregated |
| Launch register | One row per launch with prior, revisions, outcome, codes | A shared folder of presentations |

### 8.3 Where the numbers become accounting numbers

Two rules change what the ROI is measuring, and both are frequently ignored by analytics teams.

**Trade spend is usually a reduction of revenue, not a marketing expense.** Under IFRS 15,
consideration payable to a customer — which includes many promotional payments to retailers — is
generally accounted for as a reduction of the transaction price rather than a separate expense. US
generally accepted accounting principles (GAAP) treat it broadly similarly. A gross-margin-based
ROI and a net-revenue-based ROI are therefore not the same number, and a team computing
contribution from a price file without knowing which treatment applies will disagree with finance
by the size of the trade rate.

**Advertising and promotion costs are expensed, not capitalised.** IAS 38 requires expenditure on
advertising and promotional activities to be recognised as an expense when incurred; ASC 720-35
requires advertising costs to be expensed as incurred or at the first time the advertising takes
place. This is why a launch investment is a mix of capital (tooling, which may be property, plant
and equipment) and expense (launch marketing, slotting), and why a launch "payback" is a management
construct rather than a financial-statement object.

| Instrument | How it touches this topic | Grade |
|---|---|---|
| IFRS 15, consideration payable to a customer | Determines whether trade spend reduces revenue or is an expense, and therefore what the ROI denominator means | B |
| IAS 38, advertising and promotional expenditure | Requires launch marketing to be expensed, shaping the investment profile | B |
| ASC 720-35, *Advertising Costs* | US equivalent; timing of expense recognition | B |
| IAS 2 / ASC 330, *Inventories* | Excess launch stock and markdowns affect valuation and provisions | B |
| IAS 36, *Impairment of Assets* | A failed launch can trigger impairment testing; the frozen prior is audit evidence | B |
| Sarbanes-Oxley Act 2002, sections 302 and 404 | Where launch forecasts feed disclosure, changes require documented authorisation | B |

> **Note:** This is educational description, not accounting or legal advice. The classification of
> a specific promotional payment depends on its terms and the applicable framework, and it is a
> question for the controller, not the analyst.

### 8.4 Employment, privacy, and what is convention

A post-mortem that names individuals is employee monitoring, with the exposure set out in Module
07 topic 10 section 8.4: the General Data Protection Regulation (GDPR), Regulation (EU) 2016/679,
Article 88; national safeguards in the employment context; works council consultation rights in
several member states; and personal-data rules on access, retention, and purpose. The default that
avoids most of the problem is to report by mechanism and segment, never by person.

Several things that feel like rules are habits: the 26-week horizon and the weekly bucket; the
stabilisation window of weeks 3 to 8; the weight range 0.2 to 0.4; the three-level ROI ladder; the
payback threshold for capital projects; the root-cause taxonomy. Each organisation's differs, and
the only requirement is that the choice be written down and applied consistently. An analyst who
cannot tell a convention from a requirement treats the 26-week horizon as a physical constant.

## 9. Industry practice and case evidence

### 9.1 How accurate are launch forecasts

| Source | Design | Finding | Grade |
|---|---|---|---|
| Kahn (2002), *Journal of Product Innovation Management* | Survey of new product forecasting practice | Judgemental methods dominate; marketing usually owns the forecast; average reported accuracy around 58%; technique count was not associated with accuracy | A |
| IBF benchmarking report on new product forecasting | Practitioner benchmarking survey | Forecast error is highest for new products; no dominant model; post-mortem discipline varies widely | C |
| Cha, Rifai and Sarraf, as reported by IQVIA (2020) | 1,700 forecasts across 260 drugs | Peak sales missed by roughly 71% against one-year-pre-launch predictions; still around 45% off six years after launch | C |
| Kahn, in presentations reproduced by forecasting vendors | Self-reported accuracy one year post-launch, by product type | New-to-the-world around 40%; new category around 47%; line extension around 63%; cost reduction around 72% | D |

The pharmaceutical figure is worth pausing on: it is a blog summary of a published analysis, not
the analysis, and the average conceals a long tail of large overstatements. What it establishes is
direction and scale — a pre-launch number for a genuinely new product is a hypothesis.

### 9.2 Updating a diffusion model with early data

The academic work on revising new-product forecasts is mostly about the Bass model, and it
converges on a caution directly relevant here.

- **Early parameters are unstable.** Dalal, Ho and Sherman report Bass parameters from a short
  early series going badly wrong — in one case ten months of data implied a word-of-mouth
  coefficient near zero — and show that borrowing information from a neighbouring market recovered
  a sensible estimate. Grade C for the working paper; grade A for the instability result.
- **Borrowing strength is the standard remedy.** Ramírez-Hassan and Montoya-Blandón (2020),
  *International Journal of Forecasting* 36(2), estimate a generalized Bass model with priors from
  analogous markets and report better performance than traditional methods when little or no
  history exists. This is the formal version of section 3.2: the analogue is the prior, the launch
  is the likelihood. Lenk and Rao (1990) treat launches as draws from a population, so the next
  prior is the posterior — the justification for a register rather than a folder. Grade A.
- **Company evidence.** An Intel-authored paper on Bayesian updating of an extended Bass model
  reports a 33% reduction in forecast error and a saving of about 11.8 million currency units.
  Grade C.

The synthesis: the case for updating early is strong; the case for updating *naively* is not.
Every one of these methods shrinks toward a prior, and none recommends replacing the plan with four
weeks of data.

### 9.3 A short history of marketing ROI measurement

The ROI question has a long ancestry, and knowing it prevents the illusion that the current
toolkit settled something.

- **1950s and 1960s.** "Marketing mix" enters the vocabulary through Borden and Culliton. Vidale
  and Wolfe (1957) introduce saturation-and-decay for advertising, still recognisable in every
  adstock model; Koyck's distributed lag (1954) supplies carry-over. Palda on cumulative
  advertising effects and Lambin's regression-based studies establish the econometric approach.
- **1970s and 1980s.** Clarke reviews how long advertising effects last; Broadbent's adstock
  formulation (1979) fixes the carry-over convention. The universal product code (UPC) scanner
  arrives in 1974, and scanner panel data commercialise marketing-mix modelling — measurement
  moves from annual regressions to weekly store-level decomposition.
- **1990s to 2020s.** Hanssens, Parsons and Schultz's *Market Response Models* becomes the canonical
  academic reference. Digital attribution rises, its measurement problems are documented, and
  privacy changes drive a return to aggregated marketing-mix modelling — now Bayesian, and
  available open-source.

> **Note:** The timeline is drawn from practitioner histories (grade D), not from the original
> papers. The attributions are consistent across those histories but were not checked against the
> primary sources in this pass. Use it to orient, not to cite.

The recurring pattern is the point: each generation of measurement promised to settle the ROI
question and was defeated by the same two problems — the baseline is unobservable and the cost
base is incomplete. Methods improve; the identification problem does not go away.

### 9.4 What the trade-promotion evidence says

- **Blattberg and Levin (1987), *Marketing Science* 6(2)**, is titled around the question of why
  trade promotions are generally unprofitable, and models when they are not. Grade A.
- **Nijs, Misra, Anderson, Hansen and Krishnamurthi (2010), *Marketing Science* 29(2)**, estimate
  pass-through elasticities of about 0.71 at the wholesaler, 0.59 at the retailer, and 0.41 for the
  total channel, with large dispersion. A 10% off-invoice strategy reduces manufacturer and
  wholesaler profits in 56% of product-store combinations while retailers gain in 96%. Grade A.
  The finding is not that promotions never work; it is that the manufacturer frequently funds a
  transfer to the retailer.
- **Ailawadi and Harlam (2009)** and **Kumar, Rajiv and Jeuland (2001)**, both in *Marketing
  Science*, examine pass-through and retailer opportunism. Grade A.
- **Srinivasan, Pauwels, Hanssens and Dekimpe (2004), *Management Science* 50(5)**, finds the
  manufacturer-versus-retailer answer is category-dependent. Grade A.

Read with section 6.4, the evidence supports a specific practice: compute ROI on adjusted lift and
a full cost base, and expect a substantial share of events to come out negative. That is the
published norm, not a sign that the analysis is broken.

### 9.5 Do post-mortems change anything

| Source | Design | Finding | Grade |
|---|---|---|---|
| Williams (2006), PMI research conference | Survey of 522 organisations | 62% have formal lessons-learned procedures; about 12% report close adherence; 22% agree lessons transfer elsewhere; top barriers are time (67%) and management support (62.5%) | C |
| Pettiway (2015), Case Western Reserve thesis | 25 interviews with information-technology project managers | Post-mortems are largely ritual; repositories unindexed and project-scoped; managers hoard lessons; learning happens informally; short-cycle retrospectives outperform annual reviews | C |
| Pan, Pan and Newman (2007), *Journal of the American Society for Information Science and Technology* | Case study, attribution perspective | Failure is explained by attribution error: self-serving mindsets, persistence of negative beliefs, memory decay, selective recall, power dynamics | A |
| Rasper, Stanier and Carluccio (2002), PMI | Company case | An end-of-project process produced a 1,200-entry, unsearchable database nobody used; the fix was phase-based reviews, a short prioritised list, and named owners | C |

The transferable conclusion is not "run better post-mortems." It is that the binding constraint is
**reuse**, and reuse needs three things a template does not provide: a named owner for each
action, a searchable structure keyed on what you will want to look up (product type and mechanism,
not the launch name), and a decision that the finding will change. Pan and colleagues add a
fourth: an attribution process that names mechanisms rather than people, because a blame-laden
review produces selective recall and the record becomes worthless. The practice the evidence
supports is closer to an agile retrospective than to an annual post-mortem — a review at weeks 4
to 8, which is also the revision window, rather than a single review at year end.

### 9.6 What practitioners argue about

1. **Whether to revise before the stabilisation point.** One camp holds that any revision before
   eight weeks is noise-fitting; the other that the supply chain must commit before then, so
   refusing to revise is itself a decision. Both are right about their own constraint, and the
   argument is really about who bears the cost of being wrong.
2. **Whether the weight can be estimated.** Statisticians want a precision-based `w`; planners
   point out that the dominant uncertainty is model uncertainty, so a precision-based weight is
   falsely confident. The compromise is a policy weight with a sensitivity table.
3. **Whether ROI should include strategic benefits.** Defensive promotions, distribution gains,
   and trial generation have value a single-event ROI cannot capture. The argument is whether to
   monetise them in the numerator or to report them beside a negative number; the second is more
   honest and less abusable.

### 9.7 What this means for practice, and the handover

The evidence supports a short list. Adjust before you revise. Revise the remainder, not the
history. State the weight. Compute ROI on adjusted lift with a full cost base. Judge a promotion
in-season and a launch over its life. Review early enough that the finding can change a decision,
and file the finding where the next launch can find it.

None of that says how much to hold. This module's output is a probability over a horizon, revised
with evidence and accompanied by an honest interval. Module 10 takes that probability and turns it
into an order quantity, a safety stock, and a service commitment under a cost structure this module
has deliberately not modelled. The boundary is clean: here, what to believe; there, what to do
about it.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Whole-horizon restatement | The revised annual total is below the units already sold, or elapsed weeks change between versions | Rate applied to all periods, overwriting observed ones | Forecast cannot reconcile to actuals; year-end surprise | Re-forecast the remainder only |
| Unadjusted observation | The revised rate equals the raw scanner average, with no note on availability or display | Censoring and support read as demand | Over-ordering into a fading spike | Apply the four adjustments before shrinking |
| Ignored stock-outs | The revision is downward in weeks when availability was below 90% | Unobserved demand treated as absent demand | A good launch cut for a supply failure | Carry availability alongside sales |
| Weight chosen after the result | The `w` value appears for the first time on the results slide | Weight tuned to a preferred number | The revision is unfalsifiable | Fix `w` in the launch plan; publish a sensitivity |
| Anchoring on the plan | Ten launches all revise to within 3% of plan regardless of the data | Insufficient adjustment from the initial number | Slow response; the plan becomes the forecast | Score the revision's movement, not only its error |
| Over-reaction to recent weeks | The revision tracks the last two weeks; next month's reverses it | Recency and salience dominate | Variance without accuracy gain; supply whiplash | Shrink toward the prior; lengthen the window |
| No stabilisation test | Different functions cite different run rates from different windows | The window is chosen informally | The meeting argues about windows, not demand | Fix and publish the window ex ante |
| Gross-lift ROI | ROI above 3 and the event described as profitable with no adjustment | Lift valued at a margin not earned on baseline units | A loss-making event is repeated and scaled | Report the three-level ladder, or one stated definition |
| Denominator excludes late costs | The cost line is one "trade spend" figure equal to the price cut | Deductions, fees, returns arrive after the report | ROI overstated, sometimes by the whole margin | Accrue all lines at close; restate when claims land |
| Horizon mismatch | A launch rejected for not paying back in its first quarter | A season rule applied to a lifecycle investment | Good launches killed; bad promotions funded | Separate the rules and state which is in force |
| Unfrozen information set | The post-mortem is eight months later with current data and no record of what was known | Hindsight replaces the information set | The wrong lesson is learned; the mechanism is missed | Freeze prior, vintage, and decision at the moment of choice |
| Blame-coded post-mortem | Every root cause resolves to "forecasting" or to a named person | The taxonomy is a blame device | Selective recall; the register becomes worthless | Codes map to mechanisms with owners who are not the forecaster |
| Register without reuse | The lessons-learned folder has not been opened since creation | Capture without indexing or transfer | The next launch uses the same optimistic analogue | Key on product type and mechanism; review before the next prior |
| Prior judged on hindsight | "We should have known the competitor would enter", with no evidence it was knowable | The information set is not reconstructed | Punishes normal risk-taking; rewards conservative forecasts | Score the prior against as-of information |

Two deserve a note. **Whole-horizon restatement** is the most common arithmetic error and the most
damaging to trust, because it produces a forecast that cannot be reconciled to the ledger.
**Gross-lift ROI** is the most commercially consequential: section 6.4 shows an event reported at
4.04 and worth −0.40 depending on the convention, and the difference is not opinion but which units
the discount was paid on.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The early observation comes from the same population as the forecast | The rate transfers only if stores and weeks are comparable | Store additions or a changed universe make the rate a mix artifact | Compare universe and weighting week by week |
| Stock-outs are random with respect to velocity | The divisor `1/(1 - oos)` is valid only then | Concentrated stock-outs understate demand badly | Correlate out-of-stock share with velocity by store |
| Display and feature effects are confined to the weeks they ran | Stripping those weeks recovers the baseline rate | A post-display trough leaves the clean weeks contaminated | Plot weeks after display ends; look for a dip |
| The prior rate is a reasonable centre for shrinkage | A biased prior pulls a good observation toward it | An unrepresentative analogue produces persistent bias | Compare prior with outcome across the analogue set |
| The weight reflects relative precision, not preference | The update is interpretable only if `w` is principled | A convenient `w` makes the revision unfalsifiable | State `w` before data; audit whether it varies with the surprise direction |
| Stores are exchangeable within a week | The rate is an average over a comparable population | A few large-format stores dominate the average | Weight by ACV; check the rate by store format |
| The remaining-period profile is known | A flat rate over 22 weeks assumes no seasonality or second support wave | A festive peak or a second display makes the flat application biased | Compare with the analogue's profile |
| The cost base is complete when computed | ROI is comparable only if the denominator is consistent | Late deductions and markdowns change the sign | Reconcile accrued ROI to the ledger at 120 days |
| The baseline model is unbiased | ROI is a difference against it | A baseline containing the promotional effect understates lift | Validate the baseline on non-promoted weeks |
| The evaluation window includes the reversal | Pull-forward either reverses inside the window or not at all | A four-week window books forward buying as increment | Extend to event plus four to six weeks and check the dip |
| Codes are applied independently of the person reviewed | Codes are the only route to a mechanism | Blame coding produces selective recall and a useless register | Audit codes against free text; check the distribution across owners |

Two limits, stated plainly. **A better revision does not prove a better decision.** A revised
number can be more accurate and still lead to a worse order, because the order also depends on lead
times, commitments, and the cost of being wrong; those are Module 10's subject. **The
portfolio-learning claim is weaker than it sounds.** A register of thirty launches is a sample of
thirty from a non-stationary population. What it buys is not statistical precision but
institutional memory: a documented base rate, a list of mechanisms that have actually occurred, and
a prior that is not re-invented each time. That is valuable, and it is not an estimated
distribution.

## 12. Exercises

### Exercise 1 — The revision ladder by hand (calculation, 45 minutes)

A launch into 400 stores has a prior rate of 2.500 units per store-week and a 30-week plan of
30,000 units. Eight weeks have elapsed with 8,900 units sold. Weeks 1 and 2 carried display; the
clean window, weeks 3 to 8, produced 6,300 units with 5.0% of store-days out of stock. Use
`w = 0.25`.

**Deliverable:** clean rate, de-censored rate, shrunk rate, remaining-period re-forecast, revised
total, and variance against plan, with every intermediate value shown.
**Acceptance criteria:** elapsed actuals preserved as actuals; revised total not below 8,900;
variance stated to one decimal place.

### Exercise 2 — The effective sample size (calculation, 35 minutes)

A launch is observed in 20 stores for six weeks. Store-level weekly standard deviation is 0.9
units against a mean of 2.4. Assume a design effect of 4.

**Deliverable:** nominal store-weeks, effective sample size, standard error of the mean rate, and a
95% interval on the rate.
**Acceptance criteria:** design effect applied with an explanation; interval in units per
store-week and as a percentage; a statement of what it implies for an order increase.

### Exercise 3 — Three ROIs on one event (calculation, 50 minutes)

A promotion runs three weeks. Normal price 5.00, variable cost 2.60, promoted price 4.25. Baseline
12,000 units, promoted 18,000. Fixed fees 2,200, incremental logistics 0.05 per promoted unit,
markdown on 500 leftover units at 0.40. Adjustments: 2,400 units forward-bought, 1,100 units
cannibalised from an item with a contribution of 1.70, 600 units of halo at 1.10.

**Deliverable:** the three convention results, each with its ROI and a one-line statement of the
question it answers.
**Acceptance criteria:** subsidy paid on all promoted units in levels 2 and 3; displacement charge
uses the contribution difference; the three ROIs are not averaged.

### Exercise 4 — Design a revision trigger (design, 45 minutes)

Design the trigger set that would cause a launch revision between weeks 4 and 12, for a chilled
category with a 14-day shelf life and a 6-week repurchase cycle.

**Deliverable:** a table of indicators with lag, threshold, action, and owner.
**Acceptance criteria:** at least four indicators; no lag longer than the window in which its
action is still useful; every threshold a number with a unit; one trigger about availability.

### Exercise 5 — The frozen record (design, 50 minutes)

Design the record that must exist at the moment of a revision so that the revision can be
evaluated six months later.

**Deliverable:** a field list with types, the storage mechanism, the immutability rule, and the
retention period.
**Acceptance criteria:** prior, as-of vintage, adjusted rate, weight, revised total, interval,
decision, and date present; forecast change distinguished from plan change; immutability does not
depend on an analyst remembering.

### Exercise 6 — Critique the post-mortem (critique, 40 minutes)

A deck reads: "Launch volume was 12% below plan. Root cause: demand forecasting. Action: improve
forecast accuracy. Owner: demand planning." Write the critique.

**Deliverable:** a rewritten root cause, action, and owner, each with the evidence you would need.
**Acceptance criteria:** the root cause names a mechanism rather than a function; the action is
testable and dated; the owner controls the mechanism; one alternative mechanism is named with the
evidence that would distinguish it.

### Exercise 7 — Build the register (design, 60 minutes)

Design the launch register that would let a company apply the result of one launch to the next.

**Deliverable:** a schema, a review cadence, the five queries the register must answer, and a
governance rule for editing an outcome.
**Acceptance criteria:** schema keys on product type and failure mechanism as well as launch;
failed launches included; one query produces an updated prior; no retroactive revision of a
recorded outcome.

### Exercise 8 — Choose the horizon (application, 40 minutes)

A launch needs 240,000; revised contributions are 150,000, 210,000, and 230,000 over three years,
with a 12% hurdle and a 24-month discounted payback rule. A promotion needs 4,000 and returns an
adjusted 3,100 in-season.

**Deliverable:** simple payback, discounted payback, and a recommendation for each investment.
**Acceptance criteria:** discounting year by year; the two investments judged on different rules,
with the reason; the recommendation states what would change it.

## 13. Self-check answers

**Exercise 1.**

```text
clean rate      = 6,300 / (6 x 400) = 2.6250
de-censored     = 2.6250 / 0.95 = 2.7632
shrunk          = 0.75 x 2.5000 + 0.25 x 2.7632 = 2.5658
remaining 22 wk = 2.5658 x 400 x 22 = 22,579
revised total   = 8,900 + 22,579 = 31,479
vs plan         = +1,479 (+4.9%)
```

Note what the answer does not do: it does not re-state the first eight weeks. A common wrong
answer computes `2.5658 x 400 x 30 = 30,790` and reports +2.6%, which implies the elapsed weeks sold
at a rate they did not.

**Exercise 2.** Nominal `20 x 6 = 120` store-weeks; effective sample size `120 / 4 = 30`; standard
error `0.9 / sqrt(30) = 0.164`. The interval is `2.40 +/- 1.96 x 0.164 = [2.08, 2.72]`, or ±6.8% of
the mean. The decision implication is the point: an order increase justified by 2.40 is not
justified by a rate that could be 2.08. If the increase is expensive to reverse, the lower bound is
the relevant number.

**Exercise 3.** Contribution is 2.40 at normal price and 1.65 at the promoted price.

```text
level 1   6,000 x 2.40                                    = 14,400
level 2   18,000 x 1.65 - 12,000 x 2.40 = 29,700 - 28,800 =    900
level 3   900 - 2,400 x 0.75 - 1,100 x 1.70 + 600 x 1.10
        = 900 - 1,800 - 1,870 + 660                       = -2,110
costs     2,200 + 0.05 x 18,000 + 500 x 0.40             = 3,300
ROI       14,400 / 3,300 = 4.36;  900 / 3,300 = 0.27;  -2,110 / 3,300 = -0.64
```

The three numbers answer three questions: what the sales report shows, whether the event earned
before demand shifting, and whether the business earned. A submission that averages them, or
reports only the first, has not done the exercise.

**Exercise 4.** A workable trigger set for a chilled category:

| Indicator | Lag | Threshold | Action | Owner |
|---|---|---|---|---|
| On-shelf availability | 0–3 days | Below 92% two weeks running | Raise the order; escalate to retail replenishment | Supply planning |
| Sell-through vs prior | 1 week | Outside ±15% three weeks running | Re-run the revision | Demand planning |
| Weighted distribution vs plan | 1–2 weeks | More than 5 points below at week 6 | Re-base to actual distribution | Commercial |
| Repeat rate at first repurchase cycle | 6–10 weeks | Below 20% | Re-scope support; re-forecast the tail | Marketing |
| Confirmed competitor entry | 2–6 weeks | Entry in the top 20% of stores by ACV | Re-run with a competitive term | Category management |

The acceptance test is the lag column: a repeat-rate trigger at week three fails it, because the
repurchase cycle is six weeks and the reading would describe heavy users only.

**Exercise 5.** Minimum fields: `launch_id`, `revision_id`, `revision_date`, `prior_rate`,
`prior_source`, `prior_frozen_ts`, `data_vintage_ts`, `elapsed_actuals`, clean-window start and
end, `raw_rate`, `oos_share`, `adjusted_rate`, `weight_w`, `weight_justification`, `shrunk_rate`,
`remaining_periods`, `revised_total`, `interval_low`, `interval_high`, `decision_taken`,
`decision_owner`, `forecast_change_flag`, `plan_change_flag`. Append-only, written by the system at
the moment of the meeting, with no update grant for the analyst; retention at least the launch
lifecycle plus the financial record retention period. The field most designs omit is the
forecast-versus-plan change flag, and it is the one that stops a supply-driven plan change being
scored as a forecast error.

**Exercise 6.** A rewritten version: "Launch volume was 12% below plan. Two mechanisms are
candidates: (a) weighted distribution reached 78% of planned ACV by week 6 and never recovered,
accounting for roughly 9 points of the 12; (b) rate per stocked store was within 2% of plan,
accounting for the remainder. Evidence for (a): the retailer's weekly distribution file. Evidence
against a demand-side explanation: the velocity series is flat against the analogue. Action: add a
distribution gate at week 6 with a re-forecast triggered at a 5-point ACV shortfall; owner,
commercial operations; date, the next launch. Second candidate to test: competitor entry in week 9,
which would show as a step down in competitive stores and a flat series elsewhere." The point is
that the root cause becomes a mechanism with a measurement, not a function with an exhortation.

**Exercise 7.** Four tables. A `launch` table with product, category, launch date, and analogue
set. A `prior` table with the frozen forecast, assumptions, and issue timestamp. A `revision` table
with the Exercise 5 fields. An `outcome` table with horizon volumes, contribution, root-cause
codes, and review date. The five queries: distribution of outcome versus prior by product type;
which codes account for the largest share of variance; median ramp curve by category; share of
launches that missed the failure threshold; and the prior this register would recommend for the
next launch in a category. That last query is what makes it a learning system rather than an
archive. Governance: an outcome can be superseded only by a new row with a reason, never edited in
place, and the review must be held within eight weeks of launch, because section 9.5's evidence is
that delay is what destroys transfer.

**Exercise 8.** Simple payback is 17.1 months (90,000 of year two's 210,000 is 0.43 of a year).
Discounted at 12%, year one is 133,929 and year two 167,411, giving 19.6 months, so the launch
passes the 24-month rule. The promotion returns 3,100 against 4,000, an ROI of 0.78, and fails the
in-season rule. Applying either rule to the other's investment produces systematic error in both
directions.

## 14. Interview questions

**1. "The launch is four weeks in and 20% above plan. What do you do?"**

A strong answer asks what produced the four weeks before accepting the number: display, feature,
pipeline fill, availability, store universe. It then re-forecasts the remaining periods only, with
a stated weight and a sensitivity, and reports an interval rather than a point. A weak answer says
"re-forecast upward and increase the order," which is the behaviour that produces the write-down.

**2. "Why not just use the observed rate?"**

A strong answer explains that the observed rate is sales conditional on availability and on
whatever support was running, and that its precision depends on an effective sample size rather
than a count of store-weeks. It adds that the prior contains information the four weeks do not,
which is why every serious method shrinks. A weak answer says "four weeks is too short," which is
true and is not an explanation.

**3. "How would you choose the shrinkage weight?"**

A strong answer calls it a judgement about relative precision and stability, fixed before the data
arrive, justified in one sentence, and reported at two other values. It may reference the Bayesian
reading — prior precision against likelihood precision — and note that sampling-based weights are
usually falsely confident because model uncertainty dominates. A weak answer names a number with
no reasoning, or says the software chooses it.

**4. "A launch missed its plan by 15%. Was the forecast bad?"**

A strong answer reconstructs the information set at issue — distribution, competition, price, the
analogue set — and separates a forecast that was wrong from one that was wrong because the world
changed. It then asks whether the revision, when the new information arrived, responded
appropriately. A weak answer says yes because the number was wrong, which is hindsight scoring.

**5. "What is the difference between gross lift and adjusted lift?"**

A strong answer defines gross lift against a baseline, then the three adjustments: pull-forward
displacement, cannibalisation at the donor's margin, and halo at the halo item's margin. It gives
an example where the two differ in sign and notes that the denominator must include the full cost
base. A weak answer treats adjusted lift as gross lift with a haircut percentage.

**6. "Why is a promotion's ROI often lower than the sales report suggests?"**

A strong answer gives two mechanisms: the discount is paid on baseline units as well as
incremental ones, and part of the apparent lift is demand moved from adjacent weeks and items. It
notes that the trade-promotion literature has reported this for decades, and that the honest
response is not to stop promoting but to stop reporting the gross number as profit. A weak answer
blames the baseline model without naming the mechanism.

**7. "How would you judge a launch investment against a promotion investment?"**

A strong answer says the horizon and the return definition differ: a promotion is judged in-season
on adjusted incremental contribution against its own cost, a launch over its lifecycle on
discounted contribution against the launch investment. It notes that applying either rule to the
other produces systematic error in both directions, and that payback is a liquidity measure rather
than a return. A weak answer uses one rule for both.

**8. "What should be frozen at the moment of a revision, and why?"**

A strong answer lists the prior, the as-of vintage, the adjusted rate and weight, the revised total
and interval, the decision, and the date, and explains that freezing lets the revision be scored on
its own information set rather than on hindsight. It notes the freeze must be system-enforced,
because an analyst who has to remember will not do it. A weak answer says "save the spreadsheet."

**9. "Why do post-mortems so often change nothing?"**

A strong answer cites the gap between capture and reuse: procedures exist, adherence is low,
repositories are unindexed, and blame coding produces selective recall. It gives the remedies the
evidence supports — short cycle, close to the event, few issues, named owners, keyed on mechanism
rather than project name. A weak answer says "people are busy," which is a symptom, not a
mechanism.

**10. "What would you build first if a company had none of this?"**

A strong answer starts with the frozen prior and the as-of archive, because nothing else can be
evaluated without them, then the availability series, then the revision rule, and only then the ROI
machinery — an order justified by dependency rather than sophistication. A weak answer starts with
a model.

## 15. Key terms

- **Post-launch revision** — a dated change to a launch forecast after the first observations.
- **Re-forecast** — rebuilding the unelapsed portion; not a restatement of history.
- **Shrinkage weight (`w`)** — weight on new information when combining an observation with a prior.
- **Prior** — the pre-launch forecast and its supporting assumptions.
- **Observed rate** — sales per store per week over the observed window, before adjustment.
- **Store-week** — one store in one week; the unit of early launch data.
- **Effective sample size** — the independent-observation equivalent of a clustered sample.
- **Design effect** — variance inflation caused by clustering.
- **Pipeline fill** — stock shipped to fill the channel before consumer sales.
- **Sell-in / sell-through** — manufacturer shipments versus consumer offtake.
- **ACV and weighted distribution** — store weight by category sales, and the share stocked.
- **Velocity** — rate of sale per stocked store per week.
- **Repeat rate** — share of first buyers who buy again in a stated window.
- **Censoring** — truncation of observed demand by unavailability.
- **Launch spike** — the early peak from novelty, display, and trial.
- **Stabilisation point** — the week the rate stops trending and becomes forecastable.
- **Analogue and reference class** — a comparable past launch, and the population behind a base rate.
- **Anchoring** — insufficient movement from an initial number when evidence arrives.
- **As-of vintage** — the state of a dataset at a stated timestamp.
- **Information set** — everything knowable when a forecast was issued.
- **Frozen prior** — the pre-launch forecast stored immutably at issue.
- **Root-cause code** — a controlled label for the mechanism behind a variance.
- **Gross and adjusted lift** — increment before and after pull-forward, cannibalisation, and halo.
- **Trade spend** — total manufacturer funding, including fees and deductions.
- **Contribution margin** — price minus variable cost, per unit.
- **ROI** — net incremental profit divided by total intervention cost.
- **Payback and discounted payback** — time to recover the investment, undiscounted and discounted.
- **Lifecycle and season horizons** — the periods over which a launch and a promotion are judged.
- **Launch register** — the structured portfolio record of priors, revisions, and outcomes.

## 16. Sources

**Peer-reviewed and academic**

- Kahn, K. B. (2002). "An Exploratory Investigation of New Product Forecasting Practices."
  *Journal of Product Innovation Management*, 19(2). DOI: 10.1016/S0737-6782(01)00133-3. Grade A.
- Lim, J. S., and O'Connor, M. (1995). "Judgemental Adjustment of Initial Forecasts: Its
  Effectiveness and Biases." *Journal of Behavioral Decision Making*, 8(3). Grade A.
- Eroglu, C., and Croxton, K. L. (2010). "Biases in judgmental adjustments of statistical
  forecasts." *International Journal of Forecasting*, 26(1). Grade A.
- Mathews, B. P., and Diamantopoulos, A. (1990). "Judgemental revision of sales forecasts:
  Effectiveness of forecast selection." *Journal of Forecasting*, 9(4). DOI: 10.1002/for.3980090408.
  Grade A.
- Van den Broeke, M., De Baets, S., Vereecke, A., Baecke, P., and Vanderheyden, K. (2019).
  "Judgmental forecast adjustments over different time horizons." *Omega*, 87, 34–45. Grade A.
- Meub, L., and Proeger, T. (2016). "Can anchoring explain biased forecasts? Experimental
  evidence." *Journal of Behavioral and Experimental Finance*. Grade A.
- Ramírez-Hassan, A., and Montoya-Blandón, S. (2020). "Forecasting from others' experience:
  Bayesian estimation of the generalized Bass model." *International Journal of Forecasting*,
  36(2), 442–465. Grade A.
- Lenk, P. J., and Rao, A. G. (1990). "New models from old: forecasting product adoption by
  hierarchical Bayes procedures." *Marketing Science*. Grade A.
- Blattberg, R. C., and Levin, A. (1987). "Modelling the Effectiveness and Profitability of Trade
  Promotions." *Marketing Science*, 6(2), 124–146. Grade A.
- Blattberg, R. C., Briesch, R., and Fox, E. J. (1995). "How Promotions Work." *Marketing Science*,
  14(3). Grade A.
- Nijs, V. R., Misra, K., Anderson, E. T., Hansen, K., and Krishnamurthi, L. (2010). "Channel
  Pass-Through of Trade Promotions." *Marketing Science*, 29(2). Grade A.
- Ailawadi, K. L., and Harlam, B. A. (2009). "Retailer Promotion Pass-Through." *Marketing
  Science*, 28(4). Grade A.
- Kumar, N., Rajiv, S., and Jeuland, A. (2001). "Effectiveness of Trade Promotions." *Marketing
  Science*, 20(4). Grade A.
- Srinivasan, S., Pauwels, K., Hanssens, D. M., and Dekimpe, M. G. (2004). "Do Promotions Benefit
  Manufacturers, Retailers, or Both?" *Management Science*, 50(5). Grade A.
- Pan, S. L., Pan, G., and Newman, A. (2007). "Information systems project post-mortems: Insights
  from an attribution perspective." *Journal of the American Society for Information Science and
  Technology*. DOI: 10.1002/asi.20693. Grade A.

**Practitioner and industry**

- Williams, T. (2006). "How do organisations learn from projects?" PMI Research Conference.
  Grade C. Survey of 522 organisations; 62% formal procedures, about 12% close adherence, 22%
  transfer.
- Pettiway, K. (2015). "How Do Project Post-Mortems Contribute to Organizational Learning?" Case
  Western Reserve University. Grade C. Twenty-five interviews; post-mortems largely ritual.
- Rasper, P., Stanier, S., and Carluccio, P. (2002). "A real-life approach to lessons learned."
  PMI. Grade C. The 1,200-entry unsearchable database case.
- Institute of Business Forecasting, "Benchmarking New Product Forecasting and Planning." Grade C.
- Dalal, S. R., Ho, T.-S., and Sherman, R. P. "Learning from Experience to Improve Early
  Forecasts: A Posterior Mode Approach." Working paper. Grade C. Instability of Bass parameters
  from short series.
- Wu, S., and Kempf, K. "Extending Bass for Improved New Product Forecasting." Company paper.
  Grade C. Self-reported 33% error reduction and a saving of about 11.8 million.

**Vendor and practitioner guidance**

- Anaplan, "10 Steps of Demand Forecasting for New Products." Grade D. Useful framing on
  reforecasting and pre-agreed failure thresholds; the product claims are not evidence.
- IQVIA blog (2020), reporting Cha, Rifai and Sarraf on 1,700 forecasts across 260 drugs. Grade C.
- Vexpower, "History of Marketing Mix Modeling: 1948 to 2026," and Pedantics, "A Brief History of
  Marketing Mix Modeling." Grade D. Source for the section 9.3 timeline; attributions not checked
  against the primary publications.

**Standards and official**

- IFRS 15, *Revenue from Contracts with Customers*, consideration payable to a customer. Grade B.
- IAS 38, *Intangible Assets*, advertising and promotional expenditure. Grade B.
- ASC 720-35, *Advertising Costs*. Grade B.
- IAS 2 *Inventories*; ASC 330 *Inventory*. Grade B.
- IAS 36 *Impairment of Assets*. Grade B.
- Sarbanes-Oxley Act of 2002, sections 302 and 404. Grade B.
- Regulation (EU) 2016/679 (GDPR), Article 88. Grade B.
- ASCM SCOR Digital Standard. Grade C. A reference model, not a binding standard.

**Note on citations.** Every figure is reported as the source states it and was not independently
recomputed. Where a figure reached this text through a secondary summary — the pharmaceutical
accuracy figures, the marketing-mix timeline, the lessons-learned survey — the text says so. No
citation was invented, and where a DOI could not be confirmed the entry gives title and venue
without one.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Predecessor, new product forecasting and the prior this topic revises:
  [`08_npi-forecasting.md`](08_npi-forecasting.md)
- Predecessor, distribution and availability as the dominant early driver:
  [`09_distribution-assortment.md`](09_distribution-assortment.md)
- Predecessor, the baseline this topic reuses for ROI:
  [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Predecessor, the adjustments that turn lift into increment:
  [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Sibling, price and display effects: [`06_price-display-effects.md`](06_price-display-effects.md)
- Sibling, heterogeneous treatment effects: [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md)
- Sibling, demand-sensing signals and their lags: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Sibling, sales decomposition: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Prerequisite, forecast value added and the counterfactual logic reused here:
  [`../07_time_series_forecasting/10_overrides-fva.md`](../07_time_series_forecasting/10_overrides-fva.md)
- Rolling-origin evaluation: [`../07_time_series_forecasting/08_rolling-origin-backtesting.md`](../07_time_series_forecasting/08_rolling-origin-backtesting.md)
- Experiment design: [`../../01_foundations/05_experimentation_pilots_evidence/01_claim-hypothesis-design.md`](../../01_foundations/05_experimentation_pilots_evidence/01_claim-hypothesis-design.md)
- Experiments and evidence: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Next module, which turns the probability into a quantity:
  [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Return to the part map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
