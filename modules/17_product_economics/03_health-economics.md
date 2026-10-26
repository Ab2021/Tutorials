# Health Economics

> Module 17 — Product, Economics, and Leadership
> Topic 03 of 09
> Estimated study time: 5 hours
> Prerequisites: 02 (Value Quantification)
> Version 2026.3

## 1. Why this topic exists

A payer, a health technology assessment body, and a hospital finance committee ask two different
questions about the same product. The first is whether the health gained is worth the money
spent. The second is whether the money can be found in this budget year. These are
cost-effectiveness and budget impact, and they can point in opposite directions: a technology can
be excellent value and still unaffordable, or affordable and still a poor use of resources.

Health economics gives you the vocabulary and the arithmetic to answer both without conflating
them. It is the language in which coverage decisions are actually made in the United Kingdom,
Canada, Australia, and much of Europe, and increasingly the language used by United States payers
and by the Institute for Clinical and Economic Review (ICER).

### 1.1 What goes wrong without it

Three failures recur.

**The wrong comparison.** A team compares its product to doing nothing when the real alternative
is the existing standard of care. The incremental cost-effectiveness ratio (ICER) is then computed
against a comparator that no one would use, and the result is meaningless.

**The wrong perspective.** A provider builds a case counting only its own costs and its own
revenue, then presents it to a payer whose costs and benefits are different. The numbers are
correct and the argument is irrelevant to the audience.

**The wrong question.** A product with a strong ICER is presented to a budget committee that
cannot fund it, or a product with a modest ICER is rejected because no one computed the budget
impact and the committee assumed the worst.

### 1.2 Why data scientists should care

Health economic models are built from the same inputs as clinical models: base rates, effect
sizes, subgroup variation, and time-to-event. The difference is that economic models require those
inputs as distributions, with a perspective attached, and with the comparator made explicit. A
data scientist who can supply a hazard ratio but not an absolute risk difference in the payer's
population is supplying the wrong parameter.

The modelling work is also familiar: state-transition models, cohort simulation, and Monte Carlo
analysis are standard tools. The unfamiliar part is the accounting conventions, which are
arbitrary in places and binding in others, and the reporting standards, which determine whether
anyone will read the result.

### 1.3 Scope of this topic

This topic covers cost-effectiveness analysis, cost-utility analysis with quality-adjusted life
years (QALYs), the ICER, net monetary benefit (NMB), discounting, and budget impact analysis. It
does not cover cost-benefit analysis in which all outcomes are monetised, nor full Markov
modelling theory, nor the econometrics of cost estimation. It gives enough to build, read, and
critique a reference-case analysis for a digital health product.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish cost-effectiveness from budget impact and state which question each answers | Exercise 1; self-check 1 |
| 2 | Define comparators, perspective, population, and horizon for a given decision | Exercise 2; self-check 2 |
| 3 | Compute incremental cost, incremental QALYs, and an ICER with discounting | Exercise 3; self-check 4 |
| 4 | Interpret the cost-effectiveness plane and identify dominance | Exercise 4; self-check 5 |
| 5 | Compute net monetary benefit and use it to resolve an ambiguous ICER | Exercise 5; self-check 6 |
| 6 | Build a budget impact analysis with uptake and cost offsets | Exercise 6; self-check 8 |
| 7 | Apply a threshold to derive price headroom and a break-even effect | Exercise 7; interview question 7 |
| 8 | Critique an economic evaluation against reporting standards | Exercise 8; interview question 11 |

## 3. The core mental model

Two planes, one budget. The cost-effectiveness plane answers whether the technology is efficient.
The budget line answers whether it is affordable. A complete analysis draws both.

### 3.1 The cost-effectiveness plane

```text
                  Incremental cost (Delta C)
                            ^
        NW quadrant         |         NE quadrant
        MORE COST           |         MORE COST
        LESS EFFECT         |         MORE EFFECT
        "dominated"         |         "trade-off"
        reject              |         ICER = dC / dE
                            |
   <------------------------+------------------------> Incremental effect (Delta E)
                            |
        SW quadrant         |         SE quadrant
        LESS COST           |         LESS COST
        LESS EFFECT         |         MORE EFFECT
        "trade-off"         |         "dominant"
        ICER = dC / dE      |         adopt
                            |
```

Three of the four quadrants give an unambiguous answer. Only the north-east quadrant, more cost
and more effect, requires a threshold judgement. Most of the analytical work in this topic is
about that quadrant, and most of the argument in a coverage committee is about the threshold.

A technology in the south-east quadrant is dominant: cheaper and more effective. It should be
adopted on efficiency grounds, and the only remaining question is affordability. A technology in
the north-west quadrant is dominated: more expensive and less effective. It should be rejected,
and no threshold can rescue it.

### 3.2 The threshold

The threshold is the maximum the decision maker is willing to pay for a unit of health gain,
usually one QALY. It converts the ICER into a binary decision:

```text
adopt if ICER < threshold
```

Equivalently, in net monetary benefit terms:

```text
NMB = Delta E x threshold - Delta C
adopt if NMB > 0
```

The two formulations are algebraically identical when the ICER is well defined. The NMB
formulation is more robust, because it remains defined when incremental effect is negative or
zero, and it is additive across subgroups, which the ICER is not.

Thresholds are jurisdiction-specific and are policy choices, not measurements. The English
National Health Service (NHS) uses a range set by the National Institute for Health and Care
Excellence (NICE); the United States has no national threshold, and ICER uses a benchmark range
for its value-based price calculations. Section 8 gives the current values and their sources.

### 3.3 Perspective determines the numerator

The perspective decides which costs and which effects enter the model. The same intervention can
have three different ICERs under three perspectives, and all three can be correct.

```text
  Provider perspective:   costs the provider incurs; benefits the provider captures
  Payer perspective:      costs the payer covers; benefits the covered population
  Societal perspective:   all costs including patient time and productivity;
                          all effects including caregiver burden
```

A provider that bears the cost of a prevention programme and captures none of the avoided
downstream cost will compute a high ICER. A payer that captures the avoided cost will compute a
low one. Neither is wrong. The error is presenting one as if it were the other.

### 3.4 Time is a modelling decision

Costs and effects occur in different periods. A screening programme costs money now and produces
health gain in ten years. Discounting converts both to present value at a stated rate. The choice
of rate and horizon is not neutral: a long horizon with a low discount rate favours prevention,
and a short horizon with a high rate favours acute treatment.

The horizon must be long enough to capture the difference in costs and effects between the
compared strategies. Truncating at the point where the intervention looks expensive is the most
common way to manufacture a bad result, and extending beyond the evidence is the most common way
to manufacture a good one.

### 3.5 Efficiency is not affordability

The budget impact is the change in total spend for the payer in each budget period, usually
undiscounted because budgets are not discounted. A technology with an ICER of 20,000 United
States dollars (USD) per QALY can still require 200 million USD in year one if the eligible
population is large and uptake is fast.

```text
  cost-effectiveness  ->  is the price justified by the health gained?
  budget impact       ->  can the payer fund it this year?
```

Committees routinely reject cost-effective technologies on affordability grounds, and routinely
fund cost-ineffective ones for political or clinical reasons. An analyst who understands both
questions can predict which objection will be raised and prepare for it.

### 3.6 Redraw test

Draw the plane, place a technology in each quadrant, and write the decision rule for each. Then
draw a budget line underneath and write the separate decision rule. If you cannot state why the
two rules can disagree, reread Section 3.5.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or convention |
|---|---|---|---|
| Cost-effectiveness analysis (CEA) | Comparison of costs and a natural health outcome | Used loosely for any economic evaluation | Cost per event avoided |
| Cost-utility analysis (CUA) | CEA with health effect measured in QALYs | Treated as the only valid form | Cost per QALY |
| Cost-benefit analysis (CBA) | All outcomes monetised | Confused with CEA | Net currency |
| Cost-consequence analysis (CCA) | Costs and effects listed separately, not combined | Considered inferior; often appropriate for digital health | Disaggregated table |
| Comparator | The alternative against which the technology is assessed | Set to "do nothing" | Strategy |
| Incremental cost (Delta C) | Cost of intervention minus cost of comparator | Computed without offsets | Currency per patient |
| Incremental effect (Delta E) | Effect of intervention minus effect of comparator | Computed on the wrong population | QALYs per patient |
| ICER | Delta C divided by Delta E | Reported without the comparator | Currency per QALY |
| Dominant | Less costly and more effective | Confused with "cost saving" | Quadrant |
| Dominated | More costly and less effective | Confused with "expensive" | Quadrant |
| QALY | Quality-adjusted life year: one year in full health | Treated as a measure of disease severity | QALYs |
| Utility | Preference weight on a health state, 0 = death, 1 = full health | Confused with clinical severity | 0 to 1 |
| EQ-5D | Standard instrument for measuring health utility | Treated as interchangeable across value sets | Instrument |
| Health-related quality of life (HRQoL) | Patient-reported health status | Confused with utility | Score |
| Discount rate | Rate converting future costs and effects to present value | Chosen to suit the result | Percent per year |
| Time horizon | Period over which costs and effects are counted | Truncated at the payback point | Years |
| Perspective | Party whose costs and benefits are counted | Left unstated | Provider, payer, societal |
| Willingness-to-pay threshold | Maximum acceptable cost per unit of health gain | Treated as a measured constant | Currency per QALY |
| Net monetary benefit (NMB) | Delta E x threshold minus Delta C | Confused with net financial benefit | Currency |
| Net health benefit (NHB) | Delta E minus Delta C divided by threshold | Rarely used outside HTA | QALYs |
| Incremental net benefit | Either NMB or NHB depending on convention | Reported without stating which | Currency or QALYs |
| Budget impact analysis (BIA) | Change in total payer spend over budget periods | Confused with cost-effectiveness | Currency per year |
| Eligible population | Patients for whom the technology is indicated | Confused with total population | Patients |
| Uptake | Share of eligible patients treated | Assumed to be 100 percent | Proportion |
| Cost offset | Reduction in other costs caused by the intervention | Double counted with the intervention cost | Currency |
| Reference case | Standardised methods set for comparability | Treated as the only permissible analysis | Method set |
| Sensitivity analysis | Re-estimation under alternative inputs | Replaced by a single scenario | Method |
| Probabilistic sensitivity analysis (PSA) | Sampling inputs from distributions to produce an ICER distribution | Confused with one-way sensitivity | Method |
| Cost-effectiveness acceptability curve (CEAC) | Probability a technology is cost-effective at each threshold | Misread as a p-value | Curve |
| Incremental cost-effectiveness plane | Plot of Delta C against Delta E | Confused with the CEAC | Plot |
| EvLYG | Equal-value life year gained, weighting all life years equally | Confused with QALY | Life years |
| Severity modifier | Adjustment raising the threshold for severe disease | Confused with a disease severity score | Multiplier |
| Transferability | Whether an economic result applies in another setting | Assumed from a similar health system | Judgement |
| Generalisability | Whether results from a trial population apply to practice | Confused with transferability | Judgement |
| CHEERS | Reporting standard for health economic evaluations | Treated as a quality rating | Checklist |
| HTA | Health technology assessment: formal evaluation for coverage | Confused with regulatory approval | Process |
| Coverage with evidence development | Conditional coverage pending further data | Confused with full coverage | Policy instrument |
| Risk-sharing agreement | Payment conditioned on real-world performance | Confused with a discount | Contract |

## 5. The mechanics, step by step

The procedure follows the reference-case conventions of the Second Panel on Cost-Effectiveness in
Health and Medicine, with jurisdiction-specific adjustments noted. It produces a defensible ICER
and a separate budget impact.

### Step 1. State the decision problem as a question with a decision maker

**Input:** an intended product and a buyer.
**Output:** one sentence: for which patients, compared with what, should which decision maker
adopt what, at what price?
**Rationale:** an economic evaluation without a decision is an exercise. The decision maker
determines the perspective, the threshold, and the budget period, and therefore the whole
structure of the model.

Example: "For adults with symptomatic heart failure discharged from hospital, should a regional
payer cover a 90-day remote monitoring programme with nurse review, compared with usual
post-discharge care, at a negotiated price of 340 USD per patient?"

### Step 2. Define the comparators, including the real standard of care

**Input:** the decision question.
**Output:** two or more mutually exclusive strategies, each fully specified.
**Rationale:** the ICER is relative to the comparator. A weak comparator inflates the apparent
value.

Rules:

- The comparator must be what the decision maker would actually use if the technology were
  unavailable. If a care pathway already exists, it is the comparator, not "no care."
- Each strategy must be specified in enough detail to cost it: who does what, how often, with
  what adherence.
- If the technology would be added to an existing pathway rather than replacing it, the
  comparator is the pathway and the intervention is pathway plus technology.

Edge case: when several comparators are plausible, the reference case uses the most cost-effective
of the existing alternatives, because that is the one a rational decision maker would choose.

### Step 3. Choose and state the perspective

**Input:** the decision maker.
**Output:** one perspective for the base case, with alternatives reported as sensitivity.
**Rationale:** perspective determines which costs enter. A provider perspective excludes patient
travel time; a societal perspective includes it.

The Second Panel recommends two reference cases: a health care sector perspective and a societal
perspective. NICE uses a health and social care perspective for its base case. ICER uses a health
care system perspective as base case with a modified societal perspective as co-base case when
non-health impacts are large.

For a commercial value dossier, the practical choice is the payer's perspective, with a societal
analysis as supporting evidence.

### Step 4. Define the population and the eligible subgroup

**Input:** indication and epidemiology.
**Output:** an eligible population with size, age distribution, severity mix, and baseline risk.
**Rationale:** the ICER is a per-patient quantity; the budget impact depends on the population
size. Both need the same population definition.

Stratify where the effect is heterogeneous. A technology that is cost-effective in high-risk
patients and not in low-risk patients should be evaluated and priced for the high-risk subgroup,
not averaged across both.

### Step 5. Choose the time horizon

**Input:** the natural history of the condition and the duration of the effect.
**Output:** a horizon in years, with a justification.
**Rationale:** the horizon must capture all differences in costs and effects between strategies.
For a chronic disease with a mortality effect, that usually means lifetime. For a workflow tool
with an annual effect, three to five years is defensible.

Test: extend the horizon by 50 percent. If the ICER changes materially, the original horizon was
too short and the result is an artefact of truncation.

### Step 6. Build the model structure

**Input:** the strategies and the horizon.
**Output:** a decision tree for short acute events, a state-transition model for chronic
progression, or a cohort simulation for populations.
**Rationale:** the structure encodes the clinical logic. A model that cannot represent recurrence,
adherence, or discontinuation will misstate both costs and effects.

Minimum requirements:

- Health states are mutually exclusive and exhaustive.
- Transition probabilities come from the best available evidence, with the source stated.
- The cycle length is short enough to avoid material error, typically one month to one year.
- Half-cycle correction is applied where cycle length is material.
- The model is transparent enough that a reviewer can reproduce it.

For digital health technologies that affect process rather than disease progression, a
cost-consequence analysis with an impact inventory is often more appropriate than a QALY model,
because the QALY captures little of the effect. The reporting standard still applies.

### Step 7. Estimate costs by category and perspective

**Input:** unit costs and resource use.
**Output:** a cost per patient per period, disaggregated.
**Rationale:** the total is only as good as its categories. A single "cost" line cannot be
audited, and the largest errors are usually in the categories that were omitted.

Cost categories for a digital health intervention:

| Category | Examples | Typically omitted |
|---|---|---|
| Technology | Licence, hardware, hosting | Overage charges |
| Implementation | Integration, configuration, training | Buyer staff time |
| Clinical operations | Nurse review, alert triage, follow-up | The largest recurring cost |
| Patient | Device, data plan, travel, time | Travel and time in provider analyses |
| Downstream | Admissions, emergency visits, tests | Positive offsets |
| Adverse events | False positives, unnecessary follow-up | Always understated |
| Maintenance | Support, revalidation, security | Rarely in year one models |

Unit costs must carry a year and a source. A cost from a 2019 study inflated to 2026 requires an
inflation method, and health care inflation is not the consumer price index.

### Step 8. Estimate health effects in QALYs

**Input:** survival or event pathways plus utility weights.
**Output:** QALYs per patient per strategy.
**Rationale:** QALYs combine length of life and quality of life into one number, which allows
comparison across diseases. That is their strength and their weakness: the weights are
preference-based and contestable.

```text
QALYs = sum over health states of (time in state x utility of state)
```

Utility measurement:

- The EQ-5D is the most widely used instrument; NICE prefers EQ-5D-5L mapped to the 3L value set
  for adults.
- The Short Form 6D (SF-6D) and the Health Utilities Index are alternatives.
- Mapping from a disease-specific instrument to a utility is acceptable if the mapping is
  published and validated.
- Utility weights are population-specific; a United Kingdom value set applied to a United States
  population is a transferability assumption.

Where the intervention affects process rather than health, the QALY gain may be near zero even
though patients and clinicians value the change. In that case, report the QALY result and a
cost-consequence table side by side, and do not force the decision into a QALY frame.

### Step 9. Discount costs and effects

**Input:** undiscounted streams.
**Output:** present values.
**Rationale:** a cost or benefit in year five is worth less than the same amount today. The rate
must be stated, and the same rate should be applied to costs and effects in the reference case.

```text
PV = sum over t of (value_t / (1 + r)^t)
```

Conventions:

- Second Panel reference case: 3 percent real, both costs and effects.
- NICE reference case: 3.5 percent, both costs and effects, with 1.5 percent permitted only under
  specified conditions for technologies that restore near-full health for people who would
  otherwise die or have very severely impaired lives.
- ICER: 3 percent.
- State whether year one is discounted (end-of-year convention discounts year one).

Discounting is not a formality. At 3.5 percent over 20 years, a benefit is worth about half its
undiscounted value; the choice between 3 and 3.5 percent is small, but the choice between 0 and 3
percent is not.

### Step 10. Compute incremental cost and effect, then test for dominance

**Input:** discounted totals per strategy.
**Output:** Delta C and Delta E relative to the comparator.
**Rationale:** the incremental calculation is trivial arithmetic and the most common place for
sign errors.

```text
Delta C = C_intervention - C_comparator
Delta E = E_intervention - E_comparator
```

Then classify:

| Delta C | Delta E | Classification | Action |
|---|---|---|---|
| Negative | Positive | Dominant | Adopt, subject to affordability |
| Positive | Negative | Dominated | Reject |
| Positive | Positive | Trade-off | Compute ICER and compare to threshold |
| Negative | Negative | Trade-off, less effective and cheaper | ICER is positive but the direction is counter-intuitive; interpret with care |

The fourth row is a common source of confusion. A cheaper, less effective technology has a
positive ICER expressed as cost per QALY forgone, and the decision depends on whether the savings
are worth the health loss at the threshold. Report it as "cost saving per QALY forgone" to avoid
misreading.

### Step 11. Compute the ICER and the net monetary benefit

**Input:** Delta C and Delta E.
**Output:** an ICER in currency per QALY, and NMB at one or more thresholds.
**Rationale:** the ICER is the conventional currency of health technology assessment; NMB is more
robust and additive.

```text
ICER = Delta C / Delta E
NMB  = Delta E x threshold - Delta C
```

Report both. The ICER communicates to an HTA audience; the NMB supports subgroup analysis and
probabilistic results.

Edge case: when Delta E is zero or negative, the ICER is undefined or misleading. Report NMB and
a qualitative statement instead.

### Step 12. Characterise uncertainty

**Input:** the base-case model.
**Output:** a set of analyses: one-way, scenario, probabilistic, and subgroup.
**Rationale:** a point estimate is a claim of precision the model does not have. Decision makers
need to know which uncertainty matters.

| Analysis | What it does | What it answers |
|---|---|---|
| One-way sensitivity | Varies one input across a range | Which inputs drive the result |
| Tornado diagram | Ranks inputs by influence | Where to spend measurement effort |
| Scenario analysis | Replaces a structural assumption | Whether the model structure drives the answer |
| Probabilistic sensitivity analysis | Samples all inputs from distributions | Probability of cost-effectiveness |
| Subgroup analysis | Re-estimates in defined subgroups | Whether a restricted indication is better value |
| Threshold analysis | Solves for the break-even input | How wrong the input can be |

For a probabilistic analysis, assign distributions that match the parameter type: beta for
probabilities and utilities, gamma or lognormal for costs, normal for mean differences. Report
the cost-effectiveness acceptability curve and the probability of cost-effectiveness at the
relevant threshold.

### Step 13. Compute budget impact separately

**Input:** eligible population, uptake, per-patient cost, offsets, budget period.
**Output:** net budget impact per year and cumulatively.
**Rationale:** affordability is a different question, computed on cash rather than discounted
value, and over the payer's budget cycle.

The ISPOR Good Practices Task Force structure:

```text
eligible population
  x uptake in each year
  x cost per treated patient in each year
  - offsets from avoided care
  + programme fixed costs
  = net budget impact
```

Rules:

- Use undiscounted costs; budgets are not discounted.
- Model uptake as a curve, not a step.
- Include the cost of patients already on treatment in prior years, not only new starts.
- Report the peak year, not only the total, because the peak is what the budget committee must
  authorise.
- Present one-way sensitivity on uptake and price, which dominate.

### Step 14. Report to a standard and state the limits

**Input:** the completed analysis.
**Output:** a report following CHEERS 2022, plus a plain-language summary.
**Rationale:** the reporting standard is how a reviewer checks that nothing was omitted. A model
that cannot be reported to the standard is usually missing something material.

The Consolidated Health Economic Evaluation Reporting Standards (CHEERS 2022) checklist covers
title, abstract, introduction, methods, results, discussion, and other items including
perspective, horizon, discount rate, model choice, and funding. Complete the checklist even for an
internal document; it takes an hour and prevents the most common review objections.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** If no code or coverage policy exists, the payer perspective may be
empty: there is no payment to compare against. The analysis then becomes a provider cost-
consequence study, and the ICER is not the decision metric. Say so explicitly rather than
computing a QALY ratio that no one will use.

**Single-payer versus multi-payer.** In a single-payer system, one budget bears both the
intervention cost and the offset, so the ICER and the budget impact are internally consistent. In
a multi-payer system, the intervention cost may fall on a provider and the offset on a payer, or
vice versa. The analysis must name which party holds each, or it will show a benefit that no
single decision maker can capture.

**Capital versus operating budget.** Capital spending is depreciated over the asset life and
evaluated against a hurdle rate; operating spending is evaluated annually. A five-year ICER is
the right efficiency metric for either, but the budget impact must match the buyer's accounting:
capital in the purchase year, operating spread across the contract.

**Pilot-to-contract transition.** Pilot data typically show better adherence and better outcomes
than general practice. The contract-year model should use the pilot effect discounted toward
real-world performance, and should model the cost of the support that made the pilot work.

**Procurement security review.** Security requirements can add cost and delay that are not in the
clinical model. Add them to the implementation cost and to the budget impact in year one.

**Champion turnover.** If the effect depends on a champion's engagement, model a scenario in
which the champion leaves at month twelve and adherence falls. This is often the largest single
driver of the budget impact.

**Mortality effects that appear late.** A technology that reduces mortality produces QALY gains
that accrue over decades but costs that occur now. The ICER is highly sensitive to the discount
rate and the horizon; report both a 3 percent and a 1.5 percent scenario.

### 5.14 Decision table combining efficiency and affordability

The recommendation depends on both dimensions.

| ICER relative to threshold | Net budget impact | Peak year affordability | Recommendation |
|---|---|---|---|
| Below threshold | Negative or neutral | Any | Adopt; cost saving or budget neutral |
| Below threshold | Positive, within budget | Affordable | Adopt, with a phased rollout |
| Below threshold | Positive, exceeds budget | Unaffordable | Do not reject; negotiate price, restrict indication, or phase |
| At threshold | Any | Affordable | Conditional; seek price reduction or real-world evidence |
| Above threshold | Negative | Affordable | Ambiguous; the savings may justify a separate decision |
| Above threshold | Positive | Any | Do not adopt on efficiency grounds |
| Dominated | Any | Any | Reject |
| Dominant | Positive but modest | Affordable | Adopt; verify the offset assumption |

The third row is the one that most often produces a bad decision in practice: a cost-effective
technology is rejected as unaffordable without a negotiation, when a price reduction or a
risk-sharing arrangement would have resolved it.

## 6. Worked example

> **Note:** the payer and the programme below are a composite constructed for teaching. All
> figures are illustrative estimates in 2026 USD and are not reported results. Clinical parameters
> are plausible for a heart failure population but are not drawn from a specific trial.

### 6.1 The decision

A regional payer with 480,000 covered lives is considering coverage of a 90-day remote monitoring
programme for adults discharged after a heart failure admission. The programme provides a cellular
scale, a symptom questionnaire, and weekday nurse review, at a negotiated price of 340 USD per
enrolled patient plus 600,000 USD per year in programme administration.

The comparator is usual post-discharge care. The decision maker is the payer's pharmacy and
therapeutics committee, which uses a 100,000 USD per QALY benchmark and must approve any
technology with a first-year budget impact above 15 million USD at the medical policy committee.

The eligible population is 10,000 patients per year. The model horizon is five years, the
perspective is the payer, and the discount rate is 3 percent for both costs and effects.

### 6.2 Model structure and inputs

A five-state cohort model with annual cycles: stable, hospitalised, post-hospitalisation,
advanced heart failure, and death. The programme reduces the probability of hospitalisation in the
first year and modestly improves utility through symptom control.

| Parameter | Usual care | Programme | Basis |
|---|---:|---:|---|
| Year 1 cost per patient (USD) | 9,800 | 14,200 | Illustrative, payer paid basis |
| Year 2 cost | 6,400 | 8,300 | Illustrative |
| Year 3 cost | 5,200 | 6,300 | Illustrative |
| Year 4 cost | 4,200 | 4,800 | Illustrative |
| Year 5 cost | 3,400 | 3,600 | Illustrative |
| Year 1 utility-weighted survival (QALYs) | 0.620 | 0.650 | Illustrative |
| Year 2 | 0.580 | 0.610 | Illustrative |
| Year 3 | 0.550 | 0.580 | Illustrative |
| Year 4 | 0.530 | 0.555 | Illustrative |
| Year 5 | 0.510 | 0.535 | Illustrative |

The intervention costs more in every year because the programme cost is not fully offset by
avoided admissions in this parameterisation. That is the honest base case: a programme that pays
for itself in year one is unusual, and a model that shows it should be checked for a missing cost.

### 6.3 The arithmetic

Discount each stream at 3 percent with end-of-year timing.

```text
Usual care cost   = 9800/1.03 + 6400/1.03^2 + 5200/1.03^3 + 4200/1.03^4 + 3400/1.03^5
                  = 26,970 USD
Programme cost    = 14200/1.03 + 8300/1.03^2 + 6300/1.03^3 + 4800/1.03^4 + 3600/1.03^5
                  = 34,745 USD
Delta C           = 7,775 USD
```

```text
Usual care QALYs  = 2.5628
Programme QALYs   = 2.6914
Delta E           = 0.1286 QALYs
```

```text
ICER = 7,774.9 / 0.12862 = 60,442 USD per QALY
```

The division uses the unrounded values; dividing the rounded 7,775 by the rounded 0.1286 gives
60,459, a difference of 17 USD per QALY.

Against a 100,000 USD per QALY benchmark, the programme is cost-effective. Against a 50,000 USD
benchmark, it is not.

Net monetary benefit:

| Threshold (USD/QALY) | NMB per patient (USD) | Decision |
|---:|---:|---|
| 50,000 | -1,343 | Not cost-effective |
| 100,000 | 5,089 | Cost-effective |
| 150,000 | 11,520 | Cost-effective |

### 6.4 The horizon test

The same model truncated at shorter horizons:

| Horizon | Incremental cost (USD) | Incremental QALYs | ICER (USD/QALY) |
|---:|---:|---:|---:|
| 1 year | 4,272 | 0.0291 | 146,667 |
| 2 years | 6,063 | 0.0574 | 105,616 |
| 3 years | 7,069 | 0.0849 | 83,309 |
| 4 years | 7,603 | 0.1071 | 71,005 |
| 5 years | 7,775 | 0.1286 | 60,442 |

At one and two years the programme fails a 100,000 USD threshold; at five years it passes. The
cost is front-loaded and the benefit accumulates. This is a structural feature of the programme,
not a modelling artefact, and it has a commercial consequence: the payer must be willing to
commit for longer than two years, or the contract must include a mechanism that shares the
early-year cost.

### 6.5 The budget impact

Budget impact uses undiscounted cash and models each annual cohort separately. New enrolment is
4,000 patients in year one, 1,500 in year two, 1,000 in year three, and 500 in year four, for a
cumulative 7,000 of 10,000 eligible patients, a 70 percent uptake.

The per-patient incremental cost by programme year is 4,400, 1,900, 1,100, 600, and 200 USD.

| Year | New patients | Variable cost (USD) | Fixed cost (USD) | Total (USD) |
|---:|---:|---:|---:|---:|
| 1 | 4,000 | 17,600,000 | 600,000 | 18,200,000 |
| 2 | 1,500 | 14,200,000 | 600,000 | 14,800,000 |
| 3 | 1,000 | 11,650,000 | 600,000 | 12,250,000 |
| 4 | 500 | 8,150,000 | 600,000 | 8,750,000 |
| 5 | 0 | 3,750,000 | 600,000 | 4,350,000 |
| **Total** | **7,000** | **55,350,000** | **3,000,000** | **58,350,000** |

The peak year is year one at 18.2 million USD, which exceeds the 15 million USD threshold for
medical policy committee review. The technology is cost-effective and requires escalated
approval. That combination is exactly the situation Section 5.14 row three describes, and the
resolution is either a phased rollout that caps year-one enrolment at about 3,200 patients, or a
price negotiation.

Note that the budget impact total, 58.35 million USD, is not comparable to the per-patient ICER
multiplied by patients, because the ICER is discounted and the budget impact is not, and because
the budget impact includes patients enrolled in prior years.

### 6.6 Threshold price and headroom

Solving for the maximum incremental cost at each threshold:

| Threshold (USD/QALY) | Maximum Delta C (USD) | Current Delta C (USD) | Headroom |
|---:|---:|---:|---:|
| 50,000 | 6,432 | 7,775 | -1,343, price must fall 17% |
| 100,000 | 12,864 | 7,775 | +5,089, price can rise 65% |
| 150,000 | 19,295 | 7,775 | +11,520, price can rise 148% |

The headroom calculation is the most commercially useful output of a cost-effectiveness analysis.
It converts the threshold into a price ceiling, and it tells a commercial team how much room a
negotiation has before the product loses its value argument.

### 6.7 Two diagnostics that reveal a weak business case

**Diagnostic 1: the threshold swap.** The NMB per patient is -1,343 USD at 50,000 USD per QALY
and +5,089 USD at 100,000 USD per QALY. The entire conclusion depends on which benchmark the
committee uses, and the product has no control over that choice. A business case that is positive
only above 90,000 USD per QALY is a case that fails in any jurisdiction using a lower threshold,
which includes most of Europe and many United States payers. Run the ICER against three
thresholds before presenting, and if the sign changes, the product's value story must include
evidence on a non-QALY dimension, such as caregiver burden or equity.

**Diagnostic 2: the offset assumption.** Check how much of the intervention cost is assumed to be
offset by avoided admissions. If the programme cost per patient is 340 USD and the avoided
admission value is embedded in the annual cost differences, then a model in which avoided
admissions are zero has a very different result. Here, the incremental cost is positive in every
year, so the offsets are already insufficient to cover the programme. If a vendor's model shows a
negative incremental cost, ask which admission reduction produces it and whether that reduction
has been observed in a controlled setting. Unverified offsets are the largest single source of
error in digital health economic models.

A third diagnostic, not required but useful: the adherence test. Recompute the QALY gain with
adherence at 50 percent rather than the modelled level. If the ICER more than doubles, the
programme's value depends on a behaviour that the contract does not control.

### 6.8 Varying one assumption

The base case assumed an incremental QALY gain of 0.1286.

| Incremental QALYs | ICER (USD/QALY) | NMB at 100,000 USD | Decision |
|---:|---:|---:|---|
| 0.0400 | 194,376 | -3,775 | Not cost-effective |
| 0.0600 | 129,584 | -1,775 | Not cost-effective |
| 0.0800 | 97,188 | 225 | Marginal |
| 0.1000 | 77,750 | 2,225 | Cost-effective |
| 0.1286 | 60,442 | 5,089 | Cost-effective |
| 0.1600 | 48,594 | 8,225 | Cost-effective |
| 0.2000 | 38,875 | 12,225 | Cost-effective |

The decision flips at about 0.078 QALYs, which corresponds to roughly four additional weeks in
full health per patient. A QALY gain that small is within the range that different utility
instruments would produce for the same clinical effect, so the result is instrument-sensitive as
well as effect-size-sensitive.

Now vary the programme cost instead, holding the effect at 0.1286 QALYs.

| Cost multiplier | Incremental cost (USD) | ICER (USD/QALY) | NMB at 100,000 USD |
|---:|---:|---:|---:|
| 0.70 | -2,649 | dominant | 15,512 |
| 0.85 | 2,563 | 19,926 | 10,300 |
| 1.00 | 7,775 | 60,442 | 5,089 |
| 1.15 | 12,987 | 100,958 | -123 |
| 1.30 | 18,199 | 141,475 | -5,335 |

A 15 percent price increase takes the product from comfortably cost-effective to exactly at the
threshold. That is narrow headroom for a negotiated price, and it argues for contracting on a
per-patient basis with a volume commitment rather than a per-seat licence that does not scale
with the eligible population.

### 6.9 The recommendation

The analysis supports coverage with two conditions: a phased rollout that caps year-one enrolment
at 3,200 patients to stay under the 15 million USD review threshold, and a risk-sharing clause
that ties a portion of payment to observed hospitalisation at twelve months, because the QALY
gain is the input that most affects the result and the least well measured.

The economic case does not decide the coverage question. It frames it: the programme is
cost-effective at the committee's stated benchmark, unaffordable in year one at full uptake, and
sensitive to a QALY gain that the available evidence cannot pin down to better than a factor of
two.

## 7. Code, query, or calculation

The script implements discounting, ICER, NMB, horizon analysis, threshold price, one-way
sensitivity, and budget impact. Standard library only.

```python
"""Cost-utility analysis and budget impact for a heart failure monitoring programme."""

YEARS = 5
RATE = 0.03
POP = 10_000

USUAL_COST = [9800, 6400, 5200, 4200, 3400]
USUAL_QALY = [0.62, 0.58, 0.55, 0.53, 0.51]
INT_COST = [14200, 8300, 6300, 4800, 3600]
INT_QALY = [0.65, 0.61, 0.58, 0.555, 0.535]


def discount(values, rate=RATE, start=1):
    """Present value with end-of-year timing by default."""
    return sum(v / (1 + rate) ** (t + start) for t, v in enumerate(values))


def totals(cost, qaly, rate=RATE, horizon=YEARS):
    return discount(cost[:horizon], rate), discount(qaly[:horizon], rate)


def icer(dc, de):
    return dc / de if de else float("inf")


def nmb(dc, de, wtp):
    return de * wtp - dc


uc_c, uc_e = totals(USUAL_COST, USUAL_QALY)
iv_c, iv_e = totals(INT_COST, INT_QALY)
dc, de = iv_c - uc_c, iv_e - uc_e

print("DISCOUNTED TOTALS PER PATIENT (3%, years 1-5)")
print(f"  usual care cost    ${uc_c:,.0f}   QALYs {uc_e:.4f}")
print(f"  programme cost     ${iv_c:,.0f}   QALYs {iv_e:.4f}")
print(f"  incremental cost   ${dc:,.0f}   QALYs {de:.4f}")
print(f"  ICER               ${icer(dc, de):,.0f} per QALY")
for wtp in (50_000, 100_000, 150_000):
    print(f"  NMB at ${wtp:,}   ${nmb(dc, de, wtp):,.0f}")

print("\nUNDISCOUNTED TOTALS")
print(f"  incremental cost   ${sum(INT_COST) - sum(USUAL_COST):,.0f}")
print(f"  incremental QALYs  {sum(INT_QALY) - sum(USUAL_QALY):.4f}")
print(f"  ICER               ${icer(sum(INT_COST) - sum(USUAL_COST), sum(INT_QALY) - sum(USUAL_QALY)):,.0f}")

print("\nHORIZON EFFECT")
print("  horizon   inc cost   inc QALYs      ICER")
for h in range(1, 6):
    c, e = totals(INT_COST, INT_QALY, horizon=h)
    cu, eu = totals(USUAL_COST, USUAL_QALY, horizon=h)
    print(f"  {h:5d} yr  ${c - cu:9,.0f}   {e - eu:8.4f}  ${icer(c - cu, e - eu):9,.0f}")

print("\nTHRESHOLD PRICE")
for wtp in (50_000, 100_000, 150_000):
    max_dc = de * wtp
    print(f"  at ${wtp:,}/QALY  max inc cost ${max_dc:,.0f}"
          f"  headroom ${max_dc - dc:,.0f}  ({(max_dc / dc - 1) * 100:+.0f}%)")

print("\nSENSITIVITY: incremental QALY gain")
print("  dQALY     ICER      NMB at $100k")
for d in (0.04, 0.06, 0.08, 0.10, 0.1286, 0.16, 0.20):
    print(f"  {d:6.4f}  ${dc / d:9,.0f}   ${nmb(dc, d, 100_000):10,.0f}")

print("\nSENSITIVITY: programme cost multiplier")
print("  mult   inc cost       ICER      NMB at $100k")
for m in (0.7, 0.85, 1.0, 1.15, 1.3):
    c2 = discount([v * m for v in INT_COST])
    d2 = c2 - uc_c
    print(f"  {m:4.2f}  ${d2:9,.0f}  ${d2 / de:9,.0f}   ${de * 100_000 - d2:10,.0f}")

print("\nBUDGET IMPACT (undiscounted)")
NEW = [4000, 1500, 1000, 500, 0]
ANNUAL_DIFF = [INT_COST[t] - USUAL_COST[t] for t in range(5)]
FIXED = 600_000
rows = []
for yr in range(5):
    total = 0
    for cohort_year, n in enumerate(NEW):
        offset = yr - cohort_year
        if 0 <= offset < 5:
            total += n * ANNUAL_DIFF[offset]
    rows.append(total)
print("  year   new pts   variable cost   fixed cost        total")
for yr, v in enumerate(rows, start=1):
    print(f"  {yr:4d}   {NEW[yr - 1]:7,d}   ${v:12,.0f}   ${FIXED:10,.0f}   ${v + FIXED:12,.0f}")
print(f"  five-year variable ${sum(rows):,.0f}; with fixed ${sum(rows) + 5 * FIXED:,.0f}")
print(f"  peak year ${max(v + FIXED for v in rows):,.0f}")
print(f"  per treated patient, year 1 ${ANNUAL_DIFF[0]:,.0f}")

print("\nDOMINANCE")
print(f"  more costly: {dc > 0}; more effective: {de > 0}")
```

Expected output:

```text
DISCOUNTED TOTALS PER PATIENT (3%, years 1-5)
  usual care cost    $26,970   QALYs 2.5628
  programme cost     $34,745   QALYs 2.6914
  incremental cost   $7,775   QALYs 0.1286
  ICER               $60,442 per QALY
  NMB at $50,000   $-1,343
  NMB at $100,000   $5,089
  NMB at $150,000   $11,520

UNDISCOUNTED TOTALS
  incremental cost   $8,200
  incremental QALYs  0.1400
  ICER               $58,571

HORIZON EFFECT
  horizon   inc cost   inc QALYs      ICER
      1 yr  $    4,272     0.0291  $  146,667
      2 yr  $    6,063     0.0574  $  105,616
      3 yr  $    7,069     0.0849  $   83,309
      4 yr  $    7,603     0.1071  $   71,005
      5 yr  $    7,775     0.1286  $   60,442

THRESHOLD PRICE
  at $50,000/QALY  max inc cost $6,432  headroom $-1,343  (-17%)
  at $100,000/QALY  max inc cost $12,864  headroom $5,089  (+65%)
  at $150,000/QALY  max inc cost $19,295  headroom $11,520  (+148%)

SENSITIVITY: incremental QALY gain
  dQALY     ICER      NMB at $100k
  0.0400  $  194,376   $    -3,775
  0.0600  $  129,584   $    -1,775
  0.0800  $   97,188   $       225
  0.1000  $   77,750   $     2,225
  0.1286  $   60,442   $     5,089
  0.1600  $   48,594   $     8,225
  0.2000  $   38,875   $    12,225

SENSITIVITY: programme cost multiplier
  mult   inc cost       ICER      NMB at $100k
  0.70  $   -2,649  $  -20,590   $    15,512
  0.85  $    2,563  $   19,926   $    10,300
  1.00  $    7,775  $   60,442   $     5,089
  1.15  $   12,987  $  100,958   $      -123
  1.30  $   18,199  $  141,475   $    -5,335

BUDGET IMPACT (undiscounted)
  year   new pts   variable cost   fixed cost        total
     1     4,000   $  17,600,000   $   600,000   $  18,200,000
     2     1,500   $  14,200,000   $   600,000   $  14,800,000
     3     1,000   $  11,650,000   $   600,000   $  12,250,000
     4       500   $   8,150,000   $   600,000   $   8,750,000
     5         0   $   3,750,000   $   600,000   $   4,350,000
  five-year variable $55,350,000; with fixed $58,350,000
  peak year $18,200,000
  per treated patient, year 1 $4,400

DOMINANCE
  more costly: True; more effective: True
```

Two properties of the implementation matter. The `totals` function takes the horizon as a
parameter, so the horizon test is one line and cannot be omitted from the presentation. The
budget impact is computed on the undiscounted annual differences, which is the correct convention
and is different from the ICER calculation in the same script; keeping both in one file makes that
difference visible rather than accidental.

A limitation: the model has no probabilistic analysis. A proper PSA would draw probabilities,
utilities, and costs from distributions and produce a cost-effectiveness acceptability curve. At
the design stage, a one-way analysis is sufficient to identify which parameter to measure; before
a coverage submission, a PSA is expected.

## 8. Data, standards, and regulatory context

### 8.1 Reference-case conventions by jurisdiction

| Body | Perspective | Discount rate | Threshold or benchmark | Health measure |
|---|---|---|---|---|
| NICE (England) | Health and social care | 3.5% for costs and effects | 25,000 to 35,000 pounds per QALY | EQ-5D, QALY |
| ICER (United States) | Health care system, modified societal co-base | 3% | 100,000 to 150,000 USD per QALY and per evLYG | QALY and evLYG |
| Second Panel (United States) | Health care sector and societal | 3% | None; analyst supplies | QALY |
| CADTH (Canada) | Health care system | 1.5% | No fixed threshold | QALY |
| PBAC (Australia) | Health system and societal | 5% | No published threshold | QALY |

The NICE manual, PMG36, sets a maximum acceptable ICER range of 25,000 to 35,000 pounds per QALY
gained following its 2026 update, with the same 3.5 percent annual rate applied to costs and
health effects in the reference case. Earlier NICE publications cite 20,000 to 30,000 pounds, so
any citation must state which version applies. A severity modifier applies QALY weights of 1.0,
1.2, or 1.7 depending on absolute and proportional shortfall, which effectively raises the
threshold for severe conditions. Highly specialised technologies have a separate threshold of
100,000 pounds per QALY.

ICER's 2023 Value Assessment Framework uses equal-value life years gained (evLYG) as its primary
health measure, reports QALYs alongside, applies 3 percent discounting, and derives health benefit
price benchmarks at 100,000 to 150,000 USD per evLYG and per QALY, with a reporting range of
50,000 to 200,000 USD. It also conducts a five-year budget impact analysis with an access and
affordability alert.

These are conventions, not laws. A submission that ignores them will be rejected on format; a
commercial model that follows them is not thereby correct.

### 8.2 Budget impact standards

The ISPOR Good Practices Task Force report on budget impact analysis (Sullivan and colleagues,
*Value in Health*, 2014;17(1):5-14, doi:10.1016/j.jval.2013.08.2291) sets the structure used
throughout this topic: eligible population size, current and expected treatment mix, costs of each
mix, and changes in condition-related costs. It recommends a simple cost calculator where
feasible, a cohort or patient-level model where population dynamics matter, jurisdiction-specific
data, scenario-based sensitivity from the decision maker's perspective, and transparent reporting
sufficient for replication.

The key methodological difference from cost-effectiveness analysis: budget impact is undiscounted,
uses the payer's budget period, and includes only costs that the payer bears. It is a cash-flow
forecast, not a welfare calculation.

### 8.3 Utility measurement

QALYs require utility weights. The EQ-5D is the most widely used instrument and is preferred by
NICE for adults, with EQ-5D-5L data mapped to the 3L value set. The instrument has five
dimensions (mobility, self-care, usual activities, pain and discomfort, anxiety and depression)
and produces a health state that is converted to a utility through a country-specific value set.

Two consequences follow. First, utilities are not transferable across countries without an
assumption. Second, a product that improves a dimension the EQ-5D does not capture well, such as
cognitive function or caregiver burden, will show a small QALY gain even if patients value the
change highly. That is a limitation of the metric, not of the product, and the honest response is
a cost-consequence table alongside the QALY result.

### 8.4 Reporting standards

The Consolidated Health Economic Evaluation Reporting Standards (CHEERS 2022) is a 28-item
checklist covering title, abstract, introduction, methods, results, discussion, and other
elements such as funding and conflicts of interest. It replaced the 2013 statement and is
endorsed by more than a dozen journals. For a coverage submission, the relevant HTA body's own
format takes precedence, but CHEERS is the minimum.

### 8.5 Regulatory boundary

A health economic evaluation is not a regulatory submission. The Food and Drug Administration
(FDA) does not require or evaluate cost-effectiveness. A CE mark under the European Medical Device
Regulation does not imply coverage. In England, NICE appraisal is separate from Medicines and
Healthcare products Regulatory Agency (MHRA) authorisation; a device can be authorised and not
appraised, or appraised and not commissioned.

The practical consequence for a product plan: regulatory clearance, HTA recommendation, and
procurement approval are three separate gates with different evidence requirements and different
timelines. Budget for all three.

### 8.6 Where digital health differs

Digital health technologies often affect process, adherence, and access rather than disease
progression. The QALY captures little of that, which is why the ICER-PHTI framework for digital
health uses budget impact analysis as its core economic method and reports cost per user and per
user per month, and why NICE's evidence standards framework accepts cost-consequence analysis at
lower risk tiers.

A team that forces a digital workflow tool into a QALY model will usually produce a near-zero
incremental effect and an undefined or enormous ICER, which reads as "no value" when the correct
reading is "the wrong metric."

## 9. Industry practice and case evidence

### 9.1 Documented: thresholds are policy, and they move

NICE's cost-effectiveness threshold has changed over time. The manual current at the time of
writing sets a range of 25,000 to 35,000 pounds per QALY following a 2026 update, against earlier
published ranges of 20,000 to 30,000 pounds. The severity modifier introduced in the 2022 methods
review replaced the end-of-life criteria and applies a multiplier of up to 1.7 for conditions with
large shortfall.

The lesson for a commercial team is that a product's value case is partly a function of a policy
parameter outside its control. A dossier should present results at multiple thresholds so that a
reviewer in any jurisdiction can find the relevant one.

### 9.2 Documented: United States practice has no single threshold

ICER's 2023 framework retains a 100,000 to 150,000 USD per QALY and per evLYG range for its
health benefit price benchmarks, with a broader reporting range of 50,000 to 200,000 USD. An
analysis of 3,276 United States cost-effectiveness analyses found that authors increasingly cite
100,000 USD per QALY, 47.0 percent of studies in 2020-2021 against 9.1 percent in the 1990s, while
citations of 50,000 USD per QALY fell to 12.6 percent.

There is no national coverage threshold, and Medicare is statutorily prohibited from using
cost-effectiveness as a coverage criterion for certain benefits. A United States value dossier
therefore typically presents the ICER and lets the payer apply its own benchmark.

### 9.3 Documented: digital health needs a different economic frame

The ICER-PHTI framework for digital health technologies, published in 2023, uses budget impact
analysis as the core method, sets threshold pricing at 15 percent overall cost savings versus
usual care or a 3 to 1 return on investment, and reports incremental cost per user and per user
per month. That structure exists because the QALY frame fit poorly: many digital tools change
cost and process without a measurable utility gain.

Gomes, Murray, and Raftery argue in *PharmacoEconomics* (2022) that digital health interventions
have features that break standard assumptions: the product evolves during the evaluation, users
co-produce the effect, marginal cost can approach zero, and impacts diffuse beyond health. They
recommend an impact inventory and cost-consequence analysis as the reference case, with
cost-utility as complementary.

### 9.4 Documented: the reimbursement rate is not the value

The 2024 national non-facility rate for Current Procedural Terminology (CPT) code 92229, for
autonomous diabetic retinopathy screening, was reported at 45.75 USD per screening (Health
Affairs Forefront, April 2024). Whether that rate makes a screening programme viable depends on
the cost of the camera, the capture workflow, and the follow-up rate for positive findings, not on
the rate alone. A cost-effectiveness analysis of the programme would include all three; a
revenue calculation that includes only the rate will overstate the margin.

### 9.5 Inference: how payers actually use these analyses

The following is inference from the frameworks and from published coverage decisions, not a
documented finding about any specific payer.

First, payers use cost-effectiveness to set price, not to make the coverage decision. A product
that is clinically necessary will be covered at a price that satisfies the threshold, or with
restrictions. The ICER is a negotiating instrument as much as an analytic one.

Second, budget impact determines the contracting structure more often than the coverage decision.
A cost-effective product with a large first-year budget impact gets a phased rollout, a
prior-authorisation criterion, or a risk-sharing agreement, not a rejection.

Third, the analysis that changes a decision is usually the one that identifies a subgroup. A
product that is marginal across all patients is often clearly cost-effective in a high-risk
subgroup, and a restricted indication is easier to approve than a broad one.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong comparator | Comparator described as "no treatment" | Avoids the real alternative | Inflated ICER benefit | Use the standard of care |
| Perspective omitted | No perspective in the methods | Costs and benefits assigned to different parties | Incoherent result | State and justify perspective |
| Horizon truncation | Model ends where the intervention looks expensive | Benefit accrues after the horizon | Manufactured negative result | Extend horizon 50 percent as a test |
| Horizon inflation | Model extends beyond the evidence | Extrapolation drives the result | Manufactured positive result | Report the evidence boundary |
| Discount rate inconsistency | Different rates for costs and effects without justification | Arithmetic or convention error | Small but avoidable bias | Use one rate in the reference case |
| QALY for a process tool | Incremental QALYs near zero | Wrong metric for the effect | Product reads as worthless | Cost-consequence analysis alongside |
| Missing clinical review cost | Cost stack has licence only | Recurring labour omitted | ICER overstated by 2 to 3 times | Add nurse and clinician review lines |
| Unverified offsets | Avoided admissions assumed at trial effect | Generalisability error | Budget impact understated | Discount offsets toward real-world |
| Uptake set to 100 percent | Budget impact equals eligible population | Planning assumption, not forecast | Budget shock at launch | Model an uptake curve |
| Undiscounted ICER | ICER computed on undiscounted totals | Convention error | Modest overstatement over long horizons | Discount at the stated rate |
| No probabilistic analysis | Only a base case | Uncertainty hidden | Reviewer discounts the whole model | PSA before submission |
| Budget impact omitted | Only an ICER presented | Efficiency confused with affordability | Rejection at the budget committee | Compute both, always |

Two red flags are specific to this domain. The first is an ICER reported without a comparator,
which is not an ICER. The second is a budget impact that is smaller than the product of eligible
patients and per-patient cost, which usually means the uptake or the eligible population was
quietly reduced.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The comparator represents current practice | The ICER is relative | Practice varies by site | Survey actual practice |
| 2 | The effect transfers from trial to practice | Pre-launch evidence is trial-based | Adherence lower in practice | Discount the effect and report both |
| 3 | Utilities are valid for this population | QALYs require weights | Population differs from value-set sample | Use condition-specific weights in sensitivity |
| 4 | Costs are complete for the perspective | The numerator must be honest | Omitted labour or infrastructure | Compare with a bottom-up cost study |
| 5 | The horizon captures all differences | Truncation biases | Late effects excluded | Extend the horizon as a test |
| 6 | The discount rate is appropriate | Convention | Jurisdiction differs | Report at 1.5, 3, and 3.5 percent |
| 7 | Uptake follows the modelled curve | Budget impact depends on it | Supply constraints or clinician resistance | Pilot-based uptake data |
| 8 | Offsets are realised | They reduce net cost | Offsets accrue to another party | Name the party holding each offset |
| 9 | The threshold applies to this decision | Decision rule requires it | Payer uses a different benchmark | Present results at three thresholds |
| 10 | The model structure represents the disease | Structure drives results | Important states omitted | Compare with a published model |

The deepest limit is that cost-effectiveness analysis answers a question about collective
efficiency, while coverage decisions are made by committees responding to clinical need, political
pressure, budget cycles, and patient advocacy. A good analysis informs the decision; it does not
determine it. Presenting an ICER as if it settles the matter is a category error, and experienced
reviewers read it as naivety.

## 12. Exercises

### Exercise 1. Two questions

**Deliverable:** for a technology you know, write the cost-effectiveness question and the budget
impact question separately, naming the decision maker and the decision each informs.
**Criteria:** the two questions have different decision makers or different decision dates; neither
is a restatement of the other. **Time:** 30 minutes.

### Exercise 2. Comparator and perspective memo

**Deliverable:** a one-page memo defining comparator, perspective, population, and horizon for a
specified product, with a justification for each.
**Criteria:** the comparator is the actual standard of care; the perspective matches the decision
maker; the horizon is justified by the natural history. **Time:** 45 minutes.

### Exercise 3. ICER calculation

**Deliverable:** a hand calculation of incremental cost, incremental QALYs, and ICER for the
following: comparator cost 18,400 USD and 3.10 QALYs; intervention cost 25,900 USD and 3.24
QALYs, both already discounted.
**Criteria:** correct arithmetic; classification of the quadrant; a statement of the decision at
50,000 and 100,000 USD per QALY. **Time:** 20 minutes.

### Exercise 4. Quadrant classification

**Deliverable:** place six described technologies into the four quadrants and state the decision
rule for each.
**Criteria:** dominance identified correctly; the two trade-off quadrants distinguished by the
sign of incremental effect; no threshold applied to a dominated technology. **Time:** 30 minutes.

### Exercise 5. Net monetary benefit

**Deliverable:** compute NMB at three thresholds for the Exercise 3 case and identify the
break-even threshold.
**Criteria:** correct NMB; break-even threshold computed as the ICER; interpretation of the sign
change. **Time:** 30 minutes.

### Exercise 6. Budget impact model

**Deliverable:** a five-year budget impact for 25,000 eligible patients, uptake of 30, 50, 65, 70,
and 70 percent, a per-patient first-year cost of 1,200 USD declining 40 percent per year, and
fixed costs of 1.5 million USD per year.
**Criteria:** cohorts tracked separately; peak year identified; undiscounted; a sensitivity on
uptake of plus or minus 15 percentage points. **Time:** 2 hours.

### Exercise 7. Threshold price

**Deliverable:** for the Section 6 case, compute the maximum per-patient programme price at which
the ICER equals 80,000 USD per QALY, holding the effect constant.
**Criteria:** correct algebra; the answer expressed as a price, not an incremental cost; a
statement of the commercial implication. **Time:** 45 minutes.

### Exercise 8. Critique against CHEERS

**Deliverable:** a 600-word critique of a published economic evaluation against five CHEERS items
of your choice.
**Criteria:** each item quoted; the evaluation's compliance judged with evidence; at least one
substantive methodological criticism. **Time:** 2 hours.

## 13. Self-check answers

**1. Why can a cost-effective technology be rejected?**
Because cost-effectiveness answers whether the health gained justifies the price, while budget
impact answers whether the payer can fund it in the budget period. A technology can be excellent
value and still require more cash than the budget holds, especially in year one when uptake and
setup costs peak.

**2. What makes a comparator valid?**
It must be the strategy the decision maker would actually use if the technology were unavailable,
specified in enough detail to cost it. "No treatment" is valid only when no treatment is the
actual alternative.

**3. Why is the societal perspective not always preferred?**
It is the most complete for welfare analysis but it does not match any single decision maker's
budget. A payer cannot act on costs and benefits that accrue to patients, employers, and other
payers. The Second Panel recommends it as one of two reference cases; NICE and ICER use narrower
base cases.

**4. Compute the ICER for Exercise 3.** Incremental cost = 25,900 - 18,400 = 7,500 USD.
Incremental QALYs = 3.24 - 3.10 = 0.14. ICER = 7,500 / 0.14 = 53,571 USD per QALY. The
technology is cost-effective at 100,000 USD per QALY and not at 50,000 USD per QALY.

**5. What are the four quadrants and their decision rules?**
South-east, less cost and more effect: dominant, adopt. North-west, more cost and less effect:
dominated, reject. North-east, more cost and more effect: trade-off, adopt if the ICER is below
the threshold. South-west, less cost and less effect: trade-off, adopt if the saving is worth the
health forgone at the threshold.

**6. Compute NMB at 100,000 USD for Exercise 3.** NMB = 0.14 x 100,000 - 7,500 = 14,000 - 7,500 =
6,500 USD per patient. Positive, so adopt at that threshold. At 50,000 USD, NMB = 7,000 - 7,500 =
-500 USD, negative.

**7. What is the break-even threshold in Exercise 3?**
The threshold at which NMB equals zero is the ICER itself, 53,571 USD per QALY.

**8. Budget impact arithmetic for Exercise 6, year one.** 25,000 x 0.30 = 7,500 patients;
7,500 x 1,200 = 9,000,000 USD variable; plus 1,500,000 fixed = 10,500,000 USD. Year two: cohort
one at 720 USD each = 5,400,000, plus 12,500 new at 1,200 = 15,000,000, plus 1,500,000 fixed =
21,900,000 USD, which is the peak.

**9. Why does the horizon change the ICER so much in Section 6?**
Because the programme cost is concentrated in year one and the QALY gain accumulates. At one year
the ratio is 146,667 USD per QALY; at five years it is 60,442. The ratio is a function of where
the model stops, which is why the horizon must be justified by the natural history rather than by
convenience.

**10. What is the difference between the ICER and the NMB when incremental effect is negative?**
The ICER becomes negative or undefined and its interpretation reverses, which is why a negative
ICER does not mean "good value." The NMB remains well defined and correctly signs the decision:
a technology that is more costly and less effective has a negative NMB at any positive threshold.

**11. Why is a probabilistic sensitivity analysis expected before submission?**
Because a point estimate conceals the joint uncertainty across parameters. A PSA produces a
distribution of ICERs and a probability of cost-effectiveness at each threshold, which is the
form in which a committee can make a decision under uncertainty. One-way analysis identifies which
parameters matter; PSA quantifies the overall uncertainty.

**12. Rubric for Exercise 8.** Full marks require five CHEERS items quoted accurately, a judgement
on each with a specific page or table reference, at least one criticism that would change the
result if corrected, and a statement of what the authors did well. Half marks for a summary of
the paper without item-level assessment.

## 14. Interview questions

**1. What is an ICER?**
Strong: incremental cost divided by incremental effect, relative to a named comparator, in a
stated perspective and horizon, compared with a threshold. Weak: "cost per QALY" with no
comparator or perspective.

**2. Why is the comparator important?**
Strong: gives an example where a weak comparator doubles the apparent value, and explains that the
ICER is a relative measure. Weak: says the comparator should be "standard of care" without
explaining what that means operationally.

**3. What is the difference between cost-effectiveness and budget impact?**
Strong: efficiency versus affordability, discounted per-patient ratio versus undiscounted total
cash, threshold versus budget constraint, and an example where the two disagree. Weak: says budget
impact is "the total cost," which is incomplete.

**4. Why discount health effects as well as costs?**
Strong: because both are streams occurring over time and the reference case treats them
consistently; notes that some argue for a lower rate for health. Weak: says because money has a
time value, which does not explain discounting health.

**5. What perspective would you use for a payer submission?**
Strong: the payer's, with a societal analysis as supporting evidence, and explains why the two can
differ in sign. Weak: says societal because it is comprehensive.

**6. How do you handle uncertainty in an economic model?**
Strong: one-way to rank parameters, scenario for structural assumptions, PSA for decision
uncertainty, subgroup for heterogeneity, and a value-of-information framing for further research.
Weak: lists sensitivity analysis without saying what each type answers.

**7. How do you turn a threshold into a price?**
Strong: sets NMB to zero, solves for incremental cost, converts to price, and reports headroom as
a percentage. Weak: says they would price to value without the algebra.

**8. When is a QALY the wrong metric?**
Strong: when the intervention affects process, access, or caregiver burden without a measurable
utility change, and describes cost-consequence analysis as the alternative. Weak: says QALYs are
always appropriate.

**9. Why can a dominated technology still be adopted?**
Strong: identifies non-efficiency reasons such as equity, legal requirement, or clinical
necessity, and says the analysis should report the cost of that choice. Weak: says it cannot.

**10. What is the difference between transferability and generalisability?**
Strong: transferability is whether a result from one setting applies in another; generalisability
is whether trial results apply in routine practice. Weak: treats them as synonyms.

**11. How would you critique a published cost-effectiveness study?**
Strong: checks comparator, perspective, horizon, discount rate, utility source, cost completeness,
and whether the conclusions follow from the sensitivity analysis. Weak: comments on the writing.

**12. Why does NICE use a severity modifier?**
Strong: because a fixed threshold gives the same weight to a QALY regardless of how severe the
baseline condition is, and the modifier raises the acceptable threshold for conditions with large
shortfall. Weak: says it is to be fairer, with no mechanism.

## 15. Key terms

- **Budget impact analysis** — change in total payer spend over budget periods.
- **CHEERS** — reporting standard for economic evaluations.
- **Comparator** — the alternative against which a technology is assessed.
- **Cost-consequence analysis** — costs and effects listed separately.
- **Cost-effectiveness acceptability curve** — probability of cost-effectiveness by threshold.
- **Cost offset** — reduction in other costs caused by an intervention.
- **Cost-utility analysis** — cost-effectiveness with QALYs as the effect.
- **Discount rate** — rate converting future values to present value.
- **Dominant** — less costly and more effective.
- **Dominated** — more costly and less effective.
- **EQ-5D** — standard instrument for health utility.
- **evLYG** — equal-value life year gained.
- **Health technology assessment** — formal evaluation for coverage.
- **ICER** — incremental cost-effectiveness ratio.
- **Incremental cost** — cost difference between strategies.
- **Incremental effect** — effect difference between strategies.
- **Net monetary benefit** — effect times threshold minus incremental cost.
- **Net health benefit** — effect minus cost divided by threshold.
- **Perspective** — party whose costs and benefits are counted.
- **Probabilistic sensitivity analysis** — sampling inputs from distributions.
- **QALY** — quality-adjusted life year.
- **Reference case** — standardised method set for comparability.
- **Severity modifier** — threshold adjustment for severe conditions.
- **Threshold** — maximum acceptable cost per unit of health gain.
- **Time horizon** — period over which costs and effects are counted.
- **Transferability** — applicability of a result to another setting.
- **Uptake** — share of eligible patients treated.
- **Utility** — preference weight on a health state.
- **Willingness to pay** — maximum a decision maker will pay per unit of health gain.

## 16. Sources

All URLs accessed 26 October 2026.

1. National Institute for Health and Care Excellence. *NICE Health Technology Evaluations: The
   Manual* (PMG36), section 4, economic evaluation, and section 6, cost-effectiveness threshold.
   https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/
2. Sanders GD, Neumann PJ, Basu A, et al. Recommendations for conduct, methodological practices,
   and reporting of cost-effectiveness analyses: Second Panel on Cost-Effectiveness in Health and
   Medicine. *JAMA*. 2016;316(10):1093-1103. https://doi.org/10.1001/jama.2016.12195
3. Institute for Clinical and Economic Review. *2023 Value Assessment Framework*, September 2023.
   https://icer.org/wp-content/uploads/2023/09/ICER_2023_VAF_For-Publication_092523.pdf
4. Sullivan SD, Mauskopf JA, Augustovski F, et al. Budget impact analysis: principles of good
   practice. Report of the ISPOR 2012 Budget Impact Analysis Good Practice II Task Force. *Value
   in Health*. 2014;17(1):5-14. https://doi.org/10.1016/j.jval.2013.08.2291
5. Husereau D, Drummond M, Augustovski F, et al. Consolidated Health Economic Evaluation
   Reporting Standards (CHEERS) 2022 statement. *Value in Health*. 2022;25(1):3-6.
   https://www.ispor.org/heor-resources/good-practices/article/cheers-2022
6. Pearson SD, et al. ICER-Peterson Health Technology Institute value assessment framework for
   digital health technologies. *Journal of Comparative Effectiveness Research*. 2023.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10734316/
7. Gomes M, Murray E, Raftery J. Economic evaluation of digital health interventions:
   methodological issues and recommendations for practice. *PharmacoEconomics*. 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8821841/
8. Benedetto V, et al. Analytical frameworks and outcome measures in economic evaluations of
   digital health interventions. *Medical Decision Making*. 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9742632/
9. National Institute for Health and Care Excellence. *Evidence Standards Framework for Digital
   Health Technologies*, updated 2022.
   https://www.nice.org.uk/about/nice-our-programmes/evidence-standards-framework-for-digital-health-technologies
10. Neumann PJ, Kim DD. Cost-effectiveness thresholds used by study authors, 1990-2021. *JAMA*.
    2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10114019/
11. Tang PC, Wilson E, Mehrotra A. Paying for AI in healthcare. *Health Affairs Forefront*. April
    2024.
    https://www.healthaffairs.org/content/forefront/paying-artificial-intelligence-health-care-setting-right-precedent-amidst-growing-use
12. Agency for Healthcare Research and Quality. HCUP Statistical Brief #304, *Characteristics of
    30-Day All-Cause Hospital Readmissions, 2016-2020*.
    https://hcup-us.ahrq.gov/reports/statbriefs/sb304-readmissions-2016-2020.jsp

All figures in Section 6 are illustrative and constructed for teaching. Thresholds and discount
rates are attributed to the body and version that sets them; they change, and any submission
should check the current manual.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_value-quantification.md`](02_value-quantification.md)
- Next topic: [`04_reimbursement.md`](04_reimbursement.md)
- Related topic: [`07_business-models.md`](07_business-models.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
