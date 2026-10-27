# Module 05 — Experimentation, Pilots, and Evidence

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (10 lecture, 8 lab, 8 project)
> **Track fit:** Common core
> **Prerequisites:** Module 04 (estimands, counterfactuals, DiD, power concepts); basic SQL and Python
> **Tools:** Python 3.12, statsmodels, scipy, pandas, DuckDB, Jupyter; GeoLift or a synthetic geo-test generator; a planning calendar and a cost model
> **Industry question:** "What is the cheapest, fastest, most defensible design that will make this claim survive contact with a finance review?"

---

## 1. Why industry cares

Supply chain organisations spend a large share of their change budget on things nobody proved. A new slotting algorithm, a different carrier on a lane, a vendor-managed inventory (VMI) programme, a redesigned replenishment parameter set, a demand-sensing feed: each arrives with a vendor deck, a pilot, or neither. The cost is not the pilot. The cost is scaling something that did not work, or killing something that did.

Three failures recur:

| Failure | What it looks like | What it costs |
|---|---|---|
| No design | "We rolled it out in Q3 and service improved" | The improvement was seasonal; the programme is now embedded and hard to remove |
| Weak design | A pilot of 6 stores chosen because they were convenient | Effect estimate unusable; the decision is made on politics instead |
| Over-engineered design | A 12-month, 400-store trial for a $40,000 decision | Opportunity cost; the window closes before the answer arrives |

Experimentation in supply chains is harder than in digital products, and the reasons are structural. The unit of intervention is often a store, a distribution centre (DC), a lane, or a region — not a user — so there are few units and they are heterogeneous. Outcomes are measured in units, cases, and currency on weekly or monthly clocks, not clicks in real time. Interventions leak: a customer can shop at a control store, a carrier can serve both arms of a lane test, inventory can be rebalanced between clusters. And the intervention frequently cannot be withdrawn once a store has re-trained its staff.

The 2026 context adds pressure from two directions. Finance has become fluent in incrementality because trade spend and capital allocation are scrutinised line by line, so "the pilot looked good" no longer clears a stage gate. And AI vendors now sell optimisation and forecasting products whose value claims are inherently counterfactual — "our engine reduced stockouts 30%" — which means the buyer's ability to design and read an experiment is the only real protection.

This module is about producing evidence that is proportionate to the decision, cheap enough to run, and strong enough that a controller will sign it.

## 2. Learning outcomes

By the end of this module, you can:

- Translate a business claim into a testable hypothesis with a named decision, threshold, and owner.
- Choose a randomisation unit (customer, order, store, SKU, DC, lane, region) and defend the choice against interference and power constraints.
- Design an A/B, A/B/n, cluster-randomised, geo, or switchback test for a given supply chain intervention.
- Compute sample size and minimum detectable effect (MDE) for proportions, means, and clustered designs, including the design effect.
- Identify interference and spillover, estimate their direction, and choose a design that contains them.
- Design a sequential test that controls error under repeated looks, and explain why naive peeking inflates false positives.
- Distinguish marketing mix modelling (MMM) from incrementality testing and state when each is admissible evidence.
- Build and interpret an uplift model, and explain why average treatment effect is the wrong number for targeting.
- Apply the validation ladder from backtest to production and place a given claim on the evidence hierarchy.
- Appraise a vendor or analyst claim in under ten minutes using a structured checklist.
- Write an evidence plan that a finance team will accept, including cost, stopping rules, and a decision threshold.

## 3. Mental model

### 3.1 Claim, design, decision

```
CLAIM          "Replenishing twice weekly in small stores raises availability."
   │
   ├─ What exactly changes?      treatment definition, version, dose
   ├─ Compared with what?        status quo, not a strawman
   ├─ For whom?                  population, unit, geography
   ├─ Measured how?              primary metric, denominator, window
   ▼
DESIGN         unit → assignment → duration → power → analysis
   │
   ├─ Can we randomise?          at what unit? what does that cost in power?
   ├─ Can we hold the design?    compliance, contamination, seasonality
   ▼
DECISION       threshold → owner → action if above / below / ambiguous
```

The design follows from the claim, and the decision determines how much evidence is enough. A claim with no decision attached should not be tested; a decision with no threshold will be argued about regardless of the data.

### 3.2 The power triangle

```
              Precision
             /        \
            /          \
      Cost ─────────── Duration
```

You can buy any two. More units cost money or time; longer tests capture more signal but decay and contaminate. The design conversation is always about which corner to give up, and the answer depends on the decision's value and the cost of waiting.

### 3.3 The evidence ladder

```
Vendor demo  <  internal backtest  <  temporal holdout  <  external validation
             <  pilot  <  scaled deployment with monitoring
```

Move up only as far as the decision risk requires. A network design study may never need a pilot because the decision is reversible and the model is transparent. A pricing change almost always needs one because the decision is visible to customers and regulators.

## 4. Core concepts

### 4.1 From claim to hypothesis

A testable hypothesis names the metric, the direction, the size that matters, and the population.

| Claim (unusable) | Hypothesis (usable) |
|---|---|
| "The new slotting reduces pick time" | "Mean lines picked per labour hour in the 40 test DCs rises by at least 4% versus control over 8 weeks" |
| "Demand sensing improves forecast accuracy" | "Weighted absolute percentage error (WAPE) at the store-SKU-week level falls by at least 2 points over 12 weeks versus the current engine" |
| "The carrier is cheaper" | "Total cost per drop on the 120 test lanes falls by at least $1.80 with no more than a 1-point fall in on-time delivery (OTD)" |
| "The promotion paid back" | "Incremental gross profit exceeds trade spend by week 6, measured against a frozen counterfactual baseline" |

The size that matters is not the size that is statistically detectable. A 0.3% pick-rate improvement may be significant and commercially irrelevant; a 2% improvement may be commercially decisive and undetectable in the available sample. Both facts must be stated before the test.

### 4.2 Randomisation units

| Unit | Good for | Power | Interference risk | Typical constraint |
|---|---|---|---|---|
| Customer | Digital promise, delivery slot, substitution messaging | High (many units) | Low within a market, high across households | Consent, fairness of different experiences |
| Order | Packaging, carrier selection at order level | High | Low | Contamination if the same customer sees both |
| Store | Replenishment policy, layout, labour model | Moderate | Moderate (customers travel) | Cannot re-train staff mid-test |
| SKU | Pack size, facing, price | High | High (substitution within basket) | Cannibalisation contaminates control |
| DC | Automation, slotting, wave design | Low (few DCs) | Low | Very few units; huge heterogeneity |
| Lane | Carrier, mode, routing rule | Moderate | High (carrier serves both arms) | Shared equipment and drivers |
| Region | Pricing, assortment, marketing | Low (few regions) | Low | Confounded by regional shocks |

Choose the unit at which the intervention is actually implemented, then fix the power problem rather than pretending orders are independent. If the intervention is assigned to a DC, the DC is the unit, and there may be only 12 of them — which means the study is closer to a case series than an experiment, and should be reported as one.

### 4.3 A/B and A/B/n

A/B is the two-arm case. A/B/n adds arms, usually to compare several variants (three replenishment parameter sets, four carrier mixes). Three costs appear as n grows:

- **Power per arm falls.** With a fixed total sample, each arm gets less data, and the minimum detectable effect (MDE) rises roughly with √n.
- **Multiplicity rises.** Three arms against control means three comparisons; correct or pre-declare the primary contrast.
- **Operational complexity rises.** Each arm needs its own training, configuration, and exception handling, and the probability of a configuration error grows with the number of arms.

A/B/n is worth it when the variants are genuinely different hypotheses and the decision is "which one", not "does it work". It is usually a mistake when the variants are parameter tweaks that could be swept in a smaller sequential design.

### 4.4 Cluster randomisation and the design effect

When the unit of assignment is a store and the unit of analysis is an order, observations within a store are correlated. The design effect quantifies the penalty:

```
DEFF = 1 + (m − 1) · ρ
```

where m is the cluster size (orders per store) and ρ is the intraclass correlation (ICC) of the outcome within clusters.

The effective sample size is `n_effective = n_total / DEFF`, and the number of clusters required is `n_individuals_required × DEFF / m`.

**Worked example.** From §4.8, 2,551 orders per arm are required at the individual level. With 200 orders per store and ρ = 0.03:

```
DEFF = 1 + (200 − 1) × 0.03 = 1 + 5.97 = 6.97
Clusters per arm = (2,551 / 200) × 6.97 = 12.76 × 6.97 = 88.9 → 89 stores
```

Eighty-nine stores per arm, 178 in total. Had the team randomised 60 stores per arm — 12,000 orders each — the effective sample would be 12,000 / 6.97 = 1,722 per arm, and the MDE would be 2.4 percentage points rather than the 2.0 the study was designed for. The test would run, produce a null, and be read as evidence of no effect when it was simply underpowered.

Two ways to buy power back: increase m (more orders per store, which is free if stores are busy) or reduce ρ (more homogeneous clusters, which usually means narrowing the population and losing generalisability). Both trade something.

### 4.5 Geo experiments

A geo experiment assigns treatment to geographic areas — designated market areas (DMAs), metropolitan areas, or postcode clusters — and measures the aggregate response. It is the standard design when the intervention is delivered through media, pricing, or distribution reach and cannot be randomised at the individual level without contamination.

Design requirements:

| Requirement | Why |
|---|---|
| 20 or more geos, or a matched synthetic control | Fewer units means inference is fragile |
| Pre-period of at least 8 weeks for matching | Need a stable baseline relationship |
| Stable geo definitions over the test | Boundary changes destroy the panel |
| No simultaneous national activity | A national campaign contaminates all geos |
| A pre-registered test window | Post-hoc window selection invalidates inference |

Inference is usually by permutation: reassign "treatment" to each control geo in turn, build the distribution of placebo effects, and locate the observed effect in that distribution. This is honest and requires no distributional assumption, but it is low-powered with few geos.

### 4.6 Switchback designs

A switchback alternates treatment across time within the same unit: all stores get treatment in week 1, control in week 2, treatment in week 3, and so on. It is used when contamination across units is severe and the intervention can be switched quickly — pricing algorithms, dispatch rules, warehouse batching policies, delivery promise settings.

| Property | Consequence |
|---|---|
| Each unit serves as its own control | Removes between-unit heterogeneity |
| Time is the randomisation unit | Needs many switch periods for power |
| Carryover is the main threat | Need washout periods or carryover-aware models |
| Common shocks affect all units in a period | Period fixed effects absorb them |
| Sequence must be randomised, not alternating | Deterministic alternation confounds with cycle |

The analysis is a panel regression with unit and period fixed effects and treatment as the regressor, clustered by unit. If carryover is plausible — a pricing change affects demand for weeks — insert washout periods and analyse only the uncontaminated portion.

### 4.7 Pilots in stores, DCs, and lanes

A pilot is an experiment with operational constraints and political visibility. It differs from a lab test in three ways: the population is chosen for feasibility rather than representativeness, the intervention is co-designed with operators, and the results will be scrutinised by people whose work changes.

| Setting | Design pattern | Primary risks | Mitigation |
|---|---|---|---|
| Stores | Cluster-randomised, matched pairs by volume and format | Selection of enthusiastic stores; customer travel | Match on pre-period volume; randomise within pairs |
| DCs | Staggered rollout (stepped wedge) with 8–12 sites | Few units; concurrent capital projects | Pre-register the order; use site fixed effects |
| Lanes | Randomise at lane, stratify by distance and volume | Carrier serves both arms; rate changes mid-test | Randomise within carrier; lock rates for the window |
| Suppliers | Randomise purchase orders within supplier | Supplier learns and games; allocation shifts | Blind the label; pre-announce the volume band |

**The enthusiastic-store problem** is the most common pilot defect. Stores are volunteered by regional managers, so they are better staffed, better located, and more motivated than average. The pilot effect is then a mix of the intervention and the selection. Fix by randomising among stores that meet a feasibility threshold, not among stores that volunteered.

### 4.8 Power, MDE, and sample size

**Two proportions.** Equal arms, two-sided test:

```
n per arm = (z_{1−α/2} + z_{1−β})² × [p₁(1−p₁) + p₂(1−p₂)] / (p₁ − p₂)²
```

With α = 0.05 (z = 1.96) and power 0.80 (z = 0.8416):

**Worked example.** Baseline OTD = 92%; target improvement to 94%.

```
p₁ = 0.92, p₂ = 0.94
p₁(1−p₁) = 0.92 × 0.08 = 0.0736
p₂(1−p₂) = 0.94 × 0.06 = 0.0564
Sum = 0.1300
Δ = 0.02, Δ² = 0.0004
(z sum)² = (1.96 + 0.8416)² = 2.8016² = 7.849

n per arm = 7.849 × 0.1300 / 0.0004 = 1.0204 / 0.0004 = 2,551 orders per arm
```

The pooled-variance variant, using p̄ = 0.93 and 2p̄(1−p̄) = 0.1302, gives 2,555; the two forms agree to within a few orders at this effect size, which is why the exact formula matters less than stating which one was used. Total sample: 5,102 orders.

**Continuous outcome.** Equal arms:

```
n per arm = 2 (z_{1−α/2} + z_{1−β})² σ² / Δ² = 15.68 σ² / Δ²
```

**Worked example.** Cost per drop has σ = $18.00 and the decision needs a $1.50 reduction.

```
n = 15.68 × 18.00² / 1.50² = 15.68 × 324 / 2.25 = 5,080.3 / 2.25 = 2,258 drops per arm
```

**From sample to MDE.** Rearranged, MDE for a proportion with effective sample n_eff:

```
MDE ≈ (z_{1−α/2} + z_{1−β}) × √(2 p̄ (1 − p̄) / n_eff)
```

At n_eff = 1,722 and p̄ = 0.93: MDE = 2.8016 × √(0.1302 / 1,722) = 2.8016 × 0.008695 = 0.0244, or 2.4 percentage points.

**Duration, not just sample.** If 2,551 orders per arm arrive at 900 orders per week per arm, the test needs 2.84 weeks of data plus whatever washout the intervention requires. Always convert sample size into calendar time and check that the window fits inside a stable season.

### 4.9 Interference and spillover

Interference occurs when one unit's treatment changes another unit's outcome, violating the assumption that potential outcomes depend only on that unit's assignment.

| Mechanism | Example | Direction of bias |
|---|---|---|
| Substitution | Control customers buy the promoted SKU in a test store | Understates effect (control improves) |
| Physical spillover | Inventory rebalanced from control to test stores | Overstates effect |
| Carrier sharing | Same carrier serves test and control lanes, learns, reallocates equipment | Understates or overstates |
| Labour market | Test DC pays more; control DC loses staff | Overstates effect |
| Demand cannibalisation | Test region's uplift is drawn from control region | Overstates effect |
| Social/network | Customer tells a friend in a control store | Understates effect |

Three responses: **design around it** (randomise at a level above the spillover boundary, such as region), **measure it** (instrumented market-share checks in control units), or **model it** (exposure-mapping or cluster-level analysis). The common failure is to assume it away because the analysis is easier.

A useful diagnostic: if the control group's metric moves during the test in the direction of the treatment, spillover or a common shock is present.

### 4.10 Sequential testing, peeking, and false discovery

A fixed-horizon test assumes one look at one sample size. Dashboards invite continuous looking, and continuous looking at a fixed threshold inflates the false-positive rate.

| Approach | How it works | Cost |
|---|---|---|
| Fixed horizon | Pre-commit the end date; no interim decisions | Slow; cannot stop early for harm |
| Group sequential with alpha spending | Pre-plan k looks; spend α across them (O'Brien-Fleming conservative early, Pocock aggressive) | Requires planning; boundaries must be pre-computed |
| Always-valid inference | Confidence sequences valid at any stopping time | Wider intervals; lower power |
| Bayesian decision rule | Posterior probability of exceeding a threshold | Requires priors that stakeholders accept |

Five equally spaced unadjusted looks at α = 0.05 give a family-wise error rate materially above 5% — commonly around 14% in simulation [VERIFY: reproduce this by simulation in Lab 5.3 before quoting it in a client setting]. That is the mechanism behind most "we saw a significant lift at week 2" reversals.

The organisational fix is as important as the statistical one: freeze the analysis window in the pre-registration, publish interim results only as operational monitoring without a decision, and require a named approver to stop early.

### 4.11 Marketing mix modelling and incrementality

MMM regresses an aggregate outcome (sales, orders, volume) on marketing and commercial inputs (media spend, promotion depth, distribution) over time, usually with adstock and saturation transformations. It is the only practical method when no experiment is possible and the effect is spread across many channels.

| Dimension | MMM | Incrementality test |
|---|---|---|
| Unit of analysis | Market × week | Randomised geo, store, or customer |
| Identification | Model-based, with strong assumptions | Design-based |
| Strength | Covers all channels, long history, cheap | Credible causal estimate |
| Weakness | Collinearity, omitted variables, specification sensitivity | Limited to one intervention at a time; expensive |
| Admissible as | Directional evidence, budget planning | Decision evidence |

The mature pattern is **calibrate, don't choose**: run geo or holdout experiments on the largest channels, use the experimental lift to calibrate the MMM coefficients, then use the calibrated MMM to allocate across channels that cannot be tested. An uncalibrated MMM should never be the sole basis for a budget decision.

### 4.12 Uplift modelling

An uplift model estimates the conditional average treatment effect (CATE) — who responds — rather than who is likely to convert.

| Quantity | Question | Use |
|---|---|---|
| P(y=1) | Who will buy? | Ranking, not targeting |
| P(y=1 \| treated) − P(y=1 \| control) | Who buys *because* of the intervention? | Targeting, budget allocation |

Four response segments matter:

| Segment | Control outcome | Treated outcome | Action |
|---|---|---|---|
| Persuadable | Low | High | Target; the only segment that creates incremental value |
| Sure thing | High | High | Do not spend; they would have bought anyway |
| Lost cause | Low | Low | Do not spend |
| Sleeping dog | High | Low | Suppress; the intervention makes them worse |

Methods include two-model differencing (fit separate models per arm), class transformation (a single model on a transformed label), and causal forests. All require randomised or credibly identified treatment data; an uplift model fitted on observational data learns confounding, not persuasion.

Two evaluation traps: **evaluate with an uplift-specific metric** (Qini curve, uplift-by-decile) rather than AUC, and **hold out a treated and a control group in the deployment population**, because the model's ranking is only valid where the treatment prevalence matches training.

### 4.13 The validation ladder

| Rung | What it proves | What it does not prove | Typical cost |
|---|---|---|---|
| Backtest | The method fitted history | That it will fit the future | Hours |
| Temporal holdout | It generalises forward in one period | That it survives a regime change | Days |
| External or site-held-out validation | It generalises across sites | That the intervention works operationally | Weeks |
| Pilot or controlled experiment | The intervention causes the outcome | That it scales | Weeks to months |
| Scaled deployment with monitoring | It works at scale, in production, with real users | Nothing about the next regime | Continuous |

Each rung answers a different question, and skipping rungs is the standard way to be surprised. A model with a strong backtest and no pilot is a hypothesis; a pilot with no production monitoring is a snapshot.

### 4.14 Appraising vendor and analyst claims

A ten-minute checklist, applied in order:

1. **Population.** Which companies, which SKUs, which geographies, which period?
2. **Comparator.** Against the previous method, a naive baseline, or nothing?
3. **Endpoint.** Is the metric the one your decision uses, or a proxy chosen for the demo?
4. **Design.** Randomised, temporal holdout, or before/after?
5. **Constraint check.** Were capacity, case pack, minimum order quantity (MOQ), and lead time enforced in the claimed plan?
6. **Human factor.** What was the override rate, and were overrides counted as model performance?
7. **Attribution.** Who measured the benefit, and what else changed at the same time?
8. **Durability.** What happened in months 7–12, after the novelty and the vendor's attention faded?
9. **Cost.** Licence, implementation, integration, data, and internal labour over three years.
10. **Falsification.** What result would have made the vendor withdraw the claim?

Claims that fail at items 1, 2, or 4 are not evidence. They may still be a useful product demonstration.

### 4.15 An evidence plan finance will accept

Finance does not ask for statistical rigour; it asks four questions: what will change, how much will it be worth, how will we know, and what happens if we are wrong.

| Finance question | Evidence-plan element |
|---|---|
| What is the baseline? | Frozen pre-period definition, with the data source and the reconciliation to the P&L |
| What is the counterfactual? | Design, control construction, and the assumption that makes it valid |
| How big is the prize? | Effect size, unit economics, and the range, not the point |
| How will we know? | Primary metric, decision threshold, measurement owner, cadence |
| What if it fails? | Stop rule, sunk cost, and reversal cost |
| How long until we know? | Power calculation converted to calendar weeks |

A plan that answers these in two pages is worth more than a fifty-page protocol that answers none of them.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement for a valid test |
|---|---|---|---|
| Assignment log | Unit × date | Experiment platform or planning system | Immutable record of which unit got which arm, and when |
| Exposure log | Unit × event | Order, POS, WMS, TMS | Confirms the unit actually received the treatment |
| Outcome | Unit × period | POS, ERP, WMS | Same definition in both arms; no post-hoc redefinition |
| Covariates | Unit | Master data | Pre-period only, for stratification and CUPED |
| Cost | Unit or period | Finance | Fully loaded, including internal labour |
| Exclusions | Event | Analyst | Logged with reason and timestamp |

**Standards and conventions.** ASCM SCOR Digital Standard (SCOR DS) fixes the definitions of perfect order, OTD, and fill rate, so both arms are measured identically. GS1 GLN identifies stores and DCs consistently across systems; GTIN identifies items. ANSI X12 852 (product activity data) and 846 (inventory inquiry) govern partner data exchange where the test relies on retailer-supplied sell-through. ISO 3534-1 supplies the statistical vocabulary that keeps a protocol unambiguous.

**Workflow.**

1. State the claim, the decision, the owner, and the threshold.
2. Choose the unit, then compute power and convert it to calendar time.
3. Select and match units on pre-period data; check balance.
4. Write the pre-registration: primary metric, guardrails, stopping rules, analysis specification.
5. Verify instrumentation with an A/A test or a pre-period placebo run.
6. Run the test; lock the assignment log; monitor exposure but not outcomes for decisions.
7. Analyse to the pre-specified plan; report effect, interval, and segment variation.
8. Run a spillover check on the control group.
9. Report cost, and state the recommendation with a stop condition.
10. Archive assignment, exposure, code, and environment.

**Data-quality checks that commonly invalidate a pilot:** assignment log reconstructed from memory rather than recorded at launch; stores re-clustered mid-test; outcome definition changed between arms; control stores receiving the treatment through a national override; and cost lines that omit internal labour.

## 6. Industry case studies

### 6.1 The 1954 Salk polio vaccine field trial

**Context.** Poliomyelitis incidence was rising, vaccine supply was limited, and earlier observational comparisons had produced contradictory results because vaccinated children differed from unvaccinated ones in income, geography, and parental behaviour.

**Design.** A randomised, placebo-controlled, double-blind field trial across U.S. school districts, with more than 400,000 children in the placebo-controlled portion, and diagnosis by trained evaluators who did not know assignment.

**Outcome.** The trial produced a consistent estimate of protection against paralytic polio, and the design is the reason the result was accepted where observational studies had not been.

**What transferred.** Three design features remain directly applicable: randomisation at the level where contamination is controlled, a placebo or sham arm so that behaviour is held constant, and blinded outcome assessment so that the measurer cannot bias the measurement. In supply chain terms: a control store that receives the same attention and reporting cadence, and a service metric assessed from system records rather than from a manager's judgment.

**What did not transfer.** Scale and ethics. The trial was possible because the disease was severe, the population was large, and the alternative was nothing. Most supply chain interventions face a much smaller effect, a much smaller population, and a much lower cost of being wrong, which is why proportionate designs — not maximal ones — are the professional standard.

### 6.2 Microsoft and Bing online controlled experiments (2009–2013)

**Context.** Large-scale web experimentation matured at Microsoft, Google, and other platforms between 2005 and 2015, and the published practice notes became the reference for how to run tests at scale.

**Design.** Randomisation at the user level, with A/A tests to validate the platform, guardrail metrics to catch harm, and a fixed analysis horizon. The published work documents that most tested ideas fail to improve the primary metric, and that small but real effects require very large samples.

**Outcome.** The lasting contribution is methodological: the A/A test as an instrumentation check, the guardrail metric as a protection against optimising one metric into a disaster, and the observation that a large share of experiments produce null or negative results.

**What transferred.** All three practices. A/A testing maps directly onto a pre-period placebo in a store pilot. Guardrails map onto service and cost constraints that must not degrade. The expectation that most ideas fail is the antidote to pilot optimism.

**What did not transfer.** Sample size and speed. A platform can run a test on millions of users in a week; a retailer with 200 stores cannot. Supply chain designs must therefore buy power through longer windows, matched pairs, variance reduction, and cluster-level analysis, not through volume.

### 6.3 The Upworthy Research Archive (2014–2021)

**Context.** Upworthy, a media publisher, ran large numbers of headline A/B tests and in 2021 released an archive of its experiments through *Scientific Data*, creating one of the largest public corpora of real A/B tests.

**Design.** Randomised tests of alternative headlines and images, with click-through rate as the outcome, at high volume and low cost per test.

**Outcome.** The archive made it possible to study the distribution of effect sizes and the consequences of running many tests, and it became a standard cautionary dataset for multiple comparisons and for the gap between statistical significance and practical importance.

**What transferred.** Two lessons. First, high-volume testing produces a distribution of effects, most of them small, and reporting only the winners is selection. Second, a test that changes a headline in one context does not generalise to another context, which is the same failure as scaling a single-store pilot nationally.

**What did not transfer.** The metric. Click-through rate is immediate, single-stage, and cheap to measure; fill rate and cost per case are lagged, multi-stage, and expensive. A design that works at click speed does not transfer to a monthly planning cycle.

### 6.4 The Oregon Health Insurance Experiment (2008)

**Context.** Oregon expanded Medicaid to a low-income adult population but had far more eligible people than funds, so it allocated places by lottery.

**Design.** The lottery created a randomised encouragement to apply, allowing a credible comparison between lottery winners and losers on health-care use, financial outcomes, and health.

**Outcome.** The study reported increased health-care utilisation and reduced financial strain, with limited measurable improvement in physical health measures over the study window.

**What transferred.** The design pattern: when demand exceeds supply and allocation is rationed, the rationing rule can be randomised and becomes an instrument. Supply chain analogues are common — pilot allocation of constrained capacity, priority access to a new service, allocation of a limited launch quantity.

**What did not transfer.** Compliance. Lottery winners were not required to enrol, so the study estimates an intention-to-treat effect and a local effect for compliers, not the effect of insurance on everyone. Any supply chain pilot with voluntary participation faces the same dilution, and the intention-to-treat estimate is usually the one the business needs, because rollout will also be voluntary in practice.

## 7. Labs

### Lab 5.1 — Power and MDE calculator

**Objective.** Build a reusable calculator that converts a business effect into sample size, clusters, and calendar weeks.

**Data.** Course parameter file with baseline rates (OTD 92%, fill rate 96%, damage 1.4%), average weekly volume per unit, and ICC estimates by unit type.

**Steps.** (1) Implement the two-proportion and continuous-outcome formulas. (2) Add the design-effect adjustment for cluster randomisation and plot required clusters against cluster size for ρ in {0.01, 0.03, 0.10}. (3) Implement the MDE inversion. (4) Convert sample size to weeks using the volume file. (5) Produce a one-page table of five candidate designs with power, duration, and cost.

**Deliverable.** A Python module with tests, plus a one-page design comparison table.

**Acceptance criteria.** Formula reproduces 2,551 orders per arm for 92% → 94% at 80% power and α = 0.05; design effect reproduces 89 stores per arm at m = 200 and ρ = 0.03; unit tests cover both formulas and the inversion.

**Stretch.** Add a simulation-based power check that generates clustered binary data and confirms the analytic answer within Monte Carlo error.

### Lab 5.2 — Geo experiment design and placebo inference

**Objective.** Design a geo test and quantify how fragile inference is with few geos.

**Data.** Course geo panel: 40 regions × 104 weeks of weekly units, with region size and baseline correlation structure.

**Steps.** (1) Match regions into pairs or build a synthetic control for the treated set using pre-period data only. (2) Estimate the effect under a simulated 6% lift. (3) Run in-space placebos: treat each control region as if treated and record the placebo effect distribution. (4) Report the rank of the true treated effect and the implied p-value. (5) Repeat with 12 regions instead of 40 and compare.

**Deliverable.** Notebook plus a two-page note with the placebo distribution, the rank-based p-value at both region counts, and a statement of the minimum detectable effect at each.

**Acceptance criteria.** Matching uses pre-period data only; placebo procedure is correct; p-value derived from rank; the note states explicitly that the 12-region design cannot detect the simulated effect.

**Stretch.** Add a time-based placebo (fake treatment date 26 weeks earlier) and show it produces a null.

### Lab 5.3 — Peeking simulation

**Objective.** Demonstrate the inflation of false positives under repeated looks and evaluate two corrections.

**Data.** Simulated: 2,000 experiments under a true null, 500 orders per arm per week, 8 weeks.

**Steps.** (1) For each experiment, compute a p-value at every week and record whether any week crosses 0.05. (2) Report the empirical false-positive rate. (3) Implement an O'Brien-Fleming alpha-spending boundary for 8 looks and recompute. (4) Implement a Bayesian decision rule with a weakly informative prior and recompute. (5) Plot the three error rates with Monte Carlo intervals.

**Deliverable.** Notebook plus a one-page memo stating the unadjusted rate, the corrected rates, and the power cost of each correction.

**Acceptance criteria.** Monte Carlo error reported; unadjusted rate substantially above 5%; corrected rates near or below 5%; power cost quantified rather than asserted.

**Stretch.** Add a true effect of 1.5 percentage points and show how much power the corrections cost.

### Lab 5.4 — Uplift model on a store pilot

**Objective.** Estimate heterogeneous treatment effects and identify the segment where the intervention pays.

**Data.** Course VMI panel with 100 stores, 40 treated; store covariates including volume decile, format, region, and pre-period variance.

**Steps.** (1) Fit a two-model uplift estimator (separate outcome models per arm). (2) Fit a causal forest as a comparison. (3) Evaluate with a Qini curve and an uplift-by-decile table on a held-out set of stores. (4) Compute the incremental margin per store by decile and compare with the cost per store. (5) Recommend a targeting rule.

**Deliverable.** Notebook and a one-page targeting memo with the Qini curve, the decile table, and the recommended rule.

**Acceptance criteria.** Evaluation uses an uplift metric, not AUC; decile table shows incremental margin net of cost; the rule is stated in operational terms a planner can apply.

**Stretch.** Add a sleeping-dog diagnostic — stores where the estimated effect is negative — and check whether the estimate survives a bootstrap.

## 8. Assignment — Evidence plan

Produce an **evidence plan** for one supply chain intervention, formatted for a finance review. Maximum four pages plus appendices.

| Section | Required content |
|---|---|
| 1. Claim | The claim in one sentence, with the treatment version |
| 2. Decision | Decision, owner, date, and threshold for go / no-go |
| 3. Baseline | Metric definition, data source, reconciliation to the P&L, frozen pre-period |
| 4. Design | Unit, assignment, arms, duration, and why alternatives were rejected |
| 5. Power | Sample size, clusters, MDE, and calendar weeks, with the calculation shown |
| 6. Threats | Interference, contamination, seasonality, attrition, and the mitigation for each |
| 7. Analysis | Pre-specified model, clustering, primary and guardrail metrics, stopping rule |
| 8. Value | Effect size, unit economics, range, break-even, and cost of the test |
| 9. Failure handling | Stop rule, reversal cost, and what is learned if the result is null |
| 10. Appendix | Pre-registration, assignment schema, and cost model |

Grading emphasises whether the plan could be executed by someone else and whether finance would accept the value logic.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Claim and decision framing | 20% | Vague claim, no threshold | Claim testable; decision named | Claim, threshold, owner, and reversal cost all specified |
| Design choice | 25% | Picks a familiar design | Design fits the question | Alternatives compared on interference, power, and cost; choice defended |
| Power and sample size | 20% | No calculation, or wrong formula | Correct formula and arithmetic | Correct, plus design effect, calendar conversion, and sensitivity to ρ |
| Threat analysis | 15% | Lists generic risks | Names the real threats | Quantifies interference and states a mitigation with a cost |
| Finance-grade value case | 20% | Point estimate and a percentage | Range and break-even | Range, break-even, cost of the test, and a stop rule finance can enforce |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Fairness of assignment | Customers in control stores may receive worse service during a test | Cap the service gap; monitor guardrails; shorten the window |
| Consent and notice | Customer-level tests may require notice or consent under GDPR and CCPA-CPRA | Prefer store or region level; legal review for customer-level tests |
| Price testing | Differential pricing can raise antitrust and consumer-protection issues | Legal review before any price test; no competitor-signal inputs |
| EU AI Act (Regulation 2024/1689) | Automated decision systems in high-risk domains need oversight and documentation | Log the design, the assignment, and the human decision |
| Employee impact | Labour-model pilots change schedules and workload | Consult workers; report workload effects, not only cost |
| Supplier impact | Supplier pilots can shift volume and cash flow | Pre-announce volume bands; avoid tests that quietly reduce a supplier's committed volume |
| Commercial | A pilot result becomes a commitment in the business case | State the range and the assumption; do not quote the point estimate alone |

Two ethical obligations are specific to experimentation. **Do not run a test whose control arm imposes a real cost on people who did not consent** — a stockout test in a low-income neighbourhood is not the same ethical object as a headline test. And **publish null results internally**; an organisation that only records successes will keep re-running the same failed pilots.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No pre-registration | Analysis plan appears after the data | Researcher degrees of freedom | Inflated effects; unreplicable result | Freeze plan and primary metric before launch |
| Enthusiastic-store selection | Stores volunteered by managers | Selection correlated with outcome | Effect overstates by the selection term | Randomise among eligible stores |
| Clustering ignored | Orders treated as independent | Within-store correlation | False significance | Design effect; cluster-robust errors |
| Peeking | Weekly "significance" updates | Alpha inflation | Reversal after scale | Alpha spending or always-valid inference |
| Spillover unmeasured | Control group moves toward treatment | Contamination | Effect understated or overstated | Randomise above the spillover boundary; measure it |
| Underpowered test | MDE larger than the business-relevant effect | Sample too small for the decision | Null read as no effect | Power calculation before launch; state MDE |
| Definition drift | Metric definition changes mid-test | Non-comparability | Effect is an artefact | Lock definitions; reconcile to the P&L |
| Cost omission | Business case counts licence only | Internal labour and integration excluded | Negative-NPV project approved | Fully loaded three-year cost |
| Single-metric optimisation | Primary improves, guardrail degrades | Local optimisation | Service or cost damage elsewhere | Pre-declared guardrails with stop rules |
| Over-generalisation | One region's result applied nationally | Heterogeneity ignored | Effect smaller or absent at scale | Uplift analysis; staggered rollout |
| Vendor-measured benefit | Vendor reports the result | Attribution and selection | Inflated claim | Independent measurement; pre-agreed definition |
| Test too large | 12-month, 400-store trial for a small decision | Proportionality failure | Opportunity cost; window closes | Size the test to the decision |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is the first question you ask about a proposed pilot? | What decision does it change and what threshold triggers action. Weak answers start with sample size. |
| 2 | How do you choose the randomisation unit? | At the level the intervention is implemented, then fix power; consider interference and the number of available units. |
| 3 | Your pilot has 12 stores. What can you honestly conclude? | Very little inferentially; treat as a case series with a matched comparison; recommend a stepped wedge or a longer pre-period. |
| 4 | Explain the design effect to a planner. | Correlation within a store means 200 orders are not 200 independent observations; DEFF = 1 + (m−1)ρ; show the cluster count. |
| 5 | Why is peeking a problem, and what do you do about it? | Alpha inflation under repeated looks; fix with pre-committed horizon, alpha spending, or always-valid inference. |
| 6 | How would you detect spillover? | Control-group movement toward treatment, market-share checks, exposure mapping, geographic boundary analysis. |
| 7 | When is a switchback better than a cluster-randomised test? | Severe cross-unit contamination and fast switching: pricing algorithms, dispatch rules, warehouse batching. |
| 8 | What is the difference between MMM and an incrementality test? | Model-based versus design-based identification; MMM for coverage, experiments for credibility; calibrate rather than choose. |
| 9 | Why is AUC the wrong metric for an uplift model? | It ranks who will convert, not who converts because of the treatment; use Qini or uplift-by-decile. |
| 10 | A vendor claims 30% fewer stockouts. Walk me through your appraisal. | Population, comparator, endpoint, design, constraints, override rate, attribution, durability, TCO, falsification. |
| 11 | How do you write a stop rule finance will accept? | Named threshold, named metric, named date, named approver, and the sunk cost if stopped. |
| 12 | What makes a pilot result non-generalisable? | Selected population, novelty effect, season, concurrent change, voluntary compliance, and scale-dependent operations. |

## 13. Key terms

| Term | Definition |
|---|---|
| A/B test | Two-arm randomised comparison |
| A/B/n | Randomised comparison of more than two arms |
| A/A test | Both arms receive the same treatment; validates instrumentation |
| Assignment log | Immutable record of which unit received which arm and when |
| Cluster randomisation | Randomising groups rather than individuals |
| Design effect (DEFF) | Variance inflation from clustering: 1 + (m−1)ρ |
| ICC | Intraclass correlation; share of variance between clusters |
| Effective sample size | Total sample divided by the design effect |
| MDE | Minimum detectable effect at a given power and sample |
| Power | Probability of detecting a true effect of a given size |
| Alpha spending | Allocating type I error across planned interim looks |
| Peeking | Repeated unplanned looks at accumulating data |
| Guardrail metric | A metric that must not degrade, checked alongside the primary |
| Interference | One unit's treatment affecting another unit's outcome |
| Spillover | Physical or behavioural leakage between arms |
| Switchback | Alternating treatment across time within the same units |
| Stepped wedge | Staggered rollout where all units eventually receive treatment |
| Geo experiment | Randomisation at the geographic level |
| Pre-registration | Freezing the hypothesis, metric, and analysis before data collection |
| CUPED | Variance reduction using a pre-period covariate |
| MMM | Marketing mix modelling; aggregate regression of outcome on commercial inputs |
| Incrementality | The causal lift attributable to an intervention |
| Uplift model | Model estimating the conditional treatment effect |
| Qini curve | Uplift evaluation curve plotting incremental outcome against population targeted |
| Sleeping dog | Segment harmed by the treatment |
| Validation ladder | Backtest → holdout → external → pilot → production |
| Evidence hierarchy | Ranking of claim strength by design |
| Intention-to-treat | Analysis by assigned arm regardless of compliance |

## 14. Sources and further reading

**Standards and regulation**

- Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated text, EUR-Lex. Accessed 27 October 2026.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS), metric definitions. Accessed 27 October 2026.
- GS1 General Specifications (GLN, GTIN) and EPCIS 2.0. Accessed 27 October 2026.
- ANSI X12 transaction sets 846 and 852; UN/EDIFACT equivalents. Accessed 27 October 2026.
- ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1*.

**Peer-reviewed and methodological**

- Kohavi, R., Longbotham, R., Sommerfield, D., and Henne, R. M. "Controlled Experiments on the Web: Survey and Practical Guide." *Data Mining and Knowledge Discovery*, 2009.
- Kohavi, R., Tang, D., and Xu, Y. *Trustworthy Online Controlled Experiments*. Cambridge University Press, 2020.
- Athey, S., and Imbens, G. W. "The Econometrics of Randomized Experiments." In *Handbook of Economic Field Experiments*, vol. 1, 2017.
- Imbens, G. W., and Rubin, D. B. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press, 2015.
- Bojinov, I., Simchi-Levi, D., and Zhao, J. "Design and Analysis of Switchback Experiments." *Management Science*, 2023.
- Vaver, J., and Koehler, J. "Measuring Ad Effectiveness Using Geo Experiments." Google technical report, 2011.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. "A Comparison of Approaches to Advertising Measurement." *Marketing Science*, 2019.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. "Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning." *PNAS*, 2019.
- Wager, S., and Athey, S. "Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests." *Journal of the American Statistical Association*, 2018.
- Matias, J. N., Munger, K., Le Quere, M. A., and Ebersole, C. R. "The Upworthy Research Archive." *Scientific Data*, 2021.
- Francis, T., et al. "The 1954 Field Trial of Poliomyelitis Vaccine." University of Michigan, 1957. Official trial report.
- Finkelstein, A., et al. "The Oregon Health Insurance Experiment: Evidence from the First Year." *Quarterly Journal of Economics*, 2012.

**Industry and practice**

- Kohavi, R., et al. "Online Controlled Experiments at Large Scale." *KDD*, 2013.
- Gilliland, M. *The Business Forecasting Deal*. Wiley, 2010. Baseline and value-added discipline.
- Lewis, R. A., and Rao, J. M. "The Unfavorable Economics of Measuring the Returns to Advertising." *Quarterly Journal of Economics*, 2015. Why small effects are hard to measure.

**Datasets and tools**

- Meta GeoLift documentation and open-source repository. Accessed 27 October 2026.
- Google, `trimmed_match` and geo-experiment tooling. Accessed 27 October 2026.
- `statsmodels` `power` module and `scipy.stats` for sample-size calculations. Accessed 27 October 2026.
- M5 and Rossmann datasets, Kaggle, for pilot-style panels. Accessed 27 October 2026.

> **Note:** simulation results in Lab 5.3 depend on the generator, the number of looks, and the correlation structure. Report the Monte Carlo error, and never quote an error rate from a different design than the one being used.

---

> **Next:** `06_machine_learning_foundations.md` — how to build a prediction model that is calibrated, leakage-free, and tied to a decision threshold.
