---
title: "Causal Inference and A/B Experimentation for Marketing"
topic: "Estimating incremental marketing effects from randomized and observational designs"
collection: "Marketing Analytics Playbook — Cheat Sheets"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Causal Inference and A/B Experimentation for Marketing

## 1. File Header

> **Purpose:** Estimate an incremental marketing effect that survives a stated counterfactual, and convert that estimate plus its uncertainty into a bounded decision.
> **Audience:** Marketing analysts who own experiment readouts, data scientists who build the assignment and measurement pipelines, and decision makers who must act on a lift estimate.
> **Prerequisites:** Sampling distributions and confidence intervals; two-proportion hypothesis tests; SQL joins on keys with defined grain; familiarity with campaign delivery, click, and purchase event streams.
> **Key outcome:** Given an experiment or quasi-experiment specification and arm-level outcome counts, compute the intention-to-treat effect, its uncertainty under the assignment design, the economic break-even effect, and an explicit decision with a named risk.
> **Exclusions:** Auction mechanics and bid shading; media-mix regression coefficients; multi-touch attribution credit allocation; legal or privacy determinations. Those are separate decisions and separate files.
> **Level:** Core. Sections 1–7 are usable by an analyst in week one; Sections 8–10 require ownership of the measurement pipeline.
> **Version:** Draft 0.3. Not independently reviewed. All figures marked ILLUSTRATIVE are constructed for teaching and were recomputed arithmetically; no company result is claimed anywhere in this file.
> **Evidence basis:** The results in Sections 4 and 10 are derived from the stated assumptions (potential outcomes, randomization, consistency) or are closed-form standard errors. Nothing here reports an observed campaign lift.
> **Retrieval cue:** If exposed users look better than unexposed users, read Section 3 (estimand), then Section 8 F-04 (post-treatment conditioning) before you compute a single rate.

Three statements get confused constantly in marketing readouts, and they answer different questions:

1. "People who saw the ad bought more" is a description of a joint distribution. It is true, it is nearly always true, and it is not evidence of causation.
2. "Assigning people to the ad raised purchases by 2.0 percentage points" is a causal claim about a policy, with a population denominator and a comparator.
3. "The ad caused 82.5 incremental purchases across 12,000 customers" is that causal claim multiplied by a size, and it inherits every assumption of statement 2 plus a transport assumption.

This sheet is about statement 2 and the arithmetic that connects it to statement 3. The unit of analysis is always the randomized or assignable unit — customer, household, store, or geography — never the impression, never the click, never the order.

**Takeaway:** Name the intervention, the population, the outcome window, and the comparator before touching the data; a rate without those four is not an estimate.

## 2. Concept Map / Mental Model

The mental model is a funnel with three distinct stages, and errors concentrate where stages are silently merged. Assignment creates the comparison. Delivery determines what fraction of the assigned policy was actually received. Outcome measurement determines what fraction of the intended behaviour was observed. Only the first stage is under the analyst's design control; the other two are measurement and compliance problems that must be quantified, not assumed away.

```text
stage 1  ASSIGNMENT          Z in {0,1} drawn by a stated rule
         |                   invariant: Z is recorded for every eligible unit
         v
stage 2  DELIVERY            D in {0,1}; D can be zero even when Z=1
         |                   diagnostic: delivery rate by arm, by day, by channel
         v
stage 3  OUTCOME             Y measured over a declared window [0, W]
         |                   diagnostic: capture rate by arm, event-time vs ingest-time
         v
         ESTIMAND            E[Y | do(Z=1)] - E[Y | do(Z=0)] over the eligible population
```

Read the diagram as a set of constraints, not a flow of data. The invariant at stage 1 is that the assignment ledger is written before any outcome exists and is never rewritten. If the ledger is rewritten after outcomes are visible, no downstream arithmetic recovers a valid comparison — the design is gone and the file has to be re-run.

A second, orthogonal map separates the estimand from the estimator. The estimand is a property of the world; the estimator is arithmetic on a sample. Most "the test showed no effect" disputes are actually estimand disputes: the experiment identified the effect of assignment, while the stakeholder wanted the effect of exposure, and the difference between those two is the compliance rate.

```text
estimand ladder (choose one, in writing, before launch)

  ITT   effect of being assigned              always identified by randomization
  CACE  effect on units who comply            identified with an extra monotonicity assumption
  ATT   effect on the units actually treated  needs a treated-population weighting
  CATE  effect conditional on covariates      needs power in each subgroup
  ATE   population-average effect             needs the target population to be the enrolled one
```

The ladder is ordered by assumption cost, not by usefulness. ITT is cheapest because randomization is sufficient; CACE adds an exclusion restriction and monotonicity; ATT requires that the treated group's counterfactual be reconstructed; CATE requires either prespecified subgroups with adequate power or a model that is honest about its variance; ATE requires that enrolment be representative of the deployment population. The most common executive question — "how much lift would we get if we rolled this out?" — is an ATE or ATT question wearing ITT clothes, and the transport step must be labelled.

A third map is temporal, because causal claims fail on timing more often than on algebra.

```text
pre-period            assignment        exposure           outcome window        maturity
  |                      |                 |                    |                   |
covariates only    Z frozen here     D observed here     Y accumulates       Y complete
  |                      |                 |                    |                   |
eligible for        never adjust      mediator, not        truncation bias     freeze cohort,
baseline adjust     for anything      a covariate          if read early       then analyse
```

The vertical line at assignment is a wall. Anything left of it may be a covariate; anything right of it is a consequence of the intervention and adjusting for it changes the estimand at best and induces bias at worst. Section 8 F-04 works a numeric case where conditioning on a right-of-the-wall variable reverses the sign of the measured effect while the true effect is exactly zero.

**Takeaway:** Draw assignment, delivery, outcome, and the assignment wall on one page for every test; disagreements about results are usually disagreements about which stage moved.

## 3. Foundational Definitions

### 3.1 Unit, population, and eligibility

The **unit** is the entity that is assigned, and it is the entity over which the analysis must sum. If the assignment service randomizes customer IDs but the outcome pipeline returns one row per order, the analyst must aggregate orders to customers before comparing arms, or the arms are weighted by purchase frequency and the comparison silently becomes a per-order comparison.

The **target population** is the set of units to which the eventual decision applies. It is defined by eligibility rules evaluated strictly before assignment: tenure, region, prior purchase history as of `T0`, consent state, and inventory availability. Eligibility computed with data that arrives after assignment contaminates the population definition, because the treatment itself can move those fields.

The **assignment unit** and the **inference unit** need not be identical, but the mismatch must be chosen, not accidental. Assigning at household level and inferring at customer level understates variance; assigning at customer level and inferring at household level is simply wrong. Section 4.3 gives the design-effect arithmetic for the cluster case.

### 3.2 Potential outcomes and the fundamental problem

Each unit `i` has two potential outcomes: `Y_i(1)` under assignment to treatment and `Y_i(0)` under assignment to control. The individual effect `Y_i(1) - Y_i(0)` is never observed, because only one assignment happens. Everything estimable is an average over units, and every average requires an assumption that makes the assigned groups comparable.

Three assumptions carry the whole structure, and each has an operational test:

- **Consistency.** The observed outcome equals the potential outcome for the assignment actually received, `Y_i = Z_i Y_i(1) + (1 - Z_i) Y_i(0)`. It fails when "the treatment" is not one thing — for example when the offer creative, landing page, and discount change mid-flight, so assignment to `Z=1` no longer maps to a single intervention.
- **Exchangeability (ignorability).** `(Y_i(1), Y_i(0))` is independent of `Z_i`. Randomization delivers this in expectation. In observational data it is an assumption, not a finding, and it is only ever *conditionally* plausible given measured covariates.
- **Positivity (overlap).** `0 < P(Z_i = 1 | X_i) < 1` for every covariate stratum that matters. It fails when a segment is deterministically targeted, so no comparable control exists; no weighting scheme creates units that were never allowed into the other arm.

A fourth assumption, **SUTVA** (no interference and no hidden versions of treatment), is the one marketing breaks most often. Shared budgets, auction competition, and household device sharing all mean one unit's assignment changes another unit's outcome.

### 3.3 Assignment, delivery, exposure, outcome

These four are different random variables and conflating them is the single largest source of inflated lift numbers.

| Term | Symbol | Definition | Typical source | Failure if confused |
|---|---|---|---|---|
| Assignment | `Z` | Arm chosen by the randomization rule | Assignment ledger | Nothing downstream is comparable |
| Delivery | `D` | The message was actually sent or served | Send log, ad server | Undercounts ITT, hides suppression bugs |
| Exposure | `E` | The message was plausibly seen | Viewability, session join | Selection on attention |
| Outcome | `Y` | Behaviour in the declared window | Orders, CRM, store feed | Denominator drift |

Assignment is the only one that randomization makes independent of potential outcomes. Exposure is a post-treatment variable: it depends on `Z` and on latent intent, so conditioning on it selects a different population in each arm. Section 8 F-04 quantifies that selection with numbers where the true effect is zero.

### 3.4 Estimands in operational language

**ITT (intention-to-treat)** is the mean difference in outcomes between all units assigned to treatment and all units assigned to control, regardless of whether they received anything. It answers: what happens to the business if we adopt this assignment policy? It is the default because it is the only estimand that randomization identifies without extra assumptions.

**CACE (complier average causal effect)** rescales ITT by the compliance difference. If 85% of the treated arm receives the offer and 0% of controls do, `CACE = ITT / 0.85`. It answers: what would the offer do to a unit that actually takes it up? It requires monotonicity (no unit is discouraged by assignment) and an exclusion restriction (assignment affects outcomes only through receipt).

**ATT (average effect on the treated)** weights stratum effects by the treated distribution. It differs from ATE whenever treatment propensity correlates with baseline outcome level, which is normal in targeting.

**CATE (conditional average treatment effect)** is a function of covariates. It is only as good as its power in the regions where it is reported; a subgroup with 300 units and a 1.0-point effect has a standard error near 2.6 points, so the "winning segment" is usually noise.

### 3.5 Design vocabulary

**MDE (minimum detectable effect)** is the true effect size at which the test has the stated power, conventionally 80%, at the stated alpha. It is a property of sample size, baseline rate, and design — not of the observed result.

**Power** is the probability of rejecting the null when the MDE-sized effect is true. It says nothing after the fact about whether a non-significant result proves absence of effect.

**SRM (sample ratio mismatch)** is a statistical test of realized allocation against the design allocation, computed *within* randomization stratum. A globally significant SRM test on a design with deliberately unequal stratum probabilities is a false alarm; Section 5 shows exactly that case with a chi-square of 10.8.

**Sequential monitoring** is any rule that lets the analyst look at accumulating data and stop. Unadjusted repeated looks inflate the type-I error; Section 9 gives the arithmetic and the alpha-spending correction.

**Sensitivity analysis** quantifies how strong an unmeasured confounder would have to be to overturn the conclusion. The E-value in Section 4.3 is the compact version for ratio-scale outcomes.

**Takeaway:** Every one of these terms names a decision (which denominator, which assumption, which population); the definitions matter because they change the number, not because they are terminology.
## 4. Core Formulas with Full Derivation

All derivations below use the same five-part anatomy: **FORMULA** (the equation and its symbol definitions), **WHERE** (the data that must exist for it to be computable), **CONSTRAINT** (the assumption that makes it valid), **NUMERICAL EXAMPLE** (arithmetic shown at each step), **INTERPRETATION** (what a practitioner may and may not say). Each subsection closes with an explicit validity condition. Numbers marked ILLUSTRATIVE are constructed.

### 4.1 Base case — the two-arm ITT difference and its standard error

**FORMULA**

```text
Let  n1, n0     = assigned units in treatment and control arms   [count, 1..N]
     y1, y0     = units with the outcome in each arm             [count, 0..n]
     p1 = y1/n1, p0 = y0/n0                                      [proportion, 0..1]

ITT   = p1 - p0                                                 [proportion-difference]
Var   = p1(1-p1)/n1 + p0(1-p0)/n0                               [variance of difference]
SE    = sqrt(Var)
CI95  = ITT +/- 1.959964 * SE                                   [Wald interval]
```

`p1` and `p0` are arm-specific outcome rates over *assigned* units. The variance expression uses the normal approximation to the binomial separately inside each arm and adds them because the arms are independent samples. The coefficient 1.959964 is the two-sided 95% normal quantile.

**WHERE** The assignment ledger with one row per unit and columns `unit_id`, `stratum`, `arm`, `assigned_at`; the outcome table with one row per unit and `outcome_flag`, `event_at`. The join is a left join from the ledger so that zeros are counted, not dropped. Both counts must come from the same maturity cut-off timestamp.

**CONSTRAINT** Independence between arms (no shared budget, no household spillover), independent outcomes within arm (no clustering), and a fixed analysis horizon declared before data access. If outcomes are clustered, the variance formula understates `SE` by `sqrt(DEFF)` with `DEFF = 1 + (m-1)rho` — Section 4.3.

**NUMERICAL EXAMPLE** ILLUSTRATIVE. 12,000 eligible customers, 6,180 assigned to the personalized offer and 5,820 to standard email; 329 and 250 purchasers respectively within 14 days.

```text
p1 = 329 / 6180 = 0.0532362
p0 = 250 / 5820 = 0.0429553
ITT = 0.0532362 - 0.0429553 = 0.0102809   -> 1.028 percentage points

Var = 0.0532362*0.9467638/6180 + 0.0429553*0.9570447/5820
    = 0.0504021/6180 + 0.0411102/5820
    = 0.0000081557 + 0.0000070636
    = 0.0000152193
SE  = sqrt(0.0000152193) = 0.0039012   -> 0.3901 pp

CI95 = 0.0102809 +/- 1.959964*0.0039012
     = 0.0102809 +/- 0.0076462
     = [0.0026347, 0.0179271]   -> [0.26 pp, 1.79 pp]

z    = 0.0102809 / 0.0039012 = 2.6353  (two-sided p = 0.0084)
```

Every intermediate is shown because rounding at the wrong step changes the third digit: `0.3901` rounded to `0.39` gives a CI upper bound of `1.793` instead of `1.793`, but rounding `SE` to `0.4` gives `1.81`, a visible difference at the two-decimal reporting precision that executives read.

**INTERPRETATION** The assignment policy raised the 14-day purchase rate by 1.03 points, with a 95% interval excluding zero. That sentence is licensed *only* on assigned units. It is not licensed to say the offer "works for people who see it", and it is not licensed to say "2.4% lift" without stating that the relative figure uses `p0` as the denominator: `0.0102809/0.0429553 = 23.9%` relative, which sounds far larger than 1.03 points and is the same estimate.

**Validity condition:** valid iff each arm's outcomes are independent Bernoulli draws with no cross-arm interference, the two counts cover the same window, and the analyst declared this contrast and this horizon before seeing outcomes. Any deviation requires a different variance estimator, not a caveat added after the fact.

### 4.2 General case — stratified assignment with unequal probabilities

Marketing randomization is rarely simple. Targeting sends high-value segments to treatment with probability 0.6, mid-value with 0.5, and low-value with 0.4. In that design the unweighted difference is biased for the population effect because the arm composition differs by stratum.

**FORMULA**

```text
Let  s = 1..S            strata defined on pre-assignment covariates
     N_s                 eligible units in stratum s        [count]
     N = sum_s N_s       total eligible                       [count]
     w_s = N_s / N       population weight                    [-]
     p1_s, p0_s          arm rates within stratum s
     n1_s, n0_s          arm counts within stratum s

ITT_strat = sum_s w_s (p1_s - p0_s)                          [weighted average]
Var_strat = sum_s w_s^2 [ p1_s(1-p1_s)/n1_s + p0_s(1-p0_s)/n0_s ]
```

The estimand being computed is the **ATE over the eligible population**, so the weights are population shares, not arm shares. Using treated-arm shares computes an **ATT** instead; using control shares computes yet another quantity. All three are legitimate and all three give different numbers — see the numeric example.

**WHERE** The ledger needs a `stratum` column assigned before randomization, plus the prespecified stratum probabilities `pi_s`. The outcome join is identical to the base case. Strata must be defined on fields that exist at `T0`; a stratum built from the customer's outcome-adjacent behaviour is a post-treatment variable.

**CONSTRAINT** Within-stratum randomization with known `pi_s`, and enough units per stratum that the normal approximation holds (rule of thumb: at least 5 expected outcomes and 5 expected non-outcomes per arm per stratum). Strata with sparse outcomes must be collapsed before the weights are applied, or the variance explodes.

**NUMERICAL EXAMPLE** ILLUSTRATIVE. Three strata, `N = 4800 + 4200 + 3000 = 12000`.

| Stratum `s` | `N_s` | `pi_s` | `n1_s` | `y1_s` | `p1_s` | `n0_s` | `y0_s` | `p0_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High value | 4800 | 0.60 | 2880 | 187 | 0.0649306 | 1920 | 108 | 0.0562500 |
| Mid value | 4200 | 0.50 | 2100 | 99 | 0.0471429 | 2100 | 84 | 0.0400000 |
| Low value | 3000 | 0.40 | 1200 | 43 | 0.0358333 | 1800 | 58 | 0.0322222 |

```text
naive pooled:  n1 = 6180, y1 = 329 -> p1 = 0.0532362
               n0 = 5820, y0 = 250 -> p0 = 0.0429553
               naive ITT = 0.0102809  (1.0281 pp)
               naive Var = 0.0000152193, SE = 0.0039012

weights: w = (0.40, 0.35, 0.25)                         [4800/12000 etc.]
stratum effects: d_s = (0.0086806, 0.0071429, 0.0036111)
weighted ITT = 0.40*0.0086806 + 0.35*0.0071429 + 0.25*0.0036111
             = 0.0034722 + 0.0025000 + 0.0009028
             = 0.0068750                          -> 0.6875 pp

stratum variances: v_s = p1(1-p1)/n1 + p0(1-p0)/n0
   s=1: 0.0649306*0.9350694/2880 + 0.05625*0.94375/1920
      = 0.00002108145 + 0.00002764893 = 0.00004873038
   s=2: 0.0471429*0.9528571/2100 + 0.0400000*0.9600000/2100
      = 0.00002139067 + 0.00001828571 = 0.00003967638
   s=3: 0.0358333*0.9641667/1200 + 0.0322222*0.9677778/1800
      = 0.00002879109 + 0.00001732442 = 0.00004611551
Var_strat = 0.16*0.00004873038 + 0.1225*0.00003967638 + 0.0625*0.00004611551
          = 0.00000779686 + 0.00000486036 + 0.00000288222
          = 0.00001553944
SE_strat  = 0.00394201     -> 0.3942 pp
CI95      = 0.0068750 +/- 1.959964*0.00394201 = [-0.000851, 0.014601]
```

The naive estimate is 1.028 points and significant (`z = 2.635`); the correctly weighted estimate is 0.688 points and not significant at 95% (`z = 1.744`, two-sided `p = 0.081`). The gap is not noise — it is Simpson-type composition: the high-value stratum received treatment more often *and* has the highest baseline rate, so the pooled comparison credits part of the segment mix to the offer. Recovering the treated-arm weighting instead gives `0.006558` (0.66 pp), and reweighting control to treated shares gives `0.007174` (0.72 pp); three defensible estimands, three different decisions under a 0.7-point break-even threshold.

**INTERPRETATION** The number an executive hears must be paired with its weighting. "Lift of 0.69 points for the eligible population" and "lift of 1.03 points among the assigned mix" are both true statements about the same data and support different rollout decisions. The reporting template must print the weights.

**Validity condition:** valid iff strata are pre-assignment, probabilities are known rather than estimated from realized counts, and no stratum is so sparse that its arm rate is 0 or 1. If a stratum is degenerate, collapse it and renormalize, documenting the merge.

### 4.3 Variants — four corrections that change the decision

#### 4.3a Clustered assignment (design effect)

When assignment happens at store, household, or geography level, the effective sample size is not the unit count.

```text
FORMULA      DEFF = 1 + (m - 1) * rho
             SE_cluster = SE_iid * sqrt(DEFF)
             n_eff = n_total / DEFF
WHERE        m   = units per cluster [count]
             rho = intra-cluster correlation of the outcome [-]
CONSTRAINT   equal cluster sizes; unequal sizes require the
             more general 1 + (CV_m^2 + 1)*m_bar*rho - 1 form
```

ILLUSTRATIVE with `m = 50` households per block and `rho = 0.02`: `DEFF = 1 + 49*0.02 = 1.98`. Using the 12,000 units from Section 4.1, `n_eff = 12000/1.98 = 6061` and `SE` becomes `0.0039422*sqrt(1.98) = 0.005547` (0.55 pp). The 95% interval on a 0.69-point effect then runs from `-0.40` to `+1.78` points. Raising `rho` to `0.05` with the same block size gives `DEFF = 3.45` and `SE = 0.007322` (0.73 pp), at which point a 0.69-point effect is indistinguishable from zero regardless of how many customers are enrolled. The lever is block size, not budget: halving `m` to 25 at `rho = 0.05` gives `DEFF = 2.20`.

#### 4.3b Pre-period covariate adjustment (CUPED)

Subtracting a pre-period proxy scaled by its regression coefficient removes variance without touching the estimand, provided the proxy is measured before assignment.

```text
FORMULA      Y* = Y - theta * (X - E[X]),   theta = Cov(Y,X)/Var(X)
             Var(Y*) = Var(Y) * (1 - rho^2)
             rho = Corr(Y, X)
CONSTRAINT   X must be pre-assignment and unaffected by Z; theta estimated
             out-of-sample or with the appropriate degrees-of-freedom correction
```

ILLUSTRATIVE: a 90-day pre-period purchase count correlates at `rho = 0.5` with the 14-day outcome. `Var` falls by factor `1 - 0.25 = 0.75`, so `SE` falls by `sqrt(0.75) = 0.8660`. Applied to the clustered `SE` of 0.005547, the combined `SE` is `0.005547*0.8660 = 0.004804` (0.48 pp). At `rho = 0.7` the factor is `0.7141` and the combined `SE` is 0.003961. The trap: if `X` is measured after assignment — "pre-period" defined as the 90 days before the *outcome* rather than before *assignment* — CUPED absorbs part of the treatment effect and biases the estimate toward zero.

#### 4.3c Difference-in-differences for non-randomized rollouts

When a region gets the campaign and a comparable region does not, the estimator uses two differences so that a common time shock cancels.

```text
FORMULA      DiD = (p_T,post - p_T,pre) - (p_C,post - p_C,pre)
             Var = sum over the four cells of p(1-p)/n
CONSTRAINT   parallel trends: absent treatment, the treated region's change
             would have equalled the control region's change
```

ILLUSTRATIVE with 1,000 units per cell: treated `10.0% -> 13.0%`, control `10.0% -> 11.0%`.

```text
DiD = (0.13 - 0.10) - (0.11 - 0.10) = 0.03 - 0.01 = 0.02  (2.00 pp)
Var = 0.13*0.87/1000 + 0.10*0.90/1000 + 0.11*0.89/1000 + 0.10*0.90/1000
    = 0.00011310 + 0.00009000 + 0.00009790 + 0.00009000
    = 0.00039100
SE  = 0.019774   -> 1.977 pp
CI95 = 0.020 +/- 1.959964*0.019774 = [-0.01876, +0.05876]
```

The point estimate of 2.00 points looks identical to a randomized 2-point effect but its interval is five times wider at the same cell size, because DiD variance comes from four noisy cells rather than two. More importantly: if the treated region was already trending at `+0.5` pp per period for reasons unrelated to the campaign, the true campaign effect is `2.0 - 0.5 = 1.5` pp and the estimator is biased by the entire trend violation. Parallel trends is not testable with two periods; it is only falsifiable with pre-period data showing the trends tracked before treatment.

#### 4.3d Compliance rescaling (CACE) and the dilution from leakage

```text
FORMULA      CACE = ITT / (P(D=1|Z=1) - P(D=1|Z=0))
CONSTRAINT   monotonicity (no defiers) and exclusion (Z affects Y only via D)
```

ILLUSTRATIVE: ITT of 2.0 points, delivery 85% in treatment and 0% in control, gives `CACE = 0.020/0.85 = 0.02353` (2.35 pp). If 5% of controls also receive the offer through a suppression failure, the denominator becomes `0.85 - 0.05 = 0.80` and `CACE = 0.0250` (2.50 pp). With 40% delivery the rescaling yields 5.00 points — but at that point the exclusion restriction deserves scrutiny, because assignment to a mostly-undelivered campaign may itself change behaviour (inventory holds, pricing changes) through channels other than delivery.

**Validity condition for all of 4.3:** each correction changes the variance, the estimand, or both. State which. A design effect changes variance only; CUPED changes variance only if `X` is pre-assignment; DiD changes the identifying assumption; CACE changes the estimand from assignment effect to complier effect. Never present a corrected number without naming which of these happened.
## 5. Worked Numerical Example

### 5.1 The decision and the specification

A subscription retailer runs a **14-day personalized-offer test** against its standard email policy. ILLUSTRATIVE throughout: the customer counts, purchase counts, cost per contact, and contribution margin are constructed for teaching and were recomputed by hand below. No company result is represented.

| Specification item | Value in this example |
|---|---|
| Decision | Adopt the personalized-offer policy for the next quarter, or keep standard email |
| Eligible population | Active subscribers, tenure ≥ 30 days, purchasable in the US, consent on file |
| Unit | Customer (household IDs collapsed first, so one row per household) |
| Assignment | Hash of `household_id` salted with `test_id`, 50/50 within three value strata |
| Comparator | Standard email policy, same send day, same suppression list |
| Outcome | Any purchase with `event_at` in `[assigned_at, assigned_at + 14 days)` |
| Maturity cut-off | `assigned_at + 21 days`, allowing seven days of ingest lag |
| Primary estimand | ATE over eligible population, stratified weights |
| Secondary estimand | CACE at observed delivery (85% treatment, 0% control) |
| Economic inputs | Contact cost \\$0.25 per assigned customer; contribution margin \\$18 per purchase |

The specification is written before the data is touched because three of its rows are the ones analysts silently change when the result disappoints: the comparator (swapping to "no email at all" inflates the effect), the maturity cut-off (extending the window for the treated arm only), and the estimand (switching to CACE when ITT is null).

### 5.2 The counts

Assignment ran over 12,000 households. Realized allocation was 6,180 treatment and 5,820 control, which is a 51.5/48.5 split rather than the designed 50/50.

```text
FORMULA      chi2_SRM = sum_arm (observed - expected)^2 / expected,  df = 1
WHERE        expected = 6,000 per arm under a 50/50 design
NUMERICAL    (6180-6000)^2/6000 + (5820-6000)^2/6000
             = 32400/6000 + 32400/6000
             = 5.40 + 5.40 = 10.80
             p = P(chi2_1 > 10.80) = 2*(1 - Phi(sqrt(10.80)))
               = 2*(1 - Phi(3.2863)) = 0.00102
INTERPRETATION  A global SRM test rejects at 0.1%, but this design is NOT 50/50
             globally: stratum probabilities are 0.60 / 0.50 / 0.40.
```

The global test is the wrong test, and this is the most common SRM false alarm in marketing. Expected counts must be computed within stratum:

```text
stratum       N_s    pi_s   E[n1] = N_s*pi_s   observed n1   chi2 contribution
High value    4800   0.60   2880               2880          (0)^2/2880  = 0
Mid value     4200   0.50   2100               2100          (0)^2/2100  = 0
Low value     3000   0.40   1200               1200          (0)^2/1200  = 0

total chi2 = 0.000, df = 3, p = 1.000
```

The assignment service is behaving exactly as designed. The 180-unit global imbalance is the *expected consequence* of unequal stratum probabilities combined with real stratum sizes — a mathematical artifact, not a defect. Had the analyst stopped at the global test, the correct test would have been abandoned or, worse, the data re-randomized.

### 5.3 The unadjusted estimate

```text
n1 = 2880 + 2100 + 1200 = 6180      y1 = 187 + 99 + 43 = 329
n0 = 1920 + 2100 + 1800 = 5820      y0 = 108 + 84 + 58 = 250

p1 = 329/6180 = 0.0532362
p0 = 250/5820 = 0.0429553
ITT_naive = 0.0102809                       -> 1.0281 pp

Var = 0.0532362*0.9467638/6180 + 0.0429553*0.9570447/5820
    = 0.0000081557 + 0.0000070636 = 0.0000152193
SE  = 0.0039012                             -> 0.3901 pp
CI  = [0.0026347, 0.0179271]                -> [0.26 pp, 1.79 pp]
z   = 2.6353        two-sided p = 0.0084
```

An unstratified readout therefore reports a statistically significant 1.03-point lift and the team starts drafting the rollout memo. Section 5.4 is why that memo would have been wrong.

### 5.4 The stratified estimate

```text
w_s      = (0.40, 0.35, 0.25)
d_s      = (0.0086806, 0.0071429, 0.0036111)          [p1_s - p0_s]
ITT_strat= 0.40*0.0086806 + 0.35*0.0071429 + 0.25*0.0036111
         = 0.00347222 + 0.00250000 + 0.00090278
         = 0.00687500                                  -> 0.6875 pp

v_s (stratum variance of the difference):
  s=1: 0.00004873038
  s=2: 0.00003967638
  s=3: 0.00004611551
Var  = 0.16*0.00004873038 + 0.1225*0.00003967638 + 0.0625*0.00004611551
     = 0.00001553944
SE   = 0.00394201                                      -> 0.3942 pp
CI   = [-0.000851, 0.014601]                           -> [-0.09 pp, 1.46 pp]
z    = 1.7438        two-sided p = 0.0812
```

Two properties deserve attention. First, the standard error barely changed (0.3942 versus 0.3901) while the point estimate fell by a third — the correction is about bias, not precision. Second, the sign of the decision flipped: the interval now includes zero and the lower bound is slightly negative.

The three estimands that a real analysis must choose between, computed from the same table:

```text
ATE  (population weights)      = 0.0068750   (0.69 pp)
ATT  (treated-arm weights)     = 0.0065579   (0.66 pp)
  recompute: p1_std = sum_s (n0_s/n0)*p1_s
           = 0.329897*p1_1 + 0.360825*p1_2 + 0.309278*p1_3
           = 0.0214204 + 0.0170103 + 0.0110825 = 0.0495132  [4.95132%]
           ATT = 0.0495132 - 0.0429553 = 0.0065579
ATE via IPW (check)           = 0.0068750
  per stratum: w_s*(p1_s - p0_s) with weights on arm-specific means
  equals the weighted sum above because stratum probabilities are common
  within stratum across arms; it differs from ATT because the weights differ
```

The ATE-versus-ATT distinction is not academic here. The rollout decision applies to the eligible population (ATE), but a retention team that only ever touches high-value customers would be making an ATT decision and should use the 0.66-point figure. Section 8 F-11 catalogues the case where an analyst computes ATE weights but describes the result as if it applied to the treated, which overstates the benefit to the customers most likely to be targeted.

### 5.5 Design-quality corrections and what they do to power

The interval above assumes independent Bernoulli outcomes. Two realistic violations apply: household-level correlation among cohabiting customers, and the availability of a pre-period purchase count.

```text
cluster effect:  m = 50 cohabiting households per block, rho = 0.02
  DEFF  = 1 + 49*0.02 = 1.98
  SE_c  = 0.00394201 * sqrt(1.98) = 0.0055469        -> 0.55 pp
  CI_c  = [0.0068750 - 1.959964*0.0055469, ... ]
        = [-0.003997, 0.017747]                     -> [-0.40 pp, 1.77 pp]
  n_eff = 12000/1.98 = 6060.6   (half the enrolments do the work of 6,061)

CUPED with pre-period purchase count, rho_YX = 0.5
  Var factor = 1 - 0.25 = 0.75,  SE factor = 0.86603
  SE_cc = 0.0055469 * 0.86603 = 0.0048038           -> 0.48 pp
  CI_cc = [-0.002539, 0.016290]                     -> [-0.25 pp, 1.63 pp]
```

CUPED recovers roughly three quarters of the variance that clustering cost, and the combined interval is still wider than the naive one. This is the normal outcome: a properly designed and analysed test usually has a *wider* interval than the naive readout, because the naive readout omitted real sources of variance.

### 5.6 Economic break-even and the decision

```text
cost per assigned customer      c = $0.25
contribution per purchase       v = $18.00
break-even ITT                  t* = c/v = 0.25/18 = 0.0138889   -> 1.389 pp

projected incremental purchases at N = 12,000 (holds population fixed):
  point: N*0.0068750 = 82.5 purchases
         revenue 82.5*18 = $1,485 ; spend 12,000*0.25 = $3,000 ; net = -$1,515
  lower: N*(-0.000851) = -10.2 purchases
         revenue -$183.86 ; spend $3,000 ; net = -$3,183.86
  upper: N*(0.014601) = 175.2 purchases
         revenue $3,153.86 ; spend $3,000 ; net = +$153.86
```

The point estimate is unprofitable: the campaign would spend \$3,000 to earn \$1,485 in contribution, a loss of \$1,515. The 95% interval spans a \$3,337 range of outcomes and its upper end clears break-even by only \$154. A probability statement makes the decision sharper than a significance test:

```text
P(true effect > 0)        = 1 - Phi((0 - 0.006875)/0.00394201)
                          = 1 - Phi(-1.7440) = 0.9594
P(true effect > 1.389 pp) = 1 - Phi((0.0138889 - 0.006875)/0.00394201)
                          = 1 - Phi(1.7790) = 0.0376
```

The campaign is probably positive and almost certainly not profitable at these unit economics. Under the cluster-plus-CUPED standard error the second probability moves to `1 - Phi(1.4601) = 0.0721`, and the interval's upper bound no longer clears break-even. Any decision rule that ignores the design effect reports more confidence in profitability than the design supports.

### 5.7 Sensitivity to the economic inputs

Because the decision is a comparison of an estimate to a threshold, the threshold deserves as much scrutiny as the estimate. Recomputing net contribution at the point estimate and at the interval endpoints across five input pairs:

| `c` (\\$/contact) | `v` (\\$/purchase) | break-even (pp) | net at point | net at lower | net at upper | power at break-even (iid) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 18 | 0.556 | +\\$285 | -\\$1,384 | +\\$1,954 | 0.291 |
| 0.25 | 18 | 1.389 | -\\$1,515 | -\\$3,184 | +\\$154 | 0.941 |
| 0.40 | 18 | 2.222 | -\\$3,315 | -\\$4,984 | -\\$1,646 | 1.000 |
| 0.25 | 9 | 2.778 | -\\$2,258 | -\\$3,092 | -\\$1,423 | 1.000 |
| 0.25 | 36 | 0.694 | -\\$30 | -\\$3,368 | +\\$3,308 | 0.421 |

Two rows are decisive. At `c = \\$0.10` the campaign is profitable at the point estimate and break-even sits *below* the estimate; the same data supports adoption. At `v = \\$9` — a margin half the base case, plausible for a discounted first order — no point in the interval is profitable and the answer is a clean no. The estimate did not change; the economics did, and the economics carry more decision weight than the p-value.

The "power at break-even" column is a diagnostic for test design, not for this result: it shows the probability that a future test of the same size would correctly reject a null of no effect if the true effect were exactly break-even. Below about 0.8, a null result cannot distinguish "no effect" from "break-even effect", which is the worst outcome a decision maker can be handed.

### 5.8 Independent arithmetic re-check

A second path that avoids the rate subtraction entirely:

```text
purchases: treatment 329, control 250, total assigned 12,000
count difference = 329 - 250 = 79
79 / 12,000 = 0.0065833                       -> 0.658 pp

This differs from the stratified 0.6875 pp by 0.029 pp, because the
pooled denominator weights the arms by their realized sizes rather than
by stratum population shares. It is the right cross-check that the
stratified figure was NOT produced by dividing counts by a single
denominator: if an analyst's stratified result equals 0.6583 pp exactly,
they recomputed the naive number and mislabelled it.
```

Three additional re-checks, each targeting a specific arithmetic slip:

```text
1. Stratum counts must sum to arm counts:
   2880+2100+1200 = 6180 = n1   and   1920+2100+1800 = 5820 = n0   OK
   187+99+43 = 329 = y1   and   108+84+58 = 250 = y0                OK

2. Weights must sum to one: 0.40+0.35+0.25 = 1.00                   OK

3. Variance must be positive and SE must be smaller than the largest
   plausible binomial SE at these counts:
   worst-case binomial SE at p=0.5, n=6180: sqrt(0.25/6180) = 0.00636
   both SEs (0.00390, 0.00394) are below it                        OK
```

A fourth check that catches unit errors: the effect in percentage points must equal the effect in proportion multiplied by 100, and the *counts* of incremental purchases must equal the effect times the eligible population under the population estimand. `0.0068750 * 12000 = 82.5`, consistent with the table in 5.6. If a deck reports "82 incremental purchases" against a 1.03-point effect, `0.0102809*12000 = 123.4`, not 82.5 — the mismatch means the deck mixed the naive point estimate with the stratified projection, which is the most common internal inconsistency in experiment write-ups.

### 5.9 What the estimate does not license

- It does not license "the offer lifts conversion 24%" without the denominator: `0.0102809/0.0429553 = 23.93%` relative, which is the same data as 1.03 points and sounds three times larger.
- It does not license an exposure effect. Delivery was 85%, so `CACE = 0.020/0.85 = 2.35 pp` under monotonicity — but that number requires an exclusion restriction nobody verified.
- It does not license a rollout to the full 120,000-household list at the same margin. The enrolled population was filtered on tenure ≥30 days and prior purchasability; the marginal household beyond the filter has a lower baseline rate, and the same absolute effect would produce fewer incremental purchases.
- It does not license a repeated quarterly read as if each quarter were independent. The same customers appear in successive experiments; assignment histories correlate with outcome histories, so the third quarter's test is not a clean replication.

**Takeaway:** Compute the population-weighted estimate, the design-corrected standard error, and the break-even threshold; the decision is where those three meet, and all three can be computed before the test launches.

## 6. Visual / Geometric Intuition

### 6.1 The assignment wall

```text
   PRE-ASSIGNMENT (covariates)   |   POST-ASSIGNMENT (consequences)
   ----------------------------- | ------------------------------------
   tenure, prior spend, region   |   delivery, open, click, session
   consent state, list source    |   cart, purchase, return, churn
   eligibility flags as of T0    |   support ticket, unsubscribe
                                 |
   may be adjusted for           |   never a covariate in the primary model
   may define strata             |   may define a secondary, assumption-heavy estimand
```

Draw this wall once per experiment. Any variable to the right of it that appears in a model is a decision, not a default, and the decision must be written in the analysis plan with the estimand it produces. The wall is also the reason "control for engagement" fails: engagement is the first thing assignment changes.

### 6.2 The collider, drawn and quantified

```text
                    latent intent U
                    /            \
             (a)   /              \  (b)
                  v                v
   assignment Z --------> click S <-------- nothing else
                  \                /
             (c)   \              /
                    v            v
                     purchase Y
```

Paths: `Z -> S` is the treatment effect on clicking; `U -> S` and `U -> Y` are latent intent acting on both. Conditioning on `S` (restricting to clickers) opens the path `Z -> S <- U -> Y`, a collider path. The result is a comparison between treated clickers and control clickers who are not comparable on `U`.

A numeric demonstration where the **true ITT is exactly zero**:

```text
ILLUSTRATIVE model
  P(U=1) = 0.5, independent of Z                      (randomization holds)
  P(S=1 | Z=1, U=1) = 0.80    P(S=1 | Z=1, U=0) = 0.40
  P(S=1 | Z=0, U=1) = 0.30    P(S=1 | Z=0, U=0) = 0.10
  P(Y=1 | U=1) = 0.20         P(Y=1 | U=0) = 0.02     (no Z term: effect = 0)

all-assigned means:
  E[Y|Z=1] = 0.5*0.20 + 0.5*0.02 = 0.110
  E[Y|Z=0] = 0.5*0.20 + 0.5*0.02 = 0.110
  ITT = 0.000                                        <- truth

click-through rates:
  E[S|Z=1] = 0.5*0.80 + 0.5*0.40 = 0.600
  E[S|Z=0] = 0.5*0.30 + 0.5*0.10 = 0.200

composition of clickers:
  treated:   P(U=1|S=1,Z=1) = (0.5*0.80)/0.600 = 0.6667
  control:   P(U=1|S=1,Z=0) = (0.5*0.30)/0.200 = 0.7500

conversion among clickers:
  E[Y|S=1,Z=1] = 0.6667*0.20 + 0.3333*0.02 = 0.14000
  E[Y|S=1,Z=0] = 0.7500*0.20 + 0.2500*0.02 = 0.15500
  "clicker lift" = 0.14000 - 0.15500 = -0.01500     -> -1.50 pp
```

The true effect is zero, the treated arm clicks three times as often, and the clicker-restricted readout reports a *negative* 1.5-point effect. The sign reversed because treatment pulled in lower-intent clickers (0.40 click rate among low-intent treated versus 0.30 among low-intent control), diluting the clicker pool. An analyst who reasoned "treated clickers are worse, so the campaign is bad" would have concluded the opposite of the truth, and an analyst in a different parameterization would have concluded the opposite error — the point is that the direction of collider bias is not signable in advance.

The same geometry explains why "viewability-weighted lift" is untrustworthy: viewability is a post-assignment selection variable, and the treated/control compositions of viewable impressions differ for exactly the reason above.

### 6.3 Intervals as ranges of decisions, not error bars

```text
net contribution at N = 12,000, c = $0.25, v = $18

 -$3,184        -$1,515          $0          +$154
    |--------------|--------------|--------------|
    lower         point        break-even      upper
    bound        (0.69 pp)     (1.39 pp)     bound
    |<----------- loss ----------->|<-- tiny gain -->|
```

The picture makes the decision readable without any statistics vocabulary: most of the interval is loss, the profitable region is a thin sliver at the top, and the point estimate sits on the loss side. A bar chart of arm conversion rates cannot show this, because it omits the threshold. Any readout that shows two bars and no threshold is asking the audience to supply their own economics, which they will do differently from one another.

### 6.4 Sequential looks and the invisible multiplicity

```text
nominal 95% bands as a function of analysis time, unadjusted:

z
4 |                                    ....
3 |                          ....
2 |--+--+--+--+--+--+--+--+--          <- fixed 1.96 boundary
1 |     *     *
0 |  *
  +----------------------------------> fraction of planned sample
    .25   .50   .75   1.00

under H0 with 4 equally spaced unadjusted looks, the probability that
at least one look crosses the boundary is 0.127, not 0.05
```

The boundary line is what a fixed-horizon test implies; the wandering path is what the data does. Stopping when the path first crosses is what inflates the error rate — Section 9.4 gives the alpha-spending boundaries that repair it, with the actual cumulative alpha values.

**Takeaway:** Three pictures carry the whole method — the assignment wall, the collider graph, and the interval against a break-even threshold; each prevents a specific class of wrong conclusion.
## 7. Comparison / Selection Table

### 7.1 Design selection

Choose the design before choosing the estimator. The table is ordered so that earlier rows have fewer untestable assumptions; if a later row is the only feasible option, the assumption column becomes the headline of the readout rather than a footnote.

| Design | Identifying assumption | Point estimator | Variance consequence | Feasible when | Fails when |
|---|---|---|---|---|---|
| Individual randomized holdout | Randomization; no interference | `p1 - p0` | Base case | One unit's treatment does not move another's outcome | Shared budget, household sharing, social contagion |
| Randomized with unequal `pi_s` | Randomization within stratum | `sum w_s d_s` | Slightly larger, weights add variance | Targeting tiers with different treatment fractions | Strata built on post-assignment fields |
| Cluster randomized | Randomization of clusters | Difference of cluster means, or weighted unit means | Inflated by `1+(m-1)rho`; needs cluster-robust SE | Interference confined within cluster | Few clusters (`<40`) — normal approximation degrades, use `t` with `G-2` df |
| Switchback (time-sliced) | Stationarity within period; washout removes carryover | Paired difference across periods | Dominated by week-to-week variance, not unit count | One shared marketplace, no geo split | Strong seasonality or supply shocks inside the window |
| Difference-in-differences | Parallel trends absent treatment | Two differences of four cells | Four noisy cells; 5x wider than RCT at equal n per cell | Untreated comparison region exists | Differential shocks; staggered adoption with heterogeneous effects |
| Synthetic control | Donor pool spans the treated unit's factor loadings | Weighted donor counterfactual | Inference by placebo distribution, not formula | One treated geography, many donors, long pre-period | Donor pool contaminated by the treatment (spillover) |
| Propensity weighting / matching | Conditional exchangeability given `X`; overlap | IPW or matched difference | Sensitive to extreme weights; effective n collapses | Rich pre-treatment covariates, no randomization possible | Hidden targeting on unmeasured intent; poor overlap |
| Instrumental variables | Relevance, exclusion, independence, monotonicity | Wald ratio `ITT/(first stage)` | Very wide; weak instrument inflates by `1/F` | A credible nudge exists that changes exposure only | Instrument correlates with demand (e.g. price as instrument) |
| Pre/post with no comparison | Nothing; it is a description | Difference of two rates | Not a causal interval | Only for generating a hypothesis | Anything involving seasonality, trend, or concurrent promotion |

Three selection rules that resolve most real choices:

1. **If you can randomize, randomize, and then fix the analysis rather than the design.** The temptation to run a quasi-experiment because it needs no engineering is expensive: DiD at 1,000 units per cell produced a 95% interval of `[-1.88, +5.88]` points for a 2.00-point estimate in Section 4.3c, versus `[+0.26, +1.79]` for the randomized test in Section 5.3.
2. **If interference is the reason randomization looks impossible, change the randomization unit, not the method.** Section 9.2 works the switchback numbers: with `rho = 0.02` and 20,000 weekly units, a 12-week switchback yields an effective sample of about 599 units and a standard error near 1.73 points. That is worse than a clean individual test but far better than a pre/post comparison, which has no valid interval at all.
3. **If you must use observational data, spend the budget on covariates and on a negative control, not on a fancier estimator.** A propensity model with a strong negative-control outcome gives a testable prediction; a doubly-robust estimator without one gives a confident number with no falsification.

### 7.2 Estimator selection within a randomized design

| Situation | Estimator | Why not the simple difference | Numeric consequence (from this file) |
|---|---|---|---|
| Equal `pi_s`, balanced realized allocation | Unadjusted difference | Nothing to correct | 1.028 pp, SE 0.390 pp |
| Unequal `pi_s` by value tier | Stratified weighted difference | Pooled contrast confounds effect with mix | 0.688 pp, SE 0.394 pp |
| Unequal `pi_s`, small strata | Weighted difference with collapsed strata | Sparse strata produce `0/0` rates | Merge until ≥5 expected outcomes per arm |
| Pre-period outcome available | CUPED-adjusted difference | Leaves variance on the table | SE × 0.866 at `rho = 0.5` |
| Cluster assignment | Cluster-robust or aggregated analysis | `SE` understated by `sqrt(DEFF)` | SE 0.390 → 0.547 pp at `DEFF = 1.98` |
| Non-compliance and interest in receipt | IV / CACE rescaling | ITT understates effect on compliers | 2.00 → 2.35 pp at 85% delivery |
| Multiple planned looks | Group-sequential boundary | Fixed-horizon `z` inflates type I | 0.127 realised type I with 4 unadjusted looks |
| Many subgroups, exploratory | Shrinkage or prespecified hierarchy | Best-of-k reporting | A 300-unit subgroup has SE near 2.6 pp |

### 7.3 What to report, by audience

| Audience | Must see | Must not be shown without a caveat |
|---|---|---|
| Analyst peer | Estimand, weights, SE estimator, SRM test, maturity rule | Any number without its denominator |
| Marketing owner | Effect in points, interval, break-even threshold, delivery rate | Relative lift alone |
| Finance | Incremental units × margin, spend, net, interval on net | Point-estimate net as if certain |
| Exec sponsor | Decision, the risk being accepted, the monitoring trigger | Subgroup best-of-k as a targeting plan |
| Legal/privacy | Assignment mechanism, suppression list, consent basis | Any claim about individual-level causation |

**Takeaway:** The design choice is a validity decision, the estimator choice is a variance decision, and both must be written down before the numbers arrive — otherwise the estimator is chosen by which one looks better.

## 8. Failure Modes & Edge Cases

Each entry below states the mechanism, the observable symptom, at least two competing causes with a check that separates them, the numeric signal that discriminates, the correction, and the prevention. All scenarios are ILLUSTRATIVE constructions for teaching; none is an incident report.

### F-01 Post-treatment conditioning inflates or reverses the effect

**Mechanism.** The analyst restricts to units that were exposed, clicked, or opened, then compares conversion. The restriction is on a variable downstream of assignment, so the two arms' restricted populations differ in latent intent (Section 6.2).

**Symptom.** The restricted readout shows an implausibly large lift — often 3–8x the all-assigned figure — and the restricted denominator is much smaller than the assigned count.

**Competing causes.** (a) Genuine effect concentration: the campaign really does work only on engaged users. (b) Collider selection: the restriction created the association.

**Discriminating check.** Compare the arm-wise inclusion rate into the restricted set. In the Section 6.2 construction the click rates were 0.600 treated versus 0.200 control, a 40-point differential on an outcome the treatment itself moves; that asymmetry is the fingerprint of selection. A genuine effect-concentration story still requires the all-assigned ITT to be non-zero.

**Numeric signal.** True ITT 0.000, restricted readout `-0.0150` (negative 1.5 points). Any restricted estimate whose sign disagrees with a precisely estimated all-assigned ITT is presumed to be selection until proven otherwise.

**Correction.** Report ITT as primary. If a receipt effect is genuinely required, specify it as CACE with the monotonicity and exclusion assumptions named, and show the compliance rates used.

**Prevention.** The analysis plan lists every filter as either `pre-assignment` (allowed) or `post-assignment` (secondary only).

### F-02 Sample ratio mismatch from the wrong expected counts

**Mechanism.** Realized allocation deviates from the design because of a hashing bug, a logging defect, or simply because the design was stratified and the test was run pooled.

**Symptom.** Global observed counts differ from the nominal 50/50 by a visible margin, and the SRM test rejects.

**Competing causes.** (a) A genuine assignment or logging defect. (b) An expected-count calculation that ignores stratum probabilities.

**Discriminating check.** Recompute expected counts as `N_s * pi_s` within stratum. In Section 5.2 the global chi-square was 10.80 (`p = 0.001`) while the stratified chi-square was 0.000 (`p = 1.000`) — identical data, opposite conclusion, and only the stratified test is valid for a stratified design.

**Numeric signal.** If stratified chi-square is under about 3.84 for one stratum (the 5% critical value at df=1) and the pooled statistic is large, the imbalance is composition, not failure.

**Correction.** Never delete or reweight units to force the realized ratio back. Report the realized ratio in the readout and use design-consistent weights.

**Prevention.** Write the expected counts into the analysis plan as a formula, not a number.

### F-03 Delivery failures masquerading as null results

**Mechanism.** The assignment is correct but the send, the ad call, or the suppression step fails for a subset of the treated arm. The ITT dilutes toward zero in proportion to the delivery shortfall.

**Symptom.** ITT is small and non-significant, and the treatment arm's event counts show a suspiciously flat pattern in the days after a release.

**Competing causes.** (a) The campaign genuinely does not work. (b) Delivery was partial, so the ITT measures a mostly-untreated arm.

**Discriminating check.** Compute delivery rate by arm, day, and channel from the send log, joined to the immutable assignment ledger. A delivery rate of 0.85 with a null ITT is a different finding from a delivery rate of 0.30 with a null ITT: the first bounds the effect on delivered units to roughly `ITT/0.85`, the second leaves almost everything unidentified.

**Numeric signal.** With observed `ITT = 0.020` and delivery 0.85, `CACE = 0.0235`; recomputed at delivery 0.40, `CACE = 0.050`. The same null ITT implies a very different per-delivered effect depending on the delivery rate.

**Correction.** Report the delivery rate alongside the ITT and state the dilution bound. Do not re-run the analysis on delivered units only.

**Prevention.** Delivery rate is a monitored metric during the test, with an alert threshold agreed before launch.

### F-04 Maturity and truncation bias

**Mechanism.** Outcomes arrive with lag. Reading the test at a fixed calendar date gives recently assigned cohorts a shorter effective observation window than early cohorts. If the arm mix varies over time — and it does, because allocation is sequential — the truncation is arm-correlated.

**Symptom.** The effect estimate drifts day to day in the same direction, and the recent cohorts look uniformly worse in both arms.

**Competing causes.** (a) Truncation. (b) A genuine decay of effect over the test period (novelty).

**Discriminating check.** Plot the effect by assignment cohort, restricted to cohorts old enough to be fully mature. If mature cohorts show a stable effect and the pooled estimate is dragged by immature cohorts, truncation is the driver. If mature cohorts themselves decline monotonically, decay is the driver.

**Numeric signal.** A cohort that is 50% mature contributes roughly half its eventual outcomes; pooling it with mature cohorts shrinks the estimate by approximately the immaturity share times that cohort's weight.

**Correction.** Freeze the cohort set to `assigned_at <= cutoff - maturity_lag`, and report the chosen cutoff.

**Prevention.** Store event time and ingest time separately, and define maturity as a property of the cohort, not of the query date.

### F-05 Sequential peeking without correction

**Mechanism.** The team checks the dashboard daily and stops when the effect crosses significance. Each look is a test; the maximum of `k` correlated test statistics exceeds the fixed threshold far more often than alpha.

**Symptom.** The stopping date coincides suspiciously with the first crossing, and the effect at stop is larger than the effect two weeks later.

**Competing causes.** (a) Optional stopping. (b) A genuine early-peaking effect that decays.

**Discriminating check.** Inspect the analysis log: how many looks were recorded, at what sample fractions, and whether the launch plan declared a fixed horizon. A test with one planned interim at 50% under a spending function is a legitimate design; a test with fourteen unplanned looks is not.

**Numeric signal.** With four equally spaced unadjusted looks the family-wise error rate is approximately 0.127 instead of 0.05 — a 2.5x inflation. The observed effect at the stopping look is also biased upward by selection on the test statistic.

**Correction.** Apply a group-sequential boundary (Section 9.4) and re-read the data at the corrected threshold, or label the result exploratory.

**Prevention.** The stopping rule is written into the launch document, and dashboards show the corrected boundary rather than 1.96.

### F-06 Interference through a shared budget or auction

**Mechanism.** Treatment spend changes the competitive environment. The control arm's outcomes therefore depend on the treatment arm's assignment, violating SUTVA.

**Symptom.** Control-arm metrics move during the test in a way that cannot be explained by seasonality — typically control CPA rising as treatment spend rises.

**Competing causes.** (a) Interference. (b) A concurrent market change.

**Discriminating check.** Regress control-arm daily delivery and cost metrics on treatment-arm daily spend. A positive relationship between treatment spend and control CPA is direct evidence the arms are not isolated. Also check whether the platform reallocates budget across arms automatically.

**Numeric signal.** If the control conversion rate is depressed by 0.20 points by interference while the true ITT is 2.00 points, the measured contrast is 2.20 points — a 10% overstatement that no amount of extra sample size corrects.

**Correction.** Reinterpret the estimate as the effect of the whole allocation policy; run a switchback or a geo holdout with isolated budgets if the pure campaign effect is required.

**Prevention.** Prespecify whether the control arm is allowed to receive competing marketing, and log platform-level budget pacing during the test.

### F-07 Non-compliance handled by dropping units

**Mechanism.** Units in the treated arm that were not delivered are deleted from the analysis, and units in the control arm that were accidentally exposed are deleted too. Deletion is on a post-assignment variable in both arms.

**Symptom.** The analysed sample is smaller than the assigned sample, and the analysed arms have different covariate distributions from the assigned arms.

**Competing causes.** (a) Legitimate pre-specified exclusion of ineligible units discovered by a documented pre-assignment rule. (b) Post-treatment deletion.

**Discriminating check.** Compare the covariate means in the assigned sample and the analysed sample, arm by arm. A shift in a pre-assignment covariate that moves differently by arm indicates treatment-dependent selection.

**Numeric signal.** With 85% delivery, dropping undelivered treated units removes 15% of the arm — and 15% is exactly the non-complier share that CACE is designed to handle. If the analysed effect differs materially from `ITT/0.85`, the deletion did more than rescale.

**Correction.** Keep every assigned unit in the primary analysis. Use IV/CACE for the receipt estimand.

**Prevention.** The pipeline's default is left-join from the assignment ledger, so zeros cannot be dropped by accident.

### F-08 Multiplicity across segments

**Mechanism.** Fifteen segments are examined, one shows a 4-point effect at `p = 0.03`, and it becomes the targeting strategy.

**Symptom.** The "winning segment" is small, the effect is far from the pooled estimate, and it does not replicate.

**Competing causes.** (a) Real heterogeneity. (b) Selection from many comparisons.

**Discriminating check.** Count every segment examined, including the ones not shown in the deck. Compute what the expected maximum `z` would be under a null with 15 independent tests: about 2.5, which corresponds to `p = 0.012`. A single `p = 0.03` among 15 tests is below that expectation.

**Numeric signal.** A 300-unit subgroup at a 5% baseline has a standard error of about `sqrt(2*0.05*0.95/300) = 0.0178`, i.e. 1.78 points. A reported 4-point effect in that subgroup is roughly 2.2 standard errors — unremarkable given 15 tries.

**Correction.** Report all examined segments with intervals, apply a multiplicity adjustment or a hierarchical/shrinkage model, and require independent replication before targeting.

**Prevention.** Pre-register a small number of segments with power calculations, and treat everything else as exploratory.

### F-09 Outcome capture differing by arm

**Mechanism.** Outcome measurement depends on the treatment. A new creative drives app installs, and the app reports purchases differently from the web; or the offer triggers a consent flow that suppresses tracking.

**Symptom.** The treated arm has fewer observed outcomes per assigned unit while order volume from the source system is unchanged.

**Competing causes.** (a) Differential capture. (b) A genuine behavioural shift that moves purchases off the measured surface.

**Discriminating check.** Reconcile source-system counts to warehouse counts, by arm and event date. A source-to-warehouse gap that is larger in one arm is differential capture; an equal gap with differing event counts is behaviour.

**Numeric signal.** If capture is 0.95 in control and 0.88 in treatment, the observed difference is biased downward by roughly `0.07 * p0` in absolute terms — at `p0 = 0.043` that is 0.30 points of pure measurement artifact, comparable to a third of the true effect.

**Correction.** Use source-system reconciliation for the primary outcome where possible, and report the capture rates.

**Prevention.** Instrument both surfaces before launch and alert on arm-divergent capture.

### F-10 Assignment drift and population change

**Mechanism.** Eligibility logic is deployed, altered, or hot-fixed during the test, so the enrolled population changes over time and the design probabilities no longer apply uniformly.

**Symptom.** Baseline covariates shift by assignment week, and the realized allocation ratio shifts at the same moment.

**Competing causes.** (a) A code deployment changing eligibility. (b) Organic change in who is arriving.

**Discriminating check.** Version the eligibility rule and log its version on every assignment record. A step change in covariate means aligned to a deployment timestamp points to rule drift; a gradual drift points to audience change.

**Numeric signal.** A 5-point shift in the share of a high-baseline segment between the first and second half of the test moves the pooled rate by roughly `0.05 * (p_high - p_low)`, which at a 3-point gap is 0.15 points — enough to flip a marginal readout.

**Correction.** Freeze eligibility during the test, or analyse by prespecified policy period with the strata redefined.

**Prevention.** Assignment records carry the rule version; deployments during a test require an explicit sign-off.

### F-11 Estimand mismatch between the calculation and the sentence

**Mechanism.** The analyst computes ATE-weighted numbers and describes them as applying to treated customers, or the reverse. The arithmetic is right and the claim is wrong.

**Symptom.** The deck says "targeted customers gained 0.69 points" while the weighted calculation used population shares in which the targeted segment is only 40% of the base.

**Competing causes.** (a) An estimand labelling error. (b) A genuine disagreement about the decision population.

**Discriminating check.** Recompute under both weightings. In Section 5.4 the same table gives 0.6875 pp (population weights), 0.6558 pp (treated weights), and 0.7174 pp (control reweighted to treated). If the reported number matches one of these three, the label must match it exactly.

**Numeric signal.** The three estimates differ by up to 0.06 pp here; in designs with stronger propensity-outcome correlation the spread is larger than the standard error, which means the choice can change significance.

**Correction.** Print the weighting next to every number, and name the population in the sentence.

**Prevention.** The reporting template has a mandatory `estimand:` field that cannot be left empty.

**Takeaway:** Every failure above leaves a signature in a secondary table — delivery by arm, inclusion rates, covariate means by week, capture rates — so build those tables before the readout, not after a challenge.
## 9. Complexity & Cost Analysis

### 9.1 What "cost" means here

Four costs matter and they are not interchangeable: statistical cost (sample size and time needed to reach a decision), engineering cost (assignment service, logging, measurement joins), opportunity cost (revenue withheld from control), and decision cost (the expected loss from acting on an estimate that is wrong in either direction). Analysts habitually optimize the first and ignore the fourth, which is the one the business feels.

### 9.2 Cost of precision

The sample size required for a two-arm test at power `1 - beta` and two-sided alpha:

```text
FORMULA      n_per_arm = (z_{1-alpha/2} + z_{1-beta})^2 * (p1(1-p1) + p0(1-p0)) / (p1-p0)^2
             with alpha = 0.05, power = 0.80: (1.959964 + 0.841621)^2 = 7.84888

WHERE        p0 = baseline rate, p1 = p0 + delta, delta = MDE
CONSTRAINT   normal approximation; each arm needs >= 5 expected events
```

ILLUSTRATIVE evaluations at `p0 = 0.05` (a realistic purchase rate for a broad campaign):

| MDE (pp) | `p1` | numerator term `p1(1-p1)+p0(1-p0)` | `n` per arm | total assigned | weeks at 6,000/week |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.0550 | 0.099475 | 31,231 | 62,461 | 10.4 |
| 1.00 | 0.0600 | 0.103900 | 8,155 | 16,310 | 2.7 |
| 1.39 (break-even in §5.6) | 0.0639 | 0.107307 | 4,366 | 8,732 | 1.5 |
| 2.00 | 0.0700 | 0.112600 | 2,209 | 4,419 | 0.7 |
| 5.00 | 0.1000 | 0.137500 | 432 | 863 | 0.1 |

Two things are visible. First, sample size scales as `1/delta^2`: halving the target effect multiplies the enrolment by roughly 3.7-4, so detecting 1 point instead of 2 costs 8,155 versus 2,209 per arm. Second, the baseline rate enters only through `p(1-p)`, so the same absolute MDE is *harder* to detect at a 50% baseline than at 5%: at `p0 = 0.50` a 2-point effect needs `7.84888*(0.52*0.48 + 0.50*0.50)/0.0004 = 9,803` per arm, more than four times the 2,209 at `p0 = 0.05`. Teams that port a sample-size calculator from a click-through test to a purchase test routinely under-enrol by an order of magnitude because they carried the baseline forward without recomputing.

The cluster correction multiplies the requirement by `DEFF` and CUPED divides it by `(1 - rho^2)`:

```text
DEFF = 1.98, CUPED factor 0.75:
  total required = 8,732 * 1.98 * 0.75 = 12,967 assigned units at the
  break-even MDE, versus 8,732 for the iid design.
```

So a design that is "adequately powered" on paper becomes underpowered once the real correlation structure is acknowledged — and this is the single most common reason a well-run test returns an interval that cannot support a decision.

### 9.3 Cost of the four design families

ILLUSTRATIVE cost model: assignment and logging engineering measured in engineer-weeks, opportunity cost measured in withheld contribution, and calendar time measured at 6,000 eligible units per week.

| Design | Engineering | Opportunity cost | Calendar time | Effective n | Realistic use |
|---|---|---|---|---|---|
| Individual holdout, 50/50 | 2–4 eng-weeks (ledger, sticky assignment, join) | 50% of campaign contribution withheld | 1.5 weeks at break-even MDE | 8,732 | Default |
| Individual holdout, 90/10 | 2–4 eng-weeks | 10% withheld | 4x longer for same precision | ~same total needed | When withholding is expensive |
| Cluster (geo) | 4–8 eng-weeks (geo matching, cluster-robust analysis) | 50% of clusters withheld; larger blast radius | Similar | 8,732 × 1.98 | Spillover or platform-level changes |
| Switchback | 6–10 eng-weeks (scheduling, washout enforcement) | None withheld, but every period is degraded vs the better policy | 12+ weeks of calendar | ~600 (see below) | One shared auction with no geo split |
| DiD / synthetic control | 3–6 eng-weeks | None | Needs long pre-period (8–12 weeks) | 4× cells for same SE | Retrospective, or rollout already decided |
| Propensity / IV | 3–6 eng-weeks | None | Retrospective | Collapses with poor overlap | When randomization is impossible |

The switchback effective sample is the number that surprises people:

```text
12 weekly periods, 20,000 units per period, rho = 0.02
  DEFF  = 1 + (20000-1)*0.02 = 401.0
  n_eff = 12*20000/401.0 = 598.5
  SE    = sqrt(2*0.10*0.90/598.5) = 0.01734        -> 1.73 pp
```

A switchback over a quarter with 240,000 unit-periods delivers the precision of a 599-unit individual test. Increasing periods helps linearly but slowly: at 6 periods per arm the SE is 2.45 pp, at 8 it is 2.12 pp, at 4 it is 3.00 pp. The lever that matters is the within-period correlation `rho` — reducing it from 0.02 to 0.005 would cut `DEFF` roughly fourfold, which is a measurement design question (shorter periods, cleaner washout), not a budget question.

### 9.4 Cost of sequential monitoring

Fixed-horizon tests cost one number: the planned sample. Sequential designs cost a boundary and buy the right to stop early. The trade is real and quantifiable.

```text
O'Brien-Fleming spending, alpha = 0.05 two-sided, equally spaced looks:
  t = 0.25  cumulative alpha = 0.000089   -> |z| > 3.9198
  t = 0.50  cumulative alpha = 0.005575   -> |z| > 2.7718
  t = 0.75  cumulative alpha = 0.023625   -> |z| > 2.2632
  t = 1.00  cumulative alpha = 0.050000   -> |z| > 1.9600

Pocock-style constant boundary, 4 looks (simulated, 200,000 paths):
  |z| > 2.361 at every look gives one-sided+two-sided FWER 0.0500
  |z| > 2.500 gives 0.0350
  unadjusted |z| > 1.960 at every look gives 0.1269
```

The last line is the price of ignoring the problem: 2.5x the nominal error rate. The OBF boundary is nearly impossible to cross early (3.92 standard errors at the first look), which is appropriate when early evidence should be treated with suspicion; the Pocock boundary is roughly flat and easier to cross early, at the cost of a higher final threshold (2.36 versus 1.96). The choice is a policy decision about how much the organization trusts early signals.

Time cost, expressed in required effect size: an OBF design that requires `|z| > 2.7718` at the halfway point needs an effect about 41% larger to stop early than the fixed-horizon test would need at the same sample size, which is why early stopping mostly happens for unexpectedly large effects and those estimates are then biased upward.

### 9.5 Decision cost and the value of a better estimate

The decision-theoretic frame: if you roll out when the true effect is below break-even you lose `N*(c - v*effect)`; if you withhold when it is above break-even you lose `N*(v*effect - c)`. Using the Section 5 numbers with `N = 120,000` (the quarterly eligible base, ten times the test):

```text
roll out at true effect = 0 (the pessimistic case):
  loss = 120,000 * ($0.25 - $18*0) = $30,000
withhold at true effect = 0.0139 (exactly break-even): loss = $0
withhold at true effect = 0.0200: loss = 120,000*($18*0.020 - $0.25) = $13,200
roll out at true effect = 0.0069 (the point estimate):
  loss = 120,000*($0.25 - $18*0.0069) = $15,096
```

The test itself costs `12,000 * $0.25 = $3,000` in contact cost plus engineering. So a test that reduces the probability of a \$30,000 misallocation by a meaningful margin pays for itself several times over — and the relevant question is not "was the test significant" but "how much did it change the probability of the wrong action". A test with an interval that spans break-even has not resolved the decision even though it produced a number.

The counter-case matters too: at `c = $0.10` and `v = $18` the break-even is 0.56 pp, and the point estimate of 0.69 pp is above it, so the decision is "roll out" and further testing has low value. Recognizing when a test is decision-complete prevents the pathological pattern of re-running the same underpowered test every quarter.

### 9.6 Cost of the analysis pipeline itself

The arithmetic is cheap; the joins are not. A production-grade causal pipeline needs, at minimum: an immutable assignment ledger with sticky assignment and rule versioning; an outcome extract keyed to the assignment grain with event-time and ingest-time; a delivery/exposure log; a covariate snapshot at `T0`; and a readout service that recomputes SRM, balance, and maturity checks on every refresh. The recurring engineering cost is in keeping those five consistent under schema change, which is where most experiment infrastructure decay originates.

A useful sizing heuristic: the analysis code is a few hundred lines and the pipeline is a few thousand. Teams that spend the effort on the estimator and none on the ledger end up with a statistically sophisticated readout over a denominator nobody can defend.

**Takeaway:** Buy precision with sample size, buy validity with design, and buy decision quality with the break-even threshold; only the last one changes what you should do.

## 10. Key Theorems / Results Reference

### 10.1 Identification results

**Randomization identifies the ATE.** If `Z` is independent of `(Y(1), Y(0))`, then `E[Y|Z=1] - E[Y|Z=0] = E[Y(1) - Y(0)]`. Proof: by independence, `E[Y|Z=1] = E[Y(1)|Z=1] = E[Y(1)]` and similarly for control; subtract. No modelling assumption is involved, which is why randomization survives criticism that regressions do not.

**Conditional randomization identifies the stratum ATE.** If `Z` is independent of potential outcomes within stratum `s`, the within-stratum difference identifies `E[Y(1)-Y(0)|s]`, and averaging with population weights `w_s` identifies the population ATE. This is the result behind Section 4.2, and it is why stratification is a design fix rather than an analysis trick.

**Instrumental variables identify the LATE, not the ATE.** With relevance (`E[D|Z=1] > E[D|Z=0]`), independence, exclusion, and monotonicity, the Wald ratio `(E[Y|Z=1]-E[Y|Z=0])/(E[D|Z=1]-E[D|Z=0])` identifies the average effect among compliers. It says nothing about always-takers or never-takers. In marketing, compliers are exactly the units whose exposure responds to the nudge, and they are usually not representative of the population.

**Conditional exchangeability identifies the ATE under overlap.** If `(Y(1),Y(0))` is independent of `Z` given `X` and `0 < e(X) < 1`, then `E[ZY/e(X) - (1-Z)Y/(1-e(X))] = E[Y(1)-Y(0)]`. The overlap condition is not decorative: without it the ratio is undefined for some units and the estimator's variance is infinite.

**DiD identifies the ATT under parallel trends.** `E[Y(1)-Y(0)|treated] = (E[Y|treated,post]-E[Y|treated,pre]) - (E[Y|control,post]-E[Y|control,pre])` provided the untreated change in the treated group equals the observed change in controls. The assumption is about a counterfactual trend, so it can be supported but never proven by pre-period data.

### 10.2 Variance results

**Two-sample variance.** `Var(p1-p0) = p1(1-p1)/n1 + p0(1-p0)/n0` under independent Bernoulli sampling. Two refinements are worth knowing:

```text
pooled (used for testing H0: p1 = p0)
  Var_pool = p_bar(1-p_bar)(1/n1 + 1/n0),  p_bar = (y1+y0)/(n1+n0)
unpooled (used for the interval)
  Var_unpooled = p1(1-p1)/n1 + p0(1-p0)/n0

Neyman's third term, which the unpooled form omits:
  Var = S1^2/n1 + S0^2/n0 - S_tau^2/N
  with S_tau^2 the variance of the individual effect
  at n1=n0=6000 and tau = 0.01028 (effects in {0,1}):
    S_tau^2 = 0.01028*0.98972 = 0.010175
    term   = 0.010175/12000 = 8.479e-7
    Var_unpooled = 1.52193e-5   Var_Neyman = 1.43714e-5
    ratio 0.9443  ->  SE ratio 0.9717
```

The third term makes the conservative unpooled interval about 3% wider than necessary. It is almost never applied in marketing practice, and mentioning it is the difference between knowing the formula and knowing its conservatism.

**Cluster design effect.** `Var(cluster mean) = Var(unit mean) * (1 + (m-1)rho)` for equal cluster sizes `m`. For unequal sizes, replace `m` with `m_bar * (1 + CV_m^2)` where `CV_m` is the coefficient of variation of cluster size; a design with highly variable geo sizes is much less efficient than its average size suggests.

**CUPED variance.** `Var(Y - theta(X - E[X])) = Var(Y)(1 - rho^2)` at the optimal `theta`. The result is exact for the linear adjustment and asymptotically valid when `theta` is estimated; with small samples the degrees-of-freedom correction matters.

**Delta-method for ratios.** Rescaling ITT to CACE divides two random quantities. The variance is approximately `Var(ITT)/C^2` where `C` is the compliance difference, so a small compliance difference inflates variance quadratically: at `C = 0.85` the variance multiplier is 1.38; at `C = 0.40` it is 6.25.

### 10.3 Sequential and multiplicity results

**Repeated unadjusted testing inflates type I.** For `k` independent looks at the same nominal level, the family-wise error rate approaches `1 - (1-alpha)^k`; for `k = 4` and `alpha = 0.05` that is 0.185 under independence, and 0.127 with the positive correlation induced by overlapping data (simulated above). The correlation reduces but does not eliminate the inflation.

**O'Brien-Fleming spending.** Cumulative alpha at information fraction `t` is `2 - 2*Phi(z_{1-alpha/2}/sqrt(t))`, giving the boundaries tabulated in Section 9.4. The boundary is conservative early by construction and approaches the fixed-horizon critical value at `t = 1`.

**Bonferroni is conservative under positive dependence.** For `m` tests at level `alpha/m` the family-wise error is at most `alpha`; when tests are positively correlated the actual rate is well below `alpha`, so Bonferroni costs power. Holm's step-down procedure dominates it without extra assumptions.

**Selection on the maximum.** With `m` independent standard normal statistics, `E[max z] ≈ sqrt(2 ln m)`: 15 segments give about 2.33, and 100 give about 3.03. This is the number to quote against any "best segment" claim, because it is the effect size that pure noise produces.

### 10.4 Sensitivity results

**E-value for a risk ratio.** The minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need with both treatment and outcome to explain away the observed association is `RR + sqrt(RR*(RR-1))`. ILLUSTRATIVE at the Section 5 observed rates:

```text
RR = p1/p0 = 0.0532362/0.0429553 = 1.2393
E  = 1.2393 + sqrt(1.2393*0.2393) = 1.2393 + 0.5445 = 1.7838
```

An unmeasured confounder associated with both assignment and purchase by a risk ratio of 1.78 each would suffice to move the estimate to the null. In a randomized test that is a strange hypothesis (randomization should make such an association impossible); in an observational comparison it is entirely plausible, which is exactly what the E-value is for. Its role here is as a calibration of what "this observational result is fragile" means numerically.

**Bias factor for a binary confounder.** For a confounder with prevalence difference `gamma` between arms and effect `delta` on the outcome, the bias in the risk difference is approximately `gamma * delta`. ILLUSTRATIVE: if 30% more of the treated arm are high-intent (`gamma = 0.30`) and high intent raises purchase probability by 3 points (`delta = 0.03`), the bias is `0.30*0.03 = 0.9` points — larger than the entire stratified estimate of 0.69 points. This arithmetic is the fastest way to kill a retargeting "lift" claim.

**Takeaway:** Five results carry the practice — randomization identifies the ATE, stratification identifies strata, IV identifies compliers only, clustering inflates variance by `DEFF`, and repeated looks inflate error by roughly the square root of the number of looks.
## 11. Interview / Exam Questions

Fourteen questions, ordered roughly by seniority. Each answer states the number where a number is required; questions marked **[C]** require computation, **[D]** require diagnosis of a described situation, and **[J]** require a judgement call with a stated rationale.

### Q1 [C] A test has 6,180 treated with 329 purchases and 5,820 control with 250 purchases. Give the ITT in percentage points, the standard error, and the 95% interval.

`p1 = 329/6180 = 0.0532362` and `p0 = 250/5820 = 0.0429553`, so `ITT = 0.0102809`, i.e. 1.028 pp. The variance is `0.0532362*0.9467638/6180 + 0.0429553*0.9570447/5820 = 0.0000081557 + 0.0000070636 = 0.0000152193`, giving `SE = 0.0039012` (0.390 pp). The interval is `0.0102809 ± 1.959964*0.0039012 = [0.0026347, 0.0179271]`, or `[0.26 pp, 1.79 pp]`, and `z = 2.635` (two-sided `p = 0.0084`). If the candidate reports 1.03 pp with a standard error computed from the pooled rate `(329+250)/12000 = 0.04825`, they have used the testing variance rather than the interval variance; `sqrt(0.04825*0.95175*(1/6180+1/5820)) = 0.003899`, which is numerically almost identical here but is the wrong object — it is valid only under the null.

### Q2 [C] The same test was run with treatment probabilities 0.60, 0.50, and 0.40 across three strata of sizes 4,800, 4,200, and 3,000. Why does the pooled difference overstate the effect, and what is the corrected number?

The pooled difference is 1.028 pp; the population-weighted stratified difference is 0.6875 pp. The high-value stratum has both the highest baseline rate (5.63% control) and the highest treatment probability (0.60), so the treated arm is compositionally richer in high-value customers than the control arm. The pooled contrast therefore contains a mix effect of about 0.34 pp on top of the causal effect. Corrected: `0.40*0.0086806 + 0.35*0.0071429 + 0.25*0.0036111 = 0.006875`, a 33% reduction, and the `z` statistic falls from 2.635 to 1.744 so the result is no longer significant at 95%.

### Q3 [C] With the stratified estimate, what is the probability that the true effect exceeds a break-even threshold of 1.389 pp?

Using `0.006875` and `SE = 0.00394201`: `z = (0.0138889 - 0.006875)/0.00394201 = 1.7790`, so `P = 1 - Phi(1.7790) = 0.0376`. If the design is clustered with `DEFF = 1.98` and CUPED at `rho = 0.5`, `SE = 0.0048038`, `z = 1.4601`, and `P = 0.0721`. The probability roughly doubles under the correct variance, which is the practical argument for computing it on the design-corrected standard error. A candidate who reports 0.0376 while claiming the interval assumed independence has given the right arithmetic for the wrong model.

### Q4 [D] A dashboard shows treated clickers converting at 14.0% and control clickers at 15.5%, while all-assigned conversion is 11.0% in both arms. What is happening?

The restricted comparison is conditioned on a post-assignment variable, so the arms' clicker populations differ in latent intent. In the constructed example, treatment raises the click rate from 0.200 to 0.600 and the treated clicker pool is 66.7% high-intent versus 75.0% among control clickers; with purchase rates of 20% (high intent) and 2% (low intent) the restricted means are 0.140 and 0.155. The true ITT is zero. The -1.5 pp "effect" is pure collider selection, and its direction is not signable in advance. The correct action is to report the all-assigned ITT and treat the clicker table as descriptive.

### Q5 [C] Compute the MDE for a two-arm test with 6,000 per arm at a 5% baseline and `alpha = 0.05`, 80% power.

`MDE = (1.959964 + 0.841621)*sqrt(2*0.05*0.95/6000) = 2.801585*0.0039811 = 0.011154`, i.e. 1.115 pp. Repeating at other baselines with the same 6,000 per arm: a 10% baseline gives `2.801585*sqrt(2*0.10*0.90/6000) = 2.801585*0.0054772 = 0.015345` (1.535 pp), and a 20% baseline gives `2.801585*0.0073029 = 0.020460` (2.046 pp), so the absolute MDE grows by a factor of 1.83 across that range. The relative MDE moves the other way: 1.115/5 = 22.3% at the 5% baseline versus 2.046/20 = 10.2% at the 20% baseline, and at a 0.4% click baseline the same budget yields `2.801585*0.0011523 = 0.003228`, i.e. 0.323 pp = 80.7% relative. That asymmetry is the practical trap: the low-baseline click metric looks more sensitive in relative terms precisely because it cannot resolve a small number of points, and a team reporting only relative lift will believe the blunt instrument is the sharp one. Always quote the absolute MDE in points next to its baseline.

### Q6 [D] The global SRM test returns `chi2 = 10.80` on a design with 50/50 nominal allocation but 60/50/40 stratum probabilities. Diagnose.

The global test compares realized counts to 6,000 per arm, but the design implies expected counts of `4800*0.60 + 4200*0.50 + 3000*0.40 = 2,880 + 2,100 + 1,200 = 6,180` and `5,820`. Against those expectations the chi-square is exactly zero in each stratum. The global statistic of 10.80 (`p = 0.001`) is measuring the design, not a defect. The rule: expected counts are always `N_s * pi_s` within stratum, and a pooled SRM test is valid only when `pi_s` is constant across strata. Acting on the false alarm by re-randomizing or dropping units would introduce the bias the test was meant to prevent.

### Q7 [C] A geo test assigns 40 clusters. What is the correct reference distribution, and how much does it cost versus 40 independent units?

With 40 clusters the difference-in-cluster-means is approximately `t`-distributed with `G - 2 = 38` degrees of freedom, whose 97.5th percentile is 2.024 rather than 1.960 — a 3.3% wider interval before any design effect. The real cost is the design effect: at `m = 50` units per cluster and `rho = 0.02`, `DEFF = 1.98`, so `n_eff = 2000/1.98 = 1010` and the standard error is `sqrt(1.98) = 1.407` times the naive value. So 2,000 clustered units carry the information of about 1,010 independent ones, and the interval is 41% wider. With 40 clusters the `t` correction and the design effect compound, which is why cluster count — not cluster size — is the binding constraint.

### Q8 [D] Treatment spend rises 8% during the test and control CPA rises at the same time. What are the two candidate explanations and how do you separate them?

Either the arms interfere through a shared auction or budget pool, or an external market change lifted costs for everyone. Separate them by (a) regressing control-arm daily CPA on treatment-arm daily spend within the test window — a positive within-test relationship with no plausible external driver supports interference; (b) checking whether the same CPA movement appears in markets outside the test; (c) inspecting platform pacing logs for automatic cross-arm budget reallocation. If interference is real, the estimate is the effect of the whole allocation policy, not the campaign. A 0.20 pp depression in control conversion on a 2.00 pp true effect yields a measured 2.20 pp contrast, a 10% overstatement unaffected by sample size.

### Q9 [C] With 85% delivery in treatment and 5% leakage into control, convert an ITT of 2.0 pp into a CACE, and state what the number assumes.

`CACE = 0.020/(0.85 - 0.05) = 0.020/0.80 = 0.0250`, i.e. 2.50 pp. Without leakage the denominator would be 0.85 and the answer 2.35 pp, so contamination of 5% moves the receipt effect by 0.15 pp — the leakage matters more than intuition suggests because it enters the denominator directly. The number assumes monotonicity (no unit's exposure is reduced by assignment) and exclusion (assignment affects the outcome only through delivery). Exclusion is the fragile one: if assignment also triggers inventory holds or price changes, the rescaled number attributes those effects to delivery.

### Q10 [C] Four equally spaced looks, unadjusted testing, nominal 5%. What is the actual error rate, and what boundary restores 5%?

Simulating four correlated looks on a Brownian path gives a family-wise error rate of 0.127 — 2.5x nominal. A constant Pocock-style boundary at `|z| > 2.361` restores 0.0500; an O'Brien-Fleming boundary is `3.9198` at `t = 0.25`, `2.7718` at `t = 0.50`, `2.2632` at `t = 0.75`, and `1.9600` at `t = 1.00`, with cumulative alpha of 0.000089, 0.005575, 0.023625, and 0.050000. The OBF choice is right when early evidence should be discounted; the flat choice is right when the organization needs the option to stop early for operational reasons.

### Q11 [J] Your test returns an interval of `[-0.09 pp, +1.46 pp]` against a break-even of 1.39 pp. The marketing owner wants to launch. What do you say?

The correct response separates the statistical statement from the decision: the data are consistent with a small loss and with a modest gain, and the probability that the effect clears break-even is 0.038 on the iid standard error and 0.072 once clustering is acknowledged. The expected net contribution at the point estimate is negative (`12,000*(0.006875*18 - 0.25) = -\$1,515`), and only the top of the interval is profitable, by about \$154. The defensible options are: (a) do not launch at these unit economics; (b) reduce `c` or raise `v` so break-even falls below the point estimate — at `c = \$0.10` the same estimate clears break-even and the decision reverses; (c) run a larger test sized to the break-even effect, which needs about 4,366 per arm rather than the 6,000 used, so the current test was in fact sized adequately and the honest answer is that the campaign is not profitable enough. Naming the option set, rather than refusing to decide, is what distinguishes a senior answer.

### Q12 [D] A readout shows a 4-point lift in a 300-unit segment at `p = 0.03`, out of 15 segments examined. Assess.

At a 5% baseline the standard error in a 300-unit segment is about `sqrt(2*0.05*0.95/300) = 0.0178`, i.e. 1.78 pp, so a 4-point effect is about 2.2 standard errors. With 15 independent tests the expected maximum `z` under the null is `sqrt(2 ln 15) = 2.33`, so a maximum of 2.2 is *below* what pure noise produces. The segment result is unremarkable. Correct handling: report all 15 segments with intervals, apply Holm or a shrinkage model, and require independent replication before targeting. The most damaging version of this error is not the wrong number but the wrong follow-up — a targeting plan built on a segment that was selected for its noise.

### Q13 [C] An observational comparison gives a risk ratio of 1.24. Compute the E-value and interpret it in a randomized versus an observational setting.

`E = 1.2393 + sqrt(1.2393*0.2393) = 1.2393 + 0.5445 = 1.7838`. Interpretation: a confounder associated with both treatment and outcome by a risk ratio of 1.78 each could explain the association away. In a randomized test that hypothesis is implausible because randomization severs the confounder-treatment link; in a retargeting comparison where 30% more of the treated arm are high-intent and high intent adds 3 pp to purchase probability, the bias is `0.30*0.03 = 0.9 pp`, which exceeds the entire 0.69 pp estimate. The E-value is therefore a screening tool: large E-values need strong confounders to overturn, small ones need only the confounders that marketing data is full of.

### Q14 [J] When is a DiD estimate preferable to running a randomized holdout, given that DiD has four noisy cells and an untestable parallel-trends assumption?

When withholding is infeasible or when the rollout has already happened. Three concrete situations: (a) a regulatory or contractual constraint forbids denying service to a subset (some financial and healthcare communications); (b) the treatment is a market-level change — a pricing structure, a retail media placement — that cannot be applied to part of a market; (c) the decision is retrospective and the only available evidence is a natural experiment. In those cases the honest comparison is not "DiD versus RCT" but "DiD with a stated trend assumption versus no estimate at all", and the deliverable must carry the assumption in the headline. The numeric penalty is real: at 1,000 per cell, DiD gave 2.00 pp with `SE = 1.98 pp` and an interval of `[-1.88, +5.88]`, against `SE = 0.39 pp` for the randomized equivalent. A practitioner choosing DiD should also budget for 8–12 weeks of pre-period to make parallel trends at least partly falsifiable.

### Q15 [D] A readout reports a 24% relative lift. The underlying numbers are 5.3236% versus 4.2955%. What is the absolute effect, and what is the danger in the framing?

`0.0532362 - 0.0429553 = 0.0102809`, i.e. 1.03 pp, and `0.0102809/0.0429553 = 23.93%`, which is where the 24% comes from. The framing danger is that relative lift is scale-free and therefore travels badly: the same 24% relative lift on a 0.4% baseline is 0.10 pp, which may be far below break-even, while on a 40% baseline it is 9.6 pp, which may be transformational. Break-even in this file was 1.389 pp, so the 1.03 pp absolute effect fails the economic test even though the 24% headline reads like a success. Report absolute effect, baseline, and break-even together; the relative figure is a derived convenience, never the decision input.

**Takeaway:** The questions that separate candidates are not the formula recitations but the two-step ones — compute the number, then say what it does and does not license.
## 12. Quick Reference Card

| # | Item | What to do | Number / rule from this file |
|---:|---|---|---|
| 1 | Estimand | Write `ITT`, `CACE`, `ATT`, or `ATE` on the readout before computing anything | Weighting changes the answer: 0.688 / 0.656 / 0.717 pp from one table |
| 2 | ITT and interval | `p1 - p0` with `SE = sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)` | 1.028 pp, SE 0.390 pp, CI [0.26, 1.79] pp |
| 3 | Stratification | Expected counts and weights use `N_s * pi_s` and `N_s/N`, never realized arm shares | Naive 1.028 pp vs stratified 0.688 pp (33% bias) |
| 4 | Design effect | Multiply SE by `sqrt(1 + (m-1)rho)` for clustered assignment | `m=50, rho=0.02` → DEFF 1.98, SE 0.390 → 0.547 pp |
| 5 | CUPED | Pre-assignment covariate at correlation `rho` cuts SE by `sqrt(1-rho^2)` | `rho = 0.5` → factor 0.866; `rho = 0.7` → 0.714 |
| 6 | Compliance | `CACE = ITT / (delivery_treated - delivery_control)`; never delete undelivered units | 2.00 pp / (0.85 - 0.05) = 2.50 pp |
| 7 | MDE | `(1.959964 + 0.841621)*sqrt((p1(1-p1)+p0(1-p0))/n)` per arm | 1.115 pp at 5% baseline and 6,000/arm; 2.046 pp at 20% |
| 8 | Break-even | `effect* = cost_per_contact / contribution_per_purchase` | \$0.25/\$18 = 1.389 pp; compare to the interval, not the point |
| 9 | Sequential looks | Use OBF boundaries `2 - 2*Phi(1.959964/sqrt(t))`; unadjusted peeking multiplies error | 4 looks unadjusted → 0.127 FWER; OBF at `t=0.5` needs `|z|>2.7718` |
| 10 | Diagnosis | Post-assignment filters are the first suspect when a restricted readout is large | True ITT 0.000 can present as -1.50 pp among clickers |

The card in prose, for the three rows that matter most:

**Row 2 is the only arithmetic most readouts need.** Write it as a single expression so that no intermediate rounding happens: `(y1/n1 - y0/n0) ± 1.959964*sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)`. Compute the standard error from the unpooled rates; the pooled variant belongs in the hypothesis test, not the interval.

**Row 8 converts a statistical result into a business decision.** Compute it before the test launches, because its value determines whether the test can answer the question at all. If break-even is 1.389 pp and the MDE at the planned sample is 1.9 pp, the test is underpowered by construction and no analysis will repair that.

**Row 10 is the fastest diagnostic.** When a segmentation or exposure-restricted result is dramatically larger than the all-assigned result, the default hypothesis is post-treatment conditioning, and the check is comparing inclusion rates by arm. In the constructed example those rates were 0.600 versus 0.200 on a zero-effect treatment.
