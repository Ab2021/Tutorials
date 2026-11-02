> Module 04 — Statistics and Causal Inference
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.05, 04.06

# Experiments versus Quasi-Experiments

---

## 1. Why this topic exists

A retailer rolls a new shelf-replenishment policy into 180 stores in March. By June, on-shelf
availability is up 3.4 percentage points. The chief operating officer asks whether the policy
caused the improvement. The analyst compares June with the previous June and says yes.

Three things changed between those two Junes. The promotion calendar moved by two weeks. The
north distribution centre (DC) was re-sited. A competitor closed 40 stores in the same trading
area. The number is correctly computed and the claim is unsupported, because the comparison
contains three changes and attributes all of them to one.

A causal claim requires a counterfactual: what would availability have been in these stores, in
this period, had the policy not been introduced? That quantity is never observed. Every credible
method is a device for making it credible, and every device rests on an assumption that must be
named, argued, and tested.

Randomisation is the strongest device because it makes the counterfactual exchangeable by
construction. If treatment is assigned by a coin flip, the arms differ only by chance, and the
distribution of that chance difference is known. This is Fisher's insight, and it is why the
randomised controlled trial (RCT) is the reference design against which every other design is
judged.

Supply chain work complicates the picture in five specific ways, and this topic exists because
those five are usually ignored:

1. **The unit you can randomise is often not the unit you care about.** You can randomise stores,
   not customers, because the treatment is a store-level process. You can randomise DCs, not
   pallets.
2. **Units talk to each other.** Treated and control stores draw from the same DC. A stock-out in
   one arm changes the allocation to the other. The stable unit treatment value assumption
   (SUTVA) fails, and the failure biases the estimate in a direction you can often predict.
3. **Treatments persist.** A promotion produces pantry loading that survives for weeks. A routing
   change produces a learning curve that survives for months.
4. **Compliance is discretionary.** A planogram is not a line of code. Managers adapt, partially
   implement, or ignore. The effect of the policy as designed and as executed are different
   estimands.
5. **The control group costs money.** In a web experiment the control arm costs nothing. In a
   supply chain it forgoes margin, service, or capacity, and someone senior must approve that.

Most supply chain causal questions are therefore answered with quasi-experiments:
difference-in-differences (DiD), synthetic control, regression discontinuity (RDD),
instrumental variables (IV), matching, or interrupted time series (ITS). Each buys identification
with an assumption. The professional skill is knowing which assumption you just bought, what it
would look like if false in this network, and what test would show it.

This topic follows 04.05, on respecting the assignment unit, and 04.06, on which variables may be
controlled. It is about design choice; sample size, power, and execution are Module 05.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the estimand of a proposed test — unit, outcome, contrast, horizon, population — and
   **identify** the decision rule the estimate is meant to inform.
2. **Select** an assignment unit and a design from a described operational constraint, and
   **name** the identifying assumption the design carries.
3. **Compute** the design effect, effective sample size, and minimum detectable effect (MDE) for a
   cluster-randomised design given clusters, units per cluster, outcome standard deviation, and
   intra-cluster correlation (ICC).
4. **Design** a switchback schedule for a DC or lane, including block length, washout, and an
   analysis that respects the assignment path.
5. **Distinguish** the assumptions behind DiD, synthetic control, RDD, IV, matching, and ITS, and
   **describe** what each looks like as a supply chain failure.
6. **Specify** a falsification test for each design, and **say** what result would make you abandon
   the estimate.
7. **Audit** a published or internal pilot and **classify** it as randomised, quasi-experimental,
   or descriptive, with the gap stated in one sentence.
8. **Defend** a weaker design when a stronger one is infeasible, without overstating what it
   establishes.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

### 3.1 The missing cell

Every causal question is a 2 × 2 table with one cell missing for every unit.

```text
   THE 2 x 2, AND THE CELL YOU CANNOT SEE

                          treated              not treated
                        +--------------------+--------------------+
        observed        |   Y(1) observed    |   Y(0) observed    |  <- control arm
                        +--------------------+--------------------+
        counterfactual  |   Y(0) MISSING     |   Y(1) MISSING     |
                        +--------------------+--------------------+

   Randomisation does not fill the missing cell.
   It makes the missing cell exchangeable with the observed one:

        E[ Y(0) | treated ]  =  E[ Y(0) | control ]

   Quasi-experiments replace the coin flip with an argument about why
   the equality holds, or with a model of how the missing cell moves.
```

Redraw the table for any claim you are asked to evaluate. Fill in the unit, the outcome, the two
arms, and point at the empty cell. If nobody can say why the empty cell equals the observed one,
the claim is descriptive, however sophisticated the regression.

### 3.2 The design decision tree

The tree is ordered from strongest to weakest identification. Moving down costs an assumption. The
practical question is not "which design is best" but "which assumption can I defend in this
network, with the data I can obtain, in the time available".

```text
   CONSTRAINT                                 ADMISSIBLE DESIGN         ASSUMPTION IT BUYS
   ------------------------------------------------------------------------------------------
   Can you assign at the receiving unit?
       | yes, and units share no resource     -> Individual RCT            Exchangeability
       | no: stores, DCs, regions, lanes                                   SUTVA
       v
   Assignment must be at group level          -> Cluster RCT               Exchangeable clusters;
       | rollout is one-way and sequential                                 ICC estimable
       v
   Rollout order fixed by operations          -> Stepped wedge             No time-varying
       | treatment cannot persist in a market                              confounding
       v
   One market must serve both arms            -> Switchback                No carryover past
       | no randomisation possible at all                                  the washout
       v
   One treated unit, clean pre-period         -> Synthetic control         Donor pool spans it
   Sharp eligibility threshold                -> Regression discontinuity  Continuity at cutoff
   Variable moves take-up, not outcome        -> Instrumental variables    Exclusion restriction
   Treated and untreated panel                -> Difference-in-differences Parallel trends
   Alike on observables                       -> Matching                  Conditional independence
   One series, no control group               -> Interrupted time series   Pre-trend continues
   Nothing credible                           -> Descriptive + sensitivity No causal claim
```

Two reading rules.

**The design is set by the constraint, not by preference.** An analyst who proposes a store-level
RCT for a DC-level intervention has not understood the intervention. An analyst who proposes a DiD
when randomisation was available has chosen a weaker design for convenience, and should say so.

**A lower rung is not automatically worse.** A well-executed switchback with a stated washout and a
randomisation test can be more credible than a cluster RCT with contaminated arms and a naive
standard error. Design strength is local to the execution.

### 3.3 Four properties worth memorising

**Property 1 — Randomisation buys exchangeability, not truth.** It guarantees that, in
expectation, the arms are comparable on every variable, observed and unobserved. It does not
guarantee implementation, correct measurement, or generalisation. Those are separate claims.

**Property 2 — The assignment unit fixes the inference.** If you assign 40 stores, you have 40
independent units, not 1,040 store-weeks. The design effect converts rows into information.
Topic 04.05 gives the mechanics; this topic gives the design arithmetic.

**Property 3 — Every quasi-experiment is a bet on an unobserved comparison.** DiD bets that the
treated unit would have followed the control's trend. Synthetic control bets that a weighted
donor combination would have followed the treated trajectory. RDD bets that units just above and
below a threshold are alike. IV bets that the instrument affects the outcome only through
treatment. The bet is the method.

**Property 4 — Interference is a design parameter, not a nuisance.** In a network, a unit's outcome
depends on how many others are treated, so the estimand is design-dependent: the effect of
treating 10% of stores is not the effect of treating all of them. You can measure either, but you
must say which.

### 3.4 Why supply chain experiments are harder than web experiments

Analysts trained on digital experimentation import assumptions that do not survive the move to
physical operations.

| Dimension | Web A/B test | Supply chain field experiment |
|---|---|---|
| Assignment unit | User, session, cookie | Store, DC, lane, region, supplier |
| Plausible sample size | 10^5 to 10^8 users | 10 to 200 clusters |
| Interference | Mostly none; some social spillover | Pervasive: shared stock, fleet, labour, demand |
| Carryover | Minutes to days | Weeks to months (pantry loading, learning) |
| Compliance | Near-perfect; code executes | Partial; managers and drivers exercise discretion |
| Cost of the control arm | Near zero | Forgone margin, service, or capacity |
| Outcome latency | Seconds to days | Weeks to quarters |
| Reversibility | Instant rollback | Slow; contracts and resets intervene |
| Blinding | Possible | Rare; operators know which arm they are in |
| Simultaneous changes | Controlled by the platform | Many, owned by different functions |

Four consequences organise the rest of this file. You will usually randomise clusters, because
the unit receiving the treatment is a store, DC, route, or region. You must design for carryover
and interference explicitly. You will often run a switchback rather than a parallel-arm design,
because the same market must serve both arms. And half your questions will be answered without
randomisation, so the quasi-experimental family is not an appendix.

### 3.5 The design-selection table

The tree gives the logic; the table gives the operational mapping. Keep it next to the decision,
not next to the method.

| Constraint you actually face | Admissible design | Identifying assumption | First falsification test |
|---|---|---|---|
| Treatment is applied to individuals and they do not interact | Individual RCT | Exchangeability; SUTVA | Balance table on pre-period covariates |
| Treatment is a store or DC process; units share a DC | Cluster RCT | Exchangeable clusters; ICC estimable | ICC estimated from pre-period; placebo arms |
| Rollout must be sequential and one-way, region by region | Stepped wedge | No time-varying confounding; common secular trend | Compare calendar-time effects across steps |
| Same market, route, or DC must serve both arms | Switchback | No carryover beyond the washout | Vary block length; test for lagged effects |
| One region or DC changed; many comparable units did not | Synthetic control | Donor pool spans the treated unit's factor structure | Pre-period fit; in-space placebos |
| Eligibility is a sharp threshold (credit limit, order size, distance) | Regression discontinuity | Continuity of potential outcomes at the cutoff | Covariate continuity; density test; placebo cutoffs |
| A cost shifter or policy moves take-up but not the outcome | Instrumental variables | Relevance; exclusion; monotonicity | First-stage strength; over-identification; placebo outcome |
| Panel of treated and untreated units, no randomisation | Difference-in-differences | Parallel trends | Event-study pre-trends; placebo treatment date |
| Treated and control comparable on observables | Matching or weighting | Conditional independence; common support | Balance after weighting; sensitivity to unobserved confounding |
| One interrupted series, no control | Interrupted time series | Pre-trend would have continued | Placebo interruption dates; seasonality controls |
| Nothing credible is available | Descriptive comparison, labelled as such | None | Do not use causal language |

> **Note:** The table lists the *first* falsification test, not the only one. A design is credible
> when the assumption is plausible ex ante and survives at least one test that could have failed.
> A test that could not have failed is decoration.

### 3.6 What "credible" means here

Three conditions, all necessary. The assumption is stated in one business sentence, not in the
language of the estimator: "without the carrier change, on-time delivery in the treated region
would have moved in parallel with the control region" is a statement; "we assume no confounding"
is not. The assumption is plausible given the network: a control region sharing the treated
region's DC is not a counterfactual for a DC-level change. And the assumption has been tested
against a result that could have refuted it.

An estimate meeting all three is credible, not true. One meeting none is a number with a
confidence interval attached, which is worse than a number without one, because the interval
implies a warrant that does not exist.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A definition without one is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Counterfactual | The outcome a unit would have had under the treatment it did not receive | Used to mean "before the change" | Potential outcome, `Y(0)` or `Y(1)` |
| Potential outcome | The value of the outcome under a specified treatment state | Confused with the observed outcome | Outcome units |
| Treatment | The intervention whose effect is being estimated | Used to mean any change that happened | Binary indicator, or dose in natural units |
| Control | The state against which treatment is compared | Treated as "doing nothing" when it is the current process | Binary indicator |
| Estimand | The precise quantity the design is intended to estimate | Used interchangeably with "method" | Unit, outcome, contrast, horizon, population |
| Intention-to-treat (ITT) | Effect of assignment to treatment, regardless of compliance | Reported as the effect of the policy as executed | Outcome units |
| SUTVA | Stable unit treatment value assumption: a unit's outcome depends only on its own treatment | Assumed silently in every regression | Assumption, not a quantity |
| Interference | Dependence of one unit's outcome on other units' treatment | Called "spillover" and treated as nuisance | Network or sharing map |
| Carryover | Persistence of an effect after the treatment is withdrawn | Ignored when switching arms quickly | Weeks, or periods |
| Washout | Discarded periods after a switch, intended to remove carryover | Omitted to preserve sample size | Weeks per switch |
| Cluster | A group of units that receive treatment together | Treated as a synonym for "segment" | Store, DC, lane, region |
| Intra-cluster correlation (ICC) | Share of outcome variance that is between clusters | Computed on the post-period, which the treatment contaminates | Proportion, 0 to 1 |
| Design effect | Variance inflation from clustering, `1 + (m − 1) × ICC` | Ignored when the row count looks large | Ratio, unitless |
| Effective sample size | Row count divided by the design effect | Reported as the row count | Independent observations |
| Stepped wedge | A one-way sequential rollout in which all clusters are eventually treated and the order is randomised | Confused with a simple before-after rollout | Clusters per step, steps, periods |
| Switchback | Alternation of treatment and control over time within the same unit | Confused with an A/B test with a time dimension | Blocks per arm, weeks per block |
| Randomisation unit | The entity to which treatment is assigned | Confused with the unit of analysis | Store, DC, lane, region |
| Contamination | Treatment leaking from the treated arm into the control arm | Detected late, if at all | Share of control units exposed |
| Non-compliance | Failure of assigned units to receive or implement the assigned treatment | Analysed by dropping non-compliers | Proportion, 0 to 1 |
| Difference-in-differences (DiD) | Comparison of the change in a treated group with the change in a control group | Used when pre-trends diverge | Outcome units, or percentage points |
| Parallel trends | The assumption that, absent treatment, treated and control would have changed by the same amount | Asserted rather than tested | Slope difference, units per period |
| Synthetic control | A weighted combination of untreated donors built to match a treated unit's pre-period | Used with a donor pool containing treated units | Weights summing to one |
| Donor pool | The untreated units available to build a synthetic control | Chosen for convenience, not comparability | List of units |
| Regression discontinuity (RDD) | Identification from a sharp threshold in an assignment variable | Used when the running variable is manipulable | Running variable and cutoff |
| Running variable | The variable that determines treatment at a threshold | Confused with the outcome | Natural units, centred at the cutoff |
| Instrumental variable (IV) | A variable that affects take-up but affects the outcome only through treatment | Used without an exclusion argument | Natural units |
| Exclusion restriction | The assumption that the instrument affects the outcome only through treatment | Asserted because the instrument is plausibly exogenous | Assumption, not a quantity |
| Weak instrument | An instrument with too little effect on take-up to identify the effect | Diagnosed by a large standard error rather than a first-stage statistic | First-stage F, or partial R-squared |
| Matching | Constructing a comparison group with similar covariates | Matching on post-treatment variables | Covariate distance |
| Interrupted time series (ITS) | Estimation of a level or slope change at an intervention date in one series | Used with a coincident shock in the same period | Level change, slope change |
| Pre-trend | The difference in trajectory between treated and control before treatment | Tested only visually, without a placebo date | Units per period |
| Randomisation inference | Inference from the distribution of estimates over admissible assignments | Replaced by a t-test that assumes independence | p-value |

Four terms deserve expansion, because they cause the most damage when confused.

**Estimand versus method.** An estimand is a question with units attached: "the effect of the new
replenishment rule, relative to the current rule, on units sold per store-week over 26 weeks, for
the 180 stores in the test population". A method is a procedure for answering it. Two analysts can
agree on the method and disagree on the estimand, and the disagreement surfaces only when results
are read.

**ITT versus per-protocol.** ITT is the effect of *offering* treatment; per-protocol is the effect
of *receiving* it. In a promotion with 70% compliance, the ITT effect is roughly 0.7 times the
per-protocol effect if compliance is random, and neither is wrong. The error is to compute
per-protocol and report it as the policy effect.

**ICC and design effect.** The ICC is the share of variance between clusters. It is a property of
the outcome and population, not the treatment. Estimating it on post-treatment data is a mistake,
because the treatment may change between-cluster variance. Use the pre-period and state the
estimate with its uncertainty.

**SUTVA versus interference.** SUTVA is the assumption that a unit's outcome depends only on its
own treatment; interference is its violation. Interference is not binary; it is a function of the
assignment share. The effect of treating 10% of stores differs from treating 50%, because control
stores in the first case have more untreated alternatives. State the assignment share with the
estimate.

## 5. The mechanics, step by step

This section walks from a decision to a design, from a design to an estimate, and from an estimate
to a claim. The steps are ordered. Skipping step 2 because the data are available is the most
common way to produce a well-estimated answer to the wrong question.

### Phase A — Frame the decision

**Step 1. Name the decision, the owner, and the action.**

- Write one sentence: "If the estimate exceeds X, we will do Y, owned by Z, by date D."
- If no value of the estimate changes any action, stop.
- Key branch: **is the decision reversible?** A display change tolerates a weaker design than a
  carrier contract or a DC lease.

**Step 2. Write the estimand in five parts.**

- Unit: store-week, DC-day, lane-week, order.
- Outcome: units sold, on-shelf availability (percentage points), dock-to-stock hours.
- Contrast: new rule versus current rule, not new rule versus nothing.
- Horizon: 26 weeks, including at least one promotion cycle.
- Population: the 180 stores in the test, or the 1,200 in the estate.
- Key branch: **is the outcome the decision variable, or a proxy?** Availability proxies for
  sales; units sold proxies for profit. State the proxy and its failure mode.

**Step 3. Fix the decision rule before the data arrive.**

- Example: "Roll out if the lower bound of the 90% interval on units per store-week exceeds +1.5,
  and no region loses more than 0.5 percentage points of availability."
- Key branch: **is the rule in the same units as the estimate?** A rule in percentage points and
  an estimate in units cannot be compared, and the gap will be closed by rhetoric.

### Phase B — Choose the assignment unit and the design

**Step 4. Match the assignment unit to the decision unit.**

- If the decision is per store, assign per store; per region, per region; per DC, per DC.
- Key branch: **can the treatment be applied below the decision unit?** A planogram can be set per
  store. A DC's slotting policy cannot.

**Step 5. Draw the sharing map and test for interference.**

- List every channel by which one unit's treatment changes another's outcome: shared inventory,
  fleet, labour, supplier allocation, demand (cannibalisation), and customer attention.
- For each, state whether the effect is positive or negative, and how large it could be.
- Key branch: **does the control arm draw from the same pool?** If yes, the design is contaminated,
  and the bias direction depends on whether treatment raises or lowers pool availability.

**Step 6. Estimate carryover and choose a washout.**

- Promotion: two to six weeks of pantry loading and forward buying.
- Routing or slotting change: two to eight weeks of driver and picker learning.
- Planogram: until the next reset, which may be a quarter.
- Key branch: **can you measure carryover rather than assume it?** Run two block lengths and
  compare. If the estimate changes with block length, carryover is present.

**Step 7. Select the design from the tree and write the assumption.**

- One sentence, in business language, naming the comparison and why it holds.
- Key branch: **what would falsify it?** If you cannot name a test, you do not yet have a design.

### Phase C — Protect the comparison

**Step 8. Randomise at the highest level that blocks contamination.**

- If stores share a DC and the treatment affects DC allocation, randomise the DC.
- Stratify or block on strong predictors (region, volume decile, format) before randomising.
- Key branch: **does blocking cost degrees of freedom?** Block on three or four variables at
  most; many small blocks buy balance and lose power.

**Step 9. Pre-register the analysis, or at least timestamp it.**

- Primary outcome, secondary outcomes, clustering level, exclusions, MDE, decision rule. Multiple
  primary outcomes multiply the false discovery problem; Topic 04.03 applies directly.

**Step 10. Protect the control arm from resource drift.**

- Equal service allocation, equal stock, equal attention. A starved control arm is a second
  treatment.
- Key branch: **who is measured on the control arm's performance?** If a regional manager is
  measured on control-store sales, they will act, and the design fails quietly.

**Step 11. Measure compliance and exposure.**

- Record whether treatment was applied, and to what share of units.
- Key branch: **is non-compliance related to the outcome?** Stores that ignore a planogram may be
  the most disciplined stores, which biases the per-protocol estimate.

### Phase D — Analyse at the assignment level

**Step 12. Cluster the standard errors at the assignment unit.**

- Store-week rows with store-level assignment require store-clustered errors, or aggregation to
  store means. Topic 04.05 gives the estimator.
- Key branch: **how many clusters?** Below roughly 40, use a small-sample correction, a wild
  cluster bootstrap, or randomisation inference.

**Step 13. Prefer randomisation inference when the assignment is the design.**

- Enumerate or sample admissible assignments, recompute the estimate under each, and compare the
  observed estimate with that distribution.
- Key branch: **how many admissible assignments exist?** A constrained switchback with eight
  periods may have very few, and the smallest attainable p-value may be 0.125.

**Step 14. Report the estimate in decision units, with the interval and the MDE.**

- "The policy raised units per store-week by 4.1 (90% interval 0.9 to 7.3). The design could
  detect 3.9 units at 80% power."
- Key branch: **is a null being reported as evidence of no effect?** A null with an MDE above the
  decision threshold is uninformative, not negative.

### Phase E — When randomisation is impossible

**Step 15. Name the counterfactual you are constructing.**

- DiD: the control group's change. Synthetic control: the weighted donor trajectory. RDD: units
  just below the cutoff. IV: take-up variation caused by the instrument. Matching: units with the
  same covariates. ITS: the extrapolated pre-trend.
- Key branch: **could the counterfactual units have been the treated unit?** A control region
  with a different format, climate, or customer base is not a counterfactual; it is a different
  business.

**Step 16. Test the assumption on pre-period data.**

- DiD: event-study coefficients before treatment, and a placebo treatment date.
- Synthetic control: pre-period fit, and in-space placebos on donors.
- RDD: covariate continuity, a density test on the running variable, placebo cutoffs.
- IV: first-stage strength, and a test of the instrument against the outcome in a period when
  treatment could not operate.
- Key branch: **could the test have failed?** A pre-trend test with two noisy pre-periods has low
  power. Say so.

**Step 17. Add a negative control outcome and a placebo treatment.**

- Negative control: an outcome the treatment cannot affect. A nonzero effect suggests
  confounding.
- Placebo treatment: move the intervention date back a year in the pre-period. A nonzero "effect"
  suggests the comparison is picking up trend.
- Key branch: **are the placebos measured at the same level and with the same noise?** A placebo
  with ten times the sample is not a fair test.

**Step 18. Bound the bias rather than assuming it away.**

- Ask how large an unobserved confounder would need to be to overturn the estimate (Topic 04.10).
  A point estimate without a sensitivity statement invites over-reading.

**Step 19. Downgrade the language to match the design.**

- Randomised: "raised", "caused", "increased by".
- Quasi-experimental with a tested assumption: "is associated with", "we estimate that", with the
  assumption named in the same paragraph.
- Descriptive: "was higher in", "coincided with". Check that the executive summary uses the same
  verb as the methods section; the gap is where overclaiming enters the organisation.

> **Note:** Before any data are pulled, write a one-page design memo with eight rows: decision,
> estimand, assignment, interference, carryover, assumption, falsification test, and control-arm
> cost. If a row is blank, the design is not ready. Most failed pilots fail at a blank row, not at
> the estimator.

## 6. Worked example

A distributor reviews three operational changes in one quarter. Each is a different decision, and
each admits a different design. All figures are illustrative and internally consistent.

### 6.1 The three decisions

| ID | Change | Decision to be made | Outcome | Natural unit |
|---|---|---|---|---|
| D1 | New shelf-replenishment rule | Roll out to 1,200 stores or stop | Units sold | Store-week |
| D2 | New dock scheduling at one DC | Keep or revert | Throughput | Lines per hour |
| D3 | New carrier contract in one region | Extend to two regions or renegotiate | On-time delivery | Percentage points |

D1 admits a cluster-randomised design: 40 stores are available for a 26-week test. D2 admits a
switchback, because the DC must serve its own market and treatment and control must alternate in
time. D3 has already happened in one region, so the design must construct a counterfactual from
other regions.

### 6.2 D1 — cluster randomisation, computed by hand

Twenty stores are assigned to the new rule and twenty to the current rule. The outcome is units
sold per store-week. The pre-period standard deviation is 12 units, the ICC is 0.10, and the test
runs for 26 weeks.

| Quantity | Symbol | Formula | Value |
|---|---|---:|---:|
| Stores per arm | `k` | given | 20 |
| Weeks per store | `m` | given | 26 |
| Store-weeks per arm | `k × m` | 20 × 26 | 520 |
| Total store-weeks | `2 × k × m` | 2 × 520 | 1,040 |
| Outcome standard deviation | `sigma` | pre-period | 12.0 units |
| Intra-cluster correlation | `rho` | pre-period | 0.10 |
| Design effect | `DEFF` | `1 + (m − 1) × rho` | 3.50 |
| Effective sample per arm | `n_eff` | `520 / 3.50` | 148.6 |
| Effective sample, total | `2 × n_eff` | 2 × 148.6 | 297.1 |
| Standard error of the difference | `SE` | `sigma × sqrt(2 / n_eff)` | 1.392 units |
| MDE at 80% power, 5% two-sided | `MDE` | `2.802 × SE` | 3.90 units |
| MDE as share of mean sales | — | `3.90 / 60` | 6.5% |
| Naive MDE, clustering ignored | — | `2.802 × 12 × sqrt(2 / 520)` | 2.08 units |

Two readings matter. **The naive analysis overstates precision by `sqrt(3.50) = 1.87`.** Regressing
1,040 store-weeks without clustering reports an MDE of 2.1 units and suggests a 2.5-unit effect is
detectable. It is not, and the rollout decision would rest on a number the design cannot support.

**The design can detect 3.9 units, not 2.5.** If the decision rule requires 2.5, the design fails.
The honest responses are more stores, a longer panel, a lower ICC, or a paired pre-period design.
Each has a cost, and the cost belongs in the business case. The cluster count for a 3.0-unit MDE:

```text
n_eff per arm  = 2 × sigma^2 × (z_0.975 + z_0.80)^2 / MDE^2
               = 2 × 144 × (1.960 + 0.842)^2 / 9
               = 288 × 7.851 / 9
               = 251.2
store-weeks    = 251.2 × 3.50 = 879.3
stores per arm = 879.3 / 26 = 33.8  ->  34
total stores   = 68
```

With 40 stores available, the 3.0-unit target is out of reach. Reporting the 3.9-unit MDE and
letting the decision proceed at a 3.0-unit threshold is the failure this arithmetic prevents.

### 6.3 D2 — switchback at one DC, computed by hand

The DC alternates between the new dock schedule and the current schedule in two-week blocks over
16 weeks. Throughput is measured in lines per hour.

| Wave | Weeks | Arm | Throughput (lines/hour) |
|---:|---|---|---:|
| 1 | 1–2 | Control | 100 |
| 2 | 3–4 | Treatment | 106 |
| 3 | 5–6 | Treatment | 104 |
| 4 | 7–8 | Control | 98 |
| 5 | 9–10 | Control | 101 |
| 6 | 11–12 | Treatment | 107 |
| 7 | 13–14 | Treatment | 105 |
| 8 | 15–16 | Control | 99 |

```text
treatment mean = (106 + 104 + 107 + 105) / 4 = 422 / 4 = 105.5
control mean   = (100 +  98 + 101 +  99) / 4 = 398 / 4 =  99.5
difference     = 105.5 - 99.5 = 6.0 lines/hour

sample variance, treatment = 5.00 / 3 = 1.667; sd = 1.291
sample variance, control   = 5.00 / 3 = 1.667; sd = 1.291
SE of difference = sqrt(1.667/4 + 1.667/4) = sqrt(0.833) = 0.913
t = 6.0 / 0.913 = 6.57, df = 6, p < 0.01
```

The estimate is +6.0 lines per hour, about 6% of the control mean. Three qualifications decide
whether it is usable.

**The standard error treats waves as independent, and they are not.** Adjacent waves share
carryover. If the effect persists into the first week after a switch, the first control wave is
contaminated by the previous treatment wave, the control mean is too high, and the effect is
understated.

**With a one-wave washout, only four waves remain,** two per arm. The standard error roughly
doubles. The design fix is longer blocks: four weeks per block gives the same 16 weeks with four
blocks, two per arm, no washout loss, and fewer switches, hence less protection against a
slow-moving shock that coincides with one arm.

**The exact randomisation test has a floor.** With eight waves constrained to alternate in blocks,
the number of admissible assignments is small. If there are 20 admissible schedules, the smallest
attainable two-sided p-value is `2/20 = 0.10`, and no observed difference, however large, can
produce a smaller one. Compute this before the test runs.

### 6.4 D3 — difference-in-differences with a failed pre-trend

The carrier change went live in the northern region at the start of quarter 5.

| Quarter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treated region | 70 | 72 | 74 | 76 | 85 | 87 | 89 | 91 |
| Control region | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 |

```text
treated pre mean  = (70 + 72 + 74 + 76) / 4 = 73.0
treated post mean = (85 + 87 + 89 + 91) / 4 = 88.0
treated change    = +15.0 percentage points

control pre mean  = (60 + 61 + 62 + 63) / 4 = 61.5
control post mean = (64 + 65 + 66 + 67) / 4 = 65.5
control change    = +4.0 percentage points

raw DiD = 15.0 - 4.0 = +11.0 percentage points
```

The raw DiD is +11.0 points. It is also wrong, and the pre-period shows why.

```text
treated pre slope = (76 - 70) / 3 = +2.0 points per quarter
control pre slope = (63 - 60) / 3 = +1.0 points per quarter

differential trend = +1.0 point per quarter
mean post time - mean pre time = 6.5 - 2.5 = 4 quarters
bias in raw DiD = 1.0 x 4 = 4.0 points

trend-adjusted counterfactual for the treated region:
   78, 80, 82, 84   (mean 81.0)
effect = 88.0 - 81.0 = +7.0 percentage points
```

The raw estimate overstates the effect by 4.0 points, exactly the differential trend multiplied by
the four-quarter gap between the pre and post means. The control region is not admissible: it was
improving at half the treated rate before the intervention, and nothing suggests the trends would
have converged in quarter 5.

Three responses are legitimate. Find a control region with a parallel pre-trend and rerun. Model
region-specific linear trends, which identifies from deviations around those trends and needs more
than eight periods. Or build a synthetic control from a donor pool of regions weighted to match
the treated region's pre-period level and slope.

### 6.5 What the three estimates establish

| Design | Estimate | Assumption that carries it | What would overturn it |
|---|---|---|---|
| D1 cluster RCT | +4.1 units/store-week (illustrative observed) | Exchangeable stores; no DC-level contamination | A sharing channel between arms, or an ICC above 0.10 |
| D2 switchback | +6.0 lines/hour | No carryover beyond the washout | A lagged effect that survives one wave |
| D3 DiD | +11.0 points raw; +7.0 trend-adjusted | Parallel trends (raw), or correct trend specification (adjusted) | A donor region with a parallel pre-trend giving a different answer |

The three numbers are not competing estimates of one effect. They answer three decisions, in
three units, under three assumptions. The professional error is to average them or treat the
largest as truth. Report each with its assumption and let the pre-stated decision rule determine
which is fit for purpose.

### 6.6 What an analyst should have asked

1. For D1, what is the ICC on the pre-period, and its interval? An ICC of 0.10 with an interval of
   0.03 to 0.22 changes the MDE by a factor of 1.4.
2. For D3, why is this control the right counterfactual? The pre-trend says it is not, and the
   burden is on the analyst to find one that is.
3. For all three, what did the control arm cost, and who approved it? An unbudgeted control arm
   will be quietly compromised.

## 7. Code, query, or calculation

One compact block. It computes the cluster arithmetic and runs a randomisation test on the
assignment, which is the inference that matches the design.

```python
import numpy as np
from scipy import stats

sigma, m, rho = 12.0, 26, 0.10
deff = 1 + (m - 1) * rho
n_rows_per_arm = 20 * m
n_eff = n_rows_per_arm / deff
se = sigma * np.sqrt(2 / n_eff)
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
mde = z * se
print(f"DEFF={deff:.2f} n_eff={n_eff:.1f} SE={se:.3f} MDE={mde:.2f}")

rng = np.random.default_rng(7)
y = rng.normal(60, 4.40, 40)          # 40 cluster means, SD implied by sigma and rho
obs = y[:20].mean() - y[20:].mean()
null = []
for _ in range(2000):
    p = rng.permutation(40)
    null.append(y[p[:20]].mean() - y[p[20:]].mean())
pval = np.mean(np.abs(null) >= abs(obs))
print(f"observed={obs:.2f} randomisation p={pval:.3f}")
```

The first half reproduces section 6.2: `DEFF=3.50`, `n_eff=148.6`, `SE=1.392`, `MDE=3.90`. If
your output differs, check that `m` is weeks per store, not total weeks, and that `rho` is a
proportion, not a percentage.

The second half permutes the *assignment*, not the rows, and compares the observed difference in
cluster means with the distribution generated by reassignment. That is the correct null for a
randomised design and it does not require normality. With 20 clusters per arm the distribution is
discrete; when the permutation and t results disagree materially, trust the permutation.

## 8. Data, standards, and regulatory context

No regulator prescribes how a commercial supply chain experiment must be designed. The absence is
itself a risk: without an external standard, design is governed by internal convention, and
internal convention usually defaults to a before-after comparison. What follows is the data the
design must produce and the legal constraints that occasionally bind.

### 8.1 Artifacts an experiment must leave behind

| Artifact | What it contains | Why it is needed | Grade |
|---|---|---|---|
| Assignment log | Unit, arm, date, randomisation seed or mechanism | Without it, the design cannot be defended or reproduced | C |
| Exposure log | Which units actually received treatment, and when | Separates ITT from per-protocol | C |
| Baseline panel | Pre-period outcome and covariates at the assignment unit | ICC, balance, and pre-trend tests | C |
| Interference map | Sharing channels between units, and the mitigation | Makes contamination visible before it biases the estimate | C |
| Analysis plan | Primary outcome, clustering level, exclusions, MDE, decision rule | Prevents specification search after the results | C |
| Decision record | Who approved the control-arm cost, and on what basis | The control arm survives only if it is owned | C |

The grades are C because these are practitioner conventions, not standards. Their absence is not a
compliance failure but an analytical one, surfacing as an estimate nobody can replicate. The same
applies to the identifiers the design depends on: store, DC, lane, region, item–location, and
supplier legal entity. Each must be stable over the analysis window, and the hierarchy must be a
tree rather than a graph. A store renumbered mid-panel becomes two clusters in the software and
one store in the business.

### 8.2 Legal and regulatory constraints

These are general framings, not legal advice, and they are graded C because they are not verified
against statute text here. Obtain counsel before relying on any of them.

| Constraint | Where it binds | What it constrains in a design | Grade |
|---|---|---|---|
| Data protection law (for example, the EU General Data Protection Regulation) | Experiments on customer-level data | Lawful basis, minimisation, retention, and the limits of consent for operational testing | C |
| Employment and works-council law | Changes to warehouse or driving work methods | Consultation before changing routine and measurement | C |
| Price-discrimination law (for example, the U.S. Robinson–Patman Act) | Varying price across competing resellers | Whether a price test can lawfully treat two competing resellers differently | C |
| Antitrust and information-exchange rules | Sharing results with competitors or through a trade body | Whether pooled benchmark results are an unlawful exchange | C |
| Consumer-protection rules | Withholding a promised service level from a control group | Whether the control arm is a permissible service level | C |
| Revenue-recognition rules | Tests that change delivery terms or acceptance | Whether a pilot changes the timing of revenue | C |

> **Note:** The common thread is that the control arm is an operational decision with legal
> exposure, not a statistical convenience. Withholding a contractual service level from paying
> customers is a different act from withholding a website feature.

Almost everything else in experimental design is convention rather than requirement: store-level
randomisation, a 26-week window, twenty clusters per arm, and 95% confidence are inherited habits,
not rules. An analyst who can separate the two can challenge a convention instead of treating it
as a physical constant, which is why the reasoning must be explicit.

## 9. Industry practice and case evidence

The strongest public evidence on design comes from agriculture, clinical trials, public policy,
and online marketplaces. Supply chain field experiments are less publicly documented, and this
section says so rather than implying otherwise.

### 9.1 The provenance: randomisation as the physical basis of validity

Fisher's *The Design of Experiments* (Oliver and Boyd, 1935) set out the three principles that
still organise the field: randomisation, replication, and blocking. The argument was not that
randomisation guarantees balance in any one trial, but that it provides the physical basis for
the validity of the test: randomly allocating treatments to plots simulates the independence the
analysis assumes. The book also frames the null hypothesis as something never proved, only
possibly disproved. Grade A.

Cox's *Planning of Experiments* (Wiley, 1958) extended this to multi-stage randomisation, where the
lesson is that randomisation must cover every stage at which an important error arises: a
store-level draw does not cover a DC-level allocation decision. Grade A.

### 9.2 Stepped wedge: a clinical design with a supply chain shape

Hussey and Hughes (2007), "Design and analysis of stepped wedge cluster randomized trials",
*Contemporary Clinical Trials* 28(2), 182–191, is the standard reference. The motivating trial
randomised 24 county health districts in Washington State to expedited partner therapy at one of
four times, six districts per step, with five measurement points at six-month intervals. Every
cluster eventually receives the intervention; the order is randomised.

Three findings transfer to a regional rollout. The analysis must include time effects, because a
within-cluster comparison that ignores calendar time is biased when time effects exist. Power
depends on the number of steps and measurement times more than on the coefficient of variation, so
a two-step rollout is much weaker than a five-step rollout with the same number of clusters.
Delayed treatment effects reduce power substantially, and extra monitoring periods do not fully
recover it. That third point is the supply chain warning: a new planning process does not take
effect on switch-on day, because planners need weeks to trust it. Grade A.

### 9.3 Interference: why the marketplace result matters for fulfilment

Johari, Li, Liskovich, and Weintraub (2022), "Experimental Design in Two-Sided Platforms: An
Analysis of Bias", *Management Science* 68(10), compares customer randomisation, listing
randomisation, and a two-sided design. The central result is that bias depends on market balance.
In a demand-constrained market, customer randomisation is unbiased and listing randomisation is
biased; in a supply-constrained market, the reverse holds. For positive interventions, the biased
estimator overestimates.

The supply chain translation is direct. A fulfilment experiment that randomises orders to a new
pick algorithm is a demand-side design, and it is biased when capacity binds, because treated
orders consume capacity control orders would have used. A design that randomises fulfilment
centres is a supply-side design, and it is biased when demand binds, because untreated demand
shifts to treated centres. The correct design depends on which side of the market is scarce.
Grade A.

### 9.4 Switchback: time as the randomisation unit

Bojinov, Simchi-Levi, and Zhao (2023), "Design and Analysis of Switchback Experiments",
*Management Science* 69(7), 3759–3777, treats the design problem formally. When a firm alternates
treatment over time within the same unit, block length trades protection against carryover for the
number of switches. The paper frames this as a minimax robust optimisation over the assumed
carryover order, derives the worst-case assignment, and gives exact randomisation-based p-values
and a finite-population central limit theorem for conservative tests and intervals. It also
provides a data-driven procedure for identifying the carryover order. Grade A.

### 9.5 Synthetic control: one unit, many donors

Abadie, Diamond, and Hainmueller (2010), "Synthetic Control Methods for Comparative Case Studies",
*Journal of the American Statistical Association* 105(490), 493–505, formalises a weighted
combination of untreated donors that matches the treated unit's pre-period. The application is
California's Proposition 99 tobacco control programme, using annual state data from 1970 to 2000
and a 38-state donor pool. Synthetic California is a weighted mix of Colorado (0.164), Connecticut
(0.069), Montana (0.199), Nevada (0.234), and Utah (0.334). The paper reports per-capita cigarette
sales about 26 packs lower by 2000, an average reduction of about 20 packs (roughly 25%) over
1989–2000, and a permutation-based p-value of 1/39, approximately 0.026.

The supply chain analogue is a single DC that received a new warehouse management system while
comparable DCs did not. Two cautions carry over. The donor pool must be uncontaminated: a DC that
later adopted the system is not a donor. And inference is in-space placebo-based, so the smallest
attainable p-value is set by the number of donors, not the number of weeks. Grade A.

### 9.6 Regression discontinuity: the threshold as an experiment

Thistlethwaite and Campbell (1960), "Regression-discontinuity analysis: An alternative to the ex
post facto experiment", *Journal of Educational Psychology* 51(6), 309–317, is the origin. The
study compared 5,126 students who received Certificates of Merit with 2,848 who received only
letters of commendation, exploiting the sharp cutoff in qualifying test scores. The logic is the
discontinuity: students just above and just below the threshold are alike except for the award.

Imbens and Lemieux (2008), "Regression discontinuity designs: A guide to practice", *Journal of
Econometrics* 142(2), 615–635, recommends graphical analysis of binned outcomes against the
running variable, local linear regression near the cutoff, cross-validation for bandwidth, a
plug-in variance estimator, and tests for discontinuities in covariates, in the density of the
running variable, and in outcomes at other values of the running variable.

The supply chain version is a credit limit, a minimum order quantity, or a distance band. A
customer just above a credit threshold gets a different payment term; a store just inside a
delivery radius gets next-day service. The design is credible when the threshold is applied
mechanically and units cannot sort across it. It fails when the threshold is negotiable, because
then the running variable is manipulated and units just above are not comparable to units just
below. Grade A.

### 9.7 The evidence gap in supply chain

None of the strongest papers above is a supply chain experiment. They are agriculture, clinical
trials, public policy, and online marketplaces. Supply chain field experiments exist inside
companies but are rarely published with their design, assignment log, and pre-registered
analysis, so the methods transfer and the base rates do not. A 20-cluster clinical trial does not
prove a 20-cluster store test has power, and marketplace interference is not shared inventory.

### 9.8 What organisations do, versus what the design requires

| What the design requires | What organisations commonly do | Why the gap persists | Grade |
|---|---|---|---|
| Randomise at the decision unit | Compare the affected region with last year | The control arm has no budget owner | C |
| State the assumption in one sentence | Report the estimate and call it significant | The assumption is seen as a caveat, not a result | C |
| Pre-register the primary outcome | Choose the outcome after seeing the lift | Reporting incentives reward a finding | C |
| Measure compliance and report ITT | Report the subset that implemented | Per-protocol numbers look larger | C |
| Wash out carryover | Switch arms weekly to finish sooner | The timeline is set by the operating calendar | C |
| Use an uncontaminated donor pool | Use whatever units are convenient | No one checks whether donors were treated | C |

These are practitioner observations, not survey statistics. Their value is diagnostic: if three
or more rows describe your organisation, the design problem is organisational before it is
technical, and no estimator will fix it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Randomising the wrong unit | Treatment is applied by a manager covering several stores | Assignment is finer than treatment delivery | Attenuated effect and understated standard errors | Draw the delivery map before choosing the unit |
| Naive standard errors on clustered assignment | Row count is large; cluster count is small | Rows treated as independent | Intervals too narrow by `sqrt(DEFF)` | Cluster at the assignment unit; report the ICC |
| Carryover not washed out | Effect appears in the first control period after a switch | Treatment persists into the next block | Control mean contaminated; effect understated | Washout period; vary block length |
| Contamination through shared inventory | Control availability correlates with treated availability | The same DC allocates stock to both arms | The contrast measures allocation, not treatment | Randomise the DC, or isolate the pool |
| Control arm starved of resource | Control service falls for reasons unrelated to the test | Attention and stock follow the treatment | Effect inflated by neglect | Equal service protocol; monitor control KPIs |
| Non-compliance analysed per-protocol | Compliant stores differ from non-compliant at baseline | Selection into compliance | Estimate applies to a self-selected subset | Report ITT first; model compliance separately |
| DiD with divergent pre-trends | Event-study coefficients slope before treatment | Treated and control were already diverging | Bias equal to trend difference times horizon | Find a parallel control; model trends explicitly |
| Synthetic control with a contaminated donor pool | A donor later adopted the same policy | Donor trajectory contains the treatment | Counterfactual too low or too high | Screen donors for treatment; report the pool |
| RDD with a manipulable running variable | Units cluster just above the cutoff | Sorting around the threshold | Units near the cutoff are not comparable | Density test; check how the threshold is administered |
| Weak instrument | First-stage effect is small; interval is enormous | Instrument explains little take-up | Estimates unstable and biased toward the confounded estimate | Report first-stage F; find a stronger shifter |
| Matching on post-treatment variables | Covariates include a consequence of treatment | Bad control, as in 04.06 | Bias of unknown sign | Match on pre-period covariates only |
| ITS with a coincident shock | Intervention date coincides with a holiday or competitor exit | Trend extrapolation absorbs the shock | Effect is the shock | Placebo dates; add a control series |
| Control-arm cost hidden from the decision | No budget line for forgone margin | The cost is real but unowned | The test is cancelled or compromised mid-flight | Cost the control arm before approval |

Two deserve a note. **Contamination is directional.** If treatment raises the treated stores' draw
on a shared DC, control stores receive less stock, the control mean falls, and the estimated effect
rises; if treatment lowers the draw, the bias reverses, and knowing the sign often bounds it. **A
control arm with no owner is not a control arm.** Someone is measured on those units; if they are
not in the design conversation, they will optimise out of the control condition, usually by
escalating service "just to be safe".

## 11. Assumptions and limits

| Assumption | Design that needs it | Why it is needed | How to test or weaken it |
|---|---|---|---|
| Exchangeability under randomisation | Individual and cluster RCT | Treated and control would have been alike absent treatment | Balance table on pre-period covariates; placebo arms |
| No interference (SUTVA) | Individual and cluster RCT | A unit's outcome depends only on its own treatment | Draw the sharing map; randomise the sharing unit |
| No carryover beyond the washout | Switchback, crossover | The control block is uncontaminated by the previous treatment | Vary block length; estimate lagged effects |
| Exchangeable clusters and stable ICC | Cluster RCT | The design effect and variance formula depend on it | Estimate ICC on pre-period; check across strata |
| No time-varying confounding | Stepped wedge | Calendar time is not confounded with treatment status | Compare step effects across calendar time |
| Common secular trend | Stepped wedge, DiD | All clusters would have moved together absent treatment | Event-study pre-trends; placebo dates |
| Parallel trends | DiD | The control change is the treated counterfactual change | Pre-period slope comparison; sensitivity to trend specification |
| Donor pool spans the treated unit's factors | Synthetic control | A weighted donor combination can reproduce the counterfactual | Pre-period fit; in-space placebos; leave-one-out |
| Continuity at the cutoff | RDD | Units just above and below are alike except for treatment | Covariate continuity; binned outcome plot |
| No manipulation of the running variable | RDD | Units cannot sort across the threshold | Density test; institutional check on administration |
| Relevance of the instrument | IV | The instrument must move treatment take-up | First-stage F; partial R-squared |
| Exclusion restriction | IV | The instrument affects the outcome only through treatment | Argue the mechanism; placebo outcomes; over-identification |
| Conditional independence | Matching, weighting | Treated and control are alike conditional on observables | Balance after weighting; sensitivity to unobserved confounding |
| Common support | Matching, weighting | Comparable units exist in both groups | Trim to the overlap region; report the loss |
| The pre-trend would have continued | Interrupted time series | The extrapolation is the counterfactual | Placebo interruption dates; add a control series |

> **Note:** These assumptions are not equally demanding. Exchangeability under randomisation is
> guaranteed by the mechanism; parallel trends is an empirical claim about an unobserved
> trajectory. The second is a much stronger bet, and the report's language should reflect that.

Two limits of the framework are worth stating. It estimates the effect of a treatment actually
applied at a particular scale: with interference, the effect of treating 10% of stores is not the
effect of treating all of them, and no estimation recovers the latter without an explicit
interference model, itself an assumption. And randomisation does not solve external validity: a
clean test in 40 stores in one region estimates an effect for those stores in that period, and
whether it holds elsewhere is a separate question whose answer is usually "test again".

## 12. Exercises

### Exercise 1 — Audit a pilot (diagnosis, 45 minutes)

Take a pilot you know, or construct one from a public announcement: a new depot, a routing change,
a supplier consolidation. Write a one-page audit.

**Deliverable:** design classification (randomised, quasi-experimental, descriptive), the
counterfactual used, the identifying assumption, and one test that could have failed.
**Acceptance criteria:** classification justified by the assignment mechanism, not the
announcement's vocabulary; assumption in one business sentence; test specific enough to run.
**Time:** 45 minutes.

### Exercise 2 — Cluster arithmetic (calculation, 40 minutes)

A test assigns 24 stores per arm for 26 weeks. The outcome standard deviation is 15 units per
store-week, the ICC is 0.08, and mean sales are 75 units per store-week.

**Deliverable:** design effect, effective sample size per arm, standard error of the difference,
MDE at 80% power and 5% two-sided, and stores per arm required for a 3.0-unit MDE. Also compute the
MDE an analyst would report if clustering were ignored.
**Acceptance criteria:** every formula shown with units; naive and correct MDEs both stated;
required cluster count rounded up with the arithmetic visible.
**Time:** 40 minutes.

### Exercise 3 — Switchback schedule (design, 50 minutes)

A DC must alternate between a new and a current dock schedule. Expected carryover is one two-week
wave. The test window is 24 weeks.

**Deliverable:** a schedule with block lengths, washout, usable waves per arm, and the analysis.
**Acceptance criteria:** weeks per block and total weeks stated; washout explicit; number of
admissible assignments and smallest attainable p-value estimated; analysis uses the assignment
path rather than a two-sample t-test.
**Time:** 50 minutes.

### Exercise 4 — Stepped-wedge rollout (design, 50 minutes)

A retailer will roll a planning system to 60 regions over 12 months in waves. Operations want the
highest-volume regions first.

**Deliverable:** a randomisation scheme respecting the operational constraint, the time-effects
model, and the power consequence of four steps instead of six.
**Acceptance criteria:** states which parts of the order are randomised and which are fixed; model
includes calendar-time effects; power discussion names steps and pre/post measurement points.
**Time:** 50 minutes.

### Exercise 5 — Difference-in-differences with a pre-trend test (calculation, 45 minutes)

Using the table below, compute the raw DiD and the trend-adjusted effect. State whether the
control region is admissible.

| Quarter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treated region | 70 | 72 | 74 | 76 | 85 | 87 | 89 | 91 |
| Control region | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 |

**Deliverable:** pre means, post means, changes, raw DiD, pre-period slopes, bias, trend-adjusted
effect, and a one-paragraph admissibility judgement.
**Acceptance criteria:** all arithmetic shown; bias decomposition `differential slope × horizon`
stated with units; judgement names what would make a control admissible.
**Time:** 45 minutes.

### Exercise 6 — Synthetic control donor pool (design, 60 minutes)

A single DC received a new warehouse management system (WMS). You have 30 comparable DCs and five
years of monthly throughput.

**Deliverable:** donor screening rules, predictor set, pre-period fit criterion, and placebo
procedure.
**Acceptance criteria:** at least two exclusion rules (later adoption, different automation
level); predictors include mix and seasonality, not only level; fit criterion numeric; placebo
procedure states the number of placebos and smallest attainable p-value.
**Time:** 60 minutes.

### Exercise 7 — RDD feasibility (design, 45 minutes)

A credit team applies a payment-term change at a credit-score threshold. Stores just above receive
30-day terms; stores just below receive 14-day terms.

**Deliverable:** a memo on whether an RDD is credible, the running variable and cutoff, continuity
tests, and manipulation risk.
**Acceptance criteria:** names who administers the threshold and whether exceptions are granted;
at least three continuity tests; manipulation risk assessed institutionally, not only
statistically.
**Time:** 45 minutes.

### Exercise 8 — Instrument hunt (design, 50 minutes)

You want the effect of a price change on units sold, but price and quantity are set jointly.

**Deliverable:** two candidate instruments, the relevance, exclusion, and monotonicity arguments
for each.
**Acceptance criteria:** each instrument is a variable, not a method; exclusion names the direct
channel and why it is closed; monotonicity stated in terms of which customers change behaviour.
**Time:** 50 minutes.

### Exercise 9 — Interference map (application, 40 minutes)

For an omnichannel retailer, map the channels through which a fulfilment experiment in 20 stores
can affect the other 1,180.

**Deliverable:** a sharing map with at least six channels, the sign of each effect on the control
arm, and the design change that would block the two largest.
**Acceptance criteria:** each channel has a mechanism and direction; at least one positive and one
negative for control; design change is specific (randomise the DC, isolate the stock pool,
stagger timing).
**Time:** 40 minutes.

### Exercise 10 — Design-selection memo (design, 60 minutes)

A COO asks whether a new supplier collaboration programme improved on-time delivery. The programme
ran in three of eleven regions, chosen because those regions had the worst baseline.

**Deliverable:** a two-page memo selecting a design, stating the assumption, falsification test,
expected bias direction, and the language for the executive summary.
**Acceptance criteria:** selection follows from the constraint (selection on baseline rules out an
untreated comparison); assumption in one sentence; bias direction argued, not asserted; summary
language matches the design's strength.
**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 Exercise 2 arithmetic, in full

Given 24 stores per arm, 26 weeks, `sigma = 15` units per store-week, `rho = 0.08`, mean 75 units.

```text
m               = 26 weeks per store
DEFF            = 1 + 25 x 0.08 = 3.00
store-weeks/arm = 24 x 26 = 624
n_eff per arm   = 624 / 3.00 = 208.0
n_eff total     = 416.0

SE              = 15 x sqrt(2 / 208) = 15 x 0.098058 = 1.471 units
MDE (z)         = (1.960 + 0.842) x 1.471 = 4.12 units  (5.5% of 75)
MDE (t, df 46)  = (2.013 + 0.842) x 1.471 = 4.20 units
SE (naive)      = 15 x sqrt(2 / 624) = 0.849
MDE (naive)     = 2.802 x 0.849 = 2.38 units
```

The cluster-mean check confirms the arithmetic: `sigma^2 x DEFF / m = 225 x 3 / 26 = 25.96`, so
a cluster mean has standard deviation 5.095, and the difference of two means of 24 clusters has
`SE = 5.095 x sqrt(2 / 24) = 1.471`. The design effect and the cluster-mean analysis are the same
calculation in two costumes. The naive MDE understates the true 4.12 by `sqrt(3.00) = 1.732`; an
organisation planning on it would believe it could detect a 2.5-unit effect and would be wrong.

The cluster count for a 3.0-unit MDE:

```text
n_eff per arm     = 2 x sigma^2 x z^2 / MDE^2
                  = 2 x 225 x 2.802^2 / 9
                  = 450 x 7.851 / 9
                  = 392.6
store-weeks/arm   = 392.6 x 3.00 = 1,177.7
stores/arm        = 1,177.7 / 26 = 45.3  ->  46
total stores      = 92
```

A 3.0-unit MDE needs 46 stores per arm, not 24. The gap is the design finding, and it should be
reported before the test starts, not after it produces a null.

### 13.2 Exercise 5 arithmetic, in full

Exercise 5 uses the table worked in section 6.4, so the arithmetic is the same.

```text
treated change = 88.0 - 73.0 = +15.0 points
control change = 65.5 - 61.5 =  +4.0 points
raw DiD        = 15.0 - 4.0  = +11.0 percentage points
slopes         = +2.0 vs +1.0 points per quarter
bias           = 1.0 x 4.0   = +4.0 points
trend-adjusted = 88.0 - 81.0 = +7.0 percentage points
```

The control is not admissible for the raw DiD: its pre-period slope is half the treated slope, so
parallel trends is contradicted before the intervention. A control with a slope between 1.9 and
2.1 points per quarter would be admissible, provided the slope is estimated over enough periods to
rule out noise. Eight quarters is thin for that judgement, and the answer should say so.

### 13.3 Design rubric for Exercises 3, 4, 6, 7, 9, and 10

Score each criterion 0, 1, or 2. A design scoring below 9 of 16 is not ready to run.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Estimand | Not stated | Unit and outcome stated | Unit, outcome, contrast, horizon, population stated |
| Assignment unit | Not stated | Stated but mismatched to delivery | Matched to decision and delivery mechanism |
| Interference | Not considered | Sharing channels listed | Channels listed with sign and mitigation |
| Carryover | Not considered | Duration asserted | Duration estimated or bounded, with a washout rule |
| Assumption | Not stated | Named in statistical language | One business sentence with the comparison named |
| Falsification | None | One test named | One test that could fail, with the failing result |
| Analysis level | Row-level | Clustered, no small-sample treatment | Clustered at assignment; permutation or small-sample correction |
| Decision rule | Absent | Threshold stated after results | Threshold, owner, and action stated before data |

### 13.4 Rubric for the audit exercises (1 and 10)

| Criterion | Weak (0–1) | Adequate (2–3) | Strong (4) |
|---|---|---|---|
| Classification | Uses the announcement's language | Names randomised or observational | Names the assignment mechanism and the counterfactual |
| Assumption | Says "controlling for confounders" | Names one assumption | Names it, argues plausibility, and gives a test |
| Bias direction | Not discussed | Mentions possible bias | States the sign and the mechanism |
| Language | Executive summary overclaims | Mixed | Verbs match the design throughout |
| Next step | "Collect more data" | "Run a regression" | A specific design or test with a named owner |

### 13.5 Outline answers for the remaining exercises

**Exercise 1.** The depot opened where service was worst, so selection is on the outcome and a
before-after comparison is biased upward. The counterfactual is the depot's absence; any DiD
assumes the comparison region would have followed the same service trajectory. Test pre-period
slopes before examining the post-period.

**Exercise 3.** Twenty-four weeks of two-week waves gives 12 waves. A one-wave washout leaves 10
usable, 5 per arm. Four-week blocks give six blocks, 3 per arm, no washout loss but fewer
switches. State both and choose on carryover evidence; compute the smallest attainable p-value
before running.

**Exercise 4.** Randomise order within volume strata, not across them, so high-volume regions go
early without confounding volume with time. Include calendar-time fixed effects and report power
for four and six steps; 60 regions gives 15 per step, adequate for a moderate effect.

**Exercise 6.** Exclude donors that adopted the system, changed automation, or changed format.
Match on throughput, mix, seasonality, and labour model. Report pre-period RMSE and the
post-to-pre ratio; run in-space placebos on every donor and report the treated DC's rank. With 30
donors the smallest p-value is about 1/31.

**Exercise 7.** Credible only if the threshold is mechanical and exceptions are rare. Test
covariate continuity, run a density test on the credit score, and check for bunching above the
threshold. If the credit team grants exceptions, the running variable is manipulated.

**Exercise 8.** A cost shifter is relevant if it moves the price the customer faces. Exclusion
fails if the shifter is observable to customers and signals future price. Monotonicity requires
that it does not make some customers more likely to buy and others less at the same price change.

**Exercise 9.** Channels include shared DC stock, fleet capacity, labour, cannibalisation through
nearby stores, supplier allocation, and service queues. Stock and fleet usually dominate; block
them by randomising the DC or isolating a stock pool.

**Exercise 10.** The regions were chosen on the worst baseline, so selection is on the outcome and
comparison with the other eight is biased toward finding an effect. Use a synthetic control from
regions with similar pre-trends, a matched comparison on level and trend, or an ITS with a placebo
date. Say "is associated with", not "improved".

## 14. Interview questions

**1. "When would you randomise, and when would you not?"**

A strong answer starts from the decision and the assignment mechanism: if treatment can be
assigned at the unit that receives it and units do not interact, randomise; if rollout is one-way,
use a stepped wedge; if the same market must serve both arms, use a switchback; if the change has
already happened, construct a counterfactual and name the assumption. A weak answer says
"randomise whenever possible" without saying what makes it possible.

**2. "You have 40 stores and 26 weeks. How precise is your test?"**

A strong answer asks for the ICC and the outcome standard deviation, computes
`DEFF = 1 + (m − 1) × ICC`, divides the row count by it, and reports the MDE in outcome units. A
weak answer divides by the row count and reports a standard error on 1,040 independent
observations.

**3. "What is the design effect, and why does it matter more in supply chains than on the web?"**

A strong answer defines it as variance inflation from clustering, gives the formula, and explains
that supply chain experiments have tens of clusters rather than millions of users, so the design
effect can be 3 to 10 rather than negligible. A weak answer defines it and then says it does not
matter because the sample is large.

**4. "Your control stores share a DC with treated stores. What is the problem?"**

A strong answer names the contamination channel, states the direction of the bias, and proposes
randomising the DC or isolating the stock pool. A weak answer says "there may be spillover" and
does not say which way it pushes the estimate.

**5. "How would you design a switchback for a DC?"**

A strong answer states block length, washout, usable waves per arm, analysis on the assignment
path, and the smallest attainable p-value. A weak answer says "alternate weekly and compare the
averages", which ignores carryover and the discrete randomisation distribution.

**6. "What does parallel trends mean, and how would you test it?"**

A strong answer states it as a counterfactual claim about the treated group's trajectory, tests
it with pre-period event-study coefficients and a placebo date, and says that failure makes the
control inadmissible rather than calling for a robustness check. A weak answer says "the groups
should be similar" and tests balance on levels, which is not the assumption.

**7. "When is a regression discontinuity credible in a supply chain setting?"**

A strong answer requires a mechanically applied threshold, no sorting across it, and continuity of
covariates at the cutoff, and names a density test and a covariate continuity check. A weak answer
says "when there is a cutoff" and never asks who administers the threshold.

**8. "What makes an instrument valid?"**

A strong answer gives relevance, exclusion, and monotonicity, and illustrates each with a supply
chain example such as a freight cost shifter that moves price but not demand directly. A weak
answer says "it must be correlated with the treatment but not the outcome", which is the
definition without the argument or the test.

**9. "A vendor says their pilot produced a 12% lift. What do you ask?"**

A strong answer asks for the assignment mechanism, the control group, the pre-period, the
clustering, the duration, and the outcome definition, and grades the claim accordingly. A weak
answer asks only for the sample size, or accepts the number because it is large.

**10. "What is the difference between ITT and per-protocol, and which do you report?"**

A strong answer defines both, reports ITT as primary because it is the effect of the policy as
offered, and uses per-protocol as a secondary analysis with an argument about compliers. A weak
answer reports per-protocol because it is larger, without saying the estimand changed.

## 15. Key terms

- **Counterfactual** — the outcome under the treatment a unit did not receive.
- **Potential outcome** — the outcome under a specified treatment state, observed or not.
- **Estimand** — the precise quantity to estimate, with unit, outcome, contrast, horizon, population.
- **ATE / ATT / LATE** — average effect for the population, for the treated, and for instrument-driven compliers.
- **ITT** — effect of assignment to treatment, regardless of compliance.
- **SUTVA** — the assumption that a unit's outcome depends only on its own treatment.
- **Interference** — dependence of one unit's outcome on other units' treatment.
- **Carryover** — persistence of an effect after treatment is withdrawn.
- **Washout** — discarded periods after a switch, intended to remove carryover.
- **Cluster** — a group assigned to treatment together; the unit of independence.
- **ICC** — share of outcome variance that is between clusters.
- **Design effect** — variance inflation from clustering, `1 + (m − 1) × ICC`.
- **Effective sample size** — row count divided by the design effect.
- **Cluster RCT** — an RCT that assigns clusters rather than individuals.
- **Stepped wedge** — a one-way randomised rollout in which all clusters are eventually treated.
- **Switchback** — alternation of treatment and control over time within the same unit.
- **Contamination** — treatment leaking into the control arm through a shared resource.
- **Non-compliance** — failure of assigned units to receive or implement treatment.
- **DiD** — difference-in-differences, comparing changes across treated and control groups.
- **Parallel trends** — the assumption that the groups would have changed in parallel absent treatment.
- **Event study** — period-by-period treatment effects relative to a reference period.
- **Synthetic control** — a weighted donor combination built to match a treated unit's pre-period.
- **Donor pool** — the untreated units available to construct a synthetic control.
- **RDD** — regression discontinuity, identifying from a threshold in a running variable.
- **IV and exclusion restriction** — an instrument that moves take-up but affects the outcome only through treatment.
- **Matching** — constructing comparable controls from observed covariates.
- **ITS** — interrupted time series, estimating a level or slope change at an intervention date.
- **Randomisation inference** — inference from the distribution of estimates over admissible assignments.
- **External validity** — whether an effect generalises beyond the units and period studied.

## 16. Sources

**Peer-reviewed and monographs (Grade A)**

- Fisher, R. A. (1935). *The Design of Experiments*. Edinburgh: Oliver and Boyd. Source of the
  three principles and of the argument that randomisation is the physical basis of validity.
- Cox, D. R. (1958). *Planning of Experiments*. New York: Wiley. Multi-stage randomisation,
  randomised blocks, and the argument that randomisation must cover every stage at which an
  important error arises.
- Hussey, M. A., and Hughes, J. P. (2007). "Design and analysis of stepped wedge cluster randomized
  trials." *Contemporary Clinical Trials*, 28(2), 182–191. DOI: 10.1016/j.cct.2006.05.007.
- Bojinov, I., Simchi-Levi, D., and Zhao, J. (2023). "Design and Analysis of Switchback
  Experiments." *Management Science*, 69(7), 3759–3777. DOI: 10.1287/mnsc.2022.4583.
- Abadie, A., Diamond, A., and Hainmueller, J. (2010). "Synthetic Control Methods for Comparative
  Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the
  American Statistical Association*, 105(490), 493–505. DOI: 10.1198/jasa.2009.ap08746.
- Thistlethwaite, D. L., and Campbell, D. T. (1960). "Regression-discontinuity analysis: An
  alternative to the ex post facto experiment." *Journal of Educational Psychology*, 51(6),
  309–317. DOI: 10.1037/h0044319. Note: frequently miscited as a *JASA* paper; the search
  confirmed the *Journal of Educational Psychology* attribution.
- Imbens, G. W., and Lemieux, T. (2008). "Regression discontinuity designs: A guide to practice."
  *Journal of Econometrics*, 142(2), 615–635. DOI: 10.1016/j.jeconom.2007.05.001.
- Johari, R., Li, H., Liskovich, I., and Weintraub, G. Y. (2022). "Experimental Design in Two-Sided
  Platforms: An Analysis of Bias." *Management Science*, 68(10). DOI: 10.1287/mnsc.2021.4247.

**Official and regulatory (Grade B)**

- No Grade B source governs the design of commercial supply chain experiments. Clinical trial
  reporting standards exist and bind clinical research, but do not govern commercial operations.
  The legal constraints in section 8.2 are general framings, not statute text, and are graded C.

**Practitioner and company evidence (Grade C)**

- The practice-gap table in section 9.8 is a practitioner observation from reading internal pilot
  documentation, not a survey. It is a diagnostic, not a statistic, and must not be quoted as a
  base rate.

**Vendor and analyst (Grade D)**

- Public experimentation-platform and supply chain analytics vendor material, 2026. Vendor case
  studies typically report a lift without the assignment mechanism, the control group's
  treatment, the clustering, or the duration. Grade D, dated 2026, usable only as an existence
  claim about what vendors sell.

**Press and anecdote (Grade E)**

- Not used. No press summary is relied on for any quantitative claim in this file.

**Note on citations.** All bibliographic details are as returned by the searches conducted for
this topic. The Thistlethwaite and Campbell journal attribution was corrected against the search
result and recorded rather than silently applied. No vendor figure is reproduced.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite — robust errors and clustering: [`05_autocorrelation-robust-errors.md`](05_autocorrelation-robust-errors.md)
- Prerequisite — estimands and question types: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- Next topic — DAGs and bad controls: [`06_dags.md`](06_dags.md)
- Later topic — worked difference-in-differences: `08_difference-in-differences.md` (planned)
- Later topic — price elasticity and endogeneity: `09_elasticity-endogeneity.md` (planned)
- Experimentation module map: [`../05_experimentation_pilots_evidence/_index.md`](../05_experimentation_pilots_evidence/_index.md)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Survey file for pilot execution: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
