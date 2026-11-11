## 6. Visual / Geometric Intuition

### 6.1 The variance budget as a rectangle

Draw a rectangle whose total area is the variance of the estimated effect. It is partitioned into three bands, and the design moves area between them.

```text
+--------------------------------------------------------------+
| between-store      |  within-store,        | estimation      |
| variance carried   |  week-to-week         | noise from      |
| by n stores        |  averaged over m      | unequal weights |
| 2*rho*sigma^2/n    |  2*(1-rho)*sigma^2/(n*m)  | small         |
+--------------------------------------------------------------+
        ^ widening n shrinks this band
                              ^ widening m shrinks only this band, down to zero
```

The left band shrinks as `1/n` and the middle band shrinks as `1/(n*m)`. That asymmetry is the whole design problem. Doubling stores halves the left band and halves the middle band. Doubling weeks halves only the middle band, which is at most `(1-rho)` of the total. With `rho = 0.25`, weeks can address only 75% of the variance, and the left 25% is untouchable by running longer.

Concrete: `n = 12, m = 4`. Left band `= 2*0.25*0.001225/12 = 0.0000510417`. Middle band `= 2*0.75*0.001225/48 = 0.0000382813`. Total `= 0.0000893229`, which matches `2*Var` from 5.2. Now double weeks to `m = 8`: left unchanged, middle halves to `0.0000191406`, total `0.0000701823`, SE `0.0083775` — a 11.4% gain for 100% more exposure. Double stores to `n = 24` at `m = 4`: both bands halve, total `0.0000446615`, SE `0.0066833` — a 29.3% gain for 100% more stores.

The rectangle makes the trade visible: at this `rho`, a store is worth about 2.6 weeks of exposure. That ratio, `1/(1-rho) + 1 = 2.333` for the effect on the middle band, is the number to quote when procurement asks why the test needs more stores rather than more weeks.

### 6.2 The interval as a ruler, not a verdict

Picture the interval as a ruler laid over the decision threshold `D*`.

```text
   stop              iterate                  ship
 ----|------------------|----------------------|------------>
     0                 D*                     effect
          [=====CI=====]
       0.64 pp        2.05 pp      4.56 pp      (example from 5.2-5.5, D* at 0.77 pp)
```

Any interval that straddles `D*` is an *iterate*. The picture also shows the asymmetry that trips up reviewers: an interval can exclude zero (so "significant") and still straddle `D*`, because significance answers a question about zero, not about break-even.

If `D*` is drawn on the same axis before the data arrive, the analysis becomes a reading exercise rather than a debate. That is the operational value of computing `D*` first: it converts a judgement into a comparison.

### 6.3 The design-effect surface

`DEFF = 1 + (m-1)*rho` is a plane in `(m, rho)` space, and the usable region is bounded.

| `m` \ `rho` | 0.05 | 0.15 | 0.25 | 0.40 | 0.60 |
|---|---|---|---|---|---|
| 2 | 1.05 | 1.15 | 1.25 | 1.40 | 1.60 |
| 4 | 1.15 | 1.45 | 1.75 | 2.20 | 2.80 |
| 8 | 1.35 | 2.05 | 2.75 | 3.80 | 5.20 |
| 13 | 1.60 | 2.80 | 4.00 | 5.80 | 8.20 |
| 26 | 2.25 | 5.25 | 7.25 | 11.00 | 16.00 |

Read the table down a column to see what weeks buy: at `rho = 0.40`, going from 4 to 26 weeks multiplies `DEFF` by 5, so effective sample size per store goes from `4/2.20 = 1.82` to `26/11.00 = 2.36` — a 30% gain for 550% more exposure. Read across a row to see what metric choice buys: at `m = 8`, switching from a `rho = 0.40` metric to a `rho = 0.15` metric cuts `DEFF` from 3.80 to 2.05, equivalent to nearly doubling the store count.

### 6.4 Where the money goes

```text
cost per store-week
  |
  |  *
  |  *  *
  |  *  *  *  *
  |  *  *  *  *  *  *  *
  +--------------------------- stores
     2  4  8 12 16 24 32
```

Marginal cost per store is roughly flat (fixed setup per store plus weekly labour), while marginal information per store declines as `1/n^2` through the variance. The economically optimal design sits where the marginal cost per unit of variance reduction is equal across stores and weeks. In practice the cost ratio is steep enough that the optimum usually sits at "more stores, fewer weeks" — the opposite of the default proposal, which is to run the existing stores for a quarter.

## 7. Comparison / Selection Table

### 7.1 The eighteen gates

Mark each `pass`, `fail`, or `unknown`. The family column determines the consequence: a `fail` in SAFETY is a veto, a `fail` in INTEGRITY means the estimate cannot be reported, and a `fail` in ECONOMICS means stop.

| # | Gate | Family | Check (computable) | Fail consequence |
|---|---|---|---|---|
| G01 | Randomisation unit documented | INTEGRITY | written grain and assignment level match | cannot report |
| G02 | Store balance on pre-period outcome | INTEGRITY | standardised mean difference `< 0.25` | cannot report |
| G03 | Store balance on traffic and size | INTEGRITY | SMD `< 0.25` on both | report with caveat |
| G04 | No store in both arms, any week | INTEGRITY | assignment table has unique `(store, week)` | cannot report |
| G05 | Window fixed before first read | INTEGRITY | pre-registration timestamp precedes data extract | cannot report |
| G06 | Stopping rule declared | INTEGRITY | rule text exists with alpha-spending plan if interim | inflate alpha |
| G07 | Primary outcome single and named | INTEGRITY | exactly one metric in the primary slot | cannot report |
| G08 | Guardrails declared with thresholds | SAFETY | thresholds numeric and dated | veto |
| G09 | Queue or service guardrail measured | SAFETY | per-transaction minutes computed both arms | veto |
| G10 | Labour guardrail measured | SAFETY | overtime hours per store-week both arms | veto |
| G11 | Safety or compliance incident reviewed | SAFETY | incident log reviewed for the window | veto |
| G12 | Compliance `pi_t`, `pi_c` computed | ECONOMICS | both fractions reported, denominator `> 0.10` | stop |
| G13 | Fidelity measured against procedure spec | ECONOMICS | adherence steps counted per store-week | stop |
| G14 | Cost per store-week computed | ECONOMICS | labour plus system, with units | stop |
| G15 | Break-even `D*` computed before read-out | ECONOMICS | `D*` dated earlier than the effect | stop |
| G16 | Interval computed at randomisation level | INTEGRITY | `DEFF` and cluster count reported | cannot report |
| G17 | `rho` estimated, not assumed | INTEGRITY | variance decomposition on control arm | report with caveat |
| G18 | Reversal condition written in the note | ECONOMICS | sentence with numbers exists | stop |

Two `unknown` marks in SAFETY force *iterate*. One `fail` in SAFETY is a veto regardless of everything else. Three or more `unknown` in INTEGRITY means the test did not produce an estimate, and the honest note says so.

### 7.2 Why conjunctive, and what weighted scoring breaks

A weighted score `S = sum_j w_j * s_j` with `s_j` in `[0,1]` has one structural defect that matters here: it is compensatory, so a high `s` on cheap items can offset a low `s` on an expensive one. Set `w = 0.0556` for 18 equally weighted gates and consider a test that fails G09 (queue guardrail) while passing everything else at `s = 1.0`:

```text
S = 0.0556 * 17 * 1.0 + 0.0556 * 0 = 0.9452  ->  94.5% "ready"
```

A 94.5% readiness score for a test whose queue blew out. Raise the safety weights to `0.15` each for G08-G11 and re-normalise: total weight `4*0.15 + 14*0.0429 = 1.2006`; failing G09 gives `S = (1.2006 - 0.15)/1.2006 = 0.8751`. Still an 87.5% score for a veto condition. The lesson is arithmetic, not rhetorical: any additive aggregation of a veto is wrong, because the veto must map to zero, and only an AND (or a multiplication that includes a zero) does that.

The conjunctive form `ready = AND(gates)` has no such failure, but it is harsh — so split the families as in 7.1 and give each family its own aggregation rule. SAFETY is AND. INTEGRITY is AND over the reporting gates. ECONOMICS is the interval comparison from DR-1. That is three rules instead of one score, and each is defensible in a room.

### 7.3 Selecting the design

| Design | Best when | Breaks when | Store requirement | Estimand |
|---|---|---|---|---|
| Parallel two-arm | Enough similar stores; no rollout constraint | Store-level shocks correlated with assignment | `>= 12` per arm for 2-3 pp | ITT at store level |
| Pre/post same store | Only a few stores; effect large | Seasonality or reset coincides | `>= 6` stores, 8+ pre weeks | Association, not effect |
| Switchback | Between-store variance dominates; small estate | Carry-over between periods | `>= 6` stores, `>= 3` switches | Within-store contrast |
| Stepped wedge | Rollout must proceed everywhere | A shock coincides with a wave | `>= 8` stores, `>= 4` waves | Effect vs time trend |
| Synthetic control | One or two treated stores | Poor pre-period fit; no donor pool | 1 treated, `>= 10` donors | Counterfactual gap |
| Interrupted time series | No control stores at all | Any concurrent change | 1 series, 12+ pre points | Level and slope change |

The selection rule is not "which design is strongest" but "which design's failure mode can I test for". A parallel design's failure mode (imbalance) is testable from pre-period data before the test starts. A switchback's failure mode (carry-over) is testable only with a washout period. A synthetic control's failure mode (fit) is testable by leave-one-out on the donor pool. Choose the design whose assumption you can falsify cheaply.

### 7.4 Metric selection for the primary outcome

| Candidate metric | `rho` (ILLUSTRATIVE) | `sigma` | `rho*sigma^2` | Verdict for a 24-store test |
|---|---|---|---|---|
| Shelf availability, audit visit | 0.25 | 0.0350 | 0.000306 | primary; moderate persistence |
| Units per labour hour | 0.35 | 4.20 | 0.006174 | poor; store productivity is a store trait |
| Queue minutes per transaction | 0.18 | 0.62 | 0.000069 | good but narrow; use as guardrail |
| Conversion rate | 0.12 | 0.0180 | 0.000039 | good resolution, weak link to shelf work |
| Shrink share of sales | 0.30 | 0.0045 | 0.000006 | low variance but month-grain only |

The last column multiplies through: `2*rho*sigma^2/n` is the variance floor. At `n = 12`, availability's floor is `2*0.000306/12 = 0.000051` and units-per-labour-hour's is `2*0.006174/12 = 0.001029`, twenty times larger. A test that chose units per labour hour as its primary outcome could not resolve a 2 pp-equivalent change with any number of weeks.

The selection trap is choosing the metric closest to the mechanism ("shelf recovery should raise units per labour hour") rather than the metric with the resolution to detect the mechanism. Choose the resolvable proxy, and use the mechanism metric as a secondary with an explicitly weaker claim.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom as it appears in a dashboard, the competing causes that produce that same symptom, the check that discriminates between them, and the correction. The check column is the part that earns its place: a failure mode without a discriminating check is a worry, not a diagnosis.

### 8.1 Effect shrinks when the control group is re-drawn

**Symptom.** Re-running the analysis with a different control set moves the estimate by more than the interval width.

**Competing causes.** (a) Genuine imbalance — the control set was not comparable and the estimate was never identified. (b) Small donor pool — with few candidate controls, resampling produces wide variation mechanically. (c) A store-level shock in the original control set (a competitor closure, a road closure) that depressed its outcome.

**Discriminating check.** Compute the pre-period standardised mean difference between arms. If `SMD < 0.25` on the primary outcome and on traffic, cause (a) is unlikely and (c) is likely: inspect the specific control store's pre-to-post movement for a level shift. If `SMD > 0.40`, cause (a) is the explanation and the original estimate was a artefact of assignment.

```text
SMD = (mean_t,pre - mean_c,pre) / sqrt( (sd_t,pre^2 + sd_c,pre^2) / 2 )
NUMERICAL EXAMPLE: means 0.8702 and 0.8588, sds 0.0344 and 0.0358
  diff = 0.0114; pooled sd = sqrt((0.00118336+0.00128164)/2) = sqrt(0.00123250) = 0.0351070
  SMD = 0.0114/0.0351070 = 0.3247  -> above 0.25, imbalance is material
INTERPRETATION: a 1.14 pp pre-period gap explains 44% of the observed 2.60 pp effect
  if the gap persisted; report the change-in-change, not the post-period difference.
```

**Correction.** Switch the primary estimand to difference-in-differences, which removes any time-invariant arm gap, and report the post-only difference as a sensitivity. **Prevention.** Balance-check before the test using exactly this formula and pre-declare a re-randomisation or matched-pair fallback if `SMD > 0.25`.

### 8.2 A "significant" result that reverses under a different window

**Symptom.** The four-week read shows a positive effect; the eight-week read shows zero.

**Competing causes.** (a) Novelty or Hawthorne effect — the first weeks measure attention, not the procedure. (b) Genuine decay because the procedure's benefit depends on a stock of work that gets exhausted. (c) The longer window includes a seasonal or promotional period that differs between arms. (d) Random fluctuation, with the four-week read being the lucky one.

**Discriminating check.** Plot the per-week arm means. Cause (a) produces a treatment-arm bump in weeks 1-2 that decays while control stays flat. Cause (b) produces a treatment-arm decay toward the control level with no bump. Cause (c) produces a simultaneous move in both arms in the later weeks (a common shock), which is visible as a parallel shift. Cause (d) produces no pattern at all.

The distinguishing arithmetic: under (c) the arms move together, so the *difference* is stable while the *levels* jump. Under (a) and (b) the levels are stable and the difference decays. Compute the per-week difference series and look at whether variance sits in the difference or in the levels.

**Correction.** Pre-declare the window. If the window was not pre-declared, report both reads and take the shorter interval's conclusion only if the longer read is compatible with it — which is a weaker claim, stated as such. **Prevention.** Add a washout week to the exposure definition and a pre-declared primary window with a single pre-declared sensitivity window.

### 8.3 Compliance looks high but fidelity is low

**Symptom.** The procedure log shows 85% completion; the measured effect is near zero.

**Competing causes.** (a) The log measures a task that was started, not a procedure that was delivered. (b) Staff completed the checklist while skipping the substantive step. (c) The procedure was delivered as designed and genuinely does nothing.

**Discriminating check.** Compare the log's step-level completion to the audit observations of the same store-weeks. If the log says 85% but audit observes the substantive step in 55% of weeks, cause (b). If both are 85% and the effect is zero, cause (c) becomes live and a retest with a stronger dose is the right move. Cause (a) is a measurement failure and is settled by reading the log schema: does the completion flag require all mandatory steps?

```text
fidelity = (store-weeks with all mandatory steps observed) / (store-weeks marked compliant)
NUMERICAL EXAMPLE: 34/41 = 0.8293
  and dilution of the effect if the missing step carries fraction phi of the mechanism:
  observed = true * (0.8293 + 0.1707*(1 - phi)) / 1.0
  at phi = 0.60: factor = 0.8293 + 0.1707*0.40 = 0.8976 -> 2.60 pp observed implies 2.90 pp true
```

**Correction.** Report the delivered dose alongside the effect, and compute the implied full-fidelity effect only under a stated dose-response assumption. **Prevention.** Put the substantive step in the audit instrument, not only in the operator's own log.

### 8.4 Guardrail breached in the treatment arm only

**Symptom.** Overtime rises in the treated stores while sales are flat.

**Competing causes.** (a) The procedure genuinely costs labour and the benefit does not cover it. (b) The treated stores absorbed a concurrent absence or vacancy. (c) The labour hours are allocated from a central pool, so the "increase" is a reallocation artefact rather than an increase. (d) Overtime is being used to run the extra audit task that exists only because of the test.

**Discriminating check.** Compute overtime per store-week for treated and control, and separately compute the labour hours attributable to the test's own measurement activity (cause d). If removing measurement hours eliminates the delta, cause (d) is confirmed — and it is a design defect, because the test is measuring its own overhead. Cause (c) is confirmed by comparing total estate hours: if total is flat while treated stores are up, it is reallocation. Cause (b) is confirmed by the absence record in the same weeks.

**Correction.** Re-report the break-even with the corrected cost. A test whose apparent cost is 40% measurement overhead has a break-even effect inflated by the same 40%. **Prevention.** Exclude measurement-only tasks from the guardrail definition and state that exclusion in the pre-registration.

### 8.5 Spillover between arms

**Symptom.** The control arm improves during the test, more than its pre-period trend predicts.

**Competing causes.** (a) Genuine spillover through shared staff, relief managers, or stock transfers between nearby stores. (b) A common shock — a regional promotion, weather, or a supply improvement. (c) The control arm's pre-period trend was itself unusual, so the "expected" level was misestimated.

**Discriminating check.** Stratify by physical distance between treated and control stores. Cause (a) produces a gradient: control stores near treated stores improve more than distant ones. Cause (b) produces no distance gradient. Cause (c) is checked by extending the pre-period and refitting the trend; if the extended trend predicts the control arm's level, no spillover is needed to explain it.

```text
spillover contrast = mean(control near treated) - mean(control distant)
NUMERICAL EXAMPLE: near mean 0.8734 (n=6), distant mean 0.8612 (n=6)
  contrast = 0.0122; pooled sd 0.0351; SE = 0.0351*sqrt(2/6) = 0.02027
  t = 0.0122/0.02027 = 0.602 -> not distinguishable from zero at this sample size
INTERPRETATION: the gradient is in the expected direction but the test cannot resolve it,
  so report spillover as not-excluded rather than as present.
```

**Correction.** If a gradient is established, inflate the standard error for interference or re-estimate with a design that randomises at a level covering the spillover radius (district rather than store). **Prevention.** Choose the randomisation level from the spillover radius measured in the pre-period, not from convenience.
