## 3. Foundational Definitions

A field test is a measurement instrument, and like any instrument it is defined by its scale, its zero point, and its resolution. The definitions below are the scale. Each is stated as a sentence you can paste into a pre-registration document, followed by the thing it is routinely confused with.

### 3.1 The core vocabulary

| Term | Definition (paste-ready) | Units / range | Confused with | Why the confusion costs money |
|---|---|---|---|---|
| Randomisation unit | The level at which assignment to new or old process is decided | store, or store-week, or (day, store) | the analysis row | Analysis at row level shrinks intervals by `sqrt(DEFF)` |
| Cluster | The randomisation unit and everything nested inside it | store containing `m` weeks | a region or district | Region-level assignment with store-level analysis overstates precision |
| Exposure | The dated fact that a store-week ran the new procedure | binary per store-week | the planned go-live date | Planned dates produce a phantom first week of treatment |
| Window | Inclusive calendar range of store-weeks used in the primary analysis | dates, day grain | data availability range | Using whatever data is loaded biases toward stores that load fast |
| Primary outcome | The single pre-declared metric whose interval drives the decision | metric units | "everything we tracked" | Twelve outcomes guarantee at least one pleasant accident |
| Guardrail | A metric with a pre-declared harm threshold that can veto shipping | metric units, threshold | a secondary outcome | Guardrails treated as context become excuses |
| Comparator | The counterfactual construction used as the "without" arm | control stores / prior weeks / synthetic | "baseline" used loosely | Two comparators give two answers and no decision |
| Compliance | Share of assigned store-weeks that actually ran the new procedure | fraction `[0,1]` | adherence score, satisfaction | Compliance below 1 dilutes ITT toward zero |
| Fidelity | Degree to which delivered procedure matched the designed procedure | fraction `[0,1]` | compliance | High compliance with a wrong procedure is not success |
| MDE | Smallest true effect detectable with stated power at stated alpha | outcome units | the observed effect | Reporting "no effect" when the test could not see one |
| Minimum effect of interest | Smallest effect that would change the operating decision | outcome units | MDE | A test powered for an effect nobody would act on |
| Stopping rule | The pre-declared rule for when analysis happens and when it stops early | rule text + dates | "we monitor daily" | Unplanned peeks inflate the false-positive rate |
| Spillover | Treatment reaching units assigned to control, via shared staff or stock | fraction `[0,1]` | contamination by error | Attenuates the measured contrast toward zero |
| Attrition | Assignment units that leave the analysis before the window closes | count | missing rows | Attrition correlated with outcome biases the estimate |

### 3.2 Grain, and why it is the first sentence

The grain is the tuple that identifies one observation. For a store field test it is almost always `(store_id, iso_week)` and almost never `(store_id, transaction_id)`. Writing the grain down first forces three later decisions to be consistent with it.

- The randomisation unit must be a coarsening of the grain, never a refinement. You can randomise stores and analyse store-weeks with clustering; you cannot randomise weeks within a store and then claim store-level independence.
- The guardrail thresholds must be expressed at the grain. "Overtime is up" is not a guardrail; "overtime hours per store-week exceeds 6.0" is.
- The join to cost data must not change the grain. Cost joins at month grain multiply store-weeks if done carelessly, which is a Section 8 failure mode with a specific diagnostic.

### 3.3 Event time versus processing time, in a store context

Three timestamps exist for every store-week and they are all different: the week the work happened, the time the record was written to the source system, and the time it landed in the analysis table. The first is event time. The third is processing time. Availability measured by an audit belongs to event time; availability measured by a stock-count feed that is posted two days after the count belongs to processing time.

A field-test comparison is valid only when both arms are read with the same lag. If treatment stores post counts faster (because the new procedure includes a scan step), the treatment arm will appear to have complete data earlier, and any metric that depends on completeness will be biased upward in the treatment arm for mechanical reasons.

The practical rule: compute every arm mean from the same snapshot timestamp, and record that timestamp in the decision note. A test whose arms were read at different snapshot times is `unknown`, never `pass`.

### 3.4 What "no effect" means and does not mean

A test whose interval spans zero has established that the data are compatible with no effect. It has not established that the effect is zero. The distinction is quantitative: if the interval is `[-1.1, +4.0] pp`, the test is compatible with a loss of 1.1 pp and a gain of 4.0 pp, so the honest statement is "inconclusive at the resolution we bought".

Section 4.2 derives the resolution, and Section 5 shows a design where the interval is wide enough to be compatible with both the decision-relevant effect and its absence. That design does not support a ship decision and does not support a stop decision. It supports a bigger test or a cheaper intervention.

### 3.5 The decision, stated as a rule

Write the decision rule as an inequality before the data arrive, using the lower bound of the interval and the break-even threshold:

```text
DECISION RULE [DR-1]
ship     if  L(Delta) >= D*        (lower bound clears break-even)
iterate  if  L(Delta) <  D* <= U(Delta)   (interval straddles break-even)
stop     if  U(Delta) <  D*        (upper bound below break-even)
veto     if  any guardrail breaches, regardless of Delta
WHERE: Delta [metric units] estimated effect; L, U [metric units] interval bounds;
       D* [metric units] break-even effect derived in 4.5.
CONSTRAINT: D* computed before the read-out and not revised afterwards.
```

Two features make this rule usable. First, it is one-sided in the right way: the question is not "is there an effect" but "is the effect big enough to pay for itself". Second, the veto line sits outside the three-way choice, so a safety breach cannot be traded against a good effect.

Break-even `D*` is derived in 4.5 and is where most field tests silently fail: the statistics pass, and the economics do not.

### 3.6 Definitions that this sheet refuses to leave ambiguous

- **Same-store comparison.** Comparing a store with itself in a different period. Legitimate only when the period comparison is constructed to be free of the drift you are testing against — hence switchback designs in 4.3.
- **Rollout.** Deploying the change to every store in the estate. Rollout is not a test, even when monitoring continues; after rollout there is no concurrent control, so the only honest read is against a pre-registered pre-period.
- **Pilot.** A short test whose purpose is feasibility, not effect. A pilot may legitimately be small and unpowered; it must then be labelled as a pilot and must not be quoted as evidence of effect.
- **Pilot with an effect claim.** The failure this sheet exists to catch. If a run was powered for feasibility and is quoted with a point estimate, the number will be re-quoted in a capital request with the interval removed.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: the cluster-level difference and its standard error

Most store field tests randomise stores and then measure store-weeks. The naive standard error treats each store-week as an independent draw. That is wrong whenever store-weeks inside a store resemble each other, which they do: a store has a persistent traffic level, a persistent staffing model, and a persistent assortment.

Start from the variance decomposition. For outcome `Y` in store `i`, week `j`:

```text
MODEL [M-1]: Y_ij = mu + a_i + e_ij
WHERE: mu [outcome units] grand mean; a_i ~ (0, sigma_b^2) store effect;
       e_ij ~ (0, sigma_w^2) week effect inside store; a and e independent.
IMPLIED: sigma^2 = sigma_b^2 + sigma_w^2, and rho = sigma_b^2 / sigma^2.
```

Averaging `m` weeks in one store gives

```text
FORMULA [F-1]: Var(mean_i) = sigma_b^2 + sigma_w^2 / m
                          = (sigma^2 / m) * [1 + (m - 1) * rho]
                          = (sigma^2 / m) * DEFF
WHERE: Var [outcome units^2]; m [weeks per store]; DEFF = 1 + (m-1)*rho [dimensionless].
CONSTRAINT: equal m per store, common rho, balanced arms, no interference between stores.
```

The bracket is the design effect. It answers a manager's question exactly: how many independent observations is one clustered observation worth? The answer is `m / DEFF`, which is at most `1/rho` no matter how long you run (4.4).

For a two-arm comparison with `n` stores per arm:

```text
FORMULA [F-2]: SE_diff = sqrt( 2 * DEFF * sigma^2 / (n * m) )
               CI_95   = Delta +/- t_(0.975, 2n-2) * SE_diff
WHERE: Delta [outcome units] = mean_treated - mean_control;
       t_(0.975, 2n-2) [dimensionless] two-sided critical value at df = 2n - 2.
CONSTRAINT: Delta estimated as an unweighted difference of arm means at store level.
```

Worked arithmetic with `n = 12`, `m = 4`, `sigma = 0.0350`, `rho = 0.250`, `Delta = 0.0260`:

```text
DEFF      = 1 + 3 * 0.250                 = 1.7500
sigma^2   = 0.0350^2                      = 0.00122500
per-arm   = 1.7500 * 0.00122500 / 48      = 0.0000446615
SE_diff   = sqrt(2 * 0.0000446615)        = 0.0094510
t         = 0.0260 / 0.0094510            = 2.7510   (df = 22, crit 2.0739)
half-width= 2.0739 * 0.0094510            = 0.0196006
CI_95     = [0.0063994, 0.0456006]        -> [0.64 pp, 4.56 pp]
```

The naive calculation, which ignores clustering, gives `SE = sqrt(2*0.001225/48) = 0.0071443` and a half-width of `0.0148167`. The clustered interval is `1.3229x` wider, where `1.3229 = sqrt(1.75)`. Every reported `p`-value from the naive version is too small by the same mechanism, and the error is not conservative in either direction — it simply misstates precision.

Interpretation, stated with its limit: the test is compatible with an availability gain as small as 0.64 pp and as large as 4.56 pp. Any statement of the form "the procedure raises availability by 2.6 pp" is a point estimate being presented as a fact. The interval is the measurement; the point estimate is its centre.

### 4.2 General case: unequal cluster sizes and the sample-size inversion

Two generalisations matter in practice, and both are cheap to compute.

**Unequal weeks per store.** Stores drop out, open mid-window, or have audit gaps. With mean weeks per store `m_bar` and coefficient of variation of cluster size `CV_m`, the design effect becomes

```text
FORMULA [F-3]: DEFF_unequal = 1 + rho * ( m_bar * (1 + CV_m^2) - 1 )
WHERE: CV_m = sd(m_i) / m_bar [dimensionless]; m_i [weeks] per store.
CONSTRAINT: same rho for all stores; CV_m^2 small enough that the expansion holds.
BOUNDARY: at CV_m = 0 this reduces to F-1, which is the check to run on the code.
```

With `m_bar = 4`, `CV_m = 0.5`: `DEFF = 1 + 0.25*(4*1.25 - 1) = 1 + 0.25*4 = 2.000`. The standard error rises by `sqrt(2.000/1.750) = 1.0690`, about 6.9%. That is small enough to ignore when planning and large enough to matter when the interval is close to the decision threshold — which is exactly when you are reading it closely.

**Inverting for the number of stores.** Rearranging F-2 at the minimum detectable effect:

```text
FORMULA [F-4]: n_per_arm = 2 * DEFF * sigma^2 * (z_(1-alpha/2) + z_(1-beta))^2 / (delta^2 * m)
WHERE: delta [outcome units] target effect; z_(1-alpha/2) = 1.9600 at alpha = 0.05;
       z_(1-beta) = 0.8416 at power 0.80; sum = 2.8016 [dimensionless].
CONSTRAINT: normal approximation; add ~5% for unequal cluster sizes; use the t correction
            when n_per_arm is below about 20.
```

At `delta = 0.0260` (the effect the test actually saw): `n = 2*1.750*0.001225*7.8490/(0.000676*4) = 0.033651/0.002704 = 12.44`, so 13 stores per arm. The design had 12, which is why the observed effect sits almost exactly on the detectable boundary — a coincidence worth noticing, because it means the test was designed for roughly the effect it found.

At `delta = 0.0200`: `n = 0.033651/(0.0004*4) = 21.03`, so 22 stores per arm, 44 stores total. At `delta = 0.0150`: `n = 0.033651/(0.000225*4) = 37.39`, so 38 per arm, 76 stores total.

This is the arithmetic that kills most field-test proposals, and it should be run before the proposal, not after. If the estate has 24 suitable stores and the decision needs 2.0 pp resolution, the test as designed cannot support the decision. The available responses are: raise `m` (limited, see 4.4), lower `rho` by measuring a less persistent outcome, use a design that removes between-store variance (4.3), or accept a coarser decision.
