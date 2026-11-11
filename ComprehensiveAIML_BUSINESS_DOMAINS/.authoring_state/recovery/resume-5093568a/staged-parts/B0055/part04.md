## 6. Visual / Geometric Intuition

Four pictures carry most of the intuition. Each is a text diagram with a stated reading order, because a diagram you cannot read in one pass is decoration.

### 6.1 The precision-prevalence surface

Precision is not a point on the ROC curve; it is a point on a surface indexed by prevalence. Hold TPR at 0.55 and slide FPR:

```text
                       precision
   1.0 |                                     ........  FPR=0.0001
       |                            .......            FPR=0.001
   0.8 |                     ......                    FPR=0.005
       |                 ....                          FPR=0.010
   0.6 |              ...                              FPR=0.020
       |           ...
   0.4 |        ..
       |      ..
   0.2 |   ..                                            <- 0.181 at pi=.004, FPR=.01
       | ..
   0.0 +---|------|------|------|------|------|-------> prevalence
       0.0005 0.001 0.002 0.004 0.01  0.02   0.05
```

Reading order: find your prevalence on the horizontal axis, then find where it intersects the FPR curve you actually have. The vertical distance between two curves is what a model improvement buys you; the horizontal distance is what a population change costs you. The second is usually larger than the first, which is why a model validated on a high-prevalence channel and deployed on a low-prevalence one *looks* broken while being unchanged.

At the illustrative operating point (pi = 0.004, FPR = 0.01, TPR = 0.55), precision is 0.181. To reach 0.50 precision on the same prevalence, FPR must fall to 0.00221:

```text
  FPR needed for PPV = 0.5:
    pi*TPR = 0.004*0.55 = 0.0022
    0.5 = 0.0022 / (0.0022 + 0.996*FPR)
    0.0022 + 0.996*FPR = 0.0044
    0.996*FPR = 0.0022 -> FPR = 0.00221
    equivalent false positives per 1.992M legitimate events: 4,400
```

Halving FPR buys 0.32 of precision at this prevalence. Halving the false-positive *count* while keeping TPR fixed is therefore the single highest-leverage modelling change available — and it is invisible on an ROC chart, where 0.01 and 0.005 FPR are a few pixels apart.

### 6.2 The confusion-matrix budget

Think of the week as a fixed budget of 2,000,000 events and a fixed budget of 8,800 reviews. Both constraints bind simultaneously, so the diagram is a rectangle divided twice.

```text
                 total events 2,000,000
  +-------------------------------------------+-------------------+
  |          TN 1,985,312                      |  FP 6,688         |
  |          (no alert, legitimate)            |  (alert, no fraud)|
  |                                            |  cost 14,044.8    |
  +--------------------------------------------+-------------------+
  |          FN 5,888          |  TP 2,112                          |
  |          (no alert, fraud) |  (alert, fraud)                    |
  |          loss 1,884,160    |  avoided 419,020.8                 |
  +----------------------------+------------------------------------+
  <---------------- 1,992,000 legitimate ----> <---- 8,000 fraud --->
```

Reading order: the horizontal split is truth, the vertical split is action. The two right-hand cells are the entire alert budget (8,800 wide). Shrinking that column raises precision and lowers recall; growing it does the reverse, and neither changes the area of the fraud column.

The lost-loss figure: 5,888 missed frauds at 320 each = 1,884,160 of loss that screening did not prevent. Compare with the 419,020.8 avoided — the programme currently prevents 18.2% of the loss it could prevent.

```text
  419,020.8 / (419,020.8 + 1,884,160) = 419,020.8 / 2,303,180.8 = 0.1820
```

### 6.3 The value-capacity curve

Net value as capacity grows, the curve that decides hiring:

```text
  V (currency/week)
  360k |                          * 358,240 @ 17,600
  336k |                     * 336,000 @ 14,000
  314k |          * 314,125 @ 11,000
  282k |    * 281,776 @ 8,800
  204k |* 204,150 @ 6,000
       |
       |                                     * 335,580 @ 21,000
       |                                          * 311,220 @ 26,000
       +----|--------|--------|--------|--------|--------|------
           6k       9k       12k      15k      18k      21k   26k
                           alerts per week
```

Reading order: the curve rises while the marginal band's precision times 198.40 exceeds the marginal cost of 13.17 (c_m plus friction at the band's precision). It peaks at 17,600 and turns. Three properties matter:

1. The peak is flat. Between 14,000 and 17,600 the curve gains only 2.2% of value (358,240/336,000 = 1.066) for 26% more capacity. A peak this flat means the capacity decision should be made on risk appetite and staffing feasibility, not on the value maximum.
2. The decline after the peak is gentle too (335,580 at 21,000 is still 94% of peak), so over-expansion costs precision less than intuition suggests — but it costs *reputation*, because 21,000 alerts means 17,640 customers contacted unnecessarily per week.
3. The vertical distance from zero is not the story; the vertical distance to oracle (1,462,320) is. The curve is a shallow hill next to a mountain the programme cannot see from the current operating point.

### 6.4 The logit scale as a portable axis

Probability thresholds do not transport; logit thresholds transport under constant prevalence shift. The mapping:

```text
  p      0.03    0.05   0.0803   0.12    0.24    0.40
  logit -3.476  -2.944 -2.4387  -1.992  -1.153  -0.405
```

Equal steps in probability are unequal steps in logit, and model evidence accumulates additively in logit space. Two consequences:

- A model improvement worth a fixed amount of *evidence* (one more useful feature with log-likelihood ratio 0.5) shifts the logit by 0.5 at every probability — which is huge at p = 0.004 and trivial at p = 0.40.
- A threshold move stated in probability units means different things at different operating points. Moving from 0.08 to 0.12 removes 0.34 of logit; moving from 0.24 to 0.28 removes only 0.19.

The practical version: state policy changes in logit units when discussing model revisions, and in probability units when discussing business thresholds. Mixing the two produces conversations where both parties are right and neither is understood.

### 6.5 The capacity-as-knapsack picture

When alert cost varies, capacity is a 1-dimensional bin and alerts are items with value and size.

```text
  capacity 1,349.3 minutes
  +--------------------------------------------------+
  | small alerts: 5.0 min each, 32.02 value           |
  |   fill: 8,800 * 5 min = 44,000 min -> overflows   |
  |   efficient packing: value per minute = 6.40      |
  +--------------------------------------------------+
  | large alerts: 180 min each, ~240 value            |
  |   value per minute = 1.33                         |
  +--------------------------------------------------+
```

Reading order: value per minute, not value per alert, decides who gets the clock. The diagram also shows why the knapsack is not the whole story — a small alert can be *deferred* a day while a large one cannot, so the true problem is scheduling with deadlines rather than a static packing problem. Section 4.4's limit 5 covers the counterexample.

## 7. Comparison / Selection Table

### 7.1 Choosing the evaluation metric

| Metric | What it answers | Invariant to | Blind to | Use when | Do not use when |
|---|---|---|---|---|---|
| ROC-AUC | Can the score order events? | Monotone transforms, prevalence | Precision at capacity, calibration | Comparing models on the same cohort | The alert budget is tiny (AUC is optimistic at low FPR) |
| PR-AUC | How does precision behave across recall? | Monotone transforms | Absolute probability, capacity | Prevalence is low and alerts are scarce | You need a specific operating point |
| Precision at capacity | What fraction of the queue is real? | Nothing (uses the cut) | Missed fraud | Sizing and defending a queue | Model selection across cohorts with different capacity |
| Recall at capacity | What share of fraud is caught? | Nothing | Queue cost | Regulatory coverage questions | Capacity is unconstrained |
| Lift | How much better than chance? | Prevalence scaling | Everything at the operating point | Communicating to non-specialists | Any monetary decision |
| ECE | Does the probability mean what it says? | Nothing | Ordering, sharpness | Threshold setting and probability reporting | Comparing models with different sharpness |
| Brier | Squared-error accuracy of probability | Nothing | Where the error sits (before decomposition) | Model selection on probability quality | Rare events, where it is dominated by uncertainty |
| Log loss | Likelihood accuracy, punishes confidence | Nothing | The low-probability band's practical relevance | Training and probability model selection | Reporting to business stakeholders |
| Net value | Is the policy worth running? | Nothing | Breakdown by segment or failure cause | Any decision involving money or capacity | Cost inputs are unreliable (then use a range) |
| Cost per prevented fraud | Unit economics of the queue | Nothing | Scale, composition | Budget negotiation | Comparing models that catch different segments |

Three selection rules that follow from the table:

1. For a capacity decision, use **net value** and report precision and recall as its inputs. Any other headline invites optimisation of the wrong thing.
2. For a model-selection decision inside a fixed capacity, use **precision at capacity**, because it is the only ranking metric aligned with the constraint you actually face.
3. For a threshold-setting decision, use **calibration plus p\***, because a threshold is an economic quantity expressed in probability units.

### 7.2 Choosing the capacity point

| Capacity (alerts/wk) | Precision | Recall | Net value | Value/alert | Marginal band value | Recommendation |
|---|---|---|---|---|---|---|
| 6,000 | 0.2500 | 0.1875 | 204,150 | 34.02 | 30.65 (first band) | Under-uses staffed capacity |
| 8,800 | 0.2400 | 0.2640 | 281,776 | 32.02 | 30.65 (6k->8.8k) | Current operating point |
| 11,000 | 0.2227 | 0.3063 | 314,125 | 28.56 | 17.63 | Expand if hiring is approved |
| 14,000 | 0.2000 | 0.3500 | 336,000 | 24.00 | 10.22 | Expansion still pays |
| 17,600 | 0.1818 | 0.4000 | 358,240 | 20.35 | 9.11 | Value maximum; flat region |
| 21,000 | 0.1600 | 0.4200 | 335,580 | 15.98 | -3.73 | Over-expanded; loses value |
| 26,000 | 0.1400 | 0.4550 | 311,220 | 11.97 | -1.94 | Requires re-ranking before use |

The marginal column is the one to argue about. It is computed from integer counts within each band, then priced with marginal cost 11.07 rather than average 14.00. Notice the discontinuity pattern: the band from 8,800 to 11,000 carries 15.36% marginal precision and 17.63 of value; the band from 17,600 to 21,000 carries 4.71% and loses 3.73. The drop between those two bands is 12.92 per alert, and it is where the recommendation changes.

### 7.3 Choosing the calibration correction

| Situation | Correction | Data needed | Risk | Verification |
|---|---|---|---|---|
| Monotone distortion, plenty of labels | Isotonic on a temporal holdout | 2,000+ positives in the calibration window | Overfits sparse top bins | Reliability on a third, untouched period |
| Mild distortion, stable prevalence | Platt scaling (logistic on score) | 500+ positives | Cannot fix non-monotone error | Log loss vs uncalibrated baseline |
| Prevalence shift only | Prior correction on the logit: shift by ln(pi_new/pi_old) odds | Population base rate | Assumes constant likelihood ratios | Precision at capacity on the new cohort |
| Sharpness too low (predictions bunched) | Retrain with a proper loss, not post-hoc | Full training set | Post-hoc maps cannot add resolution | Brier decomposition: resolution term rises |
| Sparse top band only | Pool adjacent bins or abstain | Depends on band | Pooling hides the segment you care about | Per-band coverage report |
| No mature labels | Do nothing; report uncalibrated | — | Every downstream threshold is unjustified | Document the label gap and the window |

The prior-correction row is worth an explicit formula because it is the most common production fix and the most commonly mis-applied:

```text
  logit_new = logit_old + ln( odds(pi_new) / odds(pi_old) )
  example: pi_old = 0.004, pi_new = 0.002
    odds_old = 0.004/0.996 = 0.0040161 -> ln = -5.5175
    odds_new = 0.002/0.998 = 0.0020040 -> ln = -6.2146
    shift    = -0.6971
  so a score of 0.0803 becomes sigmoid(-2.4387 - 0.6971) = sigmoid(-3.1358) = 0.0417
```

The correction is valid only if the likelihood ratio P(x|fraud)/P(x|legitimate) is unchanged. If the new population also has different fraud *behaviour* — different attack types, different devices — the likelihood ratios have moved too, and the shift is a coincidence rather than a calibration.

### 7.4 Choosing between reviewing and automating

| Criterion | Human review | Rules / automation | Hybrid (auto + review on exception) |
|---|---|---|---|
| Fixed cost | High (FTE, training, supervision) | High build, low run | Medium |
| Marginal cost | 11.07 per alert | near zero per alert | 11.07 only on exceptions |
| Recovery rate achievable | High (0.62 in the example) | Low (0.20 typical for auto-decline) | Mixed by band |
| Friction imposed | Medium (2.10) | Higher when wrong (blocked good customer) | Lowest, if exceptions are routed well |
| Latency | Hours to days | Milliseconds | Milliseconds for the clear cases |
| Adaptability to new attack | Fast (analyst notices patterns) | Slow (rule authoring cycle) | Fast at the boundary |
| Precision needed to be viable | p > 0.0803 | p > 0.0215 at the auto-decline cost structure | band-dependent |
| Auditability | Case notes, reviewable | Rule versions, replayable | Both, split by band |

The precision thresholds in that table are the key row and they are computed, not asserted: the auto-decline action from Section 4.2 (c = 0.30, rho = 0.20, c_f = 1.10) breaks even at 0.0215, while analyst review breaks even at 0.0803. Automation is viable at lower precision precisely because it is cheap — which is why the correct architecture sends the *low* band to automation and the *middle* band to humans, not the reverse.
