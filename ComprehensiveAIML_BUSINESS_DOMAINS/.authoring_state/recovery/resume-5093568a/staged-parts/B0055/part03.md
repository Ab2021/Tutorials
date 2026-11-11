## 5. Worked Numerical Example

A complete pass through one week of a payment-screening programme, from raw counts to a defensible policy recommendation. All inputs are ILLUSTRATIVE; every arithmetic step is shown so a reader can substitute their own numbers.

### 5.1 The input set

```text
INPUT BLOCK (ILLUSTRATIVE)
  window                2026-10-05 .. 2026-10-11 (7 days, event time)
  eligible events       2,000,000      test/internal/duplicate traffic removed
  mature frauds           8,000        labels observed at >= 90 days
  alert threshold         0.081         calibrated probability cut
  alerts raised           8,800
  alerts confirmed fraud  2,112
  loss given fraud L        320.00      currency per confirmed fraud
  recovery rate rho          0.62       share avoided given timely action
  review cost c_a           14.00       currency per alert, fully loaded
  friction cost c_f          2.10       currency per false alert
  marginal cost c_m         11.07       currency per additional alert
  minutes per alert          9.20       mean, all alert types
  reviewer minutes/week    1,800.00     30 hours x 60, per FTE
```

### 5.2 Step 1 — build the confusion matrix before any ratio

```text
  TP = 2,112                 alerts that were confirmed fraud
  FP = 8,800 - 2,112 = 6,688 alerts that were not
  FN = 8,000 - 2,112 = 5,888 frauds never alerted
  TN = 1,992,000 - 6,688 = 1,985,312 legitimate events not alerted
  
  check: TP + FP + FN + TN = 2,112 + 6,688 + 5,888 + 1,985,312 = 2,000,000 ✓
```

The row check is not ceremony. In practice one of the four cells is derived from a different extract than the others, and the mismatch is how you discover it.

```text
  TPR = TP / (TP + FN) = 2,112 / 8,000      = 0.26400
  FPR = FP / (FP + TN) = 6,688 / 1,985,312  = 0.003369
  PPV = TP / (TP + FP) = 2,112 / 8,800      = 0.24000
  NPV = TN / (TN + FN) = 1,985,312/1,991,200 = 0.997044
```

Note the FPR recomputed against TN rather than against all legitimate events. The two differ by 6,688 in the denominator and give 0.003357 versus 0.003369 — a 0.36% relative difference that becomes visible when you multiply by millions of events. Always state which denominator you used.

### 5.3 Step 2 — price one alert

```text
  value per TP  = rho * L        = 0.62 * 320.00 = 198.40
  EV per alert  = PPV*198.40 - 14.00 - (1-PPV)*2.10
                = 0.24*198.40 - 14.00 - 0.76*2.10
                = 47.616 - 14.000 - 1.596
                = 32.020
  weekly value  = 8,800 * 32.020 = 281,776.0
```

Independent re-check by the other route:

```text
  benefit  = 2,112 * 198.40 = 419,020.80
  review   = 8,800 *  14.00 = 123,200.00
  friction = 6,688 *   2.10 =  14,044.80
  net      = 419,020.80 - 123,200.00 - 14,044.80 = 281,776.00 ✓
```

Two independent arithmetic paths agreeing to the cent is the minimum standard for a number that will appear in a business case.

### 5.4 Step 3 — express value per unit of the scarce resource

Capacity is not alerts; it is minutes. The conversion matters because it changes which segment looks attractive.

```text
  minutes consumed = 8,800 * 9.20          = 80,960 minutes
  hours            = 80,960 / 60           = 1,349.3 hours
  FTE equivalent   = 1,349.3 / 30          = 44.98 FTE
  value per hour   = 281,776 / 1,349.3     = 208.83
  value per minute = 208.83 / 60           = 3.48
  value per FTE-week = 281,776 / 44.98     = 6,264.5
```

Sanity check against the marginal economics: an FTE costs 30 hours at 52 per hour = 1,560 per week.
Value per FTE-week of 6,264.5 against 1,560 of cost is a 4.0x ratio — consistent with the earlier 3.05x return on total review spend, since the FTE cost line excludes tooling. If the two ratios disagreed by more than 30%, one of the inputs would be wrong.

```text
  3.05 = 419,020.80 / 137,244.80   (benefit over full review spend)
  4.02 = 6,264.5   / 1,560.0       (value per FTE over wage cost)
```

### 5.5 Step 4 — locate the threshold that the current policy implies

The published cut is 0.081. What break-even probability does that correspond to, and how much slack does it leave?

```text
  p* implied by costs   = (14.00 + 2.10) / (198.40 + 2.10) = 16.10/200.50 = 0.08030
  actual cut            = 0.08100
  slack in probability  = 0.00070  (0.87% above break-even)
  slack in value        = 0.00070 * 200.50 = 0.140 per alert
```

A cut 0.14 currency units above break-even means the marginal alert is barely worth working. That is a defensible place to sit — it means capacity and value are balanced — but it also means any of the six cost assumptions moving by more than a fraction flips the sign. Section 8 treats each assumption as a failure mode for exactly this reason.

### 5.6 Step 5 — value the current policy against two references

```text
  Reference A: no screening at all
      loss = 8,000 * 320.00 = 2,560,000 per week
      value of screening relative to nothing = 281,776 (net) versus -2,560,000
  
  Reference B: oracle ranking, same capacity
      the best 8,800 alerts contain 8,000 frauds and 800 non-frauds
      benefit  = 8,000 * 198.40 = 1,587,200
      review   = 8,800 *  14.00 =   123,200
      friction =   800 *   2.10 =     1,680
      net      = 1,462,320
  
  gap to oracle = 1,462,320 - 281,776 = 1,180,544 per week
```

The gap to oracle is the honest measure of headroom. It says that better *ranking* is worth up to 1.18 million per week at this capacity — an order of magnitude more than any plausible cost saving. Conversely, better *calibration* alone is worth almost nothing here, because the threshold can absorb a monotone scale error. This single comparison should determine where the modelling budget goes.

### 5.7 Step 6 — audit calibration on a stratified sample

A 200,000-event random sample, stratified across score bands, with mature labels:

```text
BAND       n        mean pred p   observed frauds   observed rate   deviation
0-0.001   120,000     0.00050            40           0.000333      -0.000167
0.001-005  50,000     0.00200           130           0.002600      +0.000600
0.005-015  20,000     0.00800           180           0.009000      +0.001000
0.015-040   8,000     0.02500           200           0.025000       0.000000
0.040-120   1,800     0.07000           140           0.077778      +0.007778
0.120+        200     0.20000            30           0.150000      -0.050000
TOTAL     200,000                      720           0.003600
```

```text
  ECE = sum_b (n_b/N) * |p_b - o_b|
      = 0.600*0.000167 + 0.250*0.000600 + 0.100*0.001000
      + 0.040*0.000000 + 0.009*0.007778 + 0.001*0.050000
      = 0.000100 + 0.000150 + 0.000100 + 0 + 0.000070 + 0.000050
      = 0.000470
  relative ECE = 0.000470 / 0.003600 = 13.1%
```

Brier decomposition on the same table:

```text
  reliability  = sum (n_b/N)(p_b - o_b)^2        = 0.0000033
  resolution   = sum (n_b/N)(o_b - o_bar)^2      = 0.0000988
  uncertainty  = o_bar(1 - o_bar)                = 0.0035870
  Brier        = reliability - resolution + uncertainty
               = 0.0000033 - 0.0000988 + 0.0035870 = 0.0034915
  Brier skill  = 1 - Brier/uncertainty = (resolution - reliability)/uncertainty
               = 0.0000955 / 0.0035870 = 0.0266   (2.66%)
```

Log loss on the same bins:

```text
  log loss = sum (n_b/N) * [ -o_b ln p_b - (1-o_b) ln(1-p_b) ]
           = 0.019072
  base-rate benchmark = -o_bar ln o_bar - (1-o_bar) ln(1-o_bar) = 0.023850
  log-loss skill = 1 - 0.019072/0.023850 = 0.2004   (20.0%)
```

The two skills disagree by a factor of 7.5, and the disagreement is informative rather than a mistake. Brier skill is dominated by the high-probability band where the model over-predicts (0.20 predicted, 0.15 observed): squared error there is 0.0025 per case in that bin, which swamps everything the low bands contribute. Log loss is dominated by confidence in the low bands: predicting 0.0005 where the truth is 0.0036 earns a large per-case reward on 60% of the population. Neither score is "right"; they answer different loss questions. Report both, and say which one the decision depends on.

The top bin is the one to fix: 200 events, 30 confirmed, observed 0.150 against predicted 0.200. Its binomial standard error is sqrt(0.15*0.85/200) = 0.0252, so the 0.050 gap is about 2.0 standard errors — mildly significant, and consistent with a model that is over-confident where data is scarcest. A monotone recalibration fitted on 200 events would overfit; the correct action is to wait for more labelled volume in that band or to pool it with the 0.040-0.120 band.

### 5.8 Step 7 — sensitivity, because every input is a guess

Net value under joint perturbation of rho and L, holding counts, capacity, and costs fixed:

```text
                L=200        L=320        L=400
  rho=0.45     52,835      166,883      242,915
  rho=0.55     95,075      234,467      327,395
  rho=0.62    124,643      281,776      386,531
  rho=0.75    179,555      369,635      496,355
```

Break-even probability across the same grid:

```text
                L=200        L=320        L=400
  rho=0.45     0.1748       0.1102       0.0884
  rho=0.55     0.1436       0.0904       0.0725
  rho=0.62     0.1277       0.0803       0.0644
  rho=0.75     0.1059       0.0665       0.0533
```

Reading the grid: the policy stays positive across every cell, which is reassuring, but the *threshold* moves by a factor of 3.3 (0.0533 to 0.1748). The value conclusion is robust; the operating point is not. That asymmetry is typical and it dictates the monitoring design — watch the threshold's economic justification, not just the total value.

The worst cell (rho = 0.45, L = 200) yields 52,835 per week, which is still positive but only 1.05x the 50,000 weekly fixed cost of the review function implied by 44.98 FTE at 1,560 plus tooling. In that cell the programme is marginal, and the recommendation would shift from "expand" to "hold capacity and improve ranking".

```text
  worst-case margin check: 52,835 / (44.98 * 1,560) = 52,835 / 70,169 = 0.75
```

A ratio below 1 means the function does not cover its wage bill in that scenario. This is the calculation that should appear in the quarterly review, not the point estimate.

### 5.9 Step 8 — the recommendation the numbers support

1. Hold capacity at 8,800 alerts per week; the marginal band beyond 17,600 is negative and the band from 8,800 to 11,000 is worth only 17.63 per alert, which does not justify a hiring round at 45 FTE scale.
2. Spend modelling effort on ranking, not calibration: the oracle gap (1,180,544) exceeds the entire current net value (281,776) by 4.2x.
3. Recalibrate only the top band, and only after it accumulates 400+ labelled cases; do not fit a monotone map on 200.
4. Publish value per FTE-week (6,264.5) alongside precision, so that staffing decisions use the same units as the benefit.
5. Fix the loss segmentation: with L varying by an order of magnitude, ranking by p rather than by p*L_i leaves value on the table that no threshold change recovers.
