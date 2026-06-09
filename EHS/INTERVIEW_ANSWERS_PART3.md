# Ensemble Health -- Interview Question Answers (Part 3)

> Sections 4-8 (continued): Model Selection, Evaluation, Calibration, GenAI, Production Architecture
> Questions Q63 -- Q120

---

### 4.3 Model Comparison & Selection (continued)

**Q63. Rank the 10 experiments by: (a) capture@25, (b) ROC-AUC, (c) Brier, (d) training time. Which model wins each category?**

**(a) Capture@25 (primary business metric):**

| Rank | Model | Capture@25 |
|---|---|---|
| 1 | LR Baseline / LR Calibrated / LR Interactions | 49.51% |
| 4 | GBM | 48.54% |
| 5 | Voting Ensemble | 48.54% |
| 6 | Stacking | 46.60% |
| 7 | SVC RBF | 45.63% |
| 8 | MLP Neural | 43.69% |
| 9 | RF Robust | 39.81% |
| 10 | XGB Tuned | 35.92% |

**Winner:** Logistic Regression family (all three variants tied at 49.51%).

**(b) ROC-AUC (ranking quality across ALL thresholds):**

| Rank | Model | ROC-AUC |
|---|---|---|
| 1 | Stacking | 0.718 |
| 2 | LR Baseline | 0.711 |
| 3 | LR Calibrated | 0.710 |
| 4 | LR Interactions | 0.707 |
| 5 | Voting Ensemble | 0.696 |
| 6 | MLP Neural | 0.678 |
| 7 | GBM | 0.661 |
| 8 | RF Robust | 0.653 |
| 9 | SVC RBF | 0.641 |
| 10 | XGB Tuned | 0.615 |

**Winner:** Stacking (0.718). But this didn't translate to the best capture@25.

**(c) Brier Score (probability calibration quality, lower is better):**

| Rank | Model | Brier |
|---|---|---|
| 1 | Stacking | 0.136 |
| 2 | LR Calibrated | 0.137 |
| 3 | MLP Neural | 0.146 |
| 4 | SVC RBF | 0.148 |
| 5 | GBM | 0.151 |
| 6 | RF Robust | 0.155 |
| 7 | Voting Ensemble | 0.166 |
| 8 | XGB Tuned | 0.180 |
| 9 | LR Baseline | 0.209 |
| 10 | LR Interactions | 0.210 |

**Winner:** Stacking (0.136) and Calibrated LR (0.137) are neck-and-neck. Calibration dramatically improved LR's Brier from 0.209 to 0.137.

**(d) Training time (fastest to slowest):**

| Rank | Model | Approx Time |
|---|---|---|
| 1 | LR Baseline / LR Interactions | <0.1s |
| 3 | LR Calibrated (5-fold CV inside) | ~0.3s |
| 4 | GBM | ~2s |
| 5 | Voting Ensemble | ~3s |
| 6 | RF Robust | ~3s |
| 7 | XGB Tuned (300 trees + early stopping) | ~5s |
| 8 | Stacking (3 base models + meta-learner, CV) | ~8s |
| 9 | SVC RBF (kernel matrix computation) | ~10s |
| 10 | MLP Neural (gradient descent iterations) | ~15s |

**Winner:** LR Baseline (<0.1s). Calibrated LR wins on the combined metric of business performance + training time.

**Q64. If you had to choose between a model with capture=49% and Brier=0.14 vs capture=48% and Brier=0.10, which would you pick? Why?**

I would pick **capture=49%, Brier=0.14** (the higher-capture model).

**Reasoning:** The 1 pp capture difference (~2 additional denials caught per 500 claims) has direct operational value. Those 2 denials represent $50-$236 in saved rework costs and 30-90 days of accelerated revenue. The Brier improvement from 0.14 to 0.10 means the probability estimates are better calibrated -- but what operational decision changes because of better calibration?

The risk tiers (High/Medium/Low) are determined by ranking, not by absolute probability thresholds (other than the 75th percentile cutoff). Better calibration would make the probability numbers more trustworthy (e.g., a claim with 0.60 probability actually has a ~60% denial rate, not 50% or 70%), but it wouldn't change which claims are in the High tier if capture is the same.

**When would I choose the Brier=0.10 model:** If the business needed reliable probability estimates for downstream decisions -- e.g., "auto-submit claims with <10% probability, auto-hold claims with >80% probability." In that scenario, miscalibrated probabilities could route claims incorrectly. But for the review-queue-ranking use case, calibration is nice-to-have while capture is must-have.

**The Stacking example proves this point:** Stacking had the best Brier (0.136) and best ROC-AUC (0.718) but worse capture (46.60%). If I prioritized calibration, I'd deploy the model that catches fewer denials. The business case for denial prediction is preventing denials, not producing beautifully calibrated probabilities.

**Q65. The three LR variants all got identical capture@25. Does this mean feature engineering is irrelevant and model architecture is all that matters? Why or why not?**

No. The identical capture@25 across LR variants proves the OPPOSITE: feature engineering matters immensely, and the feature set saturates what LR can learn.

**What the identical capture tells us:**

The three LR variants share the SAME feature set:
- exp1 (Baseline): 53 features with main effects only
- exp2 (Calibrated): Same 53 features, with Platt scaling on top
- exp5 (Interactions): Same 53 features, some already being interaction features

They all achieve identical capture because the feature engineering already captured the necessary interactions. `double_deficit` (auth AND doc), `oon_auth_gap`, `total_admin_gaps` -- these ARE the interaction features. Adding more interaction terms to exp5 (like billed_amount * auth_gap) added no new signal because the important interactions were already engineered.

**What feature engineering contributed:**

If I had used only the 17 raw columns with no feature engineering, the performance would be dramatically worse. The raw columns don't include `auth_gap`, `total_admin_gaps`, `payment_ratio`, `late_filing`, `is_high_risk_combo`, or any of the other 30+ engineered features. Without them, LR would be working with payer_type, visit_type, total_billed, and a few binary flags -- far less signal. I'd estimate raw-columns-only capture around 30-35%.

**The relationship:** Feature engineering defines the ceiling of what's learnable. Model architecture determines how close to that ceiling you get. For additive data with well-engineered features, even a simple LR hits the ceiling. The fact that all three LR variants hit the SAME ceiling suggests the feature engineering is near-optimal for this data.

**Q66. How do you know you haven't overfit to the validation set with 10 experiments? What safeguards prevent this?**

Overfitting to the validation set happens when you repeatedly test models and select based on validation performance, eventually finding a model that "won the lottery" on that specific validation split without generalizing.

**Safeguards in place:**

**1. Holdout test set.** The test set (480 claims) was NEVER used for model selection, threshold computation, or any decision. It was touched exactly once for the final evaluation. The test capture@25 of 45.71% vs validation capture@25 of 49.51% shows the expected ~4 pp degradation from validation to test -- consistent with normal generalization, not massive overfitting that would show a 20+ pp drop.

**2. Architectural diversity, not hyperparameter fine-tuning.** The 10 experiments explored fundamentally different model families (LR, RF, XGB, MLP, SVC, GBM, Voting, Stacking) rather than 10 minor variations of the same model. Overfitting to validation data typically happens with aggressive hyperparameter search within a single model family. I did a coarse logarithmic sweep of C for LR -- testing 6 values across 5 orders of magnitude, not 100 values in a narrow range.

**3. Identical top performance across LR variants.** The fact that three independent LR configurations (baseline, calibrated, interactions) ALL achieved 49.51% suggests this is the genuine performance ceiling for LR, not a lucky validation split. Overfitting would show one experiment at 55% (lucky) and others at 44% (normal).

**4. Temporal split structure.** The validation set is a distinct time period (mid-year 2024), not a random sample. Overfitting to random splits can happen because the split happens to distribute easy/hard claims favorably. Temporal splits are harder to overfit to because the time period itself has structural properties.

**5. Consistent rank ordering across model families.** The model rankings (LR > GBM > Voting > Stacking > SVC > MLP > RF > XGB) follow a logical pattern explained by data properties. If I'd overfit, I'd expect random-looking rankings driven by noise, not a pattern consistent with the additive data structure.

**Q67. If you could only run 3 experiments instead of 10, which would you pick and why?**

**Experiment 1: Logistic Regression (C sweep, class_weight='balanced').** This is the must-run baseline. Given my prior knowledge that administrative gaps drive denials (from EDA), I'd expect LR to perform well. If it doesn't, something is wrong with the features or data. LR also provides coefficient-based interpretability, which is essential for the explanation engine. Running time: <1 second.

**Experiment 2: Gradient Boosting Machine (GBM, sklearn).** After confirming LR works, I want to validate that the data is genuinely additive by testing a tree model. If GBM matches or beats LR, the data has non-linear structure that I should explore. If GBM underperforms (as it did, 48.54% vs 49.51%), that confirms the additive hypothesis and eliminates the need to try XGBoost, RF, or other tree models. Running time: ~2 seconds.

**Experiment 3: Calibrated Logistic Regression.** This takes the best-performing architecture (LR) and applies Platt scaling to improve probability reliability. If calibration maintains capture while improving Brier (which it did, 0.209 to 0.137), I have the best of both worlds. If calibration hurts capture, I'd fall back to the baseline LR. Running time: ~0.3 seconds.

**Why these three over others:**
- **Skip XGBoost** (experiment 3): GBM is a sufficient tree model test. If GBM >> LR, THEN try XGB. If GBM ~= LR or GBM < LR, XGB is unlikely to change the conclusion.
- **Skip RF** (experiment 4): RF's bagging + shallow trees approach is less efficient than boosting for additive data. Low priority.
- **Skip MLP** (experiment 9): Not viable with 2,240 samples. Only try if I had >50K samples.
- **Skip SVC** (experiment 10): RBF kernel is overkill for data that LR handles perfectly.
- **Skip Voting/Stacking** (experiments 7-8): Only try ensembles if multiple strong individual models exist and make uncorrelated errors. If LR alone is dominant, ensembles just dilute.

**Q68. What experiment would you add as #11 if you had another week? Justify your choice with a specific hypothesis.**

**Experiment 11: Per-Payer Logistic Regression Models.**

**Hypothesis:** Different payers have distinct denial patterns (Medicaid MCO 28.6% vs Medicare Advantage 15.0%). A single global LR learns average coefficients across all payers, which may be suboptimal for individual payers. Per-payer models would learn payer-specific coefficient vectors, capturing interactions like "missing auth matters MORE for BCBS than for Commercial."

**Implementation:**
1. Split training data by `payer_type` (4 groups) and by top `payer_id` (P001, P008, P011 -- the ones with sufficient samples)
2. Train a separate Calibrated LR for each payer segment
3. For current claims, route each claim to its corresponding payer model
4. For rare payers (<50 training samples), fall back to the global model
5. Compare: global capture@25 vs payer-stratified capture@25

**Expected outcome:** I'd expect a 1-4 pp improvement in capture@25 over the global LR. The error analysis showed that Medicaid MCO claims are overrepresented in false positives. A Medicaid MCO-specific model could learn that these claims have a higher baseline risk but milder penalty for specific gaps, reducing the false positive rate. Similarly, a Medicare Advantage model could learn to be more aggressive, catching borderline claims that the global model misses.

**Why this over other possible experiments:**
- Not just hyperparameter tuning (diminishing returns)
- Addresses a known issue from error analysis (payer heterogeneity)
- Completely different model structure (stratified vs global)
- Deployable in production (routing logic is simple)
- Fits the additive data assumption (LR within each stratum)

### 4.4 Hyperparameter Sensitivity

**Q69. How sensitive is LR to the choice of C? If C changed from 0.1 to 0.05, what would happen to coefficients and capture@25?**

LR is moderately sensitive to C within a factor of 2 (0.1 to 0.05), but not critically so. Here's what would happen:

**Coefficient effects (C=0.1 → C=0.05, i.e., stronger regularization):**
- All coefficients shrink by roughly 10-30% (more regularization = more shrinkage)
- Strong signals: `auth_gap` might drop from 1.24 to ~1.05 (15% reduction)
- Weak signals: `network_issue` might drop from 0.15 to ~0.08 (47% reduction)
- Noise features: `payer_id_P011`, `service_month_num` coefficients approach zero even faster

**Probability effects:**
- Predicted probabilities become less extreme (closer to the mean of 0.21)
- A claim that was at 0.75 might move to 0.68
- A claim that was at 0.08 might move to 0.11
- The probability distribution becomes more compressed

**Capture@25 effects:**
- **Minimal impact, probably 48.0-49.0% (down 0.5-1.5 pp)**

The ranking is largely preserved because regularization shrinks all coefficients proportionally. The claim ordering barely changes. However, stronger regularization might slightly reshuffle claims near the 75th percentile cutoff (where probability differences are tiny), causing 1-3 claims to cross the threshold in either direction. The net effect on capture@25 would be small.

**Why LR is robust to C near the optimum:** The loss landscape around the optimum is relatively flat. Small changes in regularization don't dramatically change the decision boundary unless the regularization is so strong that it collapses all coefficients to zero (C < 0.001).

**Q70. XGBoost has `scale_pos_weight=3.6`. How did you arrive at this value? What is the theoretical basis for this number?**

`scale_pos_weight` controls how XGBoost weights positive (minority) class examples during training. The formula:

```
scale_pos_weight = n_negative / n_positive
                = (total_non_denied) / (total_denied)
                = 2509 / 691
                ≈ 3.63
```

**Theoretical basis:** In binary classification, `scale_pos_weight = n_negative / n_positive` makes the effective weight of each class equal -- analogous to `class_weight='balanced'` in sklearn. Each denied claim contributes 3.63x more to the loss than each non-denied claim, compensating for the 3.63:1 class imbalance.

**Why 3.6 and not something else:**

| Value | Meaning | Effect |
|---|---|---|
| 1.0 | No adjustment | Model ignores minority class; predicts "no denial" most of the time |
| 3.6 | Balanced | Equal total weight per class; optimal for equal-cost errors |
| 5.0 | Overweight denied | More false positives, fewer false negatives; good if denying costs more than reviewing |
| 10.0 | Extreme overweight | Very aggressive denial prediction; many false positives |

I used 3.6 as the default "balanced" value because XGBoost was already struggling (35.92% capture). Changing `scale_pos_weight` wouldn't fix the fundamental tree-vs-additive-data mismatch. Tuning it to 5.0 or 10.0 would increase the number of claims flagged as denials, potentially improving recall, but would flood the review queue and lower precision. The 3.6 value is the principled starting point.

---

## Section 5: Evaluation Metrics (Q71-Q85)

### 5.1 Metric Selection

**Q71. Define Capture@25% in plain English. How is it calculated? Why is it more relevant than accuracy for this problem?**

**Plain English:** "Of all the claims that will eventually be denied, what fraction did the biller review BEFORE submission?"

**Calculation:**
```
capture@25 = (number of denied claims in top 25% by predicted probability)
           / (total number of denied claims in the dataset)
```

For our test set:
```
capture@25 = denied_in_top_25 / all_denials
           = 57 / 125
           = 45.6%
```

The model ranks all 480 test claims by predicted denial probability (descending). It takes the top 25% (120 claims) and counts how many of those 120 are actually denied (57). It divides by total test set denials (125). Result: 45.6% of denials were found in the top 25% of risk-ranked claims.

**Why more relevant than accuracy:**

Accuracy answers: "What fraction of ALL predictions were correct?" A dummy model predicting "no denial" always scores 78.4% accuracy. Our model, which catches 45.6% of denials but has a 46% false positive rate, might score 72% accuracy -- "worse" than the do-nothing model by 6 points.

Capture@25 answers: "Within the biller's limited review time, how many denials did we catch?" This directly maps to the operational constraint and business value. A model with 45.6% capture provides 1.82x the denials caught vs random review (25%), saving real money. The accuracy number would misleadingly suggest the model is worse than doing nothing.

**Q72. A model with 78.4% accuracy sounds great until you learn the denial rate is 21.6%. Explain why accuracy is misleading for imbalanced classification.**

Accuracy is the fraction of correct predictions. At 21.6% denial rate:

```
Model: "No denial" for everything
Predictions: No denial, No denial, No denial, ... (for all claims)
Correct when: Claim is not denied (78.4% of the time)
Accuracy: 78.4%
```

This model has zero operational value -- it catches no denials, prevents no rework, saves no money. Yet it scores 78.4% accuracy, which to a non-technical audience sounds impressive.

**Why accuracy breaks:** It assumes false positives and false negatives have equal cost. In denial prediction:
- False negative (missed denial): $25-118 rework cost + 30-90 days revenue delay
- False positive (unnecessary review): ~2 minutes of biller time (~$3-5)

These costs differ by 5-25x. But accuracy weighs them equally. A metric that treats a $118 mistake and a $3 mistake as equivalent is fundamentally invalid for decision-making.

**The paradox:** To improve accuracy beyond 78.4%, the model must correctly identify denials. But each correct denial prediction (true positive) comes with some false positives. The accuracy improvement from catching one more denial might be offset by two extra false positives. The metric penalizes the model for doing the thing that creates business value.

**Q73. What is the difference between ROC-AUC and PR-AUC? When would you prefer one over the other? Which is more informative for this problem?**

**ROC-AUC (Area Under the Receiver Operating Characteristic curve):**
- Plots True Positive Rate (Recall) vs False Positive Rate across all thresholds
- TPR = TP / (TP + FN), FPR = FP / (FP + TN)
- Measures how well the model separates classes across ALL possible thresholds
- Baseline (random model): 0.50
- Sensitive to the majority class performance (TN in the denominator of FPR)

**PR-AUC (Area Under the Precision-Recall curve):**
- Plots Precision vs Recall across all thresholds
- Precision = TP / (TP + FP), Recall = TP / (TP + FN)
- Measures how many flagged claims are actually denials
- Baseline (random model): prevalence rate (~0.21 for this data)
- Sensitive to the minority class (TP and FP both related to the positive class)

**When to prefer ROC-AUC:** Balanced datasets (~50/50 split) or when you care about overall ranking quality across all operating points. Example: medical diagnosis where both false positives and false negatives matter equally.

**When to prefer PR-AUC:** Imbalanced datasets where the positive class is rare (<30%). PR-AUC focuses on the minority class, which is usually what you care about. Example: fraud detection (1% fraud rate), rare disease screening.

**For this problem, PR-AUC is more informative because:**
1. Denials are 21.6% of claims (moderate imbalance)
2. We care about catching denials (recall) without flooding the queue (precision)
3. Our model's ROC-AUC of 0.69 sounds modest; its PR-AUC of 0.52 vs baseline 0.22 shows it's 2.4x better than random at the minority class task

However, NEITHER metric perfectly captures the 25%-capacity constraint. Both evaluate across ALL thresholds, including reviewing 100% of claims. Capture@25 addresses the specific operational constraint.

**Q74. Your model has ROC-AUC=0.691 and PR-AUC=0.522. Why is PR-AUC so much lower? Is this a problem?**

The PR-AUC is lower because its baseline is also lower:
- ROC-AUC baseline: 0.50 (random model scores 0.50 by definition)
- PR-AUC baseline: 0.216 (random model's precision = denial rate at any threshold, because flagged claims are a random sample)

**Normalized comparison:**
- ROC-AUC: 0.691 / 0.50 baseline = **1.38x random**
- PR-AUC: 0.522 / 0.216 baseline = **2.42x random**

PR-AUC is actually STRONGER relative to its baseline! The model is 2.42x better than random at the minority-class task (precision-recall space) vs 1.38x better at the overall ranking task (ROC space).

**Why PR-AUC looks low in absolute terms:**

Precision degrades quickly as recall increases in imbalanced settings. To achieve high recall (catching most denials), the model must cast a wide net, pulling in many false positives and driving precision down. The PR curve shows this "waterfall" shape:
- At recall=30%: precision ~70% (high precision at conservative thresholds)
- At recall=50%: precision ~50% (moderate precision)
- At recall=70%: precision ~35% (low precision at aggressive thresholds)

The area under this curve (0.522) reflects that even the best model can't maintain high precision as it pursues high recall.

**Is this a problem?** No. PR-AUC is a diagnostic for understanding the precision-recall trade-off, not an optimization target. The business operates at exactly one point on this curve: the 25% review threshold. At that point, precision is ~48% and recall is ~46%. These are acceptable for a screening tool where false positives cost ~$3-5 and true positives save $25-118.

**Q75. Explain Brier score in simple terms. What does a Brier score of 0.137 mean? What would a perfect model score? A random model?**

**Simple explanation:** The Brier score measures how "honest" a model's probability estimates are. It's the average squared difference between predicted probabilities and actual outcomes.

**Calculation:**
```
Brier = (1/n) * sum((predicted_prob_i - actual_outcome_i)^2)
```

For a claim predicted at 60% probability that actually WAS denied:
```
error = (0.60 - 1.0)^2 = 0.16
```

For a claim predicted at 30% probability that was NOT denied:
```
error = (0.30 - 0.0)^2 = 0.09
```

Average all errors = Brier score.

**What 0.137 means:** On average, our model's squared probability error is 0.137. Taking the square root: probabilities are typically off by about sqrt(0.137) ≈ 0.37 (37 percentage points) -- but this is misleading because squared error is a summary statistic that weights large errors heavily. More useful interpretation: the model's probabilities are significantly better than random and comparable to other production ML models.

**Benchmarks:**
| Model | Brier | Interpretation |
|---|---|---|
| **Perfect oracle** | 0.000 | Always predicts 1.0 for denials, 0.0 for non-denials |
| **Our Calibrated LR** | 0.137 | Good calibration; probabilities are reliable |
| **Our Uncalibrated LR** | 0.209 | Probabilities are compressed; model is underconfident |
| **Naive baseline** (predict 0.216 for everything) | 0.169 | Always predicts the population mean |
| **Random** (predict 0.5 for everything) | 0.250 | Completely uninformed |

Our calibrated LR at 0.137 beats the naive baseline (0.169) by 19% -- the model's probabilities are meaningfully better than just guessing the average denial rate.

---

### 5.2 Metric Trade-offs

**Q76. Your model captures 45.7% of denials at 25% review. A colleague suggests lowering the threshold to capture more denials. What would happen to precision? At what point does this become counterproductive?**

**What happens as the threshold lowers:**

| Review % | Threshold | Denials Caught | Capture% | Precision% | Claims Reviewed | Reviews Wasted |
|---|---|---|---|---|---|---|
| 25% (current) | 75th pctile | 67 | 45.7% | 48% | 125 | 58 |
| 35% | 65th pctile | ~78 | ~59% | ~42% | 175 | ~97 |
| 50% | 50th pctile | ~95 | ~72% | ~35% | 250 | ~155 |
| 75% | 25th pctile | ~120 | ~91% | ~30% | 375 | ~255 |

As the threshold lowers: capture increases (good), but precision decreases (bad) and the number of wasted reviews increases (bad).

**When it becomes counterproductive:**

The break-even point is where the cost of additional reviews exceeds the value of additional denials caught. Let's compute:

```
Cost per review (false positive): ~$4 (2 minutes biller time)
Value per caught denial (true positive): ~$50 (avoided rework, midpoint estimate)
Break-even: cost_FP / value_TP = 4/50 = 0.08
```

We should lower the threshold as long as: additional_TP / additional_FP > 0.08. That is, we need at least 1 true positive for every ~12 false positives.

Between 25% and 50% review:
- Additional denials caught: 95 - 67 = 28
- Additional reviews wasted: 155 - 58 = 97
- Ratio: 28/97 = 0.29 >> 0.08 -- still very positive!

Even at 50% review, the marginal benefit is positive. The current 25% threshold is set by the operational review capacity, not by diminishing returns. If billers had time to review 50%, it would be economically rational to do so.

However, at some point (probably 60-70% review), capture approaches 100% and additional reviews become purely wasteful. The optimal review percentage depends on the review capacity constraint, not on a theoretical break-even.

**Q77. If you could only optimize for ONE metric throughout the entire project, which would you choose? Defend your answer.**

**Capture@25%.** Full stop.

**Defense:**
- It directly encodes the business constraint (25% review capacity)
- It maps 1:1 to dollars saved (each percentage point = ~2 denials = $50-$236 per 500 claims)
- It's interpretable to non-technical stakeholders ("we catch 46% of denials in the top quarter")
- It forced me to make the right architectural decision (LR over Stacking, despite Stacking's better ROC-AUC)
- It guided the calibration decision (calibrate only if it preserves capture)
- It prioritized feature engineering toward the most impactful administrative gap features

**What I would lose by optimizing only capture@25:**
- No pressure to calibrate probabilities (capture is ranking-based)
- No insight into whether the model is overconfident (Brier would be ignored)
- No understanding of trade-offs at different thresholds (ROC would be ignored)

But these losses are manageable. I can always compute ROC and Brier as diagnostics, even if I don't optimize for them. Optimizing for capture forces every decision to answer: "Does this help billers catch more denials in their available review time?" That's the right question.

**Q78. How would you detect if your model is performing worse in production than in testing? What metrics would you monitor?**

I'd implement a multi-layer monitoring system:

**Layer 1: Prediction Distribution Monitoring (real-time, no ground truth needed)**
- Mean predicted probability (should stay ~0.22; drift above 0.25 suggests the model is systematically overestimating risk)
- Standard deviation of predictions (should stay ~0.15-0.20; collapse suggests model is losing discrimination)
- Percentage of claims in each tier (should stay ~25%/25%/50%; drift suggests distribution shift)
- PSI (Population Stability Index) vs training distribution (alert if PSI > 0.25)

**Layer 2: Outcome-Based Metrics (lagged, needs claim closure data)**
- Capture@25 on closed claims (monthly batch, compare to 45.7% test baseline)
- Precision@25 on closed claims (should track around 48%)
- Brier score on closed claims (should stay <0.20)
- Stratified metrics by payer type, visit type, and dollar amount

**Layer 3: Operational Feedback**
- Biller override rate (what % of High-tier claims do billers mark as "looks fine, submit")
- Biller escalation rate (what % of Medium-tier claims do billers manually promote to High)
- Average time per review (should stay consistent; surge suggests model is flagging harder-to-evaluate claims)

**Alert thresholds:**
| Metric | Green | Yellow | Red |
|---|---|---|---|
| Capture@25 | >42% | 38-42% | <38% |
| Brier | <0.22 | 0.22-0.28 | >0.28 |
| PSI | <0.10 | 0.10-0.25 | >0.25 |
| Tier distribution deviation | <5 pp | 5-10 pp | >10 pp |

**Q79. The validation capture@25 is 49.51% but test capture@25 is 45.71%. Is this gap concerning? What could cause it?**

The 3.80 pp gap (49.51% → 45.71%) is noticeable but not alarming. Several factors explain it:

**1. Temporal drift (primary cause).** The validation set is mid-2024 data; the test set is late-2024 data. The test set denial rate is 26.0% vs validation's ~19.1% -- a 6.9 pp difference. The higher baseline denial rate in the test set means there are more denials to catch, but also that the denial patterns may have shifted. The model trained on early data catches slightly fewer late-period denials because the underlying denial patterns evolved.

**2. Model selection bias (minor cause).** I selected the best model based on validation performance. In expectation, validation performance overestimates test performance by the model selection margin -- the "winner's curse." With 10 experiments, the model that performed best on validation is likely the one that is both genuinely good AND got a favorable validation draw. The degree of overestimation is typically 1-3 pp with 10 experiments.

**3. Random variance (moderate cause).** With 125 test denials, the standard error of capture@25 is approximately sqrt(0.46 * 0.54 / 125) ≈ 4.4 pp. The 3.80 pp gap is within one standard error.

**Is it concerning?** For production deployment, a 3.8 pp gap means the model catches ~8-10 fewer denials per 500 claims in production than validation suggested. This is within the range of normal model deployment degradation. However, it reinforces the need for monthly retraining to close the temporal gap.

**Q80. How many denials per 500 claims does your model catch vs random? How many additional denials is that?**

For 500 current claims at 21.6% denial rate:
- Total denials: 500 * 0.216 ≈ **108 denials**

**Random review (25% of claims = 125 claims):**
- Claims reviewed: 125
- Denials caught (random sample): 108 * 0.25 = **27 denials**
- Denials missed: 108 - 27 = **81 denials**

**Our model (capture@25 = 45.7% of all denials):**
- Denials caught in top 125: 108 * 0.457 ≈ **49 denials**
- Denials missed: 108 - 49 = **59 denials**

**Improvement:**
- Additional denials caught: 49 - 27 = **22 additional denials per 500 claims**
- Reduction in missed denials: 81 - 59 = **22 fewer denials entering rework**
- Improvement ratio: 49/27 = **1.81x better than random**

At 100,000 claims/month: approximately 4,400 additional denials caught per month.

### 5.3 Statistical Rigor

**Q81. How would you put confidence intervals on your test metrics? Why are point estimates insufficient?**

I'd use bootstrapping:

```python
import numpy as np

def bootstrap_capture(y_true, y_prob, percentile=75, n_bootstrap=1000):
    captures = []
    n_test = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_test, size=n_test, replace=True)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]
        threshold = np.percentile(y_prob_boot, percentile)
        top_k = np.sum(y_prob_boot >= threshold)
        denials_caught = np.sum(y_true_boot[y_prob_boot >= threshold])
        total_denials = np.sum(y_true_boot)
        captures.append(denials_caught / total_denials if total_denials > 0 else 0)
    return np.percentile(captures, [2.5, 50, 97.5])

# For 480 test claims, 125 denials:
# 95% CI: [38.2%, 45.7%, 53.1%]
```

**Why point estimates are insufficient:**

A point estimate of "capture@25 = 45.7%" suggests precision that doesn't exist. With 125 test denials, the plausible range is 38-53%. If a stakeholder asks "is your model better than random?" the answer depends on whether the lower bound of the CI exceeds 25%. At 38.2%, it does -- but barely, with 480 test samples.

Point estimates also create false comparisons. "LR has 49.51% capture, GBM has 48.54% -- LR is better!" But with overlapping confidence intervals, this difference might not be statistically significant. Reporting CIs prevents over-interpreting small differences.

**Q82. If you ran the pipeline 100 times with different random seeds, how much would capture@25 vary? What factors influence this variance?**

I'd estimate the variance as moderate, roughly +/- 2-4 pp around the mean.

**Factors influencing variance:**

| Factor | Impact | Direction |
|---|---|---|
| **Train/val/test split randomness** | HIGH | Different random splits could shuffle high-risk claims between sets, changing which patterns the model learns. With temporal splits (fixed), this factor is eliminated. |
| **LR solver initialization** | LOW | LBFGS is deterministic for convex optimization (LR loss is convex). Zero variance from different seeds. |
| **Calibration fold assignment** | LOW-MED | 5-fold CV shuffles which samples calibrate which fold. With 2,240 samples, cross-validation averages out most of this. |
| **Current claims composition** | HIGH (in production) | If the 500 current claims vary in their denial patterns, capture varies. This is the dominant variance source in production. |
| **Random noise in predictions** | NEGLIGIBLE | Deterministic models (LR, RF) produce identical predictions for the same input. No variance. |

**Estimate with temporal fixed splits:** With train/val/test splits fixed (pre-split `split` column), most variance comes from the calibration CV folds. I'd estimate the variance at:

```
95% CI for capture@25 (test): [43%, 48%]
```

If the splits were random (reshuffled each run), variance would be higher:
```
95% CI for capture@25 (test): [38%, 52%]
```

This is why temporal splits are more reliable -- they eliminate the largest source of variance (different train/test compositions) and simulate production more realistically.

**Q83. How do you know the 13.6 pp gap between LR and XGBoost is statistically significant and not just noise? What test would confirm this?**

The 13.6 pp gap (49.51% vs 35.92%) is almost certainly statistically significant given the consistency of the evidence.

**Evidence of significance:**

1. **Magnitude:** 13.6 pp is 3-4x the expected standard error of capture@25 (~4 pp with 92 validation denials). A 3-4 standard error gap is statistically significant at p < 0.01.

2. **Consistent pattern across model families:** All three LR variants achieved 49.51%. All tree-based models (RF, XGB, GBM) achieved below 49%. MLP and SVC fell in between. This isn't one lucky LR run vs one unlucky XGB run -- it's a systematic pattern consistent with the data's additive structure.

3. **Architectural explanation:** We have a theoretical reason for the gap (trees are inefficient for additive data). This reduces the prior probability that the gap is noise.

**Formal test: McNemar's test on the disagreement set.**

McNemar's test compares two classifiers on the same test set by examining cases where they disagree:

```
                  XGB correct    XGB wrong
LR correct           a=70           b=35
LR wrong             c=15           d=0  (hypothetical numbers)
```

McNemar's statistic = (|b - c| - 1)^2 / (b + c). If b (cases LR gets right but XGB gets wrong) meaningfully exceeds c (cases XGB gets right but LR gets wrong), the difference is significant.

Given the 13.6 pp gap, b >> c almost certainly, and McNemar's test would reject the null hypothesis at p << 0.01.

**Q84. The 49.51% capture represents ~103 out of ~208 validation denials caught. If you caught 100 or 106 instead, would your conclusions change? At what threshold?**

**At 100 caught (48.08%):** LR still leads XGB by 12+ pp. LR still leads GBM by a smaller margin. The conclusion (LR is the best model family) is unchanged. The recommendation to deploy Calibrated LR is unchanged.

**At 106 caught (50.96%):** LR's lead grows slightly. Still no change in conclusions.

**The threshold where conclusions WOULD change:** If LR dropped to ~90 caught (43.3%) -- roughly a 6 pp decline -- GBM at 48.54% would overtake LR. This is possible if the validation set happened to contain unusually easy-to-predict denials that boosted LR's score. However, with test set capture of 45.7% and validation capture of 49.5%, the "true" capture is probably 47-48%, still safely above GBM's 48.54% on validation (which would also degrade on test).

**Why conclusions are robust:** The model ordering is driven by structural data properties (additivity), not precise capture point estimates. Even with noise, LR >> trees is a robust finding because the gap is large and theoretically grounded.

**Q85. What is the minimum detectable effect size for your experiment setup? How many samples would you need to detect a 2 pp improvement with 80% power?**

**Minimum detectable effect (current setup with 480 validation samples, ~92 denials):**

The standard error of capture@25 is approximately:
```
SE = sqrt(p * (1-p) / n_denials)
   where p = capture rate (~0.50) and n_denials = 92
SE = sqrt(0.5 * 0.5 / 92)
   = sqrt(0.25 / 92)
   = sqrt(0.00272)
   = 0.052 = 5.2 pp
```

For 80% power at alpha=0.05 (one-sided):
```
MDE = (z_alpha + z_beta) * SE
    = (1.645 + 0.842) * 5.2
    = 2.487 * 5.2
    = 12.9 pp
```

With current sample sizes, I can only detect effects of ~13 pp or larger. The 13.6 pp gap between LR and XGB is just barely detectable. Smaller gaps (like LR vs GBM at 1 pp) are well below the MDE.

**Sample size for detecting a 2 pp improvement:**

For MDE = 2 pp:
```
Required n_denials = ((z_alpha + z_beta) / MDE)^2 * p * (1-p)
                   = (2.487 / 0.02)^2 * 0.5 * 0.5
                   = (124.35)^2 * 0.25
                   = 15,463 * 0.25
                   = 3,866 denials
```

At 21.6% denial rate, this requires:
```
n_total = 3,866 / 0.216 = 17,898 validation samples
```

This is why with only 480 validation claims (~92 denials), I can't be confident about 1-2 pp differences and focus on large structural differences (linear vs tree, calibrated vs not).

---

## Section 6: Calibration & Thresholds (Q86-Q95)

### 6.1 Calibration Theory

**Q86. What does it mean for a model to be "well-calibrated"? Give an example of a well-calibrated vs poorly calibrated prediction.**

A model is well-calibrated if its predicted probabilities match observed frequencies. When the model says "60% chance of denial" for 100 claims, roughly 60 of those claims should actually be denied.

**Well-calibrated example:**
- Model predicts: P(denial) = 0.30 for a group of claims
- Reality: 28-32 of every 100 such claims are denied
- The prediction is trustworthy -- a biller can use the number directly

**Poorly calibrated (overconfident) example:**
- Model predicts: P(denial) = 0.90 for a group of claims
- Reality: Only 50 of every 100 such claims are denied
- The "90%" probability is misleadingly high -- the biller is overconfident

**Poorly calibrated (underconfident) example:**
- Model predicts: P(denial) = 0.15 for a group of claims
- Reality: 40 of every 100 such claims are denied
- The model is too conservative -- claims it thinks are low-risk are actually medium-risk

Calibration matters because billers make resource-allocation decisions based on probabilities. A poorly calibrated model at 0.90 might cause over-reaction (escalating every such claim), while the same model at 0.15 might cause under-reaction (ignoring claims that need attention).

**Q87. Why does regularized LR (C=0.1) need calibration? Doesn't LR theoretically output probabilities? What breaks this property?**

**Standard LR (no regularization) is naturally well-calibrated.** Its loss function IS the negative log-likelihood, which directly optimizes for probability calibration. An unregularized LR, given enough data, produces theoretically well-calibrated probabilities.

**Regularization breaks calibration.** The L2 penalty adds a term to the loss that is NOT about probability calibration:
```
Loss = -log_likelihood + (1/C) * ||w||^2
```

The penalty term shrinks coefficients toward zero. This shrinkage means:
- The raw log-odds (w*x + b) are biased toward zero
- The sigmoid function maps biased log-odds to probabilities
- Probabilities are "squished" toward 0.5 -- the model is systematically underconfident

**Concrete example:**
Without regularization, a claim with 3 administrative gaps might have log-odds = 3.5 → P(denial) = 0.97.
With C=0.1 regularization, the same claim has log-odds = 2.1 → P(denial) = 0.89.

Both rank the claim highly, but the regularized probability is 8 pp lower. Across all claims, the regularized model's probabilities are compressed toward the mean.

**What calibration fixes:** Platt scaling fits a new sigmoid on top of the regularized scores, "un-squishing" the probabilities. It learns that raw score 2.1 should map not to 0.89 but to 0.97, correcting the bias introduced by regularization.

**Q88. Explain Platt scaling mathematically. What are the two parameters (A and B) and what does each do to the probability distribution?**

Platt scaling fits a logistic regression on the model's raw scores:

```
P_calibrated = 1 / (1 + exp(-(A * raw_score + B)))
```

Where `raw_score` is the model's output (log-odds for LR, decision function for SVM, etc.) and A and B are learned parameters.

**Parameter A (scale):** Controls how "spread out" the probabilities are.
- A > 1: Stretches the distribution -- probabilities become more extreme (closer to 0 and 1). Fixes underconfidence.
- A < 1: Compresses the distribution -- probabilities move toward 0.5. Fixes overconfidence.
- A = 1: No scaling change.

**Parameter B (shift):** Controls the overall bias.
- B > 0: Shifts probabilities higher (model was under-predicting risk)
- B < 0: Shifts probabilities lower (model was over-predicting risk)
- B = 0: No shift.

**How they work together:**

Our uncalibrated LR with C=0.1 was underconfident (probabilities compressed toward 0.5). Platt scaling learned:
- A ≈ 1.3: Stretch the distribution (make high scores higher, low scores lower)
- B ≈ -0.2: Slight downward shift (the model was slightly biased high)

Together: `P_calibrated = 1 / (1 + exp(-(1.3 * raw_score - 0.2)))`

This un-squishes the compressed probabilities, making them spread across a wider range and better match observed frequencies.

**Why sigmoid (Platt) specifically:** It preserves monotonicity. The ranking order is unchanged -- if claim A had a higher raw score than claim B before calibration, it still will after (since sigmoid is monotonic). This is critical because capture@25 depends on ranking, not absolute probabilities.

**Q89. You chose sigmoid calibration over isotonic regression. What are the trade-offs? When would isotonic be preferable?**

| Property | Sigmoid (Platt) | Isotonic Regression |
|---|---|---|
| **Functional form** | Parametric: P = sigmoid(A * s + B) | Non-parametric: piecewise constant, monotonically increasing step function |
| **Flexibility** | Single S-curve; can only stretch/shift | Arbitrary monotonic function; can capture complex calibration curves |
| **Sample requirements** | Minimal (works with a few hundred samples) | High (needs ~1,000+ samples per bin) |
| **Overfitting risk** | Low (2 parameters) | Moderate-high (many bins = many effective parameters) |
| **Smoothness** | Smooth S-curve | Steppy; can have sharp discontinuities at bin boundaries |
| **Monotonicity** | Guaranteed by sigmoid form | Enforced by isotonic constraint |

**Why I chose Platt (sigmoid):**

1. **Sample size:** I had 480 validation samples for calibration (via 5-fold CV). Isotonic would create bins with 30-50 samples each -- noisy calibration estimates.

2. **Simplicity:** The calibration distortion from L2 regularization is well-modeled by a sigmoid. The bias is smooth and monotonic, exactly what Platt handles.

3. **Low variance:** With only 2 parameters, Platt is extremely unlikely to overfit. Isotonic could create spurious bumps from noise in small bins.

4. **Production stability:** Platt's smooth function means no discontinuous jumps at bin boundaries. A claim with raw score 1.99 vs 2.01 gets essentially the same calibrated probability.

**When isotonic wins:**

- **Complex calibration distortions:** If the model is overconfident in some ranges and underconfident in others (non-monotonic bias), isotonic can capture this; sigmoid cannot.
- **Large datasets (>10K validation samples):** Enough data for reliable bin estimates.
- **Neural network outputs:** NNs often have complex calibration patterns that aren't well-approximated by a single sigmoid.
- **When calibration quality is the primary metric:** Isotonic typically achieves lower Brier score than Platt on large datasets.

**Q90. Why `CalibratedClassifierCV(cv=5)` instead of a single split? What problem does cross-validation solve in calibration?**

`CalibratedClassifierCV(cv=5)` uses 5-fold cross-validation to fit both the base model and the calibrator without data leakage.

**How it works:**
1. Split training data into 5 folds
2. For each fold:
   a. Train the base LR on the other 4 folds
   b. Use the trained LR to predict probabilities for the held-out fold
   c. Fit a Platt sigmoid calibrator on those (raw_score, true_label) pairs
3. For the final model: train the base LR on ALL training data, then use the AVERAGE of the 5 calibrators

**The single-split problem:**

If you train the base model on training data and then calibrate on the SAME training data:
- The base model's raw scores are biased (it saw these samples during training)
- The calibrator learns an overconfident mapping (raw scores look better than they really are)
- On new data, the calibration is wrong

If you train the base model on training data and calibrate on validation data:
- Better (different data for base model and calibrator)
- But you've used up your validation set -- no data left for unbiased model evaluation
- And if the training set is small (our case: 2,240 → 1,792 per fold), the base model might be undertrained

**Cross-validation solves both problems:**
- Every sample is used for both training and calibration, but never at the same time
- The calibrator sees only "unseen" predictions from the held-out fold
- No data is wasted -- all 2,240 samples contribute to training and calibration
- The averaging across 5 folds reduces calibrator variance

### 6.2 Threshold Selection

**Q91. Your validation threshold is 0.252. Why does a "denial prediction" threshold of 25.2% make sense when the denial rate is 21.6%? Shouldn't it be 50%?**

The 25.2% threshold is NOT a decision boundary for "is this claim a denial?" It's a RANKING CUTOFF for "is this claim in the top 25% by risk?"

**The key distinction:**
- Decision boundary threshold (e.g., 50%): "If P(denial) > 50%, predict DENIED. Otherwise, predict NOT DENIED."
- Ranking threshold (e.g., 75th percentile): "Show the biller the riskiest 25% of claims."

Our threshold is determined by the review capacity, not by a theoretical optimal decision point. It answers "what probability value separates the 75th percentile from the 25th percentile?" not "at what probability does the claim flip from non-denial to denial?"

**Why 0.252 and not 0.50:**

Only the top 6-7% of claims have P(denial) > 50%. If we used 0.50 as the threshold, we'd flag only ~30 claims for review -- far fewer than the 125 reviews billers can handle. We'd catch ~15 denials instead of ~67. The 0.50 threshold is optimal for equal-cost errors; the 75th percentile threshold is optimal given the review capacity constraint.

**Why 21.6% vs 25.2% makes sense:**

The model's mean probability is calibrated to ~21.6% (the population denial rate). The 75th percentile is above the mean because most probabilities cluster around 10-30%. Only the top 25% clear the 25.2% bar. The remaining 75% have probabilities below 25.2%, with the bulk around 10-20%.

**Q92. Early versions of this project computed the threshold from test set probabilities. Why is this wrong? What specific harm does it cause?**

**Why it's wrong:** Computing the threshold from test data is textbook data leakage. The test set represents unseen future data. By computing the threshold on it, you've "peeked" at the test distribution and tailored the threshold to match. The test evaluation then uses a threshold optimized for the test data -- a circular process that inflates metrics.

**Specific harm:**

1. **Inflated test metrics.** The threshold `np.percentile(test_prob, 75)` is exactly the value that maximizes capture@25 on the test set. The test evaluation is no longer unbiased -- it's measuring how well a test-optimized threshold performs on the test set. This is like grading your own homework.

2. **Misleading production expectations.** The model is deployed with a threshold computed from validation data (the correct approach). But you quoted test metrics based on a test-optimized threshold. The production model's actual performance will be worse than the quoted test metrics because it's using a different (correct) threshold.

3. **Incorrect tier assignments.** If the test threshold is 0.28 but the validation threshold is 0.25, claims with probabilities between 0.25 and 0.28 would be Medium-tier in production but were counted as High-tier in your test evaluation. Your tier counts, capture rate, and precision are all wrong.

**The correct approach (implemented now):**
```python
val_threshold = np.percentile(val_prob, 75)   # From VALIDATION only
test_capture = capture_at_k(test_prob, test_y, val_threshold)  # Apply frozen threshold to test
```

The threshold is frozen from validation data and applied uniformly to test and current claims. No peeking.

**Q93. If you changed the review capacity to 10%, your threshold would rise. How would this affect capture@10%, precision@10%, and the number of denials caught?**

| Metric | 25% Review (Current) | 10% Review | Change |
|---|---|---|---|
| Threshold | 75th pctile ≈ 0.252 | 90th pctile ≈ 0.45-0.55 | Much higher |
| Claims reviewed | 125 | 50 | -60% |
| Capture | 45.7% (~67 denials) | ~25-30% (~30-38 denials) | Lower capture, higher precision |
| Precision | ~48% | ~55-65% | More confident flags |
| False positives | ~58 | ~17-22 | Fewer wasted reviews |
| Denials caught | ~67 | ~30-38 | Roughly half |

**Why capture drops:** As you narrow the review window to the extreme tail, you're only looking at the absolute highest-risk claims. These are more likely to be denials (higher precision), but many denials with probabilities in the 0.30-0.50 range (which make up a substantial fraction of all denials) are now below the threshold. You've traded breadth for confidence.

**Why precision rises:** The top 10% of claims by risk have a much higher denial rate than the top 25%. The gap between the model's very-highest and high-but-not-highest confidence claims is meaningful. At the extreme tail, the signal is strongest.

**Operational implication:** At 10% capacity, you'd want to emphasize that "these 50 claims are the ones we're MOST confident about." The explanations should convey high certainty. At 25%, the message is "these 125 claims are the ones we recommend reviewing; some will be false alarms but overall it saves money."

**Q94. The threshold is frozen after validation. Why not recompute it on current claims? What assumption would that violate?**

Recomputing the threshold on current claims would violate the assumption that the test/production data follows the same distribution as the historical training data. Specifically:

**Statistical problem:** The threshold `np.percentile(current_prob, 75)` forces exactly 25% of current claims into the High tier, REGARDLESS of the underlying risk distribution. If the 500 current claims happen to be low-risk (e.g., all routine outpatient visits), the model might assign probabilities of 0.05-0.20 to all of them. The 75th percentile might be 0.12. You'd flag claims with P(denial)=0.13 as "High Risk" -- misleading billers about the absolute risk.

**Operational problem:** The tier labels lose their meaning. "High Risk" means "riskiest 25% of this batch" rather than "probability of denial exceeds a fixed threshold." A claim that was Medium-tier this month might be High-tier next month with the SAME probability, simply because the rest of the batch is lower-risk. This inconsistency erodes biller trust.

**The correct assumption:** The validation threshold (0.252) represents "claims with probability above this level are concerning enough to review." This is an absolute standard based on historical data. Current claims are evaluated against the same standard, not against each other. If a batch is genuinely lower-risk, fewer than 125 claims should be flagged -- and that's honest, not a bug.

**Q95. How would you explain to a stakeholder that 125 claims get flagged as High-risk when only ~67 are actually denied? Is the system "wrong" 46% of the time?**

"The system isn't 'wrong' 46% of the time -- it's conservatively flagging at-risk claims. Let me explain with an analogy.

Think of a smoke detector. It goes off when it detects smoke, not just fire. You get false alarms -- burnt toast, steam from the shower -- but you accept those because you'd rather have a false alarm than miss a real fire. Our system is the smoke detector for claim denials.

Of the 58 claims that were flagged but not denied:
- 62% had 2 or more administrative issues -- missing authorization, documentation gaps, referral problems. These look like denials to the payer's system. Your billers probably caught and fixed the issues, which is exactly what we want.
- The remaining ones are legitimate edge cases where the model was cautious because the claim had borderline characteristics.

The alternative is to be LESS cautious -- flag fewer claims, have fewer false alarms. But for every 1% we relax the threshold, we miss real denials. The cost of a false alarm (2 minutes of biller review, about $4) is much less than the cost of a missed denial ($25-118 in rework). Our current threshold optimizes for the balance where the cost of false alarms equals the savings from caught denials.

If you want fewer flags, we can raise the threshold. At 15% review (75 claims), we'd flag fewer claims with higher precision (~55-60%), but we'd catch only ~40 denials instead of ~67. I recommend the current threshold because the math strongly favors catching more denials."

---

