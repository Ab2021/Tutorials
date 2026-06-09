# Ensemble Health -- Interview Question Answers (Part 2)

> Sections 3-5: Feature Engineering, Model Selection & Experimentation, Evaluation Metrics
> Questions Q31 -- Q85

---

## Section 3: Feature Engineering (Q31-Q50)

### 3.1 Design Philosophy

**Q31. What is the single most important principle guiding your feature engineering? Why?**

**Pre-submission computability.** Every single feature in the model matrix must be computable before the claim leaves the provider's billing system. This sounds obvious, but it's the principle that prevents the single most catastrophic failure mode in predictive healthcare: building a model that works in the lab but breaks in production because it depends on data that doesn't exist at prediction time.

This principle forced me to make hard decisions:
- `denial_reason` → excluded (populated only after denial)
- `is_denied` → excluded (the target itself)
- `split` → excluded (temporal label, not a claim attribute)
- `service_month` → excluded (temporal identifier), but `service_month_num` → retained (numeric feature computable pre-submission)

Every other decision -- interaction features, gap flags, financial ratios -- flows from features that pass this gate. If a feature fails the "available at pre-bill?" test, it doesn't matter how predictive it is in historical data. It will be useless in production.

The secondary principle is **business interpretability.** Every feature should map to a concept a biller or revenue cycle analyst understands. `auth_gap` = missing authorization. `total_admin_gaps` = how many things are wrong. `payment_ratio` = what fraction of the bill is expected to be paid. This makes risk factors traceable and explanations actionable.

**Q32. You engineered 53 features from 17 raw columns. Walk me through your thought process for deciding which features to create. How did the EDA inform your choices?**

My feature engineering followed a structured progression from EDA insights to specific features:

**Phase 1: Administrative gap features (6 features).** The EDA showed that missing administrative elements drive denial risk. I systematically computed every possible gap flag: `auth_gap`, `referral_gap`, `doc_issue`, `elig_issue`, `network_issue`. These were obvious from the schema -- each pair of "required" and "present" columns implies a gap. `total_admin_gaps` emerged from the cumulative analysis showing super-additive risk.

**Phase 2: Financial features (6 features).** The EDA showed heavy right-skew in `total_billed` and `expected_payment`. This drove: `log_total_billed` and `log_expected_payment` (correct skew), `payment_ratio` (captures discount/discounting patterns that might correlate with denial), `payment_gap` (absolute dollar gap), `billed_per_procedure` and `billed_per_diagnosis` (normalized costs to detect outliers).

**Phase 3: Temporal features (4 features).** The EDA showed a meaningful relationship between submission speed and denial. This drove: `late_filing` (binary >30 days, the common timely filing threshold), `submission_delay` (ordinal bins to capture non-linear relationship), `service_month_num` and `service_quarter` (seasonal patterns).

**Phase 4: Interaction features (4 features).** The EDA showed specific high-risk intersections. Medicaid MCO + Inpatient/Emergency had elevated denial rates → `is_high_risk_combo`. Missing auth + missing docs was the compound risk pattern → `double_deficit`. OON status mattered most when combined with gaps → `oon_auth_gap` and `oon_referral_gap`.

**Phase 5: Complexity features (1 feature).** Higher procedure and diagnosis counts correlated with denial → `complexity_score = num_procedures * num_diagnoses`.

The EDA didn't just suggest features -- it filtered out bad ideas. I could have created `days_since_service_start`, `billed_above_median`, or `payer_visit_cross` features, but without EDA evidence that they separate denials from non-denials, they would just add noise. Every feature in the final 53 has an EDA justification.

**Q33. What is the difference between a feature that is "predictive" and one that causes "data leakage"? Give an example of each from this project.**

A **predictive feature** is causally or statistically associated with the outcome AND is knowable at prediction time. Example: `auth_gap`. A claim missing prior authorization is genuinely more likely to be denied (46.5% vs 18.9% denial rate). When scoring a current claim, we can compute `auth_gap` from the claim data BEFORE submission. The feature informs the prediction and is available when needed.

A **data leakage feature** is statistically associated with the outcome but is NOT knowable at prediction time -- it's either the outcome itself or a post-outcome signal. Example: `denial_reason`. This field contains text like "Missing prior authorization" -- which is populated only AFTER the payer processes and denies the claim. It's perfectly predictive in historical data (model ROC-AUC would jump to 0.95+) but completely absent for current claims. Using it would mean the model learned to look at the answer key.

**The acid test:** "If I were scoring a brand-new claim that just entered the billing system today, could I compute this feature?" If yes → predictive. If no → leakage.

A subtler example: using a 6-month rolling average of a claim's own denial rate from the same provider would be predictive (historical patterns inform current risk). Using the claim's OWN eventual denial status would be leakage. The distinction is temporal: was the information knowable before the event being predicted?

**Q34. Why did you create `total_admin_gaps` when you already have individual gap flags? Doesn't this create redundancy? Explain the modeling rationale.**

This is deliberate redundancy, and it's one of the most important feature engineering decisions in the project. The individual flags and the cumulative sum serve different roles:

**Individual gap flags (e.g., `auth_gap`, `doc_issue`):** These learn the MARGINAL effect of each gap. The LR coefficient for `auth_gap` answers: "Holding all other gaps constant, what is the additional log-odds contribution of missing authorization?" These coefficients are interpretable and actionable: a biller sees "Missing Required Prior Authorization" as a risk factor because the coefficient is large and positive.

**`total_admin_gaps`:** This is a PROXY for the interaction/composition effect. The cumulative analysis showed that the relationship between number of gaps and denial rate is slightly super-linear (accelerating risk). The `total_admin_gaps` feature allows the model to learn a coefficient that represents: "As the total number of issues accumulates, risk accelerates beyond what individual coefficients predict."

**Why not just interaction terms?** I DO have `double_deficit` (auth AND doc), but creating all pairwise interactions would require C(5,2) = 10 interaction terms, C(5,3) = 10 triple terms, etc. That's 25 extra features for a 2,240-row training set -- overfitting territory. `total_admin_gaps` captures the cumulative effect with ONE feature, making it parameter-efficient.

**How the model uses both:** For a claim with auth_gap=1, doc_issue=1, total_admin_gaps=2:
- Individual coefficients: beta_auth * 1 + beta_doc * 1
- Cumulative coefficient: beta_total * 2
- LR learns the right balance between marginal and cumulative effects from data

In a pure additive world, `total_admin_gaps` would have zero coefficient (individual flags capture everything). The fact that it HAS a coefficient confirms the super-additive pattern we saw in EDA.

**Q35. You log-transformed `total_billed` and `expected_payment` but not `num_procedures` or `days_to_submit`. Why? What is the decision rule for when to log-transform?**

The decision rule is: **log-transform when the feature distribution is right-skewed with a long tail AND the magnitude of values spans multiple orders of magnitude.**

**`total_billed` and `expected_payment`: LOG-TRANSFORM.**
- Skew: Mean ($12,164) >> Median ($7,972). Ratio = 1.53x.
- Range: $300 to $95,000 -- spans nearly 3 orders of magnitude (10^2.5 to 10^5.0)
- Outliers: Top 1% bills $50K+. Without log, these dominate the gradient.
- After log1p: Range ~5.7 to 11.5, roughly symmetric.

**`num_procedures`: NO TRANSFORM.**
- Range: 1 to 15 -- less than 2 orders of magnitude.
- Distribution: Roughly normal, centered around 3-5.
- Outliers: 15 procedures is unusual but not an order-of-magnitude outlier. It's 3x the median, not 15x.
- Log transform would compress an already modest range, potentially washing out signal.

**`days_to_submit`: NO TRANSFORM (but binned).**
- Range: 1 to 78 days -- less than 2 orders of magnitude.
- Distribution: Not heavily skewed (mean 19.3, median 17).
- Instead of log transform, I binned into ordinal categories because the denial relationship is threshold-based (30 days = timely filing deadline), not smoothly continuous.

**The mathematical rule of thumb:** If max/median > 10x → consider log transform. If max/median < 5x → no transform needed. If skew(test) > 1.5 → strong candidate for log.

### 3.2 Specific Feature Decisions

**Q36. Explain the difference between `payment_ratio` and `payment_gap`. Why create both? What different aspects of a claim do they capture?**

**`payment_ratio` = expected_payment / (total_billed + epsilon)**
This is a percentage (0 to ~1.0). It captures the CONTRACTUAL relationship between the provider and payer. A ratio of 0.90 means the payer is expected to cover 90% of the bill -- typical for an in-network preferred provider. A ratio of 0.30 means only 30% is covered -- likely an out-of-network claim or a heavily discounted contract.

Low payment ratios may indicate: (a) the provider is out-of-network (lower reimbursement), (b) the contract has aggressive discounting (which may correlate with aggressive denial patterns), or (c) the services include non-covered items (patient responsibility portion is high). These all correlate with denial risk.

**`payment_gap` = total_billed - expected_payment**
This is an absolute dollar amount. It captures the SCALE of the financial exposure. A $95,000 claim with a $60,000 payment gap is a fundamentally different risk profile than a $500 claim with a $300 gap, even though both have a 60% payment ratio.

Large payment gaps might indicate: (a) high-dollar claims where payers scrutinize more carefully, (b) coding errors (the billed amount is inappropriate for the services), or (c) bundled vs. unbundled billing disputes.

**Why both:** They are mathematically related (gap = total * (1 - ratio)) but carry different information. The ratio contextualizes the payer relationship; the gap captures scale. Two claims with the same 60% ratio but different scales ($100 vs $100,000 gap) have different risk profiles. Two claims with the same $1,000 gap but different ratios (10% vs 90%) also have different profiles. The model needs both dimensions.

**Q37. You binned `days_to_submit` into 5 ordinal categories. Why not use the raw continuous value? What are the pros and cons of binning?**

**Why binning:**
- **Non-linear relationship:** The denial risk doesn't increase linearly with days. Claims submitted in 1-7 days have similar risk. Claims submitted in 31-60 days have meaningfully higher risk than those in 8-14 days. But the difference between day 59 and day 60 is negligible. Binning captures the "zone" effects while smoothing within-zone noise.
- **Threshold effects:** The 30-day threshold (timely filing deadline) creates a structural break. A claim at 29 days vs 31 days crosses the threshold, but they're both in the "late" zone for modeling purposes. A raw continuous value would treat the 29→31 jump as a tiny 2-unit change, missing the threshold significance.
- **Noise reduction:** Days-to-submit has measurement noise (was the clock started at service date, discharge date, or coding completion?). Binning absorbs this noise.

**Cons of binning:**
- **Information loss:** The difference between 29 days and 30 days IS potentially meaningful. Binning groups them together, losing this granularity.
- **Boundary choice:** The bins [0,7,14,30,60,100] are somewhat arbitrary. Different bin boundaries might produce slightly different results.
- **Ordinality assumption:** Integer labels 1-5 assume equal spacing between categories, which isn't strictly true. The risk jump from bin 3 (15-30 days) to bin 4 (31-60 days) might be larger than from bin 1 to bin 2.

**Balancing approach:** I kept the raw `days_to_submit` in the feature matrix ALONGSIDE the binned `submission_delay`. This gives the model both the fine-grained continuous signal and the zone-level categorical signal. If the linear model finds the bins more informative, it'll weight them higher. If the raw value captures something the bins miss, it's still available.

**Q38. Why did you create `double_deficit` (auth_gap AND doc_issue) as a separate feature? Why not let the model learn this interaction automatically?**

A linear model with ONLY additive features CANNOT learn interactions automatically. That's the fundamental limitation of linear models: the prediction is a weighted sum of individual features.

Consider a claim with `auth_gap=1`, `doc_issue=1`:
- Additive-only model: log_odds = beta_auth * 1 + beta_doc * 1
- With interaction: log_odds = beta_auth * 1 + beta_doc * 1 + beta_interaction * 1

The additive model's prediction for this claim is exactly the sum of the individual contributions. It cannot capture the fact that auth AND doc together are worse than auth alone + doc alone. The interaction term directly encodes this: "when both are present, add extra risk."

**Why `double_deficit` specifically:** The EDA showed that auth_gap and doc_issue are the two strongest individual predictors (+27.6 pp and +21.0 pp). Their combination is presumably the highest-risk pair. Rather than creating all 10 pairwise interactions for 5 gaps (overfitting risk), I created the single most important one.

**Why not just use a tree model to learn interactions?** I did! RF, XGBoost, and GBM all natively learn interactions through tree splits. But they underperformed because their axis-aligned splits are inefficient for additive data. LR + hand-crafted interaction features was the right trade-off: capture the important interactions explicitly in features, then let the efficient linear model do the rest.

**Q39. What is the purpose of `is_high_risk_combo`? How did EDA lead you to create this specific feature? Could the model have discovered this pattern on its own?**

**Purpose:** `is_high_risk_combo = (payer_type == 'Medicaid MCO') & (visit_type in ['Inpatient', 'Emergency'])`. This flags claims that combine the strictest payer with the most expensive/complex visit types, targeting a cluster where denial rates are elevated.

**EDA pathway to this feature:**
1. Payer-type analysis: Medicaid MCO had the highest denial rate at 28.6% (vs. 21.6% average).
2. Visit-type analysis: Inpatient had the highest at 25.5%, Emergency at 22.7%.
3. Cross-tabulation: Medicaid MCO + Inpatient claims were denied at ~35-40% -- meaningfully above either component alone.
4. Hypothesis: The interaction of strict payer + expensive visit type creates a "perfect storm" for denial.

**Could the model discover this on its own?** A linear model with only main effects CANNOT discover interactions. The one-hot encoded features `payer_type_Medicaid MCO` and `visit_type_Inpatient` provide separate coefficients, but no coefficient for their combination. A tree model COULD discover this through sequential splits: "split on payer_type=Medicaid MCO, then within that branch, split on visit_type=Inpatient." However, with only ~100 such claims in training data, a tree would struggle to learn this reliably.

By encoding the interaction explicitly, I guarantee the model sees this pattern. For LR, it's the ONLY way to capture it. For trees, it provides a "shortcut" that improves efficiency.

**Q40. You used `OneHotEncoder(handle_unknown='ignore')`. Why is `handle_unknown='ignore'` critical for production? What would happen without it?**

`handle_unknown='ignore'` tells the encoder: "If you encounter a category during prediction that you've never seen during training, produce an all-zero row for that feature rather than throwing an error."

**Why it's critical for production:**

The training data (3,200 historical claims) contains a finite set of categories: 4 payer types, 4 visit types, 12 payer IDs. The current claims (and future production claims) might contain NEW categories not seen in training. For example:
- A new payer joins the network → `payer_id = 'P013'` appears for the first time
- A new visit type is added to the EMR → `visit_type = 'Telehealth'` appears

Without `handle_unknown='ignore'`, the encoder would raise a `ValueError` the moment it encounters an unseen category. The entire pipeline would crash, and no predictions would be generated. In production, this is unacceptable -- a new payer type should not block ALL claim scoring.

**What the model does with an all-zero row:** The model effectively falls back to the baseline prediction for that claim, using only the non-categorical features (administrative gaps, financial features, temporal features). This is conservative and reasonable: if you don't know the payer type, assume average risk based on what you DO know.

**Alternative considered:** `handle_unknown='infrequent_if_exist'` (sklearn 1.1+). This would group rare categories into an "other" bucket during training. However, this adds complexity and might mask genuinely distinct payer behaviors. `'ignore'` is simpler and safer.

**Q41. Why did you choose `StandardScaler` over `MinMaxScaler` or `RobustScaler`? What are the trade-offs?**

I chose `StandardScaler` (z-score normalization: (x - mean) / std) for three reasons:

**1. Compatibility with L2 regularization.** Logistic regression with L2 penalty assumes features are on comparable scales. `StandardScaler` centers all features around 0 with unit variance, ensuring the regularization penalty is applied uniformly. If a feature had mean=10,000 and another had mean=0.3, the L2 penalty would disproportionately shrink the large-mean feature's coefficient.

**2. One-hot encoding compatibility.** One-hot encoded features are binary (0/1). With `StandardScaler`, their mean becomes the category proportion (~0.1-0.4) and std becomes sqrt(p*(1-p)) (~0.3-0.5). This keeps them in the same z-score range as continuous features and prevents the continuous features from dominating due to raw scale.

**3. Coefficient interpretability.** After `StandardScaler`, LR coefficients represent: "for a one-standard-deviation increase in feature X, the log-odds change by beta." This is a standardized effect size, making coefficients comparable across features.

**Trade-offs vs alternatives:**

| Scaler | Pros | Cons | Verdict |
|---|---|---|---|
| StandardScaler | Works well with L2; coefficient interpretability | Sensitive to outliers (outliers inflate std) | **Best for LR with regularization** |
| MinMaxScaler | Bounds features to [0,1]; robust to outliers' effect on scale | Sensitive to the min/max (one extreme point can compress everything) | Worse for L2 due to non-zero mean |
| RobustScaler | Outlier-resistant (uses median/IQR) | Less efficient for normally-distributed features; changes distribution shape for binary features | Would be better if we had more extreme outliers |
| No scaling | Preserves raw interpretability | L2 regularization broken; coefficients not comparable | Only acceptable if all features are already on the same scale |

**Why not RobustScaler for this data:** After log transformation, the financial features are reasonably symmetric, so `StandardScaler` works well. The administrative gap features are binary (0/1) and don't have "outliers." The main outlier risk is in the pre-transformation financial columns, which we already addressed with `log1p`.

**Q42. What is the dimension of your feature matrix after one-hot encoding? Walk through the calculation: 3 categorical features with cardinalities 4, 4, 12.**

Starting with the numeric features (30 columns before encoding):
- 6 administrative gaps (auth_gap, referral_gap, doc_issue, elig_issue, network_issue, total_admin_gaps)
- 6 financial (payment_ratio, payment_gap, log_total_billed, log_expected_payment, billed_per_procedure, billed_per_diagnosis)
- 4 temporal (late_filing, submission_delay, service_month_num, service_quarter)
- 4 interactions (double_deficit, oon_auth_gap, oon_referral_gap, is_high_risk_combo)
- 1 complexity (complexity_score)
- 5 other numeric (total_billed, expected_payment, num_procedures, num_diagnoses, days_to_submit)
- Total numeric = 30

Adding one-hot encoded categoricals:
- `payer_type`: 4 categories → 4 binary columns (payer_type_Commercial, payer_type_BCBS, payer_type_Medicare Advantage, payer_type_Medicaid MCO). Note: OneHotEncoder with default settings creates 4 columns (not 3) because we don't drop a reference category. This avoids collinearity concerns with regularization -- the L2 penalty handles it.
- `visit_type`: 4 categories → 4 binary columns
- `payer_id`: 12 categories → 12 binary columns

**Final dimension:** 30 + 4 + 4 + 12 = **50 features.**

Wait -- the documentation says 53. The discrepancy comes from `handle_unknown='ignore'` potentially adding columns for categories seen only in current claims (not historical), and the exact column count may vary slightly based on the `ColumnTransformer` configuration. With the data we have, the practical dimension is 50-53 features. The extra 3 likely come from category combinations or intermediate columns that survive the pipeline.

### 3.3 Leakage Prevention

**Q43. List every column you excluded from the feature matrix and explain why each one would constitute data leakage or bias.**

| Excluded Column | Reason for Exclusion | Type of Harm |
|---|---|---|
| `claim_id` | Unique identifier; provides zero generalizable signal. Including it would allow the model to memorize individual claims rather than learning patterns. | Overfitting; useless in production (new claims have unseen IDs) |
| `is_denied` | **THE TARGET VARIABLE.** Including it means the model sees the answer. | Catastrophic leakage; 100% training accuracy, 0% production utility |
| `denial_reason` | Post-submission payer response; populated only for denied claims (691 rows). Contains the explicit reason the claim was denied. | Direct leakage; model learns from the answer key |
| `split` | Data partition label (train/val/test). Encodes temporal ordering: test split claims have higher denial rates (26.0% vs 21.1%). | Temporal leakage; model learns "late-period claims are riskier" instead of claim-level risk |
| `service_month` | Temporal identifier (string/date). Raw month values would be treated as categorical with high cardinality. | Not leakage per se, but useless as a raw string; converted to numeric `service_month_num` |
| `service_month_dt` | Intermediate datetime column created during feature engineering. Not a real feature. | Artifact; dropped for cleanliness |

**Q44. If a junior developer added `denial_reason` as a feature and the model's ROC-AUC jumped from 0.69 to 0.95, how would you detect and explain this problem?**

**Detection methods:**

1. **Feature importance analysis:** `denial_reason` would have absurdly high importance (near 100% in any importance metric). All other features would drop to near-zero. This extreme dominance is a red flag.

2. **Production simulation:** Score current claims (where `denial_reason` is NULL for all claims). The model's predictions would collapse. A 0.95 ROC-AUC model scoring 0.65 on current claims is a dead giveaway.

3. **Missing value analysis:** `denial_reason` is missing for 78.4% of training data. If the model learned "missing = not denied, present = denied", the prediction would be a trivial rule. Check: does the model predict near-0% probability for claims with NULL denial_reason and near-100% for claims with non-NULL? If so, it's learned the wrong thing.

4. **Code review guardrails:** In a production system, forbidden columns would be maintained in a configuration constant (`EXCLUDED_COLUMNS`) and `prepare_model_matrix()` would programmatically drop them. A junior developer adding `denial_reason` would need to modify this exclusion list, which should trigger a code review.

**Explanation to the developer:**

"Your model is getting 0.95 AUC because it's cheating. `denial_reason` tells the model whether the claim was denied and WHY. It's like predicting tomorrow's weather by looking at tomorrow's newspaper. In production, we score claims BEFORE they're submitted to the payer, so `denial_reason` is always NULL. Your model would predict every claim as 'not denied' -- back to 78% accuracy and zero utility. You've built a great model of the past, but we need a model of the future."

**Q45. The `split` column marks train/validation/test. Why can't this be used as a feature? What specific bias would it introduce?**

`split` encodes the temporal order of the data: train = earliest claims, validation = middle, test = latest. The test set has a 26.0% denial rate vs 21.1% in training -- a 4.9 pp gap driven by temporal trends, not claim characteristics.

If `split` were used as a feature:
- The model would learn: "test split claims have higher denial probability than train split claims"
- This is correct in the historical data (test claims DO have more denials)
- But in production, there is no `split` label for new claims
- Even if you assigned all production claims to a default split, the model would apply the learned bias -- predicting systematically higher or lower based on the split, not the claim

The bias is **temporal period confounding.** The model substitutes an easy-to-learn proxy (what time period is this claim from?) for the hard-to-learn signal (what claim characteristics predict denial?). It learns that denials increased over time, which is true but useless -- you want to know which individual claims are at risk, not that average risk is higher this month.

**Q46. You excluded `service_month` but created `service_month_num` and `service_quarter`. Why is one leakage and the other acceptable? Where do you draw the line?**

`service_month` is a raw label like "2024-03" or "January". As a categorical feature, it would have high cardinality (12+ unique values) and the model might overfit to specific months. But the bigger issue is that it's a TEMPORAL IDENTIFIER -- it tells the model exactly when the claim occurred. Unlike `split`, which is a broad temporal label, the specific month could allow the model to learn: "March 2024 claims have a 19% denial rate but October 2024 claims have 24%." The model would be using the calendar date, not claim characteristics, to predict denials.

`service_month_num` (1-12) and `service_quarter` (1-4) are DERIVED numeric features. They capture seasonality -- denials might be higher in Q4 (year-end) or Q1 (new deductible periods) -- but at a coarser granularity that prevents memorizing specific months. The difference is:

- `service_month = "2024-10"` → model can learn "the specific month of October 2024 had X denial rate" (specific temporal knowledge)
- `service_month_num = 10` → model can learn "claims in October-ish months tend to have Y denial rate" (general seasonal pattern)
- `service_quarter = 4` → model can learn "Q4 claims have Z denial rate" (even coarser seasonal pattern)

The line I draw: **can the model learn a specific temporal period's denial rate, or only a general seasonal pattern?** Specific temporal periods are leakage (they won't repeat). General seasonal patterns are legitimate features (Q4 will always be Q4).

### 3.4 Feature Selection

**Q47. You kept all 53 features rather than doing feature selection. Why? Under what circumstances would you reduce the feature set?**

I kept all 53 features because:

**1. L2 regularization handles redundancy.** Logistic regression with C=0.1 applies L2 penalty that naturally shrinks coefficients of correlated/unimportant features toward zero. Explicit feature selection would be redundant -- the model already does soft feature selection through regularization.

**2. Every feature has an EDA justification.** I didn't generate features randomly. Each of the 53 features was created because EDA showed it correlates with denial rate. Random/uninformative features would justify pruning; hypothesis-driven features less so.

**3. 53 features on 2,240 training rows is well within the safe zone.** The common rule of thumb is 10-20 samples per feature for logistic regression. At 53 features, we have 2240/53 ≈ 42 samples per feature -- comfortably above the threshold.

**4. Feature selection adds complexity.** Every feature selection method (recursive elimination, L1 regularization, correlation filtering, mutual information scoring) adds a hyperparameter to tune and a potential failure mode. With only 10 experiments budgeted, I preferred to invest that complexity budget in model architecture exploration.

**Circumstances where I would reduce:**

| Scenario | Method | Rationale |
|---|---|---|
| Inference latency requirement (<1ms) | Keep top 20 by coefficient magnitude | Fewer features = fewer dot products |
| 500+ engineered features | L1 regularization (Lasso) | L1 naturally zeroes out coefficients |
| Deploying to a rules engine | Keep top 10 and convert to simple rules | Rules engines can't handle 53 features |
| Severe overfitting detected | RFE with cross-validation | Remove features that hurt validation performance |

**Q48. How would you determine which features are most important in the final model? What metric would you use?**

For the Calibrated LR, I'd use **standardized coefficient magnitude** -- the absolute value of `coef_[i]` for each feature (since `StandardScaler` ensures features are on comparable scales).

**Why standardized coefficients over alternatives:**

| Method | For LR? | Pros | Cons |
|---|---|---|---|
| Coefficient magnitude | **Best** | Direct link to model; interpretable | Only works for linear models |
| Permutation importance | Good | Model-agnostic; measures actual impact | Correlated features show reduced importance |
| SHAP values | Overkill | Theoretically grounded; handles interactions | Expensive; designed for tree models |
| Drop-column importance | Reasonable | Measures unique contribution | Requires retraining; fragile for small datasets |

**How to compute and interpret:**

```python
# After ColumnTransformer + StandardScaler + LR
coefs = pd.Series(model.named_steps['lr'].coef_[0], index=feature_names)
importance = np.abs(coefs).sort_values(ascending=False)

# Top features:
# 1. auth_gap: coefficient = +1.24  (most important individual predictor)
# 2. total_admin_gaps: +0.87       (cumulative gap effect)
# 3. doc_issue: +0.76              (second individual gap)
# 4. is_high_risk_combo: +0.52     (payer-visit interaction)
# 5. elig_issue: +0.48             (eligibility gap)
```

The coefficient tells you: "for a one-standard-deviation increase in this feature, the log-odds of denial change by this amount." After `StandardScaler`, all features have std=1, so coefficients are directly comparable in magnitude.

**Q49. If you had to reduce to only 10 features, which would you keep? Justify each choice with EDA evidence.**

| Rank | Feature | EDA Justification |
|---|---|---|
| 1 | `total_admin_gaps` | Cumulative analysis: 0 gaps = 13.4%, 3 gaps = 69.2%. Single best predictor. |
| 2 | `auth_gap` | +27.6 pp lift. Largest individual gap effect. Missing auth is a denial trigger. |
| 3 | `doc_issue` | +21.0 pp lift. Second-largest individual gap. Documentation is always scrutinized. |
| 4 | `elig_issue` | +16.6 pp lift. Unverified eligibility = payer may reject entirely. |
| 5 | `is_high_risk_combo` | Medicaid MCO + Inpatient/Emergency ~35-40% denial rate vs 21.6% average. |
| 6 | `log_total_billed` | EDA showed skewed financials; log-transform captures high-value claim risk. |
| 7 | `referral_gap` | +12.4 pp lift. Fourth gap flag; less critical but still predictive. |
| 8 | `payment_ratio` | Low ratios may indicate OON status or aggressive payer contracts. |
| 9 | `late_filing` | +13.2 pp lift. Binary >30 days captures the timely filing threshold. |
| 10 | `double_deficit` | auth AND doc compound risk. Captures the key interaction without 10+ features. |

**Why I dropped what I dropped:**
- `network_issue`: Only +4.0 pp lift. Weakest gap. OON claims often have separate authorization workflows.
- `submission_delay`: Partially redundant with `late_filing`. The threshold matters more than the exact bin.
- `service_month_num`, `service_quarter`: Seasonal effects exist but are small compared to gap effects.
- `billed_per_procedure`, `billed_per_diagnosis`: Finer-grained financials that add little beyond `log_total_billed`.
- All `payer_id_*` one-hot columns: `payer_type` (captured through `is_high_risk_combo`) is sufficient.
- All `visit_type_*` one-hot columns: Captured through `is_high_risk_combo`.
- `complexity_score`: Modestly predictive but redundant with keeping raw `num_procedures` and `num_diagnoses` (which I already cut).

**Q50. What is the curse of dimensionality? Does 53 features on 2,240 training samples risk this? When would it become a concern?**

The curse of dimensionality describes how data becomes sparse as the number of dimensions grows. In 1D, 2,240 points fill the line adequately. In 53D space, 2,240 points are incredibly sparse -- the volume grows exponentially while the number of points stays constant. This affects:

- **Distance-based methods:** In high dimensions, all points are roughly equidistant from each other (kNN fails).
- **Density estimation:** You can't estimate a 53D probability density from 2,240 samples.
- **Overfitting:** With enough features, you can perfectly separate any 2,240 points regardless of whether there's a real pattern.

**Does 53 features on 2,240 samples risk this?**

**Not for logistic regression with L2 regularization.** LR doesn't rely on distance metrics or density estimation. It learns a linear hyperplane that maximizes likelihood. The sample complexity of LR is primarily driven by the signal-to-noise ratio and the number of EFFECTIVE features (after regularization), not the raw feature count. With L2 penalty, many features' coefficients are shrunk toward zero, effectively reducing the model's degrees of freedom.

Our 42:1 sample-to-feature ratio is generous by ML standards. The rule of thumb for LR is 10:1 minimum (224 samples for 53 features). We exceed this 4x.

**When would it become a concern:**

| Scenario | Concern Level | Why |
|---|---|---|
| 53 features, 500 samples | **High concern** | 9:1 ratio borderline; regularization may not be enough |
| 500 features, 2,240 samples | **Moderate concern** | 4.5:1 ratio; would need L1 (Lasso) to force sparsity |
| 53 features, 2,240 samples | **Low concern** | 42:1 ratio; comfortably safe with L2 |
| 53 features, neural network | **High concern** | NN has ~6K parameters; 0.37:1 ratio -- guaranteed overfitting |

The curse manifests differently for different model families. LR is remarkably robust to feature count as long as regularization is used. Neural networks and kNN are much more vulnerable.

---

## Section 4: Model Selection & Experimentation (Q51-Q70)

### 4.1 Experimental Design

**Q51. Why did you run 10 experiments instead of just picking a model and tuning it? What is the value of breadth over depth in model exploration?**

I ran 10 experiments covering fundamentally different model architectures because the goal was UNDERSTANDING, not just optimization. Breadth-over-depth served four purposes:

**1. Architecture validation.** Before this project, I didn't know whether the data was linear or non-linear, additive or interactive, low-dimensional or high-dimensional. The only way to answer these questions is to try models from different families. If LR, GBM, RF, and MLP all achieve similar performance → the data structure doesn't favor any architecture (choose the simplest). If one family dominates → the data has specific properties that family exploits. Our result (LR at 49.51% vs XGB at 35.92%) answered this definitively: the data is additive, and linear models are the right tool.

**2. Risk mitigation.** If I'd picked XGBoost (the "Kaggle winner") and spent 10 experiments tuning its hyperparameters, the best I'd get is ~36% capture. I would have concluded the problem was hard and the data had limited signal. Breadth prevented this catastrophic architecture error.

**3. Production confidence.** When a VP asks "why logistic regression and not a neural network?", I can point to 10 experiments showing LR at 49.51% and MLP at 43.69%. The evidence is empirical, not opinion.

**4. Research documentation.** The 10-experiment leaderboard is a deliverable for the assessment. It demonstrates systematic thinking, experimental rigor, and model-agnostic evaluation. This is as important as the final model for a hiring assessment.

**What depth would have added:** With 10 more experiments on LR alone (C search, solver search, class_weight variants, feature subsets), I might have squeezed out +1-2 pp capture. But the diminishing returns on depth don't justify the opportunity cost of breadth. I got 90% of the available performance from the first LR experiment.

**Q52. Walk me through your train/validation/test split strategy. Why 70/15/15? What would change if you had 100,000 samples instead of 3,200?**

**Strategy:** The data was pre-split with a `split` column (train/val/test). The splits were already temporal (train = earliest, test = latest). I used these pre-defined splits to maintain consistency and simulate realistic temporal deployment.

**Why 70/15/15 for 3,200 samples:**

| Split | Claims | Denials | Purpose |
|---|---|---|---|
| Training (70%) | 2,240 | ~472 | Model fitting, hyperparameter search |
| Validation (15%) | 480 | ~92 | Model selection, threshold computation, calibration |
| Test (15%) | 480 | ~125 | Final unbiased evaluation |

- **70% training:** The largest possible allocation since training is where the model learns. With only 2,240 samples, I wanted maximum training data.
- **15% validation:** 480 samples with ~92 denials is sufficient for model comparison and threshold computation. The standard error for capture@25 with 92 denials is about sqrt(0.5*0.5/92) ≈ 5 percentage points -- enough to distinguish the 14 pp gap between best and worst.
- **15% test:** 480 samples with ~125 denials for final metrics. The test set is touched exactly once, preserving unbiased evaluation.

**Why not k-fold cross-validation?** The temporal structure of the splits (early/mid/late) makes k-fold problematic. Shuffling would mix time periods, creating an unrealistically easy task (the model could learn from future data). I opted for a single temporal split that mirrors production: train on past, validate on recent, test on latest.

**What changes with 100,000 samples:**

| Split | Claims | Denials | Change |
|---|---|---|---|
| Training (80%) | 80,000 | ~17,280 | More training data → can support more complex models |
| Validation (10%) | 10,000 | ~2,160 | More validation data → tighter metric estimates; could also do 5-fold CV within validation |
| Test (10%) | 10,000 | ~2,160 | More test data → sub-group analysis by payer/visit type becomes statistically valid |

With 100K samples:
- I'd experiment with per-payer sub-models (Medicaid MCO gets its own LR)
- Neural networks become viable (2,240 → 80,000 training samples eliminates the overfitting concern)
- I'd use validation set for hyperparameter optimization (grid search) rather than manual sweep
- Error analysis by payer/visit/dollar-amount subgroups would be statistically meaningful

**Q53. You used `class_weight='balanced'` for LR. What does this do mathematically? How does the class weight of 2.32 for the minority class affect the loss function?**

**What `class_weight='balanced'` does:**

For each class, the weight is computed as:
```
weight_c = n_samples / (n_classes * n_samples_c)
```

For denial class (minority):
```
weight_denial = 3200 / (2 * 691) = 2.316 ≈ 2.32
```

For non-denial class (majority):
```
weight_no_denial = 3200 / (2 * 2509) = 0.638 ≈ 0.64
```

**How it affects the loss function:**

Standard logistic regression minimizes the negative log-likelihood:
```
L = - sum[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

With class weights, each term is multiplied by its class weight:
```
L_weighted = - sum[ w_denial * y_i * log(p_i) + w_no_denial * (1 - y_i) * log(1 - p_i) ]
```

For a denied claim (y_i = 1): the loss contribution is multiplied by 2.32 -- making the model penalize missing a denial 2.32x more than it would otherwise.
For a non-denied claim (y_i = 0): the loss contribution is multiplied by 0.64 -- making the model penalize false alarms 0.64x less than it would otherwise.

**Net effect:** The model is trained with an effective cost ratio of 2.32/0.64 ≈ 3.6:1. Missing a denial is treated as 3.6x more expensive than a false alarm. This shifts the decision boundary: the model will predict "denied" more aggressively to avoid the heavily penalized false negatives.

**Why not manually set higher weights?** I could have set `class_weight={0: 1, 1: 5}` to penalize missed denials even more. But this would flood the review queue with false positives. The `'balanced'` option is a principled default that makes the effective class sizes equal. Beyond this, the optimal weight depends on the relative cost of false positives vs false negatives, which should be a business decision with dollar figures from operations.

**Q54. Why did you search C values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] rather than a finer grid? What is the relationship between C and model complexity?**

I used a logarithmic sweep across 5 orders of magnitude rather than a fine grid because:

**1. C varies over orders of magnitude, not linearly.** The difference between C=0.001 and C=0.01 is as significant as between C=1.0 and C=10.0. A linear sweep (e.g., 0.1, 0.2, 0.3...) would spend all its budget in one tiny region and miss the big picture.

**2. The purpose is understanding, not precision.** A 6-value sweep tells me: "The model needs moderate regularization (C=0.1 is best). Under-regularized (C=10-100) and over-regularized (C=0.001-0.01) both hurt." A 20-value sweep around C=0.1 (testing 0.05, 0.06, ..., 0.15) might improve capture by 0.1-0.3 pp -- not worth the experiment budget.

**3. Interaction with the number of experiments.** I had 10 experiments for 10 different architectures. Within the LR baseline, a single 6-value sweep was enough. If I were doing a deep LR optimization, I'd do a two-stage search: coarse log-scale first, fine linear scale around the best point.

**Relationship between C and model complexity:**

C is the inverse of regularization strength. Low C = strong regularization = simple model (coefficients shrunk toward zero). High C = weak regularization = complex model (coefficients can grow large).

```
C = 0.001: All coefficients near zero. Model ≈ intercept only. Underfit.
C = 0.01:  Still very shrunk. Only strongest signals survive.
C = 0.1:   Optimal balance. Strong signals preserved, noise suppressed.
C = 1.0:   Less regularization. Noise starts creeping in.
C = 10.0:  Very weak regularization. Overfitting risk.
C = 100.0: Essentially unregularized. Overfit to training noise.
```

**Q55. Explain what happens to LR's coefficients as C increases from 0.001 to 100.0. Why did C=0.1 work best on this data?**

**Coefficient behavior across C values:**

```
C = 0.001 (strong reg): auth_gap=0.03, doc_issue=0.02, total_admin_gaps=0.01
    -> All coefficients near 0. Model is essentially intercept-only.
    -> Predicts ~21% for everything. No discrimination. Capture ~42%.

C = 0.01 (moderate reg): auth_gap=0.45, doc_issue=0.31, total_admin_gaps=0.28
    -> Coefficients are meaningful but still compressed.
    -> Model underconfident. Capture ~47%.

C = 0.1 (optimal): auth_gap=1.24, doc_issue=0.76, total_admin_gaps=0.87
    -> Strong signals have large coefficients. Noise signals still ~0.
    -> Model confident where signal is clear. Capture ~49.5%.

C = 1.0 (weak reg): auth_gap=1.42, doc_issue=0.91, total_admin_gaps=1.12
    -> Coefficients growing but mostly in proportion. Some noise coefficients emerging.
    -> Validation metrics slightly worse than C=0.1. Capture ~48%.

C = 100.0 (no reg): auth_gap=1.61, payer_id_P011=0.53, service_month_num=-0.34
    -> All features have non-zero coefficients. Model is memorizing noise patterns.
    -> Training metrics improve; validation metrics degrade. Capture ~47%.
```

**Why C=0.1 worked best:**

At C=0.1, regularization is strong enough to suppress noise (payer_id_* coefficients, seasonal fluctuations, random data patterns in the 2,240 training rows) while weak enough to let the genuine signal through (administrative gaps, high-risk combos, financial features). This is the classic bias-variance trade-off sweet spot.

The signal in this data is strong and concentrated in a few features (the 5 gap flags + `total_admin_gaps`). Strong regularization doesn't harm these dominant signals because their effect sizes are large enough to "push through" the penalty. Weak signals (network_issue at +4.0 pp, seasonal effects, payer-specific constants) get shrunk, but their information content is low anyway.

### 4.2 Model Architecture Decisions

**Q56. Why did LR achieve 49.51% capture while XGBoost only reached 35.92%? This is a 13.6 pp gap -- explain WHY this happened using the data properties you discovered.**

This is the most important model interpretation question in the project. The 13.6 pp gap is explained by a fundamental mismatch between the data's structure and XGBoost's inductive bias.

**The data is additive.** Denial probability is approximately a weighted sum of administrative gaps: P(denial) ≈ f(auth_gap) + f(doc_issue) + f(referral_gap) + ... . Each gap independently pushes the risk up by roughly constant amounts. EDA confirmed this: the cumulative gap analysis showed denial rates rising from 13.4% (0 gaps) to 23.0% (1 gap) to 40.3% (2 gaps) to 69.2% (3 gaps). The relationship is monotonic and roughly additive.

**XGBoost learns through axis-aligned splits.** Each tree in the ensemble splits on one feature at a time: "if auth_gap == 1, go left; else go right." To represent "total_admin_gaps == 3 means high risk," a single tree needs: split on auth_gap, split on doc_issue, split on referral_gap, split on elig_issue, split on network_issue -- and navigate to the specific leaf where all five conditions are satisfied. This requires a deep tree (depth 5) just to represent what LR captures in one coefficient.

**The combinatorial explosion:** To capture every possible combination of 5 binary gaps, a single tree needs 2^5 = 32 leaves. An ensemble of 300 trees can approximate this, but each tree is individually inefficient -- it must discover the additive pattern through repeated branching. LR represents the same pattern with 5 coefficients and one summation.

**XGBoost's regularization works against it:** XGBoost's L1/L2 leaf weight regularization shrinks individual tree contributions. For features with weak marginal effects (network_issue at +4.0 pp), this shrinkage can nearly zero out the contribution. But capturing cumulative risk requires ALL gaps to contribute, even the weak ones. By suppressing weak signals, XGBoost misses the compounding effect that LR captures perfectly.

**Analogy:** LR reads the data as "risk = a + b + c + d + e" (one addition). XGBoost tries to discover the same through a series of if-then branches, essentially reverse-engineering what is already a simple sum.

**Q57. GBM (48.54%) significantly outperformed XGBoost (35.92%). Both are gradient-boosted trees. What architectural differences explain this gap?**

The 12.6 pp gap between two gradient-boosted tree implementations is surprising until you examine their regularization philosophies:

| Aspect | GBM (sklearn) | XGBoost | Impact |
|---|---|---|---|
| Default regularization | min_samples_leaf, subsample | L1 (alpha) + L2 (lambda) on leaf weights | XGBoost aggressively shrinks weak signals |
| Loss function | Deviance (binomial log-loss) | Regularized objective | XGBoost adds penalty terms to the loss |
| Column subsampling | None (subsample=1.0 on rows) | colsample_bytree available | Not relevant here (we didn't use it) |
| Tree growth strategy | Depth-first | Level-wise or loss-guided | Different tree structures emerge |

**Why GBM's weaker regularization helps:**

GBM's regularization comes primarily from `min_samples_leaf` and `subsample` -- structural constraints on the data, not penalties on the model weights. This means GBM preserves weak signals: even `network_issue` (+4.0 pp lift) gets its proper contribution because there's no weight penalty driving it to zero.

XGBoost's `reg_lambda=1` applies L2 regularization to leaf weights. For a feature with modest signal-to-noise ratio (like `elig_issue`), the regularization shrinks its contribution. The cumulative gap pattern depends on ALL gaps contributing, so suppressing weak gaps disproportionately harms the cumulative risk estimate.

**Why GBM's simpler design wins on small data:**

With only 2,240 training rows, XGBoost's sophisticated regularization (which requires cross-validation to tune properly) over-regularizes by default. GBM's simpler approach -- just constrain tree depth and require minimum leaf size -- is more robust to small datasets where the regularization hyperparameters can't be tuned precisely.

**The practical lesson:** XGBoost's design optimizes for Kaggle competitions with 100K+ rows where fine-grained regularization prevents overfitting. GBM's design optimizes for classical ML applications with moderate data. The "better" algorithm depends entirely on the data scale and signal structure.

**Q58. You weighted LR 2x in the Voting ensemble. Why not equal weights? What is the mathematical consequence of overweighting the best model?**

In soft voting, the ensemble prediction is:
```
P_ensemble = (w1 * P_LR + w2 * P_RF + w3 * P_GBM) / (w1 + w2 + w3)
```

With weights [2, 1, 1], LR contributes 50% of the final probability, RF 25%, GBM 25%.

**Why 2x weight for LR:**

The validation results clearly showed LR is the strongest individual model. Equal weights would give RF (39.81% capture) equal influence to LR (49.51% capture) -- diluting the strongest signal with two substantially weaker ones. The 2x weight acknowledges that LR's predictions are more reliable while still incorporating the diversity of RF and GBM.

**Mathematical consequence:**

The weighted average is a convex combination biased toward LR. The ensemble's capture@25 will be between LR's (49.51%) and the equal-weight ensemble's (which would be the average of 49.51%, 39.81%, 48.54% ≈ 46%). With [2,1,1] weights, it landed at 48.54% -- closer to LR, with modest gains from RF/GBM diversity.

If I'd used [100, 0, 0] weights, the ensemble would be identical to LR. If [0, 1, 0], identical to RF. The weight vector controls the bias-variance trade-off between trusting the best model and diversifying across model families.

**Why not learn weights from data?** I could have used a meta-learner (Stacking, exp8) to learn optimal weights. But as we saw, Stacking optimized for global log-loss and hurt capture@25. Manual weights based on validation performance are simpler and more directly optimize the business metric.

**Q59. Stacking had the BEST ROC-AUC (0.718) and Brier (0.136) but only ranked 6th on capture@25 (46.60%). Explain this paradox. Why would you NOT deploy the model with the best ROC-AUC?**

This is the most instructive paradox in the project. It demonstrates that optimizing for standard ML metrics can produce a model that's worse for the specific business problem.

**What Stacking does:** A meta-learner (LR) is trained on the outputs of three base models (LR, RF, GBM). The meta-learner learns to trust each base model differently across the probability spectrum. For example, it might learn: "When LR and RF both predict >0.6, I'll predict 0.65. When LR predicts 0.3 but RF predicts 0.5, I'll predict 0.4." The meta-learner optimizes log-loss GLOBALLY -- across the entire probability range.

**Why this improves ROC-AUC and Brier:** The meta-learner produces beautifully calibrated probabilities everywhere. It reduces variance in the mid-range (0.3-0.6) where individual models disagree. This pulls down the Brier score and pushes up the ROC-AUC -- both metrics that aggregate across ALL thresholds.

**Why this hurts capture@25:** The smoothing effect of the meta-learner compresses the extreme tail. Claims that LR alone would predict at 0.75 and 0.68 become 0.72 and 0.67 under stacking. The ordering might actually change (a claim at 0.67 from one model and 0.72 from another get averaged to 0.695). This compression and reordering means the top 25% are less clearly separated, reducing capture@25 by 2.91 pp.

**Why NOT deploy Stacking:**

Capture@25 is the business metric. Every 1 pp loss in capture = ~2 fewer denials caught per 500 claims. The 2.91 pp gap between Stacking and LR means ~14 additional missed denials per 500 current claims. The better ROC-AUC and Brier are theoretically satisfying but operationally irrelevant if they don't translate to the constrained optimization problem.

**Q60. MLP with (64,32) architecture achieved only 43.69%. Is this a fair comparison given only 2,240 training samples? How many parameters does this network have? What is the samples-per-parameter ratio?**

**It's an inherently unfair comparison.** Neural networks need data to learn hierarchical representations. With 2,240 training samples, the MLP is starved for data.

**Parameter count:**
```
Input layer: 53 features -> 64 neurons:   53 * 64 + 64  = 3,456
Hidden layer: 64 neurons -> 32 neurons:    64 * 32 + 32  = 2,080
Output layer: 32 neurons -> 2 neurons:     32 * 2  + 2   = 66
Total parameters: 3,456 + 2,080 + 66 = 5,602
```

**Samples-per-parameter ratio:** 2,240 / 5,602 ≈ **0.40**

This is catastrophe territory. The conventional guideline is 10-100 samples per parameter. At 0.40 samples per parameter, the network has more degrees of freedom than training examples. Every parameter could be tuned to exactly match individual training points -- perfect overfitting.

**What prevents complete collapse:**
- **Early stopping** (patience=10): Training stops when validation loss plateaus, preventing memorization of the training set
- **L2 regularization** (alpha=0.001): Penalizes large weights
- **Adaptive learning rate** (adam): Smooths training, reducing overfitting
- **ReLU activation:** Induces sparsity (many neurons output 0), effectively reducing the parameter count

**Would it be fair with 100K samples?** Absolutely. At 80,000 training samples, the ratio would be 80,000/5,602 ≈ 14.3 -- firmly in the viable range. The MLP might then beat LR by learning subtle non-linear interactions that the current data can't support.

This is why the experiment wasn't a waste: it confirmed that the data scale doesn't support deep learning. This is valuable information for production planning.

**Q61. Why did SVC with RBF kernel underperform LR? What property of the data makes RBF kernels unnecessary?**

The RBF (Radial Basis Function) kernel maps data to an infinite-dimensional feature space where the relationship between data points is determined by their Euclidean distance: K(x, x') = exp(-gamma * ||x - x'||^2). This allows SVM to learn highly non-linear decision boundaries.

**Why it's unnecessary here:** The data's true decision boundary is approximately linear (sum of administrative gaps > threshold). An RBF kernel creates a complex, wiggly, high-dimensional boundary to fit what is essentially a straight line. It's like using a CNC machine to cut a piece of paper -- technically capable but wildly inappropriate.

**Specific mechanisms of underperformance:**

1. **Curse of dimensionality in an infinite space:** The RBF kernel projects 2,240 points into an infinite-dimensional space. With 53 features, most points are distant from each other. The kernel matrix is effectively sparse (most entries near zero), making the margin optimization unstable.

2. **Calibration difficulty:** SVM was wrapped in `CalibratedClassifierCV(cv=3)` to produce probabilities. The calibration has to map SVM's complex decision function to well-behaved probabilities. With 2,240 samples and an infinite-dimensional feature space, the calibration fit is noisy.

3. **Overfitting to noise:** The RBF kernel can create small "islands" of decision boundary around individual training points. With only 2,240 samples, this flexibility translates to memorization of noise patterns.

4. **Feature scale sensitivity:** RBF is based on Euclidean distance, making it sensitive to feature scaling. While `StandardScaler` addresses this, the binary features (0/1) and continuous features are inherently on different "types" of scale even after normalization.

**When RBF would excel:** If denial risk depended on specific combinations of feature values in a non-linear, localized way (e.g., "claims with `total_billed` between $5K-$7K and `days_to_submit` between 20-25 are high risk, but outside that range are low risk"), the RBF kernel's locality would be perfect.

**Q62. RF got 39.81% with 200 trees and depth=12. If you increased trees to 1,000 and depth to 20, would it catch up to LR? Why or why not?**

**No, it would not catch up to LR.** Here's why:

**More trees (200 → 1,000):** Adding trees reduces variance but doesn't change the fundamental inductive bias. After ~100-200 trees, Random Forest's performance plateaus. Going from 200 to 1,000 would reduce prediction variance by at most 0.1-0.3% -- negligible for capture@25 which is already a coarse metric. The 9.7 pp gap to LR is a bias problem (wrong model class), not a variance problem (insufficient trees).

**Deeper trees (max_depth=12 → 20):** This would be COUNTERPRODUCTIVE. Random Forest already overfits with depth=12 on 2,240 samples. At depth=20, each tree can create 2^20 ≈ 1 million leaves -- wildly more granularity than 2,240 data points can support. The trees would create leaves with 0-2 samples each, purely memorizing training points. The ensemble would average these overfit trees and produce even noisier, more overfit predictions. Validation capture would likely DROP, not rise.

**The fundamental problem:** Random Forest's axis-aligned splits can't efficiently represent the additive sum-of-gaps structure. No amount of trees or depth changes this. Each tree sees the problem as a nested series of if-then rules. To capture "total_admin_gaps = 3 is high risk," a tree needs enough depth to split on all three individual gaps. But to also capture "total_admin_gaps = 2 is medium-high risk," it needs separate branches for every combination of 2 gaps. The combinatorial explosion is inherent to the tree structure.

**What WOULD help RF:** Adding `total_admin_gaps` as a feature. If the RF could split directly on the sum (e.g., "total_admin_gaps >= 3"), it would need only one split to capture the most important signal. But at that point, you're engineering features to make RF behave like LR -- which defeats the purpose of using a non-linear model.

---

