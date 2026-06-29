# ML ALGORITHMS CHEAT SHEET — Interview-Ready Deep Dive
> Critical gap from interviews: You say "I used XGBoost" but can't explain WHY convincingly. This file fixes that.

---

## THE GOLDEN RULE

> **Always start with the simplest model that could work. Escalate to complexity only when you can PROVE baseline fails on the key business metric.**

Interviewer feedback (direct quote): *"Simple solutions exist. You don't need a GBM. You don't even need ML in some of those problems."*

---

## SECTION 1: LOGISTIC REGRESSION — THE BASELINE KING

### How It Works
```
P(y=1 | X) = sigmoid(w·X + b) = 1 / (1 + exp(-w·X - b))
```
- Output is a **probability** (0 to 1) — unlike linear regression which can go negative
- Decision boundary: predict fraud if P(y=1|X) > threshold (default 0.5, tune this!)
- Log-odds: log(P(fraud)/P(not-fraud)) = w·X + b = **linear combination of features**
- Each coefficient w_i = change in log-odds per unit increase in feature X_i

### Regularization Options

**L1 (Lasso) — Feature Selection:**
- Adds |w| penalty to loss: Loss + alpha * sum(|w_i|)
- Effect: drives some weights to **exactly zero** → automatic feature selection
- Use when: you have many irrelevant features, want a sparse model
- In sklearn: `LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha)`

**L2 (Ridge) — Multicollinearity:**
- Adds w^2 penalty to loss: Loss + alpha * sum(w_i^2)
- Effect: shrinks all weights toward zero, **none go to exactly zero**
- Use when: features are correlated (multicollinearity), want to keep all features
- In sklearn: `LogisticRegression(penalty='l2', C=1/alpha)` (default)

**Elastic Net — Both:**
- Combines L1 and L2: alpha1 * sum(|w_i|) + alpha2 * sum(w_i^2)
- Use when: many correlated features AND some are irrelevant
- In sklearn: `LogisticRegression(penalty='elasticnet', l1_ratio=0.5)`

**Key Hyperparameter: C (inverse of regularization)**
- C = 1/lambda. Higher C = less regularization (more complex, risk overfit)
- C = 0.01 = strong regularization (penalizes complex weights hard)
- C = 100 = weak regularization (allows complex weights)
- Default C=1 is a reasonable starting point

### Interview One-Liner
> "Logistic regression is my default baseline. Its coefficients are interpretable as log-odds changes, it's fast to train and serve (<1ms inference), handles multicollinearity with Ridge penalty, and for many structured fraud problems matches complex models within 1-2% AUC-PR. I escalate to XGBoost only when LR misses non-linear interactions."

---

## SECTION 2: DECISION TREES — INTERNAL MECHANICS

### Split Selection Algorithm
```
For each feature f:
  For each unique value v in sorted(feature f):
    Split: left = samples where X[f] <= v, right = samples where X[f] > v
    Compute: Gain = Gini(parent) - (|left|/|parent|)*Gini(left) - (|right|/|parent|)*Gini(right)
Choose: (f, v) with maximum Gain
```

### Gini Impurity — Worked Example
Node with 100 claims, 10 fraud:
```
Gini(parent) = 1 - (10/100)^2 - (90/100)^2 = 1 - 0.01 - 0.81 = 0.18
```

Split on claim_amount > $50K: Left=60 claims (8 fraud), Right=40 claims (2 fraud)
```
Gini(left) = 1 - (8/60)^2 - (52/60)^2 = 1 - 0.0178 - 0.7511 = 0.231
Gini(right) = 1 - (2/40)^2 - (38/40)^2 = 1 - 0.0025 - 0.9025 = 0.095
Weighted Gini = (60/100)*0.231 + (40/100)*0.095 = 0.1386 + 0.038 = 0.177
Gain = 0.18 - 0.177 = 0.003
```

Tree picks the split (feature, threshold) that MAXIMIZES this Gain.

### Key Hyperparameters
| Parameter | Controls | Default | Overfitting? |
|-----------|----------|---------|--------------|
| max_depth | Max tree depth | None | Higher = overfit |
| min_samples_leaf | Min samples per leaf | 1 | Lower = overfit |
| min_samples_split | Min samples to split | 2 | Lower = overfit |
| max_features | Features per split | 'auto' | More = overfit |
| ccp_alpha | Complexity pruning | 0 | Higher = simpler tree |

---

## SECTION 3: RANDOM FOREST — BAGGING ENSEMBLE

### How Bagging Works
1. Sample N training examples **with replacement** (bootstrap) → ~63% unique, ~37% duplicates
2. Fit a decision tree on this bootstrap sample
3. At each split: consider only **sqrt(p) randomly selected features** (reduces tree correlation)
4. Repeat for n_estimators trees
5. Predict: majority vote (classification) or mean (regression) across all trees

**Why this works:**
- Individual trees have HIGH variance (sensitive to training data)
- Averaging N uncorrelated trees: Variance(average) = Variance(tree) / N
- Trees are decorrelated by random feature subsets (otherwise all trees look similar)

### Out-of-Bag (OOB) Error
- Each tree was trained on ~63% of data
- The remaining 37% = "out-of-bag" samples for that tree
- Average OOB prediction across all trees = built-in cross-validation
- Eliminates need for separate validation set (though still use one for final evaluation)

### Feature Importance in Random Forest
**Mean Decrease in Impurity (MDI):**
- Sum of Gini reduction from all splits on feature X across all trees
- Fast to compute, but biased toward high-cardinality features

**Permutation Importance (recommended):**
- Shuffle feature X → measure how much validation performance drops
- More reliable, works with any model, not biased by feature cardinality

**SHAP Values:**
- Game-theory-based attribution: how much does each feature push prediction up or down?
- Most accurate, most computationally expensive
- Use for model explanability in regulated domains (insurance, banking)

### When to Choose Random Forest
- Smaller datasets (< 500K rows) where XGBoost overfits
- Need parallel training (each tree independent → fast on multi-core)
- Quick feature importance needed (MDI fast)
- Less hyperparameter tuning needed (more robust to settings)
- Need built-in OOB validation

### When NOT to Choose Random Forest
- Need best accuracy → XGBoost/LightGBM usually better
- Very large datasets → too slow
- Need extremely fast inference → single trees faster

---

## SECTION 4: XGBOOST — DETAILED INTERNALS

### How Gradient Boosting Works (Step by Step)
```
1. Initialize: F_0(x) = argmin_gamma sum(L(y_i, gamma))  [usually mean of target]
2. For m = 1 to M trees:
   a. Compute pseudo-residuals: r_im = -[dL(y_i, F(x_i))/dF(x_i)]  [negative gradient]
   b. Fit decision tree h_m to residuals
   c. Find step size: gamma_m = argmin_gamma sum(L(y_i, F_m-1(x_i) + gamma*h_m(x_i)))
   d. Update: F_m(x) = F_m-1(x) + learning_rate * gamma_m * h_m(x)
3. Final model: F_M(x) = sum of all trees
```

### XGBoost Specific Enhancements

**Regularized Objective:**
```
Obj = Sum(Loss(y_i, y_hat_i)) + Sum over trees of (gamma*T + 0.5*lambda*sum(w_j^2))
```
Where T = number of leaves, w_j = leaf weight, gamma = leaf complexity penalty, lambda = L2 on leaves

This prevents overfitting at the leaf level — standard GBM doesn't have this.

**Newton's Method (2nd order optimization):**
- Standard GBM: uses only gradient (1st derivative)
- XGBoost: uses gradient AND hessian (2nd derivative) → faster convergence
- Result: fewer trees needed for same accuracy vs standard GBM

**Level-wise Tree Growth:**
- Grows tree level by level (all nodes at depth d before depth d+1)
- More conservative, better for small/medium datasets
- Less risk of overfitting than leaf-wise (LightGBM)

### CRITICAL HYPERPARAMETERS — MEMORIZE FOR INTERVIEWS

| Parameter | What It Does | Typical Range | Interview Note |
|-----------|-------------|---------------|----------------|
| n_estimators | Number of trees | 100-2000 | Use with early_stopping_rounds |
| learning_rate (eta) | Shrinkage per tree | 0.01-0.3 | Lower rate → more trees needed |
| max_depth | Tree depth | 3-8 | Start with 6, tune down for overfit |
| subsample | Row sampling ratio | 0.6-0.9 | Adds randomness, reduces overfit |
| colsample_bytree | Column sampling per tree | 0.6-0.9 | Reduces feature dominance |
| min_child_weight | Min sum of hessians in leaf | 1-10 | Regularization, higher = simpler |
| gamma | Min loss reduction to split | 0-5 | Higher = more conservative splits |
| reg_alpha | L1 on leaf weights | 0-1 | Feature selection |
| reg_lambda | L2 on leaf weights | 0-5 | Shrinkage, default=1 |
| scale_pos_weight | Imbalance correction | neg/pos ratio | Set to ~49 for 2% fraud rate |
| early_stopping_rounds | Stop if no improvement | 10-50 | Prevents overfit, saves time |

**Interview answer for XGBoost hyperparameter tuning:**
> "I start with n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=49 for 2% fraud. I use early stopping on validation PR-AUC (stops at 30 rounds no improvement). Then I tune max_depth (3-8) and min_child_weight (1-10) for regularization. I use Optuna for Bayesian optimization, not grid search — it's 10x faster."

### Crisp WHY XGBOOST Answer for Interviews
> "I chose XGBoost over LightGBM because our dataset was ~300K rows — medium sized, where XGBoost's level-wise growth gives better-calibrated probability outputs. I chose XGBoost over logistic regression because EDA showed strong non-linear interactions between claim_amount, claim_tenure, and geographic_risk_score that LR couldn't capture (validated by 9-point PR-AUC lift). scale_pos_weight handles the 2% fraud rate without needing SMOTE."

### When NOT to Use XGBoost
- Very small datasets (<1000 rows): logistic regression or RF preferable
- Strictly linear/additive data: GLM is faster, more interpretable
- Need ultra-low latency (<5ms): ONNX-converted LightGBM or even a simple rule system
- Feature types are mostly categorical + need GPU training: LightGBM is faster

---

## SECTION 5: LIGHTGBM — KEY DIFFERENCES FROM XGBOOST

### Leaf-wise vs Level-wise Tree Growth — THE KEY DIFFERENCE

**XGBoost (Level-wise):**
```
Level 0:    [root]
Level 1:  [left]  [right]
Level 2: [LL] [LR]  [RL] [RR]  ← All 4 split before going deeper
```

**LightGBM (Leaf-wise):**
```
                [root]
              /       \
           [left]   [right]
          /    \
       [LL]   [LR]        ← Picks leaf with max loss reduction, splits it
      /    \
   [LLL] [LLR]            ← Goes deeper into promising subtree
```

**Consequences:**
- LightGBM grows deeper, more expressive trees
- Typically higher accuracy on same number of trees
- Risk of overfit on small datasets (use min_data_in_leaf to control)
- MUCH faster for large datasets (fewer total nodes)

### Algorithm Innovations

**GOSS (Gradient-based One-Side Sampling):**
- Problem: standard GBM must look at all training samples to find best split
- GOSS: keep all samples with large gradients (hard examples) + random 10% of small-gradient samples
- Effect: 80% speed improvement with <1% accuracy loss on large datasets

**EFB (Exclusive Feature Bundling):**
- Many real-world features are sparse and mutually exclusive (e.g., one-hot encoded)
- EFB bundles these into single dense features
- Reduces number of features from 1000 to 100-200 → faster training

### Critical LightGBM Hyperparameters

| Parameter | What It Does | Note vs XGBoost |
|-----------|-------------|-----------------|
| num_leaves | Max leaves per tree | Use instead of max_depth |
| min_data_in_leaf | Min samples per leaf | Critical for overfit control |
| max_depth | Tree depth (-1=unlimited) | Set -1 and control via num_leaves |
| feature_fraction | Column sampling | Same as colsample_bytree |
| bagging_fraction | Row sampling | Same as subsample |
| bagging_freq | How often to bag | 0=no bagging, 5=every 5th iteration |
| learning_rate | Shrinkage | Same as XGBoost |
| n_estimators | Number of trees | Same as XGBoost |
| class_weight | Handle imbalance | Use instead of scale_pos_weight |
| is_unbalance | Auto-balance | Alternative to class_weight |
| num_class | Number of classes | For multi-class |

**num_leaves is the KEY parameter:** 
- Controls model complexity more directly than max_depth
- Typical: 31 (default), 63-127 for complex problems
- 2^max_depth as upper bound: depth=6 → num_leaves <= 64

### When LightGBM over XGBoost

| Situation | Choose LightGBM |
|-----------|-----------------|
| Dataset > 500K rows | Yes — 10x faster training |
| Many categorical features | Yes — native categorical support |
| Need real-time <100ms inference | Yes — faster prediction |
| Training speed is bottleneck | Yes |
| Small dataset (<50K rows) | No — XGBoost safer (less overfit risk) |
| Need probability calibration | No — XGBoost level-wise gives better calibration |

**Interview answer for LightGBM:**
> "For real-time fraud scoring with sub-100ms latency requirement, I chose LightGBM. Its leaf-wise growth and histogram algorithm make inference about 3x faster than XGBoost. The ONNX-converted LightGBM model fits in 50MB RAM and scores in <15ms. For batch overnight models where calibration quality matters more, I used XGBoost."

---

## SECTION 6: SURVIVAL MODELS — FROM YOUR RESUME

### When Survival Analysis, Not Classification

Standard classification: "Will this customer churn? Yes/No"
Survival analysis: "WHEN will this customer churn, given they haven't yet?"

**Key difference: RIGHT CENSORING**
- Customer signed up 6 months ago, hasn't churned yet
- We don't know WHEN they'll churn — we just know it hasn't happened yet
- Standard classification treats this as "not churned" (ignores timing)
- Survival analysis: this customer is "censored" at 6 months

### Kaplan-Meier Estimator

**Formula:**
```
S(t) = P(T > t) = product over all event times t_i <= t of (1 - d_i/n_i)
```
- S(t) = probability of surviving past time t
- d_i = number who churned at exactly time t_i
- n_i = number still active (at risk) just before time t_i
- Censored observations contribute to n_i until their censoring time, then drop out

**Worked Example:**
- t=1 month: 1000 customers active, 50 churn → S(1) = 1 - 50/1000 = 0.95
- t=2 months: 940 active (10 censored at t=1), 30 churn → S(2) = 0.95 * (1 - 30/940) = 0.95 * 0.968 = 0.92
- S(t) = step-function that drops at each event time

**Interview answer for Kaplan-Meier:**
> "For customer churn in healthcare, I used Kaplan-Meier to estimate the probability a patient stays engaged after N weeks. The key advantage over simple classification is handling censored patients — those who are still active at study end. The KM curve tells us: at month 6, 75% of patients are still active. This guides retention intervention timing."

### Cox Proportional Hazards Model

**Formula:**
```
h(t|x) = h_0(t) * exp(beta_1*x_1 + beta_2*x_2 + ... + beta_p*x_p)
```
- h(t|x) = hazard rate at time t given features x = instantaneous risk of event
- h_0(t) = baseline hazard (unspecified — semi-parametric model)
- exp(beta_i) = HAZARD RATIO for feature x_i

**Hazard Ratio Interpretation:**
- exp(beta) = 1.5 for "fraud_history" → customers with fraud history have 1.5x higher risk of churn
- exp(beta) = 0.7 for "email_engagement" → engaged customers have 30% lower churn hazard
- HR > 1 = increases hazard (risk factor), HR < 1 = decreases hazard (protective factor)

**Interview answer for Cox:**
> "Cox PH gave me hazard ratios for each feature — exp(beta) tells me the multiplier effect on churn risk. For example, tenure under 3 months had HR=3.2, meaning new customers churn 3.2x faster than baseline. This drove our targeted retention program for new customers."

---

## SECTION 7: MARKETING MIX MODELS — FROM AXTRIA RESUME

### What MMM Actually Is

**NOT classification. NOT clustering. It's TIME SERIES REGRESSION.**

```
Sales(t) = beta_0 + beta_1*TV_spend(t) + beta_2*Digital_spend(t) + 
           beta_3*Print_spend(t) + beta_4*Seasonality(t) + 
           beta_5*Competitor_promo(t) + error(t)
```

**Output of the model (what interviewers kept asking about):**
- Predicted sales per time period (continuous, in dollars)
- Channel contribution = beta_i * spend_i = incremental sales from that channel
- ROI = channel_contribution / channel_spend

**Why XGBoost for MMM (your resume claim):**
- Linear regression assumes: doubling TV spend doubles sales (false — diminishing returns!)
- Non-linear saturation: XGBoost captures S-curve and diminishing returns naturally
- Interaction effects: TV amplifies digital ROI (synergy)
- Feature importance ≈ channel contribution proxy (use SHAP for exact attribution)

**Adstock/Carryover Effect:**
```
Adstock(t) = Spend(t) + decay_rate * Adstock(t-1)
```
- Marketing effects persist and decay over time
- TV campaign this week affects sales for 4+ weeks
- decay_rate = 0.3-0.8 (higher = longer carry-over)
- Must include as lagged features in XGBoost

**Interview answer (crisp):**
> "MMM takes weekly channel spend as input and outputs predicted sales. In linear MMM, coefficients ARE the channel contributions. In XGBoost MMM, SHAP values give channel attribution accounting for non-linear saturation and interactions. We validate with holdout periods, measuring MAPE < 10% and out-of-sample R-squared > 0.85. The optimization layer takes model predictions and uses constraint optimization to find budget allocation that maximizes predicted sales."

### Output to interviewer's question "what does XGBoost give you?"
> "A single number: predicted sales in dollars for that time period, given the input spend levels. We run this for multiple scenarios — what if we shift 20% budget from TV to digital? The model predicts the resulting sales change. Channel contributions are extracted via SHAP values."

---

## SECTION 8: COMPLEXITY ANALYSIS — THE GAP FROM INTERVIEWS

### Why O(N^2) Doesn't Scale (Interviewer's Point)

Brute-force pairwise similarity for 100,000 nodes:
```
Comparisons = N*(N-1)/2 = 100,000 * 99,999 / 2 ≈ 5 billion
At 1 microsecond per comparison = 5,000,000 seconds = 1,389 hours
```
This is WHY the interviewer said "your approach won't scale."

### Algorithm Complexity Reference Table

| Operation | Algorithm | Time Complexity | Space |
|-----------|-----------|-----------------|-------|
| Pairwise similarity (all pairs) | Brute force | O(N^2) | O(N^2) |
| Nearest neighbor search | FAISS (HNSW) | O(N log N) build, O(log N) query | O(N) |
| Exact pairwise matching | Block + Group-By | O(N) with SQL index | O(N) |
| Clustering | DBSCAN | O(N^2) worst, O(N log N) with index | O(N) |
| Clustering | KMeans | O(N*k*iterations) | O(N*k) |
| Decision tree training | CART | O(N*p*log N) | O(N) |
| Random Forest | RF | O(trees * N * sqrt(p) * log N) | O(trees) |
| Graph BFS/DFS | Graph search | O(V + E) | O(V) |
| Community detection | Louvain | O(N log N) | O(N + E) |

### The Right Answer for Fraud Ring Detection at 100K Nodes

**Interviewer's expected answer (simple, scalable):**
```sql
-- Step 1: Block by shared attributes (O(N) with index)
SELECT phone, email_domain, address_hash, COUNT(*) as group_size, 
       array_agg(claim_id) as claims
FROM claims
GROUP BY phone, email_domain, address_hash
HAVING COUNT(*) >= 2  -- Flag groups with 2+ members
```

**That's it. This is O(N) and finds 90% of fraud rings.**

Only then, for the remaining suspicious groups, do we add graph analysis.

**Interview answer:**
> "I start with SQL blocking — GROUP BY on high-signal attributes like phone, address, and email domain. This runs in O(N) with indexes and identifies most fraud rings in seconds. For sophisticated rings that obfuscate direct connections, I then build a graph on the flagged groups only (much smaller N) and apply community detection. This two-stage approach is both scalable and accurate."

---

## SECTION 9: MODEL SELECTION DECISION MATRIX

| Problem | Data Size | Interpretability | Latency | First Choice | When to Upgrade |
|---------|-----------|-----------------|---------|--------------|-----------------|
| Binary fraud (tabular) | Any | High needed | Not critical | Logistic Regression | PR-AUC gap > 3% |
| Binary fraud (tabular) | < 1M rows | Medium | Batch OK | XGBoost | - |
| Binary fraud (tabular) | > 1M rows | Medium | <100ms | LightGBM | - |
| Fraud + text features | Any | Low OK | Batch | XGBoost + BERT | - |
| Time-to-event (churn) | Any | High needed | N/A | Kaplan-Meier + Cox PH | - |
| Revenue forecasting | Time series | Medium | Batch | Linear + adstock | XGBoost for non-linear |
| Fraud rings | Any | High needed | Batch | SQL GROUP BY | Neo4j only for complex |
| Real-time scoring | Any | Low OK | <100ms | LightGBM + ONNX | - |
| Anomaly detection | Any | Low OK | Depends | Isolation Forest | AutoEncoder |
| NLP feature extraction | Text | Low OK | Batch | BERT fine-tuned | GPT-4o + RAG |

---

## SECTION 10: QUICK RECALL CARD

**XGBoost in one sentence:**
> Sequentially builds trees to correct previous errors, with L1/L2 regularization on leaf weights and second-order optimization for fast convergence.

**LightGBM in one sentence:**
> Like XGBoost but grows trees leaf-wise (not level-wise), uses histograms and GOSS sampling for 10x speed on large datasets.

**Random Forest in one sentence:**
> Trains many independent trees on bootstrap samples with random feature subsets, averages predictions to reduce variance.

**Why I chose [X] over [Y] — TEMPLATE:**
> "I chose [X] because [specific data property/business constraint]. I ruled out [Y] because [specific reason it fails here]. The evidence was [metric difference that justified the choice]."

**Scale_pos_weight calculation:**
> scale_pos_weight = (number of negative examples) / (number of positive examples)
> For 1% fraud: 99/1 = 99. For 2% fraud: 49. For 5% fraud: 19.

**LightGBM num_leaves vs max_depth:**
> max_depth=6 allows at most 2^6=64 leaves. Set num_leaves <= 64 as a rule of thumb.
> Start with num_leaves=31 (default), increase to 63 or 127 for complex problems.

---

## SECTION 11: DEEPER MODEL SELECTION — THE LEAD-LEVEL RATIONALE

### The Baseline-First Rule

Every model selection story must start with a baseline. The baseline proves that complexity is earned, not assumed.

**Baseline hierarchy:**
1. Business rules and heuristics
2. Logistic regression or linear regression
3. Decision tree or single interpretable model
4. Random Forest or Gradient Boosting
5. Deep learning or specialized architectures

**How to tell the story in an interview:**
> "I started with logistic regression as a baseline because it is fast, interpretable, and gives calibrated probabilities. It achieved 0.65 PR-AUC. The business target was 0.78. EDA showed non-linear interactions between claim amount, policy tenure, and claim frequency that logistic regression could not capture. XGBoost closed the gap to 0.81, so the added complexity was justified."

### The Three-Part Justification

For any model choice, answer three things:

**1. Why this model fits the data structure:**
- Tree-based models handle non-linear interactions and mixed feature types
- Linear models work when relationships are additive and interpretability matters
- Deep learning needs large data and raw unstructured signals

**2. Why this model fits the business constraint:**
- Latency requirements may force LightGBM over XGBoost
- Regulatory requirements may force logistic regression over black-box models
- Interpretability requirements may force SHAP plus a simpler model

**3. What evidence justifies it over alternatives:**
- A metric lift on a held-out validation set
- A business KPI improvement
- A constraint that only this model satisfies

### Model Selection by Data Characteristics

**Small dataset (< 10K rows):**
- Prefer logistic regression, decision trees, or small random forests
- Avoid deep learning and complex ensembles
- Use regularization heavily to prevent overfitting

**Medium dataset (10K to 1M rows), tabular:**
- XGBoost and LightGBM are usually strongest
- Random Forest is a safe, parallelizable alternative
- Start with logistic regression to establish baseline

**Large dataset (> 1M rows), tabular:**
- LightGBM for speed
- Distributed XGBoost if cluster resources allow
- Sampling and feature selection become important

**Text-heavy data:**
- BERT or domain-specific transformers for feature extraction
- RAG when knowledge changes frequently
- Fine-tuning when style, format, or stable domain knowledge matters

**Time-to-event data:**
- Kaplan-Meier for descriptive survival curves
- Cox PH when you need hazard ratios and interpretability
- Gradient-boosted survival models when you need non-linear interactions

### Defending Complexity in Interviews

**XGBoost over logistic regression:**
> "Logistic regression achieved 0.65 PR-AUC versus a 0.78 business target. EDA showed threshold effects and interactions — for example, claim amount only becomes suspicious when policy tenure is under 60 days. XGBoost captured these and reached 0.81."

**LightGBM over XGBoost for real-time:**
> "XGBoost inference was 90ms, which left no margin for feature extraction and network latency within our 200ms SLA. LightGBM inference was 15ms after ONNX conversion, giving us headroom."

**Random Forest over XGBoost:**
> "For a 50K-row healthcare readmission dataset, Random Forest achieved 0.78 AUROC while XGBoost overfit to 0.82 train / 0.74 validation. The Random Forest model was more robust and required less tuning."

**Deep learning over gradient boosting:**
> "We had 2M+ claims with free-text notes. A fine-tuned BERT for feature extraction gave +4 PR-AUC points over bag-of-words plus XGBoost because it captured semantic patterns like 'whiplash' versus 'soft tissue injury'."

### When NOT to Use Each Algorithm

**Logistic regression:**
- When relationships are strongly non-linear and interactions dominate
- When you need to capture complex feature interactions without manual engineering
- When the dataset is large enough that a more flexible model generalizes better

**Decision tree:**
- As a final production model for high-stakes decisions because it overfits easily
- When stability and smoothness of predictions matter

**Random Forest:**
- When you need the absolute best accuracy and have time to tune XGBoost
- When inference speed is critical and a single tree or linear model suffices
- When you need probability calibration without post-processing

**XGBoost:**
- On very small datasets where it will overfit
- When inference latency is below 10ms and model size matters
- When the data is strictly linear and GLM gives comparable metrics

**LightGBM:**
- On small datasets where leaf-wise growth overfits
- When probability calibration quality is more important than speed
- When categorical features have extremely high cardinality without proper handling

**Deep learning:**
- On small tabular datasets where gradient boosting dominates
- When interpretability is required and SHAP is not acceptable
- When training data is limited and you cannot afford validation data

---

## SECTION 12: LIGHTGBM INTERNALS — BEYOND LEAF-WISE GROWTH

### Histogram-Based Decision Tree Learning

LightGBM bins continuous features into a fixed number of bins, typically 256.

**Why this speeds up training:**
- Instead of evaluating every unique value as a split point, LightGBM evaluates bin boundaries
- Memory usage is lower because feature values are stored as integers instead of floats
- Communication cost is reduced in distributed training

**Tradeoff:**
- Histogram binning can slightly reduce split precision because the exact optimal split point may fall inside a bin
- In practice the accuracy loss is minimal and the speed gain is large

### Gradient-Based One-Side Sampling (GOSS)

GOSS keeps all data instances with large gradients and randomly samples a small fraction of instances with small gradients.

**Why it works:**
- Instances with large gradients are the hard examples that need more correction
- Instances with small gradients are already well-predicted and contribute little to split improvement
- By keeping the hard examples and sampling the easy ones, LightGBM speeds up training without sacrificing much accuracy

### Exclusive Feature Bundling (EFB)

Many high-dimensional datasets have sparse, mutually exclusive features such as one-hot encoded categories.

**How EFB helps:**
- Bundles mutually exclusive features into a single dense feature
- Reduces the effective number of features from thousands to hundreds
- Speeds up histogram construction

### Native Categorical Feature Handling

LightGBM can handle categorical features directly without one-hot encoding.

**How it works:**
- For a categorical feature with k levels, LightGBM finds the optimal split by trying subsets of categories
- It uses a smart algorithm to avoid evaluating all 2^(k-1) possible splits
- This is more efficient than one-hot encoding for high-cardinality categories

**Best practices:**
- Use for categorical features with moderate cardinality (dozens to a few hundred levels)
- For very high cardinality, target encoding or embeddings may still be better
- Ensure missing or rare categories are handled consistently between training and serving

**Interview one-liner:**
> "LightGBM is faster than XGBoost because of three things: histogram binning reduces split search cost, GOSS focuses computation on hard examples, and EFB bundles sparse categorical features. Its native categorical support also avoids expensive one-hot encoding."

---

## SECTION 13: SURVIVAL ANALYSIS — BUSINESS INTERPRETATION DEPTH

### When to Use Survival Analysis Instead of Classification

Use survival analysis when the question is "when will the event happen?" rather than "will the event happen?"

**Examples:**
- Customer churn: "When will this customer churn?" not just "Will they churn?"
- Patient readmission: "When is the patient likely to be readmitted?"
- Loan default: "When will the borrower default?"
- Fraud detection: "When in the claim lifecycle will fraud signals emerge?"

**Why classification is wrong here:**
- A customer who joined yesterday and has not churned is not the same as a ten-year customer who has not churned
- Classification treats both as "not churned" and ignores the time dimension
- Survival analysis uses censored observations correctly

### Kaplan-Meier Business Story

The KM curve tells stakeholders the probability of survival past any time point.

**How to present it:**
> "The KM curve shows that 90% of new customers remain active at month 3, but only 60% remain at month 12. This means the highest-risk window for churn is between month 3 and month 9, and retention interventions should be concentrated there."

### Cox PH Business Story

Cox PH produces hazard ratios that are directly interpretable as risk multipliers.

**How to present hazard ratios:**
> "A hazard ratio of 2.5 for 'no email engagement' means customers who never open emails are 2.5 times more likely to churn at any given time, holding other factors constant. This justifies investing in email re-engagement campaigns."

**Proportional hazards assumption:**
- The effect of each feature must be constant over time
- If the effect changes over time, consider time-varying coefficients or a different model
- Check with Schoenfeld residuals

### Competing Risks

In healthcare and insurance, multiple events can prevent the event of interest.

**Example:**
- Studying time to fraud confirmation, but some claims are closed without investigation
- Studying time to readmission, but patient may die before readmission

**When to mention:**
> "If competing events are common, standard survival analysis may overestimate the risk of the target event. I would use competing-risks models or report cumulative incidence functions instead of naive survival curves."

---

## SECTION 14: MARKETING MIX MODELING — DEFENDING ADVANCED ATTRIBUTION

### Why MMM Is Time-Series Regression

MMM predicts a continuous outcome — usually sales in dollars — from marketing spend and other drivers over time.

**Inputs:**
- Weekly spend per channel
- Seasonality indices
- Economic indicators
- Competitor activity

**Outputs:**
- Predicted sales per week
- Channel contribution
- Return on investment per channel
- Optimal budget allocation

### Adstock and Saturation Are Mandatory

**Adstock:** Models the carryover effect of marketing. A TV ad this week may influence sales for several weeks.
> "Adstock(t) = Spend(t) + decay * Adstock(t-1). The decay parameter, often 0.3 to 0.8, controls how long the effect lasts."

**Saturation:** Models diminishing returns. Doubling spend does not double sales indefinitely.
> "Saturation(t) = 1 - exp(-k * Spend(t)). As spend grows, incremental sales increase but approach a ceiling."

### Attribution Alternatives and When to Use Each

**Last-touch attribution:**
- Simple, easy to implement
- Credits the final touchpoint before conversion
- Problem: undervalues awareness and upper-funnel channels

**First-touch attribution:**
- Credits the first touchpoint
- Problem: undervalues conversion-focused channels

**Linear attribution:**
- Gives equal credit to all touchpoints
- Problem: ignores that some touchpoints matter more than others

**Markov chain attribution:**
- Models transitions between touchpoints
- Computes removal effects: what happens to conversion probability if a channel is removed?
- Best when: sequence matters, you need removal-effect insights, data is sequential

**Data-driven attribution (Google Analytics 360):**
- Black box but easy to deploy
- Best when: you need an off-the-shelf solution and removal effects are not critical

**Shapley value attribution:**
- Game-theory based fair allocation
- More rigorous but computationally expensive
- Best when: you need theoretically sound allocation and can handle complexity

**How to defend Markov chain use:**
> "I don't default to Markov chains. I use them only when the business question is 'what happens to conversion probability if we remove a channel?' Last-touch and equal-credit cannot answer that. Markov chains can, because they model transition probabilities and compute removal effects."

### Holdout Validation for MMM

Because you cannot easily A/B test TV spend, use holdout time periods for validation.
> "I hold out the most recent 3 months of data. The model is trained on earlier data and evaluated on the holdout period using MAPE and out-of-sample R-squared. If MAPE is below 10% and R-squared is above 0.85, the model is credible for budget optimization."

---

## SECTION 15: ALGORITHM INTERVIEW SCENARIOS

### "Why did you choose XGBoost over LightGBM?"

> "Both are gradient boosting frameworks. I chose XGBoost because our dataset was around 300K rows and calibration quality was more important than raw speed for batch scoring. XGBoost's level-wise growth gave better-calibrated probabilities, which we verified with Brier score and reliability diagrams. LightGBM would be my choice for real-time inference or datasets above 1M rows."

### "Why not use deep learning for your fraud model?"

> "Our fraud signals were mostly tabular — claim amounts, frequencies, provider history. Gradient boosting handles tabular interactions well with less data and faster training. Deep learning would only make sense if unstructured text or images were the primary signal. We used BERT for text feature extraction, but the final classifier remained XGBoost."

### "When would you use Random Forest instead of XGBoost?"

> "I use Random Forest on smaller datasets where XGBoost might overfit, when I need built-in out-of-bag validation, when parallel training matters, or when I want robust feature importance with minimal tuning. For example, a 40K-row healthcare readmission model where Random Forest gave better validation performance than XGBoost because XGBoost memorized training noise."

### "Explain gradient boosting in simple terms."

> "Gradient boosting builds trees one at a time. The first tree makes predictions. The second tree is trained to predict the mistakes of the first tree. The third tree predicts the mistakes of the combined first two trees. Each new tree corrects the residual errors. A learning rate shrinks each tree's contribution so the model does not overfit."

### "Why does LightGBM use leaf-wise growth?"

> "Leaf-wise growth lets the tree focus on the leaf that gives the biggest error reduction, rather than growing all leaves level by level. This creates deeper, more expressive trees and usually higher accuracy on large datasets. The risk is overfitting on small data, which we control with min_data_in_leaf and num_leaves."

---

## SECTION 16: MARKETING MIX MODELING — SCALABILITY AND OPERATIONAL LIMITS

### When MMM Becomes Hard to Scale

MMM is powerful, but it has practical ceilings:

- **Granularity**: weekly national data works well; daily SKU-store data strains the regression assumptions and requires many parameters.
- **Heterogeneity**: a single model across very different markets can hide local dynamics. Regional hierarchical models or separate models may be needed.
- **Feature engineering load**: adstock, saturation, seasonality, and holiday transforms must be maintained across channels.
- **Retraining cadence**: spend and response patterns change; MMM is not a set-and-forget model.
- **Validation difficulty**: you cannot A/B test TV or outdoor spend easily, so holdout time periods and causal reasoning matter.

### Scaling Strategies

- **Modular pipelines**: separate data prep, transformation, modeling, and allocation optimization so each piece can evolve.
- **Hierarchical models**: national-level MMM plus market-level adjustment layers.
- **Bayesian MMM**: encodes prior beliefs and produces uncertainty intervals for budget decisions.
- **Automated diagnostics**: MAPE, R-squared, residual checks, and parameter stability reports after every retrain.

### Interview One-Liner

> "MMM works best at weekly national granularity. When scaling to many regions or SKUs, I use hierarchical or Bayesian structures, modular pipelines, and automated holdout diagnostics. The real bottleneck is validation, because you cannot A/B test offline media."

---

## SECTION 17: PROD.TXT CONNECTION â€” ALGORITHMS TO AGENTIC PLATFORM DECISIONS

The algorithm selection principles in this file also apply to the Axtria GenAI platform:

| Algorithm Principle | GenAI Platform Decision |
|---|---|
| Start with simple baseline (logistic regression) | Start with deterministic workflow; add agency only with evidence. |
| Model selection matrix | Model routing: GPT-4o-mini for classification/extraction, GPT-4o for reasoning/reflection. |
| XGBoost vs LightGBM tradeoff | Plan-and-execute vs ReAct tradeoff: predictability/auditability vs flexibility. |
| Histogram binning / GOSS efficiency | Token budget enforcement and per-step cost caps in LangGraph state. |
| Probability calibration | Faithfulness scoring and citation requirements ground LLM outputs in evidence. |
| Ensemble hierarchy | Orchestrator-workers pattern: specialist agents assembled by a coordinator. |
| Complexity must earn its place | 6 AI surfaces + multi-tenancy + streaming justify LangGraph, hybrid RAG, Redis, and RLS. |

### Interview One-Liner

> "The same algorithm-selection discipline I use for XGBoost vs LightGBM â€” start simple, justify complexity with a metric failure, and monitor production behavior â€” is how I design agentic AI platforms: deterministic workflow first, agency only where it solves a specific problem, and always with observability and cost guardrails."
