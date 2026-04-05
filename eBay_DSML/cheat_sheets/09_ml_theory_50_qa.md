# eBay DS/ML — ML Theory Q&A with Detailed Answers

> 50 questions with interview-quality answers.
> For each: the concept, why eBay cares, and the trade-off.

---

## 🌳 Section 1: Supervised Learning (15 Questions)

### Q1: Explain the bias-variance tradeoff.
**Answer:** Bias = error from simplifying assumptions (underfitting). Variance = sensitivity to training data (overfitting). Total error = bias² + variance + irreducible noise. Goal: find the sweet spot.

**eBay context:** A simple linear model for search ranking has high bias (misses complex user-item interactions). A deep neural net with few data has high variance (memorizes training clicks). Solution: start with gradient boosted trees (good bias-variance balance for tabular data).

### Q2: L1 (Lasso) vs L2 (Ridge) regularization?
**Answer:**
- L1 adds |w| penalty → drives weights to exactly zero → feature selection
- L2 adds w² penalty → shrinks weights toward zero → keeps all features
- Elastic Net combines both

**When to use:** L1 when you suspect many irrelevant features (listing attributes). L2 when all features contribute but you want to prevent overfitting.

### Q3: How does a Random Forest work?
**Answer:** Ensemble of decision trees, each trained on a bootstrap sample (bagging) with a random subset of features at each split. Final prediction = majority vote (classification) or average (regression).

**Why it works:** Each tree is a high-variance, low-bias learner. Averaging many decorrelated trees reduces variance without increasing bias.

### Q4: Random Forest vs XGBoost/LightGBM?
| Aspect | Random Forest | XGBoost/LightGBM |
|---|---|---|
| Training | Parallel (independent trees) | Sequential (each tree fixes errors of previous) |
| Bias | Higher | Lower |
| Overfitting risk | Lower | Higher (needs regularization) |
| Feature importance | Permutation-based | Gain/split-based |
| Speed | Good | LightGBM is fastest |
| Best for | Quick baseline, robust | Maximum accuracy, competitions, production ranking |

### Q5: How does gradient boosting work? Walk through 3 iterations.
**Answer:**
```
Iteration 0: Predict mean → F₀(x) = ȳ = 10
  Residuals: y - F₀(x) = [2, -3, 5, -1]

Iteration 1: Fit tree h₁ to residuals → h₁(x) ≈ residuals
  F₁(x) = F₀(x) + η·h₁(x)  [η = learning rate, e.g., 0.1]
  New residuals = y - F₁(x)

Iteration 2: Fit tree h₂ to new residuals
  F₂(x) = F₁(x) + η·h₂(x)
  ...continue until convergence or max iterations
```
Each tree corrects the mistakes of the ensemble so far.

### Q6: How do you handle class imbalance (e.g., 0.1% fraud)?
**Techniques:**
1. **Class weights:** Increase misclassification cost for minority class
2. **SMOTE:** Synthetic oversampling of minority class (create interpolated samples)
3. **Undersampling:** Reduce majority class (risk: lose information)
4. **Anomaly detection:** Treat minority as anomalies (isolation forest, autoencoders)
5. **Threshold tuning:** Train normally, adjust decision threshold on validation set
6. **Focal loss:** Down-weight easy examples, focus on hard ones

**Recommendation for eBay fraud:** Class weights + threshold tuning + ensemble. Avoid SMOTE for very high imbalance (0.1%).

### Q7: AUC-ROC vs AUC-PR — when to use which?
- **AUC-ROC:** Good for balanced classes. Can be misleadingly high for imbalanced data.
- **AUC-PR:** Better for imbalanced data. Focuses on positive class performance.

**Rule:** If positive class is rare (<5%), use AUC-PR. For eBay fraud (0.1% positive), AUC-PR is essential.

### Q8: Precision vs Recall — which matters more?
**It depends on the cost of errors:**
- **High precision needed:** Fraud blocking (false positives = blocking legitimate users = lost revenue)
- **High recall needed:** Counterfeit detection (false negatives = counterfeit items reach buyers = trust damage)
**F1:** Harmonic mean when both matter equally.
**Fβ:** F2 (weights recall higher) or F0.5 (weights precision higher).

### Q9: What is cross-validation? When is k-fold NOT appropriate?
**k-fold:** Split data into k folds, train on k-1, validate on 1, rotate. Gives robust estimate of model performance.

**When NOT to use standard k-fold:**
- **Time series:** Use time-based split (train on past, test on future) — TimeSeriesSplit
- **Grouped data:** Use GroupKFold (e.g., all listings from one seller in same fold)
- **Very small data:** Use Leave-One-Out CV

### Q10: What is feature importance? Which method is more reliable?
- **Impurity-based (Gini/Entropy):** Fast but biased toward high-cardinality features
- **Permutation importance:** Shuffle one feature, measure accuracy drop. More reliable but slower.
- **SHAP values:** Most accurate. Provides both global and local explanations.

### Q11: What is a decision tree's splitting criterion?
- **Classification:** Gini impurity or Information Gain (entropy reduction)
- **Regression:** Variance reduction (MSE)
- Gini: `1 - Σ(pᵢ²)` — fast to compute
- Entropy: `-Σ(pᵢ log₂ pᵢ)` — theoretically sound

### Q12: Explain overfitting. How do you detect and prevent it?
**Detection:** Training accuracy >> validation accuracy (gap grows with complexity)

**Prevention:**
1. More training data
2. Regularization (L1/L2, dropout, early stopping)
3. Simpler model (fewer parameters, shallower trees)
4. Cross-validation for hyperparameter tuning
5. Feature selection (remove noise features)
6. Ensemble methods (averaging reduces variance)

### Q13: What is a Support Vector Machine? When would you use it?
**SVM:** Finds hyperplane that maximizes margin between classes. Kernel trick for non-linear boundaries.

**When to use:** Small-medium datasets, high-dimensional spaces (text classification with TF-IDF). NOT for eBay-scale data (doesn't scale well to millions of rows).

### Q14: Explain logistic regression. What are its assumptions?
**Model:** P(y=1|x) = σ(w·x + b) where σ = sigmoid function

**Assumptions:**
1. Linear relationship between features and log-odds
2. No perfect multicollinearity
3. Independent observations
4. Large sample size

**Why it's useful:** Interpretable (coefficients → odds ratios), fast, good baseline, works as building block in larger systems.

### Q15: What is multicollinearity and why does it matter?
**Definition:** Features are highly correlated with each other.

**Impact:** Inflates coefficient variance → unstable/unreliable feature importance. Doesn't affect prediction accuracy for tree-based models but kills interpretability.

**Detection:** VIF (Variance Inflation Factor) > 5-10, correlation matrix heatmap

**Solutions:** Drop one of correlated features, PCA, regularization (L1/L2)

---

## 🔮 Section 2: Unsupervised Learning (8 Questions)

### Q16: K-Means vs DBSCAN vs Hierarchical Clustering?
| Aspect | K-Means | DBSCAN | Hierarchical |
|---|---|---|---|
| Clusters needed upfront? | Yes (K) | No | No |
| Cluster shape | Spherical | Arbitrary | Arbitrary |
| Handles outliers? | Poorly | Well (labels as noise) | Medium |
| Scalability | O(n·K·iter) fast | O(n²) or O(n log n) | O(n²) or O(n³) |
| Best for | Large data, round clusters | Spatial data, noise | Small data, hierarchies |

### Q17: How do you choose K in K-Means?
- **Elbow method:** Plot inertia vs K, find the "elbow"
- **Silhouette score:** Ranges [-1, 1], higher = better separation
- **Domain knowledge:** Sometimes K is defined by business (3 buyer segments)
- **Gap statistic:** Compare to random baseline

### Q18: What is PCA? When do you use it?
**PCA:** Projects data onto directions of maximum variance.

**Use cases:**
- Dimensionality reduction before modeling (reduce noise)
- Visualization (reduce to 2-3D)
- Decorrelate features (removes multicollinearity)

**Caution:** Loses interpretability. Don't use if feature interpretability matters (explain to stakeholders which features drive predictions).

### Q19: Collaborative filtering — user-based vs item-based?
- **User-based:** Find users similar to you → recommend what they liked. Sparse, scales poorly.
- **Item-based:** Find items similar to what you bought → recommend those. More stable, scales better.
- **Matrix Factorization (ALS/SVD):** Factor user-item matrix into latent factors. Better than both for large-scale.
- **Modern:** Two-tower neural networks with learned embeddings (eBay's approach).

### Q20: What is the cold-start problem?
**Problem:** Can't recommend for new users (no history) or new items (no interactions).

**Solutions:**
- New users: popularity-based recs → content-based (use demographics) → CF as data accumulates
- New items: content-based similarity to existing items → promote in explore/experiment slots
- Multi-armed bandit for exploration vs exploitation

### Q21-23: Association Rules, Anomaly Detection, t-SNE
*Prepare brief answers for: Apriori algorithm, Isolation Forest, t-SNE vs UMAP for visualization.*

---

## ⚙️ Section 3: Deep Learning (10 Questions)

### Q24: How does backpropagation work?
**Answer:** Chain rule applied recursively through the network:
1. Forward pass: compute output and loss
2. Backward pass: compute gradient of loss w.r.t each weight using chain rule
3. Update: weights -= learning_rate × gradients

### Q25: Vanishing/Exploding gradient problem?
- **Vanishing:** Deep networks with sigmoid/tanh → gradients shrink toward zero → early layers don't learn
- **Exploding:** Gradients grow exponentially → unstable training
- **Solutions:** ReLU activation, batch normalization, residual connections, gradient clipping, proper initialization (He/Xavier)

### Q26: SGD vs Adam optimizer?
| | SGD | Adam |
|---|---|---|
| Learning rate | Fixed (needs careful tuning) | Adaptive per-parameter |
| Convergence | Slower, can generalize better | Faster, sometimes overfits |
| Memory | O(1) per param | O(2) per param (stores m,v) |
| Best for | Large-scale (vision) | Default choice, NLP, smaller datasets |

### Q27: Dropout — what is it and why does it work?
Randomly zero out neurons with probability p during training. Forces network to learn redundant representations. Acts as ensemble of sub-networks. At test time, scale outputs by (1-p).

### Q28: Batch Normalization — what and why?
Normalize activations within each mini-batch to zero mean, unit variance. Then scale/shift with learnable parameters. Benefits: faster training, higher learning rates, reduces sensitivity to initialization.

### Q29: CNNs — key components and why they work for images?
- Convolution: local patterns (edges, textures) via learnable filters
- Pooling: spatial downsampling, translation invariance
- Parameter sharing: same filter applied everywhere → much fewer params than fully-connected
- eBay use: product image classification, visual search, condition grading

### Q30: RNNs/LSTMs — why replaced by Transformers?
- RNNs: sequential processing → slow, vanishing gradients on long sequences
- LSTMs: gating mechanism solves vanishing gradient → still sequential
- Transformers: parallel attention → faster training, captures long-range dependencies better

### Q31-33: Autoencoders, GANs, Transfer Learning
*Prepare brief answers for: VAE vs AE, GAN applications in data augmentation, when to fine-tune vs feature-extract.*

---

## 🏭 Section 4: Production ML (12 Questions)

### Q34: What is train-serve skew?
**Problem:** Features computed differently during training vs serving → model performs worse in production.

**Causes:** Different code paths, stale data, different libraries, time-zone bugs

**Solution:** Use a Feature Store (shared feature definitions for train + serve).

### Q35: Online vs Offline evaluation?
- **Offline:** AUC, NDCG on held-out test set. Fast, cheap. But doesn't capture real user behavior.
- **Online:** A/B test with real users. Ground truth. But expensive and slow.
- **Gap:** Model can have high offline AUC but fail online (distribution shift, latency issues, feedback loops).

### Q36: What is model drift?
- **Data drift:** Input distribution changes (e.g., new product categories, COVID buying patterns)
- **Concept drift:** Relationship between features and target changes (e.g., what "relevant" means shifts)
- **Detection:** PSI (Population Stability Index), KL-divergence on feature/prediction distributions
- **Action:** Retrain on recent data, set up automated monitoring + alerts

### Q37: Feature Store — what and why?
A centralized store for feature definitions and values, shared between training and serving pipelines.

**Benefits:**
1. Prevents train-serve skew
2. Feature reuse across teams/models
3. Point-in-time correctness (avoid future data leakage)
4. Faster feature engineering
**eBay uses NuKV** for storing precomputed embeddings for real-time serving.

### Q38: Model deployment strategies?
- **Shadow mode:** New model runs alongside old, compare outputs, don't serve to users
- **Canary release:** Serve to 1-5% of traffic first
- **Blue-green:** Instant switchover with rollback capability
- **A/B test:** Split traffic, measure metrics, decide

### Q39: Monitoring checklist for production models?
```
1. Prediction distribution (PSI for drift)
2. Feature distribution (per-feature drift)
3. Latency (p50, p95, p99)
4. Error rate (null predictions, timeouts)
5. Business metrics (conversion, CTR — lagging indicator)
6. Data freshness (how stale is the feature store?)
7. Model staleness (when was the model last retrained?)
```

### Q40: Batch vs Real-time inference?
| | Batch | Real-time |
|---|---|---|
| Latency | Hours | Milliseconds |
| Freshness | Stale | Fresh |
| Cost | Lower | Higher |
| Use case | Email recommendations, reports | Search ranking, fraud |
| Complexity | Simple (Spark/Airflow) | Complex (feature store + model server) |

### Q41-45: MLOps topics
- **Q41:** What is a model registry? (Version control for models)
- **Q42:** What is A/B testing for models? (Statistical comparison of model versions)
- **Q43:** How do you handle model retraining? (Scheduled vs triggered by drift)
- **Q44:** What is data versioning? (DVC, delta tables — track training data changes)
- **Q45:** How do you handle model explainability? (SHAP, LIME, feature importance reports)

### Q46-50: System Design Quick Questions
- **Q46:** *"Model latency is 500ms but budget is 200ms. How do you optimize?"*
  → Quantization, distillation, feature pruning, caching, batching
- **Q47:** *"Training takes 3 days. How do you speed it up?"*
  → Distributed training, mixed precision, data sampling, feature selection
- **Q48:** *"You have 0.95 AUC offline but poor production results. Why?"*
  → Train-serve skew, data leakage, distribution shift, feature staleness, latency timeout
- **Q49:** *"How do you handle feedback loops in recommendation systems?"*
  → Exploration (epsilon-greedy, UCB), exposure-weighted metrics, debiasing
- **Q50:** *"What is the difference between model accuracy and model calibration?"*
  → Accuracy = % correct. Calibration = predicted probabilities match actual frequencies.
  → Use calibration plots and Brier score. Important for bid optimization and fraud scoring.
