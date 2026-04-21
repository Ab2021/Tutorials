# 📐 Round 2: Depth in Mathematical Modeling (DMM) — Master Guide
### Flipkart Senior Data Scientist | Math & Theory Round

> **Round Format:** 60 min | Typically with a Principal DS or Research Director
> **What They Test:** Mathematical foundations, derivations, probabilistic reasoning, algorithm internals, "WHY" behind every technique
> **Your Winning Strategy:** Start with intuition → build to math → acknowledge limitations → connect to production reality

---

## 🎯 WHAT THIS ROUND IS REALLY ABOUT

DMM probes whether you **understand** ML at the math level, not just apply it. Expect:
- "Derive the gradient of cross-entropy loss from first principles"
- "Why does L2 regularization create a Gaussian prior on weights?"
- "What's the math behind attention? Now explain why multi-head helps."
- "How does XGBoost differ from AdaBoost mathematically?"
- "When does gradient descent diverge and how do you fix it?"

The interviewer (PhD background) will push past surface answers. Prepare to go 4 levels deep.

---

## 🗺️ THE 8 MATHEMATICAL DOMAINS TO MASTER

```
Domain 1: PROBABILITY THEORY
├── Bayes' theorem, conditional probability
├── Distributions: Gaussian, Bernoulli, Poisson, Beta, Dirichlet
├── Expectation, variance, covariance, central limit theorem
└── MLE vs. MAP estimation

Domain 2: LINEAR ALGEBRA
├── Matrix operations, SVD, PCA, eigendecomposition
├── The role of matrix rank, conditioning
├── Gram matrix and kernel methods
└── Vector norms (L1, L2, Frobenius)

Domain 3: OPTIMIZATION
├── Gradient descent, SGD, Adam, RMSProp
├── Convexity, saddle points, local vs. global minima
├── Learning rate schedules, momentum
└── Lagrangian, KKT conditions (for SVMs)

Domain 4: CLASSICAL ML ALGORITHMS (Mathematical Deep Dive)
├── Logistic Regression: MLE → cross-entropy loss → gradient
├── Decision Trees: information gain, gini impurity
├── Random Forests: bias-variance + bagging math
├── Gradient Boosting: Taylor expansion, XGBoost objective
└── SVM: max-margin, kernel trick, dual formulation

Domain 5: DEEP LEARNING MATH
├── Backpropagation: chain rule derivation
├── Vanishing/exploding gradients: why and how to fix
├── Batch Normalization: why it stabilizes training
├── Dropout: connection to ensemble methods
└── Convolutional math: receptive field, parameter sharing

Domain 6: TRANSFORMER & ATTENTION MATH
├── Scaled dot-product attention derivation
├── Multi-head attention: why multiple heads?
├── Positional encoding: sinusoidal math
├── BERT vs GPT pre-training objectives
└── KV cache and inference optimization math

Domain 7: STATISTICAL INFERENCE
├── Hypothesis testing: t-test, z-test, chi-square
├── p-values, Type I/II errors, power
├── Confidence intervals vs. credible intervals
├── A/B test design: sample size, MDE
└── Multiple comparison corrections (Bonferroni, FDR)

Domain 8: INFORMATION THEORY
├── Entropy, cross-entropy, KL divergence
├── Mutual information for feature selection
├── Information gain in decision trees
└── Connection: cross-entropy loss = KL divergence minimization
```

---

## 🔥 TOP 40 DMM QUESTIONS WITH DERIVATIONS

### BLOCK A: Loss Functions & Optimization

**Q1: Derive logistic regression loss from first principles (MLE approach).**

Model: $P(y=1|x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}$

Likelihood for one sample:
$$P(y|x; w) = \hat{y}^y (1-\hat{y})^{1-y}$$

Log-likelihood over N samples:
$$\ell(w) = \sum_{i=1}^N [y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

Minimizing **negative** log-likelihood = Binary Cross-Entropy Loss:
$$\mathcal{L}(w) = -\frac{1}{N}\sum_{i=1}^N [y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

Gradient w.r.t. w:
$$\frac{\partial\mathcal{L}}{\partial w} = \frac{1}{N}X^T(\hat{y} - y)$$

**Key insight:** The gradient is simply (prediction - label), not a complex expression — this is why cross-entropy is preferred over MSE for classification.

---

**Q2: Why does MSE perform poorly for classification (sigmoid output)?**

With MSE loss: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$

Gradient: $\frac{\partial\mathcal{L}}{\partial w} = (\hat{y}-y) \cdot \sigma'(z) \cdot x$

Problem: $\sigma'(z) = \sigma(z)(1-\sigma(z))$ → **near 0 when $\sigma(z)$ is near 0 or 1** → vanishing gradient → slow learning when prediction is very wrong (exactly when we need fast learning).

Cross-entropy gradient: $\frac{\partial\mathcal{L}_{CE}}{\partial w} = (\hat{y}-y) \cdot x$ → no saturation issue.

---

**Q3: Derive XGBoost's objective function and explain the Taylor expansion trick.**

At step $m$, XGBoost minimizes:
$$\text{Obj}^{(m)} = \sum_i [l(y_i, \hat{y}_i^{(m-1)} + f_m(x_i))] + \Omega(f_m)$$

2nd-order Taylor expansion around $\hat{y}_i^{(m-1)}$:
$$\text{Obj}^{(m)} \approx \sum_i [l(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(x_i) + \frac{1}{2}h_i f_m(x_i)^2] + \Omega(f_m)$$

Where: $g_i = \frac{\partial l}{\partial \hat{y}^{(m-1)}}$ (1st derivative, gradient), $h_i = \frac{\partial^2 l}{\partial (\hat{y}^{(m-1)})^2}$ (2nd derivative, hessian)

**Why 2nd order?** Captures curvature of the loss — leads to better step sizes than vanilla GBM (which only uses 1st order). Also allows **closed-form optimal leaf scores**:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Regularization:** $\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_j w_j^2$ — penalizes number of leaves ($T$) and leaf weight magnitudes.

---

**Q4: Why does L2 regularization correspond to a Gaussian prior on weights?**

MAP estimation with Gaussian prior $w \sim \mathcal{N}(0, \sigma^2)$:
$$\log P(w|D) \propto \log P(D|w) + \log P(w)$$
$$= \ell(\text{log-likelihood}) - \frac{1}{2\sigma^2}\|w\|^2$$

Maximizing MAP = minimizing negative log-likelihood + $\frac{1}{2\sigma^2}\|w\|^2$

This **is** L2 regularization with $\lambda = \frac{1}{2\sigma^2}$.

L1 regularization = Laplace prior (encourages sparsity because the Laplace has a sharp peak at 0).

---

**Q5: Prove that the optimal leaf score in a simple regression tree is the mean of its leaf samples.**

For MSE loss: $l(y_i, \hat{y}) = (y_i - \hat{y})^2$

For leaf $j$ with samples $I_j$: minimize $\sum_{i \in I_j}(y_i - \hat{y}_j)^2$

Taking derivative w.r.t. $\hat{y}_j$ and setting to 0:
$$\frac{\partial}{\partial \hat{y}_j}\sum_{i \in I_j}(y_i - \hat{y}_j)^2 = -2\sum_{i \in I_j}(y_i - \hat{y}_j) = 0$$
$$\hat{y}_j^* = \frac{1}{|I_j|}\sum_{i \in I_j} y_i$$

Hence: mean of leaf samples. This is the "leaf value" in vanilla gradient boosting with MSE loss.

---

### BLOCK B: Deep Learning Math

**Q6: Derive backpropagation for a 2-layer network (chain rule application).**

Network: $\hat{y} = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)$

Loss: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$ (MSE for simplicity)

```
Forward pass:
  z1 = W1·x + b1
  a1 = σ(z1)          ← activation of hidden layer
  z2 = W2·a1 + b2
  ŷ = σ(z2)            ← final prediction

Backward pass (chain rule):
  δL/δŷ = (ŷ - y)
  δL/δz2 = (ŷ - y) · σ'(z2)          ← δL/δŷ · δŷ/δz2
  δL/δW2 = δL/δz2 · a1^T              ← outer product
  δL/δa1 = W2^T · δL/δz2              ← pass gradient back through W2
  δL/δz1 = δL/δa1 ⊙ σ'(z1)           ← element-wise multiply by activation derivative
  δL/δW1 = δL/δz1 · x^T
```

**Key insight:** Vanishing gradients occur when $\sigma'(z) \ll 1$ at many layers. ReLU avoids this: $\text{ReLU}'(z) = 1$ if $z>0$, else 0 — gradient flows through without decay.

---

**Q7: Explain Batch Normalization mathematically. Why does it help?**

For a mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta \quad (\gamma, \beta \text{ learnable})$$

**Why it helps:**
1. Reduces **internal covariate shift** — each layer receives normalized inputs regardless of upstream weight changes
2. Allows **higher learning rates** without divergence
3. Has mild **regularization effect** (noise from batch statistics)
4. **Gradient flow:** normalizing activations keeps them in the non-saturating region of sigmoid/tanh

**Issue:** Unstable for small batch sizes → use Group Normalization / Layer Normalization instead.

---

**Q8: Derive the attention mechanism. Why does scaling by √d_k matter?**

Scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Derivation intuition:**
- $QK^T$: dot product gives "relevance" between each query-key pair
- Raw dot products grow with $d_k$ (dimension) — variance of $q \cdot k = d_k\sigma^2$ if entries ~N(0,σ²)
- Large dot products → softmax saturates → near-zero gradients
- Dividing by $\sqrt{d_k}$ normalizes variance to 1 → softmax in gradient-friendly regime

**Why multi-head?**
$$\text{MultiHead}(Q,K,V) = \text{Concat}(h_1,...,h_H)W^O$$
where $h_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

Each head can **attend to different positions and relationship types** simultaneously. Different heads often specialize: one learns syntactic dependencies, another semantic dependencies.

---

**Q9: Why does BERT use masked language modeling instead of next-token prediction?**

GPT (causal): $P(w_t | w_1, ..., w_{t-1})$ — only left context
- Useful for generation tasks
- Unidirectional: position $t$ cannot see position $t+1$

BERT (masked): Randomly mask 15% of tokens, predict them using both directions:
$P(w_{masked} | w_{context\_left}, w_{context\_right})$
- **Bidirectional context** — position $t$ sees all other positions
- Better for discriminative tasks (classification, NER, QA)
- Pre-training task closely mimics fine-tuning tasks (fill-in-the-blank → classify)

**The 15% masking detail:** Of masked tokens, 80% replaced with [MASK], 10% replaced with random word, 10% kept unchanged → prevents model from simply learning to identify [MASK] positions.

---

**Q10: Derive the information gain formula for decision tree splitting.**

Entropy: $H(S) = -\sum_{c} p_c \log_2 p_c$

For binary classification (fraud/not-fraud):
$H(S) = -p_{fraud}\log_2 p_{fraud} - p_{clean}\log_2 p_{clean}$

Information gain from split on feature A:
$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

Gini Impurity (used in CART, sklearn default):
$$G(S) = 1 - \sum_c p_c^2$$

**Gini vs. Entropy:**
- Gini: faster (no log computation), slightly favors larger partitions
- Entropy: information-theoretic motivation, better for multi-class
- Empirically: similar performance in most real-world tasks

---

### BLOCK C: Statistics & Probability

**Q11: What is the bias-variance tradeoff? Derive it mathematically.**

Expected prediction error at point $x$:
$$E[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- $\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$ — systematic error
- $\text{Var}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ — model sensitivity to training set
- $\sigma^2$ — irreducible noise in data

**In fraud detection context:**
- High bias: logistic regression on non-linear fraud patterns → misses complex schemes
- High variance: deep neural net on 50K labeled samples → memorizes training fraud, misses new patterns
- Optimal: XGBoost with regularization + cross-validation tuning

---

**Q12: Explain the Central Limit Theorem and its role in A/B testing.**

CLT: For iid samples $X_1,...,X_n$ from ANY distribution with mean $\mu$ and variance $\sigma^2$:
$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \text{ as } n \to \infty$$

**In A/B testing:**
- Even if fraud rates are non-normal (heavy-tailed), sample means of treatment/control become approximately normal for n > 30
- This justifies using z-tests/t-tests for comparing fraud rates between groups
- Rule of thumb: need n > 30 per group for CLT approximation to hold

**When CLT fails:** Very heavy-tailed metrics (e.g., transaction amount with rare very-large transactions). Use: bootstrap resampling or non-parametric tests (Mann-Whitney U).

---

**Q13: Bayesian vs. Frequentist — how do they differ for fraud risk scoring?**

| | Frequentist | Bayesian |
|---|---|---|
| Parameters | Fixed, unknown | Random variables with distributions |
| Inference | p-values, CIs (long-run frequency) | Posterior distributions, credible intervals |
| Prior info | Ignored | Incorporated via prior |
| Uncertainty | Sampling variability | Epistemic uncertainty in parameters |

**Fraud risk Bayesian example (new seller):**
- Prior: $p_{fraud} \sim \text{Beta}(\alpha_0, \beta_0)$ based on population fraud rate
- Observe: k fraud orders out of n
- Posterior: $p_{fraud} | \text{data} \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$
- Credible interval: 95% probability that true fraud rate is in [lower, upper]
- **Advantage:** Handles cold-start — new seller borrows strength from population prior

---

**Q14: What is the precision-recall tradeoff? Derive the F-beta score.**

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$

F1 = harmonic mean of Precision and Recall:
$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

F-beta (weighted): $\beta^2$ is the weight of recall relative to precision:
$$F_\beta = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

- $\beta < 1$: precision more important (too many FP is costly — e.g., customer friction)
- $\beta > 1$: recall more important (missing fraud is costly — financial loss)
- $\beta = 2$: recall twice as important (typical for high-stakes fraud with large FN cost)

**In your system:** With claims averaging $8,500 each, FN cost >> FP cost → use F2 score.

---

**Q15: Explain KL Divergence and its connection to cross-entropy loss.**

KL Divergence from P to Q:
$$D_{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)} = \sum_x P(x)\log P(x) - \sum_x P(x)\log Q(x)$$
$$= -H(P) + H(P, Q)$$

Where $H(P,Q) = -\sum_x P(x)\log Q(x)$ is the cross-entropy.

Since entropy $H(P)$ is constant w.r.t. model parameters Q:
$$\min_Q D_{KL}(P\|Q) \equiv \min_Q H(P, Q)$$

**Connection to ML:** Minimizing cross-entropy loss = minimizing KL divergence between data distribution P and model distribution Q. This is the probabilistic justification for the cross-entropy loss function.

---

### BLOCK D: Advanced Topics

**Q16: How does dropout regularize a neural network? Show the connection to ensemble methods.**

At training: each neuron dropped with probability $p$ → different "sub-network" at each forward pass.
At inference: scale weights by $(1-p)$ (or use inverted dropout: scale during training by $\frac{1}{1-p}$).

**Connection to ensembles:** Training with dropout is like training $2^n$ different networks (exponential in number of neurons) that share weights. At inference, we're averaging over all these models → ensemble effect.

**Mathematical effect:** Prevents co-adaptation of neurons — if a neuron cannot rely on the presence of other specific neurons, it must learn more robust features. Equivalent to adding a regularization term that encourages weight redundancy.

---

**Q17: Explain the math behind contrastive learning (used in your embedding model training).**

Contrastive loss (SimCLR style):
$$\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where:
- $\text{sim}(u, v) = u^T v / (\|u\| \|v\|)$ (cosine similarity)
- $\tau$ is temperature hyperparameter (lower → sharper distributions, trains harder negatives harder)
- $(z_i, z_j)$ is a positive pair (same fraud pattern)
- Denominator sums over all other samples in batch (in-batch negatives)

**In your fraud embedding model:**
- Positive pairs: Two claims from same fraud scheme (labeled by investigators)
- Negatives: Random legitimate claims + hard negatives (similar-looking legitimate claims)
- Goal: fraud claims cluster together, separated from legitimate claims

**Temperature $\tau$ effect:** Low $\tau$ → model focuses more on hard negatives → better discrimination but harder to train. Typical: $\tau = 0.07$ for image models, $0.05$–$0.1$ for text.

---

**Q18: What is NDCG? Derive it. When is it better than precision@k?**

Discounted Cumulative Gain:
$$DCG@k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

Where $rel_i$ is the graded relevance of document at position $i$ (e.g., 0, 1, 2, 3).

Normalized DCG:
$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

Where IDCG = DCG of ideal ranking (perfectly sorted by relevance).

**When NDCG > Precision@k:**
- When relevance is graded (not binary) — "highly relevant" > "somewhat relevant"
- When position matters — missing a highly relevant result at position 1 is worse than at position 5
- Recommendation systems: personalized ranking with multiple quality levels

**In your pharma recommendation:** Recommend correct drug (binary) at right position (rank matters for rep's attention) → NDCG@5 is appropriate.

---

**Q19: Cox Proportional Hazards model — when and why from your EXL work?**

Model: $h(t|x) = h_0(t) \exp(\beta^Tx)$

Where:
- $h(t|x)$: hazard at time $t$ for patient with features $x$
- $h_0(t)$: baseline hazard (non-parametric — key advantage)
- $\exp(\beta^Tx)$: proportional deviation from baseline

**Partial likelihood** (Cox's innovation — avoids estimating $h_0(t)$):
$$L(\beta) = \prod_{i:\text{event}} \frac{\exp(\beta^T x_i)}{\sum_{j \in R(t_i)} \exp(\beta^T x_j)}$$

Where $R(t_i)$ = risk set (all patients still at risk at time $t_i$).

**Why Cox for patient attrition:** Handles censored data (patients who leave the study without experiencing the event). BERT + Cox hybrid: use BERT embeddings as features in the proportional hazards model.

---

**Q20: How does the kernel trick work in SVMs? What's the math?**

SVM dual formulation maximizes:
$$\sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^T x_j$$

Only depends on $x_i^T x_j$ (dot products). Replace with kernel function $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$.

**Key insight:** We never compute $\phi(x)$ explicitly — only $K(x_i, x_j)$. This allows **infinite-dimensional feature spaces** at tractable cost.

Common kernels:
- **RBF:** $K(x,z) = \exp(-\gamma\|x-z\|^2)$ — infinite feature space
- **Polynomial:** $K(x,z) = (x^Tz + c)^d$
- **Linear:** $K(x,z) = x^Tz$ (standard SVM, no feature expansion)

**Mercer's theorem:** $K$ is valid kernel iff the Gram matrix $[K(x_i,x_j)]$ is positive semi-definite.

---

### BLOCK E: Probability Puzzles & Reasoning

**Q21: You flip a biased coin. P(H)=0.7. What is the expected number of flips to get HH (two heads in a row)?**

Let $E$ = expected flips from start
Let $E_H$ = expected flips when last flip was H

$E = 1 + 0.7 E_H + 0.3 E$
$E_H = 1 + 0.7 \cdot 0 + 0.3 E$

From second equation: $E_H = 1 + 0.3E$
Substitute in first: $E = 1 + 0.7(1+0.3E) + 0.3E$
$E = 1 + 0.7 + 0.21E + 0.3E$
$E - 0.51E = 1.7 \Rightarrow 0.49E = 1.7 \Rightarrow E \approx 3.47$

---

**Q22: In a fraud detection system, P(fraud) = 0.01, P(model says fraud | fraud) = 0.90, P(model says fraud | no fraud) = 0.05. A claim is flagged. What's P(actually fraud)?**

Bayes' theorem:
$$P(F|\text{flag}) = \frac{P(\text{flag}|F) \cdot P(F)}{P(\text{flag})}$$

$P(\text{flag}) = P(\text{flag}|F)P(F) + P(\text{flag}|\neg F)P(\neg F)$
$= 0.90 \times 0.01 + 0.05 \times 0.99 = 0.009 + 0.0495 = 0.0585$

$$P(F|\text{flag}) = \frac{0.90 \times 0.01}{0.0585} = \frac{0.009}{0.0585} \approx 15.4\%$$

**Insight:** Even with 90% recall and 95% specificity, precision is only 15.4% when fraud rate is 1%. This is why PR-AUC matters more than ROC-AUC in imbalanced settings.

---

**Q23: 3 cards: one red on both sides, one blue on both sides, one red on one side blue on another. You see a red face. What's P(other side is red)?**

P(other side red | see red) = P(see red AND both sides red) / P(see red)

P(see red) = P(pick RR card) × P(see red | RR) + P(pick BR card) × P(see red | BR)
= 1/3 × 1 + 1/3 × 1/2 = 1/3 + 1/6 = 1/2

P(see red AND both sides red) = 1/3 × 1 = 1/3

P(other side red | see red) = (1/3) / (1/2) = **2/3**

---

**Q24: Your model outputs probabilities. How do you check if they're calibrated?**

**Reliability Diagram (calibration curve):**
1. Bin predictions into 10 buckets (0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. For each bucket: compute mean predicted probability and actual event rate
3. Plot: well-calibrated model lies on the diagonal (y=x)

**Brier Score:** $\text{BS} = \frac{1}{N}\sum_i(\hat{y}_i - y_i)^2$ — lower is better (0=perfect, 0.25=random)

**ECE (Expected Calibration Error):** $\sum_b \frac{|B_b|}{N}|\overline{pred}_b - \overline{actual}_b|$

**Platt Scaling:** Fit logistic regression on top of model outputs: $P(y=1|\hat{p}) = \sigma(a\hat{p} + b)$

**Isotonic Regression:** Non-parametric calibration (more flexible but needs more data).

---

**Q25: Prove that the gradient of softmax cross-entropy with respect to logits simplifies nicely.**

For multi-class: $\hat{y}_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_k e^{z_k}}$

Loss: $\mathcal{L} = -\sum_c y_c \log \hat{y}_c$ (only true class contributes: $\mathcal{L} = -\log\hat{y}_j$ for true class j)

$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

**Proof:**
$\frac{\partial\hat{y}_j}{\partial z_i} = \hat{y}_j(\delta_{ij} - \hat{y}_i)$ (softmax Jacobian)

$\frac{\partial\mathcal{L}}{\partial z_i} = -\frac{y_j}{\hat{y}_j} \cdot \hat{y}_j(\delta_{ij} - \hat{y}_i) = -y_j(\delta_{ij} - \hat{y}_i)$

For one-hot $y_j = 1$: $\frac{\partial\mathcal{L}}{\partial z_i} = -(1\cdot\delta_{ij} - \hat{y}_i) = \hat{y}_i - y_i$

**Beautiful result:** Gradient = prediction - label. Same form as MSE gradient, but without the saturation problem!

---

### BLOCK F: Questions From Your Specific Projects

**Q26: In your BERT + XGBoost model for readmission: why concatenate BERT embeddings with tabular features for XGBoost rather than fine-tuning an end-to-end BERT classifier?**

Arguments for hybrid approach:
1. **Clinical notes were short (<512 tokens):** BERT doesn't gain over simpler aggregation for very short texts
2. **Structured data dominated signal:** Comorbidity codes, LOS, vital signs had high predictive power independently — throwing them away for pure BERT would lose signal
3. **Training data size:** Only 50K labeled patients — insufficient to fine-tune BERT end-to-end for classification without catastrophic forgetting. Used ClinicalBERT for feature extraction (frozen) + XGBoost for classification
4. **Interpretability:** XGBoost provides SHAP values per feature — including the BERT embedding as a feature group — maintaining explainability required by healthcare regulations
5. **Compute:** Fine-tuning BERT end-to-end requires GPU, XGBoost training is CPU-bound and faster to iterate

**When end-to-end would be better:** Larger dataset (>500K), notes are long and complex (>512 tokens needing hierarchical attention), explainability not required.

---

**Q27: In your PySpark CLV model: what's the serialization challenge with distributed Random Forest?**

Challenge: In PySpark MLlib, models are serialized and sent to each worker — for very large forests (**200 trees × features**), serialization overhead can exceed computation time.

Solutions applied:
1. **Column pruning:** Only broadcast features actually used in the forest (drop zero-importance features)
2. **Broadcast join for small lookup tables:** Instead of shuffle join on worker → reduce network I/O
3. **Caching intermediate DataFrames:** Persist feature matrices that are reused across multiple scoring passes (`.cache()` / `.persist(StorageLevel.MEMORY_AND_DISK)`)
4. **Partition tuning:** Set `spark.sql.shuffle.partitions` = 2-3× number of cores to minimize partition skew
5. **Model prediction vectorization:** Score all records in a partition together rather than row-by-row UDF

---

**Q28: How did your contrastive learning approach for fraud embeddings handle hard negative mining?**

Standard random negatives are "easy" — any random legitimate claim is clearly different from fraud. Model stops learning once it easily separates them.

**Hard negatives:** Legitimate claims that LOOK like fraud (unusual language, timing patterns) but are actually legitimate. These are the most valuable training signals.

Hard negative mining strategy:
1. After initial training, embed all claims
2. For each fraud query, find top-20 nearest legitimate claims in embedding space
3. Use these as hard negatives for next training round
4. Repeat: **iterative hard negative mining**

This is similar to the approach in: FAISS-based ANN search → find semantic neighbors → use for contrastive training.

---

**Q29: In your Marketing Mix Model: why genetic algorithms for budget allocation?**

Marketing budget allocation is a **multi-objective optimization** problem:
- **Objective 1:** Maximize revenue
- **Objective 2:** Subject to budget constraint ($B$ total)
- **Objective 3:** Satisfy channel-specific constraints (min/max per channel)
- **Objective 4:** Minimize concentration risk (cap any one channel at 40%)

Why genetic algorithm:
1. **Non-convex:** Response curves (log, saturation effects) make the space non-convex → gradient descent finds local optima
2. **Mixed constraints:** Hard constraints (budget sum must = B) are easy to enforce via crossover/mutation operators
3. **Multi-objective:** Pareto front discovery — GA naturally produces diverse solution population

Alternative: **Bayesian optimization** (used for hyperparameter tuning — fewer evaluations needed). **Why not Bayesian for budget?** Budget space is high-dimensional (many channels); Bayesian optimization scales poorly to >20 dimensions.

---

**Q30: In the Agentic BI tool: what's the math behind token budget management for long contexts?**

Challenge: LangChain agent accumulates conversation history. At 4K tokens/step × 5 tool calls → 20K tokens → exceeds context window of most LLMs.

Strategies used:
1. **Sliding window:** Only keep last $W$ turns in context → $O(W)$ memory
2. **Summarization compression:** After every $k$ turns, summarize older history with LLM → $O(1)$ growth but adds latency and summary quality risk
3. **Retrieval-augmented memory:** Store full history in vector DB → retrieve relevant past steps based on current query → only inject top-k relevant memories
4. **Selective context:** Score each step's relevance to current query → only include high-scoring steps → greedy context packing

**Token counting:** BPE tokenization — rule of thumb: 1 token ≈ 4 characters in English. Pre-count context before each LLM call, budget allocation: `system(500) + history(2000) + tool_results(1000) + response_budget(500) = 4000`.

---

## 🔢 MATHEMATICAL FORMULAS CHEAT SHEET

| Concept | Formula |
|---|---|
| Logistic regression gradient | $\frac{1}{N}X^T(\hat{y} - y)$ |
| XGBoost leaf score | $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ |
| XGBoost gain from split | $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$ |
| Attention | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| KL Divergence | $\sum P(x)\log\frac{P(x)}{Q(x)}$ |
| Brier Score | $\frac{1}{N}\sum(\hat{y}_i - y_i)^2$ |
| F-beta | $\frac{(1+\beta^2)PR}{\beta^2 P + R}$ |
| Bayes' Theorem | $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ |
| Information Gain | $H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v)$ |
| Cox PH Hazard | $h(t|x) = h_0(t)\exp(\beta^Tx)$ |
| Contrastive Loss | $-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)}$ |
| NDCG | $\sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}$ / IDCG |

---

*See companion files: 02_ML_Algorithm_Derivations.md, 03_Deep_Learning_Math.md, 04_Transformer_Attention_Math.md, 08_Statistical_Testing_AandB.md*
