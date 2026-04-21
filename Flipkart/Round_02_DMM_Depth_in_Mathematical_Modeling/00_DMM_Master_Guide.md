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

---
---

# 🔥 BLOCK G: SURVIVAL ANALYSIS & CLINICAL ML MATH
### (From EXL Patient Readmission + Survival Analysis Projects)

---

## Q31: Derive the Kaplan-Meier estimator. Why is it the correct estimator for right-censored data?

**Setup:** We observe $n$ patients. Each has an event time $T_i$ or a censoring time $C_i$ (whichever comes first). We observe $t_i = \min(T_i, C_i)$ and the indicator $\delta_i = \mathbf{1}(T_i \leq C_i)$.

**Kaplan-Meier (Product-Limit) Estimator:**

At each observed event time $t_{(j)}$ (sorted, distinct event times), let:
- $d_j$ = number of events (deaths/readmissions) at time $t_{(j)}$
- $n_j$ = number of subjects at risk *just before* $t_{(j)}$ (haven't had event or been censored yet)

$$\hat{S}(t) = \prod_{t_{(j)} \leq t} \left(1 - \frac{d_j}{n_j}\right)$$

**Why censoring is handled correctly:**
- Censored subjects reduce $n_j$ at their censoring time — they "leave the risk set"
- They do NOT contribute a death at that time
- This is the key insight: by reducing the denominator only (not numerator), KM correctly accounts for the fact that censored patients *could* have experienced the event if followed longer
- If censoring is non-informative (independent of $T$), this gives an unbiased estimate of the true survival function

**Example (3 patients, events at t=1,3, censored at t=2):**
```
t=1: n=3, d=1  →  factor = (1 - 1/3) = 2/3
t=2: n=2, d=0, censored=1  →  no factor (no event), n drops to 1
t=3: n=1, d=1  →  factor = (1 - 1/1) = 0

S(t):
  t < 1:  1.0
  1 ≤ t < 3:  2/3 ≈ 0.667
  t ≥ 3:  2/3 × 0 = 0.0
```

**Log-rank test** (comparing two survival curves): Tests $H_0$: $S_1(t) = S_2(t)$ for all $t$.
$$\chi^2 = \frac{(O_1 - E_1)^2}{E_1} + \frac{(O_2 - E_2)^2}{E_2}$$
where $O$ = observed events, $E$ = expected events under $H_0$.

---

## Q32: Derive the Cox Proportional Hazards partial likelihood. Why doesn't it need to estimate the baseline hazard?

**Model:** $h(t | x_i) = h_0(t) \exp(\beta^T x_i)$

The "trick" in Cox (1972) is to construct a likelihood that focuses only on the *ordering* of events, not their timing:

**Partial Likelihood:**

At each event time $t_{(i)}$, given that one event occurred, the probability that it was specifically subject $i$ (given the risk set $\mathcal{R}(t_{(i)})$):

$$P(\text{subject } i \text{ fails} | \text{one failure from } \mathcal{R}(t_{(i)})) = \frac{h(t_{(i)}|x_i)}{\sum_{j \in \mathcal{R}(t_{(i)})} h(t_{(i)}|x_j)}$$

$$= \frac{h_0(t_{(i)}) \exp(\beta^T x_i)}{\sum_{j \in \mathcal{R}(t_{(i)})} h_0(t_{(i)}) \exp(\beta^T x_j)}$$

$$= \frac{\exp(\beta^T x_i)}{\sum_{j \in \mathcal{R}(t_{(i)})} \exp(\beta^T x_j)}$$

🔑 **The $h_0(t)$ cancels out!** Both numerator and denominator contain it, so it divides away.

**Full partial likelihood (over all D events):**
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(\beta^T x_i)}{\sum_{j \in \mathcal{R}(t_{(i)})} \exp(\beta^T x_j)}$$

Maximize this to get $\hat{\beta}$ — **no need to estimate $h_0(t)$**. This is why Cox is "semi-parametric":
- Non-parametric part: $h_0(t)$ — left completely unspecified
- Parametric part: $\exp(\beta^T x)$ — modeled explicitly

**Log partial likelihood gradient (score equation):**
$$\frac{\partial \log L}{\partial \beta} = \sum_{i:\delta_i=1} \left[ x_i - \frac{\sum_{j \in \mathcal{R}(t_i)} x_j \exp(\beta^T x_j)}{\sum_{j \in \mathcal{R}(t_i)} \exp(\beta^T x_j)} \right] = 0$$

This is the observed minus expected covariate for the failing subject — same structure as logistic regression!

---

## Q33: What is the Proportional Hazards (PH) assumption? How do you test and fix violations?

**The PH assumption:** The hazard ratio between two subjects is constant over time.

$$\frac{h(t | x_i)}{h(t | x_j)} = \frac{\exp(\beta^T x_i)}{\exp(\beta^T x_j)} = \text{constant in } t$$

**Testing with Schoenfeld residuals:**

For each event $i$ and covariate $k$, the Schoenfeld residual is:
$$r_{ik} = x_{ik} - \hat{E}[x_k | \text{failing at } t_i]$$

Under PH, these residuals should be *uncorrelated with time*. Plot $r_{ik}$ vs. $t_i$:
- Flat/random pattern → PH holds ✅
- Trend (upward/downward) → PH violated ❌

**Fixes when PH is violated:**

| Fix | When to Use |
|---|---|
| **Stratified Cox** | PH violated for a categorical covariate (e.g., age group). Allow each stratum its own $h_0(t)$, share $\beta$. |
| **Time-varying coefficients** | PH violated continuously. Model $\beta(t) = \beta_0 + \beta_1 \cdot g(t)$ where $g(t)$ is a smooth function of time. |
| **Accelerated Failure Time (AFT)** | Alternative parametric model: $\log T = \beta^T x + \sigma \epsilon$. Doesn't require PH at all. |

**From your EXL work:** Young patients' readmission hazard spiked in first 7 days then dropped; elderly patients' hazard monotonically increased → PH violated for age group → used **stratified Cox** by age bracket.

---

## Q34: Derive the expected value and variance of the exponential distribution. Connect to survival modeling.

**Exponential distribution:** $f(t) = \lambda e^{-\lambda t}$, $t \geq 0$, $\lambda > 0$

**Expected value:**
$$E[T] = \int_0^\infty t \lambda e^{-\lambda t} dt$$

Integration by parts ($u = t$, $dv = \lambda e^{-\lambda t} dt$):
$$= \left[-t e^{-\lambda t}\right]_0^\infty + \int_0^\infty e^{-\lambda t} dt = 0 + \left[-\frac{1}{\lambda} e^{-\lambda t}\right]_0^\infty = \frac{1}{\lambda}$$

**Variance:**
$$E[T^2] = \int_0^\infty t^2 \lambda e^{-\lambda t} dt = \frac{2}{\lambda^2}$$
$$\text{Var}(T) = E[T^2] - (E[T])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}$$

**Survival and hazard functions:**
$$S(t) = P(T > t) = e^{-\lambda t}$$
$$h(t) = \frac{f(t)}{S(t)} = \frac{\lambda e^{-\lambda t}}{e^{-\lambda t}} = \lambda \quad \text{(constant!)}$$

**Key insight:** Exponential distribution has **constant hazard** — the "memoryless" property. A patient who has survived 30 days has the same hazard as a new patient. This is often unrealistic in clinical settings (recovery usually reduces readmission risk over time) → use Weibull ($h(t) = \lambda\alpha t^{\alpha-1}$) which allows increasing or decreasing hazard.

---

---

# 🔥 BLOCK H: CAUSAL INFERENCE & A/B TESTING MATH
### (From EXL Propensity Modeling + A/B Testing + Axtria MMM Attribution)

---

## Q35: Derive the Potential Outcomes Framework (Rubin Causal Model). What is the Fundamental Problem of Causal Inference?

**Notation:**
- $Y_i(1)$ = potential outcome for unit $i$ if treated ($T_i = 1$)
- $Y_i(0)$ = potential outcome for unit $i$ if untreated ($T_i = 0$)
- We observe: $Y_i = T_i \cdot Y_i(1) + (1-T_i) \cdot Y_i(0)$

**Individual Treatment Effect (ITE):**
$$\tau_i = Y_i(1) - Y_i(0)$$

**The Fundamental Problem:** We can NEVER observe both $Y_i(1)$ and $Y_i(0)$ for the same unit simultaneously. One is always the **counterfactual** (unobserved).

**Average Treatment Effect (ATE) — what we estimate instead:**
$$\text{ATE} = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

**Why simple comparison $E[Y|T=1] - E[Y|T=0]$ is biased (without randomization):**

$$E[Y|T=1] - E[Y|T=0] = \underbrace{E[Y(1)|T=1] - E[Y(0)|T=1]}_{\text{ATT (what we want)}} + \underbrace{E[Y(0)|T=1] - E[Y(0)|T=0]}_{\text{Selection Bias}}$$

Selection bias arises because treated and untreated groups differ in their baseline outcomes. Example: sicker patients (lower $Y(0)$) are more likely to receive treatment → treated group looks worse even if treatment helps.

**Randomization eliminates selection bias:** If $T \perp (Y(0), Y(1))$, then $E[Y(0)|T=1] = E[Y(0)|T=0]$. Selection bias = 0.

---

## Q36: Derive the Inverse Probability Weighting (IPW) estimator for ATE. Why does it work?

**Setting:** Observational study. Treatment assignment not random, confounded by $X$.

**Key assumption (Unconfoundedness):** $(Y(0), Y(1)) \perp T | X$

**Propensity score:** $e(x) = P(T=1|X=x)$

**IPW estimator:**
$$\hat{\tau}_{IPW} = \frac{1}{n}\sum_{i=1}^n \left[\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)}\right]$$

**Why it works — intuition via reweighting:**

Treated units with $e(x) = P(T=1|X=x) = 0.1$ look like they "accidentally" got treated. They are very similar to the control group. Upweighting them (by $1/0.1 = 10$) makes the treated group representative of the full population.

**Formal proof of unbiasedness:**
$$E\left[\frac{T_i Y_i}{e(X_i)}\right] = E\left[\frac{T_i Y_i(1)}{e(X_i)}\right]$$
$$= E\left[E\left[\frac{T_i Y_i(1)}{e(X_i)} \bigg| X_i, Y_i(1)\right]\right]$$
$$= E\left[\frac{Y_i(1)}{e(X_i)} \cdot E[T_i | X_i]\right] = E\left[\frac{Y_i(1)}{e(X_i)} \cdot e(X_i)\right] = E[Y_i(1)]$$

Similarly for the control term. Therefore $E[\hat{\tau}_{IPW}] = E[Y(1)] - E[Y(0)] = \text{ATE}$. ✅

**Practical issue:** IPW has high variance when propensity scores are near 0 or 1 (some weights become very large). Fix: **Doubly Robust estimator** — combines IPW with outcome model. Consistent if EITHER the propensity model OR the outcome model is correctly specified (but not necessarily both).

---

## Q37: Derive the Difference-in-Differences (DiD) estimator. What is the parallel trends assumption?

**Setup:** Two groups (Treatment, Control), two time periods (Pre, Post). Treatment happens between Pre and Post.

| | Pre | Post |
|---|---|---|
| **Treatment** | $\bar{Y}_{T,pre}$ | $\bar{Y}_{T,post}$ |
| **Control** | $\bar{Y}_{C,pre}$ | $\bar{Y}_{C,post}$ |

**DiD estimator:**
$$\hat{\tau}_{DiD} = \underbrace{(\bar{Y}_{T,post} - \bar{Y}_{T,pre})}_{\text{change in treatment}} - \underbrace{(\bar{Y}_{C,post} - \bar{Y}_{C,pre})}_{\text{change in control}}$$

**Why it works:** Subtracting the control group's change removes the **time trend** (any outcome change that would have happened regardless of treatment).

**Regression formulation:**
$$Y_{it} = \alpha + \beta_1 \text{Treatment}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treatment}_i \times \text{Post}_t) + \epsilon_{it}$$

$\hat{\beta}_3$ is the DiD estimate. $\beta_3 = \hat{\tau}_{DiD}$.

**Parallel Trends Assumption:** In the absence of treatment, the treatment and control groups would have evolved in parallel:
$$E[Y_{T}(0)_{post} - Y_{T}(0)_{pre}] = E[Y_{C}(0)_{post} - Y_{C}(0)_{pre}]$$

**Testing:** Plot pre-period trends for both groups. If they're parallel before treatment, we have evidence the assumption holds. **Falsification test:** Check for "anticipation effects" 1-2 periods before treatment — there should be none.

**Flipkart application:** If Flipkart launches a new seller support feature in South India (treatment) but not North India (control), DiD estimates the causal impact on seller revenue by comparing revenue changes between regions.

---

## Q38: Explain CUPED (Controlled-Experiment Using Pre-Experiment Data) mathematically. How much variance does it reduce?

**Problem:** Standard A/B test variance is large because users are noisy. We need 2-4 weeks of data to get enough power.

**CUPED insight:** Pre-experiment behavior ($X_i$) correlates with post-experiment outcome ($Y_i$). Use $X_i$ to predict and subtract out the predictable part of $Y_i$, reducing residual variance.

**CUPED estimator:**
$$\tilde{Y}_i = Y_i - \theta(X_i - \bar{X})$$

where $\theta$ chosen to minimize variance of $\tilde{Y}_i$:
$$\theta^* = \frac{\text{Cov}(Y_i, X_i)}{\text{Var}(X_i)} = \text{Pearson correlation} \times \frac{\sigma_Y}{\sigma_X}$$

This is exactly the OLS coefficient of regressing $Y$ on $X$!

**Variance reduction:**
$$\text{Var}(\tilde{Y}) = \text{Var}(Y) - \frac{\text{Cov}(Y,X)^2}{\text{Var}(X)} = \text{Var}(Y)(1 - \rho^2)$$

**Variance reduction factor** = $1 - \rho^2$, where $\rho$ = correlation between pre/post metrics.

**Concrete example:** If $\rho = 0.7$ (typical for purchase frequency in e-commerce):
- Variance reduced by $1 - 0.49 = 51\%$
- Required sample size reduced by $51\%$
- Or you get same statistical power in **half the time** → cut A/B test duration from 4 weeks to 2 weeks

**Key property:** CUPED is unbiased. The treatment estimate $\hat{\tau}_{CUPED} = \bar{\tilde{Y}}_T - \bar{\tilde{Y}}_C = \bar{Y}_T - \bar{Y}_C$ (same as original estimator in expectation, lower variance).

---

## Q39: Derive the Markov Chain attribution model used in your Axtria MMM project.

**Setup:** Customer journey as a Markov Chain. States = marketing channels + Start + Conversion + Null (dropout).

**Transition matrix $P$:** $P_{ij}$ = probability of transitioning from channel $i$ to channel $j$.

**Absorption analysis:** Conversion and Null are absorbing states. The probability of being absorbed into Conversion = the **conversion probability**.

**Computing conversion probability:**

Partition the transition matrix:
$$P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}$$

where:
- $Q$ = transitions among transient (non-absorbing) states
- $R$ = transitions from transient states to absorbing states
- $I$ = identity (absorbing states stay absorbed)

**Fundamental matrix:** $N = (I - Q)^{-1}$

$N_{ij}$ = expected number of times the chain is in transient state $j$, given it started in state $i$.

**Absorption probability matrix:**
$$B = N \cdot R$$

$B_{i, \text{Conv}}$ = probability of eventually converting, starting from state $i$.

**Removal Effect for channel $k$:**
1. Compute $P(\text{Conv})$ using full transition matrix $P$
2. Set row $k$ of $P$ to route entirely to Null: $P_{k, \text{Null}} = 1$, $P_{k,j} = 0$ for all other $j$
3. Recompute $P(\text{Conv}_{-k})$ with channel $k$ removed
4. Removal Effect of channel $k$ = $1 - P(\text{Conv}_{-k}) / P(\text{Conv})$

**Normalization:** Attribution credit for channel $k$ = $\frac{RE_k}{\sum_j RE_j}$ (so all credits sum to 1).

**Why the homepage bias problem you faced:** Channels with near-100% pass-through probability (everyone visits homepage) have high RE because removing them routes everyone to Null. Fix: weight RE by channel's *exclusive* contribution (customers who visited ONLY that channel).

---

## Q40: Explain Thompson Sampling for Multi-Armed Bandits. Derive the regret bound.

**Problem:** $K$ arms (ad campaigns, recommendation policies). Each arm has unknown reward distribution. Explore to learn vs. exploit to earn.

**Thompson Sampling (Bayesian approach):**

For Bernoulli rewards, maintain a Beta posterior for each arm:
- Prior: $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$ (typically $\alpha_k = \beta_k = 1$, uniform)
- After observing $s_k$ successes and $f_k$ failures:
  - Posterior: $\theta_k | \text{data} \sim \text{Beta}(\alpha_k + s_k, \beta_k + f_k)$

**At each round:**
1. For each arm $k$, **sample** $\tilde{\theta}_k \sim \text{Beta}(\alpha_k + s_k, \beta_k + f_k)$
2. Pull arm $k^* = \arg\max_k \tilde{\theta}_k$
3. Observe reward, update posterior

**Why this balances exploration-exploitation:**
- Arms with uncertain estimates (few pulls) have wide Beta distributions → high sampling variance → frequently get lucky sample and get pulled → **exploration**
- Arms with many successful pulls have narrow, high-mean distributions → consistently sampled high → **exploitation**

**Regret:** $R_T = \sum_{t=1}^T \mu^* - \mu_{k_t}$ (difference from optimal arm total reward)

**Thompson Sampling achieves:** $E[R_T] = O\left(\sum_{k:\mu_k < \mu^*} \frac{\log T}{\Delta_k}\right)$

where $\Delta_k = \mu^* - \mu_k$ is the gap from optimal. This is **asymptotically optimal** (matches Lai-Robbins lower bound).

**Versus UCB (Upper Confidence Bound):**
$$UCB_k(t) = \bar{\mu}_k + \sqrt{\frac{2\ln t}{n_k}}$$

| | Thompson Sampling | UCB |
|---|---|---|
| Framework | Bayesian | Frequentist |
| Mechanism | Sample from posterior | Optimism under uncertainty |
| Prior knowledge | Can incorporate | Cannot |
| Empirical performance | Often better | Theoretically tighter |

---

---

# 🔥 BLOCK I: OPTIMIZATION, DISTRIBUTED COMPUTING & ADVANCED TOPICS
### (From MMM Genetic Algorithms + PySpark CLV + Bayesian Optimization)

---

## Q41: Explain Bayesian Optimization mathematically. Why did you use it for hyperparameter tuning instead of grid search?

**Setting:** Optimize $f(\theta)$ (e.g., validation AUC as function of hyperparameters) where each evaluation is expensive (costs minutes of training).

**Core idea:** Build a probabilistic surrogate model of $f$ using observed evaluations. Use this model to decide where to evaluate next — balancing exploration (uncertain regions) and exploitation (promising regions).

**Surrogate model: Gaussian Process (GP)**

A GP defines a distribution over functions:
$$f(\theta) \sim \mathcal{GP}(m(\theta), k(\theta, \theta'))$$

where $m(\theta)$ = mean function, $k(\theta, \theta')$ = kernel (covariance between any two points).

After observing $\mathcal{D} = \{(\theta_i, f_i)\}_{i=1}^n$, the posterior is:
$$f(\theta^*) | \mathcal{D} \sim \mathcal{N}(\mu(\theta^*), \sigma^2(\theta^*))$$

$$\mu(\theta^*) = k(\theta^*, \mathbf{\theta})[K(\mathbf{\theta}, \mathbf{\theta}) + \sigma_n^2 I]^{-1}\mathbf{f}$$
$$\sigma^2(\theta^*) = k(\theta^*, \theta^*) - k(\theta^*, \mathbf{\theta})[K + \sigma_n^2 I]^{-1}k(\mathbf{\theta}, \theta^*)$$

**Acquisition function (where to sample next):**

**Expected Improvement (EI):**
$$\text{EI}(\theta) = E[\max(f(\theta) - f^+, 0)]$$

where $f^+ = \max_i f(\theta_i)$ (current best). For a GP with posterior mean $\mu$ and std $\sigma$:

$$\text{EI}(\theta) = (\mu(\theta) - f^+)\Phi(Z) + \sigma(\theta)\phi(Z)$$

where $Z = \frac{\mu(\theta) - f^+}{\sigma(\theta)}$, $\Phi$ = CDF, $\phi$ = PDF of standard normal.

**Next point:** $\theta_{n+1} = \arg\max_\theta \text{EI}(\theta)$

**Why better than grid/random search:**
- Grid search: exponential in dimensions ($10^d$ for $d$ hyperparameters, each with 10 values)
- Random search: ignores what was learned from previous evaluations
- Bayesian Opt: each new point is chosen based on ALL previous evaluations → 15% improvement in accuracy using only 20 evaluations vs. grid search's 1000 (from your Axtria work)

---

## Q42: Explain Genetic Algorithms for multi-objective optimization. How did you use them in Marketing Mix Modeling?

**Biological analogy → Mathematical operation:**

| Biology | Math |
|---|---|
| Chromosome | Solution vector $x = [x_1, x_2, ..., x_d]$ (budget allocation) |
| Population | Set of $N$ candidate solutions |
| Fitness | Objective function value $f(x)$ (revenue) |
| Selection | Keep top-$k$ solutions by fitness |
| Crossover | Combine two parent solutions: child inherits parts from each |
| Mutation | Random perturbation: $x_j \leftarrow x_j + \mathcal{N}(0, \sigma^2)$ |

**Single-objective GA for budget allocation:**
1. Initialize: $N=100$ random budget allocations, subject to $\sum x_k = B$ (total budget)
2. Evaluate: score each allocation using MMM response curves
3. Select: keep top 50% by revenue
4. Crossover: pair parents, for each dimension sample from one parent
5. Mutate: with probability $p_m = 0.01$, perturb a random channel's allocation
6. Repeat steps 2-5 for $G=500$ generations

**Multi-objective GA (NSGA-II) for your case:**

You had competing objectives: maximize revenue AND maximize brand awareness AND satisfy budget constraints.

**Pareto front:** A solution is Pareto-optimal if no other solution is better on ALL objectives simultaneously. NSGA-II finds the entire Pareto front.

**Why GA over convex optimization?**
- MMM response curves are non-convex (saturation effects create local optima)
- Hard constraints on channel minimums/maximums are easy to enforce via mutation operators (project back to feasible region)
- Multi-objective: GA naturally discovers the Pareto front, while gradient methods find a single point

**The Corner Solution problem you solved:**
Without constraints, GA found $x_{TV} = 0$ (zero TV budget). Added a mutation operator that enforces $x_k \geq 0.5 \times x_k^{historical}$. This creates a "feasibility boundary" that prevents degenerate solutions while still allowing significant reallocation.

---

## Q43: Derive the math behind PySpark's Salting technique for data skew. What's the theoretical speedup?

**Problem (Data Skew):**

In your EXL CLV project, `groupBy("customer_id")` caused massive skew. One VIP corporate account had 10 million transactions vs. average user's 50.

**Without salting:**
- All 10M VIP transactions route to ONE reducer (by hash of customer_id)
- That reducer takes $O(10M)$ while all other reducers finish quickly
- Total time = max across reducers = $O(10M)$ for that one partition

**Salting algorithm:**

**Step 1 — Add salt:**
```python
n_salts = 10
df = df.withColumn("salt", (F.rand() * n_salts).cast("int"))
df = df.withColumn("salted_key", 
    F.concat(F.col("customer_id").cast("string"), 
             F.lit("_"), 
             F.col("salt"))
)
```

**Step 2 — First aggregation on salted key:**
```python
partial_agg = df.groupBy("salted_key", "customer_id") \
    .agg(F.sum("amount").alias("partial_sum"), 
         F.count("*").alias("partial_count"))
```

**Step 3 — Final aggregation, remove salt:**
```python
final_agg = partial_agg.groupBy("customer_id") \
    .agg(F.sum("partial_sum").alias("total_amount"),
         F.sum("partial_count").alias("total_count"))
```

**Theoretical speedup:**

Without salting: max partition size $= M$ (VIP record count), time $\propto M$

With $S$ salts: VIP's 10M transactions split across $S$ reducers $\approx M/S$ each.

New max $\approx \max\left(\frac{M}{S}, \frac{\sum_{\text{normal}}}{N_{\text{partitions}}}\right)$

In your case: $M = 10M$ transactions, $S = 10$ salts → each salt partition $= 1M$ transactions. Normal users: $\sim 50$ each across 2000 partitions = 25K avg. New bottleneck = 1M vs. old 10M → **10x speedup** on that join operation.

**Cost:** Two groupBy passes instead of one. For $n$ rows: $O(n \log n)$ → $O(n \log n)$ (same asymptotic, different constant). Practically: 10-40% additional CPU time which is well worth the I/O savings from eliminating skew.

---

## Q44: Explain the mathematics of TF-IDF and why character n-grams outperform word tokens for fuzzy entity matching.

**TF-IDF:**

**Term Frequency:** $\text{TF}(t, d) = \frac{\text{count of term }t\text{ in document }d}{\text{total terms in }d}$

(Variants: raw count, log-normalized, binary)

**Inverse Document Frequency:** $\text{IDF}(t) = \log\frac{N}{|\{d: t \in d\}|}$

(Smoothed variant: $\log\frac{N + 1}{|\{d: t \in d\}| + 1} + 1$ to avoid division by zero)

**TF-IDF:** $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$

High TF-IDF → term appears frequently in THIS document but rarely in the corpus → discriminative.

**Why character n-grams dominate for entity matching:**

Consider "Johnson & Johnson" vs. "Jhonson & Jhonson" (typo):

**Word token comparison:**
- Vocabulary: {Johnson, Jhonson, &}
- Vector A: [2, 0, 1], Vector B: [0, 2, 1]
- Cosine similarity = $\frac{0 \times 0 + 0 \times 0 + 1 \times 1}{\sqrt{5} \times \sqrt{5}} = \frac{1}{5} = 0.2$

Very low! A single character typo completely destroys word-level similarity.

**Character 3-gram comparison:**
- A: {"Joh", "ohn", "hns", "nso", "son", ... }
- B: {"Jho", "hns", "nso", "son", ... } ← shares "hns", "nso", "son"

Jaccard similarity = $\frac{|A \cap B|}{|A \cup B|}$. With 12 unique grams in A, 12 in B, 6 shared: $J = 6/18 = 0.33$

More robust — ONE typo changes only 3 grams (the affected 3-grams containing the typo character) vs. the entire word token.

**Mathematical characterization:**

For a string of length $L$ with a single character substitution at position $p$:
- Word tokens: similarity drops to 0 for the affected word
- Character n-grams: similarity drops by at most $\frac{n}{L-n+1}$ (proportional to n-gram length over total grams)

For $L=10$, $n=3$: drop at most $3/8 = 37.5\%$. This degradation is gradual and proportional to the error — exactly what you want for fuzzy matching.

---

## Q45: Prove that SMOTE creates synthetic minority samples in the convex hull of existing minority samples. Why is this a limitation?

**SMOTE Algorithm:**

For each minority sample $x_i$:
1. Find K nearest neighbors in minority class: $\{x_{i1}, x_{i2}, ..., x_{iK}\}$
2. Randomly choose one neighbor $x_{il}$
3. Generate synthetic sample: $\tilde{x} = x_i + \lambda(x_{il} - x_i)$ where $\lambda \sim \text{Uniform}(0,1)$

**Proof that $\tilde{x}$ lies in the convex hull:**

$\tilde{x} = x_i + \lambda(x_{il} - x_i) = (1-\lambda)x_i + \lambda x_{il}$

This is a **convex combination** of $x_i$ and $x_{il}$ (since $0 \leq \lambda \leq 1$ and $(1-\lambda) + \lambda = 1$). Therefore $\tilde{x}$ lies on the line segment between $x_i$ and $x_{il}$ — inside the convex hull of minority samples. ✅

**Limitations:**

1. **Does not extrapolate beyond observed minority region:** If minority class occupies a small, irregular manifold in feature space, SMOTE only fills that manifold — doesn't extend it. Novel fraud patterns that exist NEAR but outside the observed region are missed.

2. **Ignores majority class:** SMOTE can generate synthetic points that fall into dense majority regions (the boundary between classes), creating mislabeled training examples. **SMOTE+Tomek Links** addresses this by removing synthetic points too close to majority.

3. **Inapplicable to non-tabular data:** Cannot interpolate between claim text documents or fraud network embeddings. Only use SMOTE on tabular numerical features.

**Alternative — class weight scaling (your preferred approach):**

`scale_pos_weight = n_negatives / n_positives` in XGBoost achieves similar effect mathematically by up-weighting minority class gradient contributions, without creating synthetic samples. Zero risk of generating boundary-violating samples.

---

## Q46: Derive the ELBO (Evidence Lower BOund) used in Variational Autoencoders. Why is it a lower bound?

**Setup:** VAE learns a latent variable model. We want to maximize $\log P(x)$ but it's intractable (requires integrating over all possible $z$).

**ELBO derivation:**

$$\log P(x) = \log \int P(x, z) dz = \log \int P(x|z)P(z) dz$$

Multiply and divide by posterior approximation $Q(z|x)$:

$$= \log \int Q(z|x) \frac{P(x,z)}{Q(z|x)} dz = \log E_{Q(z|x)}\left[\frac{P(x,z)}{Q(z|x)}\right]$$

**Jensen's inequality** ($\log$ is concave, so $\log E[X] \geq E[\log X]$):

$$\log P(x) \geq E_{Q(z|x)}\left[\log \frac{P(x,z)}{Q(z|x)}\right]$$

This is the **ELBO** — a lower bound on $\log P(x)$.

**Decompose ELBO:**
$$\text{ELBO} = E_{Q(z|x)}[\log P(x|z)] - D_{KL}(Q(z|x) \| P(z))$$

| Term | Interpretation |
|---|---|
| $E_{Q(z|x)}[\log P(x|z)]$ | **Reconstruction loss** — how well does decoded $z$ reproduce $x$? |
| $D_{KL}(Q(z|x) \| P(z))$ | **Regularization** — how close is the learned posterior to the prior? |

**The gap:** $\log P(x) - \text{ELBO} = D_{KL}(Q(z|x) \| P(x|z)) \geq 0$

As we improve the approximation $Q$, ELBO tightens toward $\log P(x)$.

**Connection to fraud embedding:** VAE learns a structured latent space. Fraud claims cluster in certain regions; legitimate claims in others. The KL term prevents the latent space from collapsing to arbitrary codes.

---

---

## 🔢 EXPANDED MATHEMATICAL FORMULAS CHEAT SHEET

| Concept | Formula |
|---|---|
| Logistic regression gradient | $\frac{1}{N}X^T(\hat{y} - y)$ |
| XGBoost leaf score | $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ |
| XGBoost split gain | $\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$ |
| Attention | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| KL Divergence | $\sum P(x)\log\frac{P(x)}{Q(x)}$ |
| Brier Score | $\frac{1}{N}\sum(\hat{y}_i - y_i)^2$ |
| F-beta | $\frac{(1+\beta^2)PR}{\beta^2 P + R}$ |
| Bayes' Theorem | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ |
| Information Gain | $H(S) - \sum_v \frac{\|S_v\|}{\|S\|}H(S_v)$ |
| Cox PH Hazard | $h(t\|x) = h_0(t)\exp(\beta^Tx)$ |
| Cox Partial Likelihood | $L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(\beta^T x_i)}{\sum_{j \in R(t_i)} \exp(\beta^T x_j)}$ |
| Kaplan-Meier | $\hat{S}(t) = \prod_{t_{(j)} \leq t}\left(1 - \frac{d_j}{n_j}\right)$ |
| Contrastive Loss (InfoNCE) | $-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)}$ |
| NDCG | $\sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}$ / IDCG |
| IPW ATE Estimator | $\frac{1}{n}\sum\left[\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)}\right]$ |
| DiD Estimator | $(\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})$ |
| CUPED Variance Reduction | $\text{Var}(\tilde{Y}) = \text{Var}(Y)(1 - \rho^2)$ |
| Thompson Sampling Regret | $O\left(\sum_{k:\mu_k<\mu^*}\frac{\log T}{\Delta_k}\right)$ |
| Beta-Binomial Conjugate | Prior $\text{Beta}(\alpha,\beta)$ + $k$ successes → Posterior $\text{Beta}(\alpha+k, \beta+n-k)$ |
| SMOTE Synthetic Sample | $\tilde{x} = (1-\lambda)x_i + \lambda x_{il}$ where $\lambda \sim U(0,1)$ |
| ELBO (VAE) | $E_{Q(z\|x)}[\log P(x\|z)] - D_{KL}(Q(z\|x)\|P(z))$ |
| Markov Removal Effect | $1 - P(\text{Conv}_{-k}) / P(\text{Conv})$ |
| Jaro-Winkler | Jaro + $p \cdot l \cdot (1 - \text{Jaro})$ where $l$=common prefix len, $p \leq 0.25$ |
| MinHash Jaccard Error | $\text{Var}(\hat{J}) = \frac{J(1-J)}{k}$ for $k$ hash functions |

---

## 🚦 COMMON DMM PITFALLS TO AVOID IN THE INTERVIEW

| ❌ Mistake | ✅ What to Do Instead |
|---|---|
| Deriving from memory without structure | Always state assumptions first, then derive step-by-step |
| Confusing AUC-ROC with PR-AUC | Explicitly say: "For imbalanced data like fraud, PR-AUC is more informative because..." |
| Forgetting the $\sqrt{d_k}$ scaling in attention | Always justify why: variance of dot products grows with dimension → softmax saturates |
| Saying "XGBoost uses gradient" without specifying order | XGBoost uses 2nd-order (gradient + hessian); sklearn GBM uses 1st-order only |
| Not connecting math to your resume | After every derivation, add: "In my [project], this manifested as..." |
| Forgetting log-likelihood vs. likelihood | Always take log for numerical stability and to convert products to sums |
| Confusing MLE and MAP | MLE = maximize $P(D\|w)$; MAP = maximize $P(w\|D) \propto P(D\|w) P(w)$ |
| Can't explain WHY KL divergence is asymmetric | $D_{KL}(P\|Q)$ weights by $P$; infinite when $Q(x)=0$ but $P(x)>0$ → mode-seeking vs mode-covering |
| Forgetting the PH assumption for Cox | Always mention: "subject to the proportional hazards assumption, which I verified with Schoenfeld residuals" |
| Not knowing when correlation ≠ causation | Always connect to causal inference: "correlation doesn't imply causation because of selection bias / confounders" |

---

*See companion files: 02_Causal_Inference_Math_Grind.md, 03_Optimization_Loss_Functions_Grind.md, 04_Transformer_Attention_Math.md, 08_Statistical_Testing_AandB.md*
