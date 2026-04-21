# 📐 DMM GRIND — 50 Mathematical & Theory Questions with Full Derivations
### Round 2: Depth in Mathematical Modeling | Answers with first-principles math

> Level: PhD-interview depth. Every answer goes: intuition → math → production connection.

---

## ═══════════════════════════════════════
## SECTION A: PROBABILITY & STATISTICS
## ═══════════════════════════════════════

### Q1: Derive the Maximum Likelihood Estimator (MLE) for Gaussian distribution.

**Setup:** Observe i.i.d. samples $x_1, ..., x_n$ from $\mathcal{N}(\mu, \sigma^2)$. Derive MLE for $\mu$ and $\sigma^2$.

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**MLE for μ:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$
$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

→ **Sample mean is the MLE.** Intuitive: Gaussian is symmetric around μ, so the center of mass.

**MLE for σ²:**
$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0$$
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

**Key insight:** MLE gives $\frac{1}{n}$ (biased). Unbiased estimator uses $\frac{1}{n-1}$ (Bessel's correction). Why biased? We use $\bar{x}$ instead of true $\mu$ — $\bar{x}$ is closer to the data than $\mu$, so variance is underestimated.

**Production connection:** Gaussian assumption in credit scoring — income distributions are NOT Gaussian (right-skewed). Log-transform first, then apply Gaussian MLE. Verify with Shapiro-Wilk test before assuming normality.

---

### Q2: Derive the Beta-Binomial conjugate pair and explain cold-start fraud modeling.

**Setup:** Prior: $p \sim \text{Beta}(\alpha, \beta)$. Likelihood: $k$ fraud cases from $n$ transactions (Binomial).

**Posterior derivation:**
$$P(p | k, n) \propto P(k | p) \cdot P(p)$$
$$= \binom{n}{k}p^k(1-p)^{n-k} \cdot \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$$
$$\propto p^{k+\alpha-1}(1-p)^{n-k+\beta-1}$$

This IS a Beta distribution with updated parameters:
$$p | k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**Bayesian conjugate posterior:** $\text{Beta}(\alpha + k_{observed}, \beta + n_{observed} - k_{observed})$

**Cold-start application:**
- **New seller arrives:** No transaction history. What's their fraud probability?
- **Prior:** Set $\alpha, \beta$ from population fraud rate. If population fraud = 2%, set $\text{Beta}(2, 98)$ (mean = 2%).
- **After 10 transactions, 0 fraud:** Posterior = $\text{Beta}(2 + 0, 98 + 10) = \text{Beta}(2, 108)$. Mean = 2/110 ≈ 1.8%.
- **After 10 transactions, 3 fraud:** Posterior = $\text{Beta}(5, 105)$. Mean = 5/110 ≈ 4.5%.

**Key property:** With few observations, posterior is dominated by prior (shrinkage toward population rate). With many observations, posterior is dominated by data. This is EXACTLY the right behavior for cold-start.

---

### Q3: Prove that the OLS estimator is BLUE (Best Linear Unbiased Estimator) — Gauss-Markov Theorem.

**OLS estimator:** $\hat{\beta} = (X^TX)^{-1}X^Ty$

**Step 1: Show Unbiasedness**
$$E[\hat{\beta}] = E[(X^TX)^{-1}X^Ty] = (X^TX)^{-1}X^TE[y]$$
Since $E[y] = X\beta$ (from linear model $y = X\beta + \epsilon$, $E[\epsilon]=0$):
$$E[\hat{\beta}] = (X^TX)^{-1}X^TX\beta = \beta \checkmark$$

**Step 2: Variance of OLS**
$$\text{Var}(\hat{\beta}) = (X^TX)^{-1}X^T\text{Var}(\epsilon)X(X^TX)^{-1} = \sigma^2(X^TX)^{-1}$$
(using $\text{Var}(\epsilon) = \sigma^2 I$, homoskedasticity assumption)

**Step 3: BLUE Proof (sketch)**
Let $\tilde{\beta} = Cy$ be any other linear unbiased estimator. Unbiasedness requires $CX = I$.
$$\text{Var}(\tilde{\beta}) = \sigma^2 CC^T \geq \sigma^2(X^TX)^{-1} = \text{Var}(\hat{\beta})$$

The inequality follows from $CC^T - (X^TX)^{-1}$ being positive semi-definite (proven by writing $C = (X^TX)^{-1}X^T + D$ where $DX = 0$, then $CC^T = (X^TX)^{-1} + DD^T \geq (X^TX)^{-1}$).

**Why Gauss-Markov breaks in fraud modeling:**
1. **Heteroskedasticity:** fraud patterns are more volatile than non-fraud → $\text{Var}(\epsilon)$ is not diagonal. Fix: Weighted Least Squares.
2. **Non-linearity:** fraud relationships are non-linear. Fix: XGBoost > OLS.
3. **Label noise:** fraud labels are noisy (missed fraud labeled as non-fraud). Fix: Robust loss functions.

---

### Q4: Derive the Expected Calibration Error (ECE) and explain why calibration matters in fraud scoring.

**Calibration definition:** A model is calibrated if for all predicted probabilities $p_i$, the actual fraction of positives among samples with predicted probability $p_i$ equals $p_i$.

**ECE Formula:**
$$ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |\overline{\text{acc}}(B_b) - \overline{\text{conf}}(B_b)|$$

Where:
- $B_b$ = set of predictions in bucket $b$ (e.g., [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
- $|\overline{\text{acc}}(B_b) - \overline{\text{conf}}(B_b)|$ = gap between average actual positive rate and average predicted probability in bucket $b$
- Weighted by bucket size $|B_b|/N$

**Why calibration matters in fraud risk scoring:**
1. **Business decisions use probabilities:** "Flag claims with P(fraud) > 0.3" — this ONLY makes sense if 30% of claims scored 0.3 actually ARE fraud. If P=0.3 → 60% are actually fraud (overconfident), you're under-investigating.
2. **Risk reserve setting (insurance):** Reserves = $\sum_i P_i \times \text{claim\_amount}_i$. If probabilities are miscalibrated, financial reserves are wrong.
3. **Communication:** "This claim has a 75% fraud probability" must mean something to investigators.

**Platt Scaling (calibration method):**
Fit logistic regression on top of uncalibrated scores:
$P_{calibrated} = \sigma(as_i + b)$ where $s_i$ is uncalibrated score, $a, b$ learned on validation set.

**Isotonic Regression calibration:** Non-parametric; fit monotone step function. Better when relationship is non-linear. Needs more data than Platt.

---

### Q5: Derive Gradient Descent convergence and explain Adam optimizer advantages.

**Gradient Descent update:**
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**Convergence for convex loss (L-smooth, μ-strongly convex):**
- L-smooth: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$ (Lipschitz gradient)
- Convergence guarantee: $\|\theta_t - \theta^*\|^2 \leq (1 - \mu/L)^t \|\theta_0 - \theta^*\|^2$
- Linear convergence rate; $\eta^* = 2/(\mu + L)$ for fastest convergence

**Problem 1: Bad conditioning** If $L/\mu \gg 1$ (ill-conditioned loss landscape), convergence is extremely slow.

**Problem 2: Non-convex loss** (neural networks) — gradient descent can get stuck in saddle points.

**Adam optimizer (Adaptive Moment Estimation):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad[\text{1st moment estimate}]$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad[\text{2nd moment estimate}]$$
$$\hat{m}_t = m_t/(1-\beta_1^t) \quad[\text{bias correction}]$$
$$\hat{v}_t = v_t/(1-\beta_2^t) \quad[\text{bias correction}]$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

**Advantages:**
1. **Adaptive learning rate per parameter:** Features with large gradients (common) get smaller effective LR; rare features (sparse) get larger effective LR. Crucial for NLP where word embeddings are sparse.
2. **Momentum:** $m_t$ accumulates gradient direction → dampens oscillations in narrow valleys
3. **Bias correction:** Early steps: $m_0 = v_0 = 0$ → without correction, estimates are biased toward zero. Bias correction prevents slow start.

**Common pitfalls: L2 regularization + Adam:**
Standard Adam doesn't handle weight decay correctly. Use **AdamW** (decoupled weight decay):
$$\theta_{t+1} = \theta_t - \eta[\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon) + \lambda\theta_t]$$
This is the correct way — weight decay is applied to weights directly, not mixed into gradient statistics.

---

### Q6: Explain the Expectation-Maximization (EM) algorithm. How does it apply to Gaussian Mixture Models?

**EM Goal:** Find MLE when observations $X$ are incomplete (latent variable $Z$ is unobserved).
$$\hat{\theta}^{MLE} = \arg\max_\theta \log P(X|\theta) = \arg\max_\theta \log \sum_Z P(X,Z|\theta)$$

**Problem:** The sum inside log makes direct maximization intractable.

**EM Solution:**
$$\text{E-step: } Q(\theta|\theta^{old}) = E_{Z|X,\theta^{old}}[\log P(X,Z|\theta)]$$
$$\text{M-step: } \theta^{new} = \arg\max_\theta Q(\theta|\theta^{old})$$

**Guaranteed:** Log-likelihood $\log P(X|\theta)$ is non-decreasing at each EM step (proven via Jensen's inequality).

**Applied to GMM (Gaussian Mixture Models):**

Model: $P(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$

Latent variable: $z_i \in \{1,...,K\}$ = which component generated $x_i$

**E-step (compute responsibilities):**
$$r_{ik} = P(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

**M-step (update parameters):**
$$\pi_k^{new} = \frac{1}{N}\sum_i r_{ik}, \quad \mu_k^{new} = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \Sigma_k^{new} = \frac{\sum_i r_{ik}(x_i-\mu_k^{new})(x_i-\mu_k^{new})^T}{\sum_i r_{ik}}$$

**Fraud detection application:** GMM for anomaly detection. Train GMM on legitimate transactions. At inference: $P(x_{new})$ = probability density. Low density → anomalous → fraud flag.

**EM limitation:** Only finds local maximum. Solution: Multiple random restarts; report best.

---

### Q7: What is the Rademacher Complexity and why is it more precise than VC dimension for deep learning?

**VC Dimension:** Largest number of points that can be shattered (labeled arbitrarily correctly) by the hypothesis class. Integer-valued. Example: Linear classifiers in $d$ dimensions: VC dim = $d+1$.

**Problem with VC:** For neural networks, VC dimension ≈ O(number of parameters) → millions → gives vacuous generalization bounds. Says nothing useful about deep learning generalization in practice.

**Rademacher Complexity:**
$$\hat{\mathcal{R}}_n(\mathcal{F}) = E_\sigma\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(x_i)\right]$$

Where $\sigma_i \in \{-1, +1\}$ are independent Rademacher random variables.

**Intuition:** How well can the function class $\mathcal{F}$ fit random noise labels? Higher Rademacher = more complex = fewer points needed to overfit = worse generalization.

**Generalization bound:** With probability ≥ 1-δ:
$$\mathcal{L}_{true}(f) \leq \hat{\mathcal{L}}_{train}(f) + 2\mathcal{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

**Why better than VC for DL:**
- Empirically computed on actual training data (not worst-case)
- Captures spectral norms, weight magnitudes (explains why regularization helps)
- Explains the "benign overfitting" phenomenon (overparameterized models generalize despite zero training loss)

---

### Q8: Derive the Markov Chain Monte Carlo (MCMC) — Metropolis-Hastings algorithm.

**Goal:** Sample from intractable posterior $P(\theta|X) \propto P(X|\theta)P(\theta)$.

**Algorithm:**
```
Initialize θ_0 randomly
For t = 1, 2, ..., T:
  1. Propose: θ* ~ q(θ*|θ_{t-1})   (proposal distribution, e.g., Gaussian random walk)
  2. Acceptance ratio:
     α = min(1, [P(X|θ*) P(θ*) q(θ_{t-1}|θ*)] / [P(X|θ_{t-1}) P(θ_{t-1}) q(θ*|θ_{t-1})])
  3. Accept: with probability α → θ_t = θ*
     Reject: with prob 1-α → θ_t = θ_{t-1}
```

**Key property — Detailed Balance:**
The stationary distribution of this Markov chain IS $P(\theta|X)$.
Proven by showing: $\pi(\theta)T(\theta'|\theta) = \pi(\theta')T(\theta|\theta')$ (detailed balance)

**Bayesian fraudster risk estimation:**
- $\theta$ = true fraud rate for a specific fraud ring
- Prior $P(\theta)$ = population fraud rate distribution
- Likelihood $P(X|\theta)$ = probability of observing these transaction patterns given fraud rate $\theta$
- MCMC samples from posterior $P(\theta|X)$ → full uncertainty quantification
- **Output:** Not just a point estimate but a distribution → credible intervals + risk-based decisions

**Burn-in and mixing:** First 10-20% of MCMC samples discarded (burn-in period, chain not yet at stationary distribution). Thinning: retain every k-th sample (reduces autocorrelation).

---

### Q9: Derive the Singular Value Decomposition (SVD) and explain its connection to PCA.

**SVD:** For any matrix $A \in \mathbb{R}^{m \times n}$:
$$A = U \Sigma V^T$$

Where:
- $U \in \mathbb{R}^{m \times m}$: left singular vectors (columns are orthonormal)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal with singular values $\sigma_1 \geq \sigma_2 \geq ... \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (columns are orthonormal)

**Derivation sketch:** $A^TA \in \mathbb{R}^{n\times n}$ is symmetric PSD → eigendecomposition: $A^TA = V\Lambda V^T$ where $\Lambda = \Sigma^T\Sigma$. Then $\sigma_i = \sqrt{\lambda_i}$, $U = AV\Sigma^{-1}$.

**Connection to PCA:**
PCA on data matrix $X \in \mathbb{R}^{n \times d}$ (centered: $X := X - \bar{X}$):
- Covariance matrix: $C = \frac{1}{n}X^TX$
- PCA: find eigenvectors of $C$ → PC directions
- SVD: $X = U\Sigma V^T$ → right singular vectors $V$ are the PC directions!
- PC scores (projections): $XV = U\Sigma$

**Low-rank approximation (Eckart-Young theorem):**
$$\min_{\text{rank-k} \hat{A}} \|A - \hat{A}\|_F = \|A - U_k\Sigma_k V_k^T\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$$

Best rank-k approximation = keep top-k singular values. Optimal in Frobenius norm.

**Application in your recommendation system:**
- User-item interaction matrix $R \in \mathbb{R}^{n_{users} \times n_{items}}$: mostly zeros (sparse)
- SVD: $R \approx U_k\Sigma_k V_k^T$ — low-rank factorization into user latent factors × item latent factors
- Prediction: $\hat{R}_{ij} = u_i^T v_j$ (dot product of user and item latent factors)
- This IS matrix factorization collaborative filtering!

---

### Q10: What is Mutual Information and how do you use it for feature selection?

**Definition:**
$$I(X;Y) = \sum_x \sum_y P(x,y) \log \frac{P(x,y)}{P(x)P(y)} = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

**Properties:**
- $I(X;Y) \geq 0$ always
- $I(X;Y) = 0$ iff X and Y are independent
- $I(X;Y) = H(X) + H(Y) - H(X,Y)$ (information overlap)
- **More expressive than correlation:** Pearson r only captures LINEAR relationships. MI captures ALL dependencies (linear + non-linear).

**Feature selection using MI:**

```python
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def select_features_mi(X_train, y_train, n_features=30):
    """
    Select top features by mutual information with target.
    Better than correlation for fraud (non-linear relationships).
    """
    mi_scores = mutual_info_classif(
        X_train, y_train, 
        discrete_features='auto',
        n_neighbors=5    # k-NN estimator for continuous features
    )
    
    # Rank features
    mi_ranking = pd.Series(mi_scores, index=X_train.columns)
                    .sort_values(ascending=False)
    
    # Select top features
    selected = mi_ranking.head(n_features).index.tolist()
    
    return selected, mi_ranking

# Why MI over correlation for fraud:
# - Account age vs. fraud: non-linear (very new AND very old accounts are risky)
# - Pearson r captures only monotone linear: r ≈ 0 (misses both ends)
# - MI captures the full dependency: MI > 0 (finds both wings)
```

**MI-based feature selection limitation:**
- Greedy: selects features individually, ignores joint information (may select redundant features)
- mRMR (Minimum Redundancy Maximum Relevance): Also minimizes MI between selected features → less redundancy
$$\max_S \left[\frac{1}{|S|}\sum_{f \in S} I(f; y) - \frac{1}{|S|^2}\sum_{f,g \in S} I(f; g)\right]$$

---

## ═══════════════════════════════════════
## SECTION B: DEEP LEARNING MATH
## ═══════════════════════════════════════

### Q11: Derive LSTM equations and explain what problem it solves over vanilla RNN.

**Vanilla RNN Problem:** For sequence $x_1, ..., x_T$:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**Vanishing gradient:** Backprop through time multiplies by $W_h^T$ repeatedly:
$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} \cdot \text{gate terms}$$

If $\|W_h\| < 1$ → product → 0 exponentially. If $\|W_h\| > 1$ → explodes.
→ RNN cannot learn dependencies longer than ~20 steps.

**LSTM Solution — The Full Equations:**

$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate: what to erase from cell)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate: what new info to write)}$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \quad \text{(candidate cell: new info)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) \quad \text{(output gate: what to expose)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

**Why gradients don't vanish:** The cell state update is ADDITIVE:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

If forget gate $f_t \approx 1$ (remember everything) and input gate $i_t \approx 0$: 
$c_t \approx c_{t-1}$ → gradient flows back: $\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1$ → no vanishing!

**Gate count:**
- Forget gate: "How much of past cell state to keep?"
- Input gate: "How much of new candidate to write?"
- Output gate: "How much of cell state to expose as hidden state?"

**When LSTM > Transformer:**
- Very long sequences where positional encoding struggles to generalize (>10K tokens)
- Streaming/online data (no fixed sequence length)
- But: For most NLP tasks with moderate lengths → Transformer wins due to parallelization

---

### Q12: Prove that ReLU activations help with vanishing gradient problem.

**Sigmoid gradient:** $\sigma'(z) = \sigma(z)(1-\sigma(z))$
Maximum at z=0: $\sigma'(0) = 0.25$. For $|z| > 4$: $\sigma'(z) < 0.03$.
→ In deep networks, product of sigmoid derivatives → 0 exponentially.

**ReLU:** $\text{ReLU}(z) = \max(0, z)$

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Chain rule for L-layer network with ReLU:**
$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \prod_{l=2}^{L} \text{ReLU}'(z_l) \cdot W_l$$

Since $\text{ReLU}'(z_l) \in \{0, 1\}$ (never >1), only "dead" neurons (z≤0) have gradient 0.
Active neurons: gradient = 1 × W_l = passes full gradient.

**Dead ReLU problem:** If weights initialized badly, a neuron outputs <0 for ALL training examples → gradient always 0 → neuron never learns.

**Fix:** Leaky ReLU: $\max(0.01z, z)$ — small slope for z<0, never truly dies.
Or: ELU, SELU, GELU (smooth approximation to ReLU, used in BERT).

**GELU (used in BERT):**
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(\mathcal{N}(0,1) \leq x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

Smooth, non-monotone, probabilistic interpretation: scales input by how much above median it is.

---

### Q13: Derive the Transformer's positional encoding and explain why it enables length generalization.

**Problem:** Self-attention is permutation invariant — a shuffled sequence has same attention scores. Positions must be explicitly encoded.

**Sinusoidal positional encoding:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Why sinusoidal?** Key property:
$$PE_{pos+k} = f(PE_{pos})$$

Specifically: $PE_{pos+k}$ can be expressed as a linear transformation of $PE_{pos}$:
$$\begin{pmatrix}PE_{pos+k,2i} \\ PE_{pos+k,2i+1}\end{pmatrix} = \begin{pmatrix}\cos(k/\omega_i) & \sin(k/\omega_i) \\ -\sin(k/\omega_i) & \cos(k/\omega_i)\end{pmatrix}\begin{pmatrix}PE_{pos,2i} \\ PE_{pos,2i+1}\end{pmatrix}$$

→ Attention can compute RELATIVE positions without seeing them explicitly.

**Length generalization:** Sinusoidal PE uses fixed functions → can compute PE for ANY position, even if longer than training. Learnable PE (BERT): lookup table up to max_position → can't generalize beyond training length.

**Modern alternative — RoPE (Rotary Position Embedding, used in LLaMA):**
Apply rotation matrix to query and key vectors, where rotation angle = function of position.
$$f_q(x_m, m) = W_q x_m e^{im\theta}$$
$$f_k(x_n, n) = W_k x_n e^{in\theta}$$

Inner product: $Q_m \cdot K_n$ = function of $(m-n)$ only → encodes relative position naturally.
**Key advantage:** Attention depends only on relative position $(m-n)$, not absolute → better length extrapolation.

---

### Q14: What is Layer Normalization vs. Batch Normalization? Derive the gradient of Layer Norm.

**Batch Normalization:**
$$BN(x_i) = \gamma \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \beta$$

Where $\mu_\mathcal{B}, \sigma_\mathcal{B}^2$ are computed over the batch dimension.

**Layer Normalization:**
$$LN(x_i) = \gamma \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta$$

Where $\mu_i, \sigma_i^2$ are computed over the feature dimension for sample i:
$$\mu_i = \frac{1}{H}\sum_{j=1}^H x_{ij}, \quad \sigma_i^2 = \frac{1}{H}\sum_{j=1}^H (x_{ij} - \mu_i)^2$$

**Gradient of Layer Norm (for implementing custom training):**
$$\frac{\partial LN}{\partial x_j} = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\left[\delta_{ij} - \frac{1}{H} - \frac{(x_j-\mu)(x_i-\mu)}{H(\sigma^2+\epsilon)}\right]$$

**Key differences:**

| Property | Batch Norm | Layer Norm |
|---|---|---|
| Normalization over | Batch dimension | Feature dimension |
| Works with batch_size=1? | ❌ Unstable | ✅ Yes |
| Variable length sequences? | ❌ Problematic | ✅ Works perfectly |
| Training vs. inference | Need running statistics | Same computation |
| Use case | CNNs, feed-forward | Transformers, RNNs |

**Pre-LN vs. Post-LN:**
- Post-LN (original paper): LN after residual → training instability in deep networks
- Pre-LN (current practice): LN before attention/FFN → more stable training, slightly different optimization landscape

---

### Q15: Explain the math behind attention masking for causal (autoregressive) models.

**GPT causal attention requirement:** Position $i$ can only attend to positions $j \leq i$ (past and present, not future).

Without masking, full attention:
$$A_{ij} = \text{softmax}(QK^T/\sqrt{d_k})_{ij}$$
Every position can attend to every other.

**Causal mask implementation:**
$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$$A_{ij} = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)_{ij}$$

After adding M: positions with $i < j$ get $-\infty$ before softmax → $e^{-\infty} = 0$ → zero attention weight.

**Why $-\infty$ not 0?** If we directly set weights to 0, they wouldn't sum to 1, violating softmax normalization. By setting logits to $-\infty$ before softmax, softmax redistributes weight only over valid positions while summing to 1.

**Flash Attention optimization:**
Standard attention: stores $O(n^2)$ attention matrix in HBM (GPU memory) for backprop.
Flash Attention: computes attention in tiles that fit in SRAM → $O(n)$ HBM reads. 
No full attention matrix materializes → 2-4x memory reduction, faster in practice.

---

## ═══════════════════════════════════════
## SECTION C: INFORMATION THEORY
## ═══════════════════════════════════════

### Q16: Prove that cross-entropy loss minimization = KL divergence minimization.

**Setup:** Data distribution $P$ (empirical), model distribution $Q_\theta$.

**KL divergence:**
$$D_{KL}(P \| Q_\theta) = \sum_x P(x)\log\frac{P(x)}{Q_\theta(x)} = \sum_x P(x)\log P(x) - \sum_x P(x)\log Q_\theta(x)$$
$$= -H(P) + H(P, Q_\theta)$$

$H(P)$ is the entropy of the data distribution — **constant** w.r.t. model parameters $\theta$.

Therefore:
$$\min_\theta D_{KL}(P \| Q_\theta) \equiv \min_\theta H(P, Q_\theta) = \min_\theta \left[-\sum_x P(x)\log Q_\theta(x)\right]$$

The cross-entropy loss IS the cross-entropy $H(P, Q_\theta)$:
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \log Q_\theta(y_i | x_i) \approx H(P, Q_\theta)$$

Therefore: **minimizing cross-entropy loss = minimizing KL divergence from model to data distribution.**

**Why this matters:** KL divergence is not symmetric ($D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$). We minimize $D_{KL}(P\|Q)$ (data → model): where P=0 but Q>0 doesn't matter (model can assign probability to unseen events). Where P>0 but Q=0 → infinite KL → model must cover all data support.

---

### Q17: What is Entropy regularization in RL and how does it prevent mode collapse in LLM RLHF?

**Standard RL objective:**
$$\max_\pi E_{\tau \sim \pi}[\sum_t r(s_t, a_t)]$$

**Entropy regularized RL:**
$$\max_\pi E_{\tau \sim \pi}[\sum_t r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$

Where $H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$ is policy entropy.

**Effect:** Encourages exploration — high entropy means diverse action choices. Prevents the policy from greedily collapsing to a single action.

**In RLHF for LLMs:**
$$\max_\pi E_{y \sim \pi}[r(x, y) - \beta D_{KL}(\pi \| \pi_{ref})]$$

The KL divergence term acts like entropy regularization — prevents the policy (LLM) from drifting too far from the reference model:
- Without KL penalty: LLM learns to maximize reward at any cost — might ignore language quality, safety constraints, calibration
- With KL penalty: Stay "close" to the reference model (pre-RLHF LLM)
- $\beta$ controls the tradeoff: small $\beta$ = optimize reward aggressively; large $\beta$ = conservative

**RLHF mode collapse:** Without KL penalty or with very small $\beta$, RLHF can cause the LLM to repeatedly output a small set of highly rewarded responses (mode collapse). Example: if human raters always prefer shorter responses, LLM learns to give one-word answers to everything.

---

## ═══════════════════════════════════════
## SECTION D: GRAPH THEORY & GNNs
## ═══════════════════════════════════════

### Q18: Derive the GraphSAGE aggregation equations and explain the choices.

**GraphSAGE (Graph Sample and Aggregate):**

For each node $v$ at layer $k$:
$$h_{\mathcal{N}(v)}^k = \text{AGGREGATE}^k\left(\{h_u^{k-1} : u \in \mathcal{N}(v)\}\right)$$
$$h_v^k = \sigma\left(W^k \cdot \text{CONCAT}(h_v^{k-1}, h_{\mathcal{N}(v)}^k)\right)$$

**Aggregation options:**
1. **Mean aggregator:** $h_{\mathcal{N}(v)}^k = \text{mean}(\{h_u^{k-1}\})$ — simple, ignores graph structure depth
2. **LSTM aggregator:** Order nodes randomly, feed through LSTM — can capture sequence signal
3. **Pooling aggregator:** $h_{\mathcal{N}(v)}^k = \max(\{\sigma(W_{pool}h_u + b)\})$ — element-wise max captures most salient features

**Why CONCAT vs ADD:**
- ADD: $h_v^k = \sigma(W_1 h_v^{k-1} + W_2 \text{AGG})$ — can't distinguish "node has strong self-feature" from "strong neighbor feature"
- CONCAT: always preserves both signals separately → more expressive

**In fraud ring detection:**
- Layer 0: Node features = [GMV, return_rate, cancel_rate, account_age]
- Layer 1: Each seller aggregates from their device/address connections
- Layer 2: Each seller aggregates from their layer-1 neighbors' embeddings
- Final: Embedding contains information from 2-hop neighborhood
- Predict: P(this seller is in a fraud ring) from 2-hop embedding

**Why 2 layers?** Fraud rings typically operate within 2-hop clusters. Deeper → more computation, diminishing signal, risk of over-smoothing.

**Over-smoothing problem:** Too many layers → all node embeddings converge → lose discriminative information. Fix: Residual connections in GNN (add skip connection: $h_v^k = h_v^k + h_v^{k-1}$).

---

### Q19: What is Random Walk and Node2Vec? How do they differ from GNN for fraud detection?

**Random Walk Embeddings (Node2Vec, DeepWalk):**

1. **Simulate random walks** from each node (BFS or DFS controlled by π,q parameters in Node2Vec)
2. Treat walk sequences as "sentences", nodes as "words"
3. Apply Skip-gram (Word2Vec) to learn node embeddings:
   $$\max \sum_{v \in V}\left[\log P(\text{context}|\text{center})\right] = \sum \log \sigma(z_v^T z_u) + \sum \log \sigma(-z_v^T z_{neg})$$
4. Nodes with similar graph neighborhoods → similar embeddings

**Node2Vec BFS vs. DFS (p, q hyperparameters):**
- $p$ = return probability (tendency to revisit recent node)
- $q$ = in-out ratio (DFS vs. BFS tendency)
- $q > 1$: Biased toward BFS → samples structural equivalences (nodes with similar roles)
- $q < 1$: Biased toward DFS → samples community structure

**Node2Vec vs. GNN for fraud:**

| | Node2Vec | GNN |
|---|---|---|
| Feature usage | Node embeddings only | Can use rich node features |
| Transductive | Fixed graph → must retrain for new nodes | Inductive → can embed new nodes instantly |
| Scalability | Good (random walk sampling) | Expensive (recursive neighborhood aggregation) |
| Expressive power | Limited (ignores node attributes) | High (node features + structure) |
| **Best for fraud** | Static fraud rings (known at training) | New sellers (inductive embedding) |

---

## ═══════════════════════════════════════
## SECTION E: RAPID FIRE DERIVATION QUESTIONS
## ═══════════════════════════════════════

### Q20-Q50: QUICK DERIVATIONS AND THEORETICAL Q&A

**Q20: Why is the gradient of softmax with respect to its inputs a Jacobian matrix?**
> Softmax maps a vector to a vector: $f: \mathbb{R}^n → \mathbb{R}^n$. Gradient of vector-to-vector is a matrix (Jacobian). The $(i,j)$ entry: $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$ where $\delta_{ij}$ is Kronecker delta. This is why the combined gradient of cross-entropy + softmax simplifies beautifully to $\hat{y} - y$ (matrix-vector product of Jacobian with $\partial L/\partial s$ collapses cleanly).

**Q21: What is the difference between L1 and L2 regularization geometrically?**
> **L2 (Ridge):** Constraint region is a sphere $\|w\|_2^2 \leq t$. Smooth surface everywhere → optimal intersection with loss function ellipses tends to be off-axis (non-sparse). **L1 (Lasso):** Constraint region is a diamond $\|w\|_1 \leq t$. Has corners at axis-aligned points → optimal intersection tends to occur exactly at corners → exactly 0 for some weights → sparsity. **Intuition:** L1 penalty = linear cost for each weight → pays a fixed price to keep it nonzero. L2 penalty = quadratic cost → shrinks weights toward 0 but rarely exactly 0.

**Q22: What is the kernel trick in SVMs? Why can't we use it in deep learning?**
> **Kernel trick:** Replace dot product $x_i^T x_j$ with kernel function $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ where $\phi$ may be infinite-dimensional. Only need to compute $K$, not $\phi$ explicitly. Works because SVM prediction $f(x) = \sum \alpha_i y_i K(x_i, x)$ depends only on dot products. **Why not in DL:** Deep networks have non-quadratic complexity in feature space — there's no equivalent "kernel trick" because the feature transformation is not fixed (weights are learned). DL avoids this by learning feature representations directly in finite-dimensional space.

**Q23: Prove that Logistic Regression has a convex loss function.**
> Cross-entropy loss $\mathcal{L}(w) = -\frac{1}{N}\sum [y_i\log\sigma(w^Tx_i) + (1-y_i)\log(1-\sigma(w^Tx_i))]$. The Hessian $H = \frac{1}{N}X^T\text{diag}(\hat{y}_i(1-\hat{y}_i))X$ = semidefinite positive (since $\hat{y}_i(1-\hat{y}_i) \geq 0$ always). PSD Hessian → convex. **Implication:** Every local minimum is the global minimum — gradient descent guaranteed to find global optimum.

**Q24: What is the VC dimension of a single perceptron in d dimensions?**
> VC dim = $d+1$. A perceptron (linear classifier) can shatter any set of $d+1$ points in general position in $\mathbb{R}^d$ but cannot shatter any set of $d+2$ points. Proof for shattering $d+1$ points: $d+1$ points in general position → any labeling = some hyperplane separates them (by linear algebra — $d+1$ equations in $d+1$ unknowns (d weights + bias) is exactly determined). Cannot shatter $d+2$ points: by Radon's partition theorem.

**Q25: What is Jensen's inequality and how is it used in EM algorithm?**
> **Jensen's:** For convex function $f$: $f(E[X]) \leq E[f(X)]$. For concave function: reversed. **In EM:** $\log$ is concave. ELBO derivation:
> $$\log P(X|\theta) = \log \sum_Z P(X,Z|\theta) = \log \sum_Z Q(Z)\frac{P(X,Z|\theta)}{Q(Z)} \geq \sum_Z Q(Z)\log\frac{P(X,Z|\theta)}{Q(Z)} = \mathcal{L}(Q,\theta)$$
> Jensen's inequality (applied to concave log) → ELBO ≤ log-likelihood. E-step maximizes ELBO over Q; M-step maximizes over θ. At convergence: ELBO = log-likelihood (tight bound).

**Q26: What is the difference between covariance and correlation?**
> **Covariance:** $Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$. Measures joint variability. Units: product of X and Y units. **Correlation:** $\rho(X,Y) = Cov(X,Y)/(\sigma_X\sigma_Y)$. Dimensionless. Range [-1, 1]. **Key:** Correlation is standardized covariance. Correlation = 0 does NOT imply independence (only for Gaussian). Example: $Y = X^2$, $X \sim \mathcal{N}(0,1)$: $Cov(X,Y) = E[X \cdot X^2] = E[X^3] = 0$ but Y is deterministically a function of X.

**Q27: Explain the connection between dropout and Gaussian noise injection.**
> Dropout: each neuron output multiplied by Bernoulli$(1-p)$ mask. For large networks, by CLT, the sum of independent Bernoulli-masked activations approaches Gaussian. Specifically, multiplicative Bernoulli noise ≈ multiplicative Gaussian noise in expectation. This connection means: instead of dropout, could inject Gaussian noise $\mathcal{N}(1, p/(1-p))$ to activations — equivalent regularization effect. **Practical implication:** Gaussian noise injection is differentiable everywhere (Bernoulli is not); useful for gradient-based analysis of dropout effects.

**Q28: What is the relationship between margin and generalization in SVMs?**
> **Margin** $= 2/\|w\|$ = distance between support vectors. Maximizing margin = minimizing $\|w\|^2$. **Generalization bound (structural risk minimization):** $\mathcal{L}_{true} \leq \mathcal{L}_{train} + O\left(\sqrt{\frac{VC\_dim \cdot \log n}{n}}\right)$. For SVMs, VC dim scales with margin: VC dim $\approx (R/\gamma)^2$ where $R$ = data radius, $\gamma$ = margin. Maximizing margin → smaller VC dim → better generalization bound. This is the theoretical justification for max-margin classification.

**Q29: What is Hebbain learning and how does it connect to self-supervised learning?**
> **Hebbian:** "Neurons that fire together wire together." $\Delta W_{ij} \propto h_i h_j$ — if pre- and post-synaptic neuron are co-active, strengthen their connection. **Connection to self-supervised:** Contrastive learning (SimCLR, your fraud embeddings) learns similar representations for augmented views of the same sample → embeddings "fire together" for positive pairs. InfoNCE loss = Hebbian update in expectation. BERT's MLM: predicting masked tokens strengthens connections between contextual and semantic representations of nearby tokens.

**Q30: Prove that the variance of sample mean is σ²/n.**
> $\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$. (Using independence → covariance terms = 0). **Implication:** Standard error of mean = $\sigma/\sqrt{n}$. This is the denominator in z-tests and t-tests. **A/B test implication:** To halve standard error, need 4× more samples. This is why A/B tests are expensive.

**Q31: What is the Johnson-Lindenstrauss lemma and why does it matter for high-dimensional embeddings?**
> **JL Lemma:** For any set of $n$ points in high-dimensional space, there exists a random projection $f: \mathbb{R}^d \to \mathbb{R}^k$ where $k = O(\log n / \epsilon^2)$ such that Euclidean distances are preserved to $(1\pm\epsilon)$ factor: $(1-\epsilon)\|u-v\|^2 \leq \|f(u)-f(v)\|^2 \leq (1+\epsilon)\|u-v\|^2$. **Why it matters:** You can randomly project 768-dim BERT embeddings to 50-100 dimensions and preserve approximate nearest-neighbor relationships. This is the mathematical justification for RAG system efficiency — you don't need to search in full 768-dim space; random projection preserves relative distances.

**Q32: What is Lipschitz continuity and why is it important for GAN training?**
> $f$ is L-Lipschitz if $|f(x) - f(y)| \leq L\|x-y\|$ for all $x, y$. **In GANs (Wasserstein GAN):** The discriminator must be 1-Lipschitz. Method: gradient penalty — $\mathcal{L}_{GP} = \lambda E[(|\nabla_{\hat{x}} D(\hat{x})| - 1)^2]$ added to discriminator loss, penalizing gradient norm deviating from 1. **Why:** Wasserstein distance requires discriminator to be 1-Lipschitz for the GAN training signal to be meaningful. Without this constraint, training is unstable (discriminator becomes too powerful, gradient vanishes for generator).

**Q33: Explain the No Free Lunch theorem and its practical implication.**
> **NFL Theorem:** Averaged over all possible data distributions, no ML algorithm outperforms any other. Every algorithm has the same average performance over all problems. **Practical implication:** There is no universally best algorithm — the optimal algorithm depends on the problem structure. When XGBoost consistently outperforms neural networks on tabular fraud data: it's because the inductive biases of XGBoost (axis-aligned splits, piecewise constant functions) happen to match tabular data structure better than neural net's smooth function approximation. NFL mathematically justifies why we must benchmark algorithms on the specific problem at hand.

**Q34: What is Kolmogorov Complexity and why is it the "ideal" data compression?**
> $K(x)$ = length of shortest program that outputs $x$. The ultimate description length. **Connection to ML:** Learning = finding the shortest description of data patterns (Minimum Description Length principle). Regularization = penalizing description length. Overfitting = using unnecessarily long description (memorizing rather than generalizing). **Uncomputable:** Kolmogorov complexity is not computable in general (halting problem). Practical approximations: MDL (Minimum Description Length), Bayesian model selection with prior proportional to code length.

**Q35: Derive the update rule for the Perceptron algorithm and prove its convergence.**
> **Update rule:** If $y_i(w^Tx_i) \leq 0$ (misclassified): $w_{t+1} = w_t + y_i x_i$. **Perceptron convergence theorem:** If data is linearly separable with margin $\gamma$ and $\|x_i\| \leq R$: the algorithm converges in at most $\lfloor R^2/\gamma^2 \rfloor$ updates. **Proof sketch:** Show $w^* \cdot w_t$ increases by at least $\gamma$ per update; $\|w_t\|$ increases by at most $R^2$ per update. After $T$ updates: $\cos\theta \geq T\gamma / (\|w^*\| \sqrt{TR^2}) = \sqrt{T}\gamma/(\|w^*\|R)$. Since $\cos\theta \leq 1$: $T \leq (R\|w^*\|/\gamma)^2$.

**Q36: What happens when features are perfectly multicollinear in linear regression?**
> $X^TX$ becomes singular (non-invertible) → OLS has no unique solution. $\hat{\beta} = (X^TX)^{-1}X^Ty$ fails. **Fix:** Ridge regression adds $\lambda I$: $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$ — always invertible for $\lambda > 0$. **In credit scoring:** Income and annual salary are highly collinear. Standard regression → unstable coefficients (can be arbitrarily large and opposite signs). Ridge shrinks both toward 0 → stable, interpretable coefficients. Alternative: Drop one of the collinear features + PCA to decorrelate.

**Q37: Derive the Gini impurity and prove it's maximized when class distribution is uniform.**
> $G(p) = 1 - \sum_c p_c^2$ for class probabilities $p_1, ..., p_C$ with $\sum_c p_c = 1$.
> **Proof of maximum at uniform distribution:** Use Lagrange multiplier: $\nabla_p G = \lambda \nabla_p(\sum p_c - 1)$. $\frac{\partial G}{\partial p_c} = -2p_c = \lambda$ → all $p_c$ equal → $p_c = 1/C$ for all c. Maximum value: $G_{max} = 1 - C \cdot (1/C)^2 = 1 - 1/C$. For binary (C=2): max Gini = 0.5. **Minimum** (= 0) at pure node: all samples same class.

**Q38: What is the difference between parametric and non-parametric tests? When do you use each?**
> **Parametric:** Assumes distribution (usually Gaussian). t-test, z-test, ANOVA. Use when: sample sizes large (CLT applies) OR distribution assumption verified. More powerful when assumptions hold. **Non-parametric:** No distributional assumption. Mann-Whitney U (compares rank-sum), Kruskal-Wallis, Wilcoxon. Use when: small samples, non-Gaussian distribution (heavy tails), ordinal data. For fraud A/B testing: fraud transaction amounts have heavy tails (few very large frauds) → non-parametric test is safer.

**Q39: What is a sufficient statistic? Give an example.**
> $T(X)$ is sufficient for $\theta$ if $P(X|T(X), \theta) = P(X|T(X))$ — no additional information about $\theta$ in X given T(X). **Example:** For Gaussian $\mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$: $T(x_1,...,x_n) = \bar{x}$ is sufficient for $\mu$. Knowing all individual $x_i$ gives no additional info about $\mu$ beyond knowing $\bar{x}$. **Practical implication:** You can summarize data by its sufficient statistics without loss for parameter estimation — justifies computing summary statistics rather than raw data features in ML preprocessing.

**Q40: Explain bootstrapping and its application in fraud model confidence intervals.**
> **Bootstrapping:** Resample training data WITH REPLACEMENT $B$ times (e.g., B=1000). Train model on each bootstrap sample. Compute metric of interest (AUC, PR-AUC) on each. Confidence interval = [2.5th percentile, 97.5th percentile] of bootstrap distribution. **Why it works:** Bootstrap distribution approximates sampling distribution of the statistic (by CLT analog for bootstrap). **Fraud application:** "Our model achieves AUC 0.91 ± 0.02 (95% CI: [0.87, 0.95])" — bootstrapped CI. Needed when: analytical formula for CI doesn't exist (PR-AUC has complex distribution), or sample size is small.

**Q41: What is the Cramér-Rao lower bound?**
> **CRB:** The variance of any unbiased estimator $\hat{\theta}$ is bounded below: $\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$ where $I(\theta) = E\left[-\frac{\partial^2}{\partial\theta^2}\log P(X|\theta)\right]$ is the Fisher Information. **Achievability:** MLE achieves the CRB asymptotically (for large n) — it's the most efficient unbiased estimator. **In GMM:** The Fisher Information matrix tells you the theoretical limit on how precisely you can estimate cluster parameters — useful for assessing if your fraud cluster estimates have enough resolution.

**Q42: Explain the Markov property and how it's used in your CLV survival analysis.**
> **Markov property:** $P(X_{t+1}|X_0,...,X_t) = P(X_{t+1}|X_t)$ — future state depends only on present, not history. **Cox PH model Markov connection:** The hazard $h(t|x)$ depends only on the current state (feature vector $x$) and time $t$, not on the trajectory of how the patient arrived at this state. This is an implicit Markov assumption. **When it breaks:** Cancer progression is NOT Markovian — prior treatment history matters. Fix: Add time-varying covariates (treatment history as evolving features) to make the model approximately Markov.

**Q43: What is the curse of dimensionality? Give a concrete example for fraud NLP.**
> As dimensionality $d$ increases: (1) Volume of unit ball goes to 0 relative to unit cube → all points are "near" the boundary. (2) All pairwise distances converge → nearest neighbor becomes meaningless. Concretely: For $n=1000$ points uniformly in [0,1]^d, the average distance between any two points = $\sqrt{d/6}$, and the ratio of max to min distance → 1 as d→∞. **Fraud NLP:** If you use raw bag-of-words (50K vocabulary = 50K dimensions) for claim similarity, all claims are equally "far" from each other — cosine similarity becomes uniformly low → RAG retrieval becomes garbage. Fix: Dimensionality reduction (SVD/PCA → 100-300 dims) or learned dense representations (BERT → 768 dims but semantically meaningful).

**Q44: What is backpropagation through time (BPTT) and why is gradient clipping needed?**
> **BPTT:** Unrolling RNN across T time steps, then applying standard backpropagation. Gradient at time t: $\frac{\partial L}{\partial W} = \sum_{t} \frac{\partial L_t}{\partial W}$ where $\frac{\partial L_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{t} W_h^T \text{diag}(\tanh'(z_k))$. **Exploding gradient:** If $\|W_h\| > 1$, product grows exponentially → parameter updates catastrophically large. **Gradient clipping:** If $\|g\| > \text{max\_norm}$: $g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}$. Prevents parameter explosion without eliminating gradient direction. Standard max_norm: 1.0 (BERT) to 5.0 (LSTMs).

**Q45: What is self-supervised learning and how does BERT use it?**
> **Self-supervised:** Labels are generated automatically from the data itself — no human annotation needed. **BERT pre-training objectives:** (1) MLM (Masked Language Model): randomly mask 15% of tokens; predict them. Self-supervised because the original token IS the label. (2) NSP (Next Sentence Prediction): classify if sentence B follows sentence A. Label = is_next/not_next from corpus structure. **Why it's powerful:** Trains on entire text corpus (Wikipedia + Books = billions of tokens) → learns rich language representations without any human labeling. Fine-tuning on small labeled dataset then adapts these representations → data efficiency.

**Q46: Explain the Gumbel-Softmax trick for discrete sampling in neural networks.**
> **Problem:** Sampling from categorical distribution is not differentiable → can't backpropagate through discrete choices (e.g., attention head selection, token sampling in seq2seq). **Gumbel trick:** $z_k = \arg\max_k [\log \pi_k + G_k]$ where $G_k \sim \text{Gumbel}(0,1) = -\log(-\log U)$, $U \sim \text{Uniform}(0,1)$. This samples from categorical distribution with probabilities $\pi_k$. **Gumbel-Softmax (differentiable):** Replace $\arg\max$ with $\text{softmax}$ at temperature $\tau$: $y_k = \frac{\exp((\log\pi_k + G_k)/\tau)}{\sum_j \exp((\log\pi_j + G_j)/\tau)}$. As $\tau \to 0$: Gumbel-Softmax → one-hot (discrete). As $\tau \to \infty$: uniform. Use for: learning discrete structures in NLP, coding agent tool selection as differentiable operation.

**Q47: What is catastrophic forgetting and how does EWC (Elastic Weight Consolidation) prevent it?**
> **Catastrophic forgetting:** When training on task B (fraud claims), model forgets task A (general NLP) — gradients for task B overwrite weights important for task A. **EWC:** Identifies which parameters were important for task A (high Fisher Information). Adds regularization that penalizes changing important parameters:
> $$\mathcal{L}_{EWC} = \mathcal{L}(\text{task B}) + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta^*_{A,i})^2$$
> Where $F_i = E\left[\left(\frac{\partial \log P(y|x,\theta^*_A)}{\partial \theta_i}\right)^2\right]$ (Fisher Information = importance of parameter for task A).
> **Intuition:** Don't change weights that mattered for task A; freely optimize weights that didn't.

**Q48: What is the difference between frequentist p-value and Bayesian credible interval? Common misconception?**
> **p-value:** Probability of seeing data at least as extreme as observed, ASSUMING $H_0$ is true. $P(\text{data} \geq \text{observed} | H_0)$. **Common misconception:** "p < 0.05 means 95% probability that H₁ is true." WRONG. p-value says nothing about probability of hypotheses. **Bayesian credible interval [a,b]:** "95% probability that parameter θ is in [a,b]" — this IS a probability statement about θ. Uses posterior $P(\theta|data)$. **Key difference:** Bayesian requires a prior. Frequentist: no prior needed. **In A/B testing:** Bayesian "P(treatment is better) = 93%" is more actionable than "p=0.04" — directly answers the business question.

**Q49: Derive the update rule for word2vec skip-gram with negative sampling.**
> **Objective:** For each center word $w_c$, predict context words $w_o$:
> $$\mathcal{L} = -\log\sigma(v_{w_o}^T v_{w_c}) - \sum_{k=1}^K \log\sigma(-v_{w_k}^T v_{w_c})$$
> First term: maximize compatibility with positive context. K terms: minimize with random (negative) samples.
> **Gradient w.r.t. center word embedding $v_{w_c}$:**
> $$\frac{\partial \mathcal{L}}{\partial v_{w_c}} = -(1-\sigma(v_{w_o}^T v_{w_c}))v_{w_o} + \sum_k \sigma(v_{w_k}^T v_{w_c})v_{w_k}$$
> **Connection to your fraud embedding:** Your contrastive loss is essentially this negative sampling loss: maximize similarity with positive (same fraud scheme), minimize with negatives (random claims). InfoNCE vs. negative sampling: InfoNCE normalizes over all negatives in batch (softmax denominator) vs. word2vec normalizes over K explicit negatives (sigmoid).

**Q50: What are the 3 assumptions of Cox Proportional Hazards and how do you test each?**
> **Assumption 1: Proportional Hazards** — The hazard ratio between two individuals is CONSTANT over time: $h(t|x_1)/h(t|x_2) = \exp(\beta^T(x_1-x_2))$. Test: Schoenfeld residuals plot — should be random around 0 over time (not trending). Fix if violated: Stratified Cox (different $h_0(t)$ per stratum), or time-varying coefficients.
> **Assumption 2: Log-linearity** — log hazard is LINEAR in covariates. Test: Martingale residuals vs. each covariate — should be flat (not curved). Fix: Transform covariates (log, spline).
> **Assumption 3: Non-informative censoring** — Censored patients are equivalent to still-at-risk patients (not sicker/healthier). Test: Compare baseline characteristics of censored vs. event patients. Fix: Improve data collection (reduce censoring), sensitivity analysis.

---

*End of 50 DMM Questions. Cross-reference: 04_Transformer_Attention_Math.md, 08_Statistical_Testing_AandB.md*
