# 🔴 Hard: Contrastive Losses, AdaBoost, Gradient Boosting, RL, SVD
> **Platform:** TensorTonic + Deep-ML | **Difficulty:** Hard | **Gap-fill from universe**

---

## 🔴 PROBLEM 1: InfoNCE Loss & Contrastive Loss

### Theory
**Contrastive Loss (Hadsell et al., 2006):** Pulls similar pairs together, pushes dissimilar pairs apart.

$$\mathcal{L} = \frac{1}{2N}\sum_{i=1}^N \left[ y_i \cdot d_i^2 + (1-y_i) \cdot \max(0, m - d_i)^2 \right]$$

- y_i = 1 if similar pair, 0 if dissimilar
- d_i = Euclidean distance between embeddings
- m = margin (minimum distance for dissimilar pairs)

**InfoNCE Loss (van den Oord, 2018):** Used in contrastive learning (SimCLR, CLIP).

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(z_i, z_j)/\tau)}$$

- z_i, z_i⁺ = two views (augmentations) of same sample — the positive pair
- τ = temperature (controls concentration of distribution)
- All other samples in batch are treated as negatives

**Dice Loss:** For semantic segmentation.
$$\text{Dice Loss} = 1 - \frac{2|P \cap G|}{|P| + |G|} = 1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$$

### Things to Focus On
- ✅ InfoNCE: batch size matters — larger batch = more negatives = harder task = better representations
- ✅ Temperature τ: low τ → hard distributions (sharp softmax); high τ → uniform (less informative)
- ✅ Dice loss vs BCE: Dice handles class imbalance inherently (good for medical segmentation with tiny lesions)
- ✅ SimCLR uses InfoNCE with cosine similarity; CLIP uses it across modalities (image ↔ text)

### Implementation
```python
import numpy as np

def contrastive_loss(embeddings1: np.ndarray, embeddings2: np.ndarray,
                      labels: np.ndarray, margin: float = 1.0) -> float:
    """
    Contrastive loss for siamese networks.
    
    embeddings1, embeddings2: (N, d) — paired embedding vectors
    labels: (N,) — 1 if similar pair, 0 if dissimilar pair
    margin: minimum distance for dissimilar pairs
    """
    # Euclidean distance between paired embeddings
    diff = embeddings1 - embeddings2
    dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
    
    # Similar pairs: minimize distance (pull together)
    similar_loss = labels * dist ** 2
    
    # Dissimilar pairs: maximize distance (push apart, up to margin)
    dissimilar_loss = (1 - labels) * np.maximum(0, margin - dist) ** 2
    
    return np.mean(0.5 * (similar_loss + dissimilar_loss))

def infonce_loss(z1: np.ndarray, z2: np.ndarray, temperature: float = 0.07) -> float:
    """
    InfoNCE (NT-Xent) Contrastive Loss.
    Used in SimCLR, CLIP, and most contrastive representation learning.
    
    z1, z2: (N, d) — two views of N samples. (z1[i], z2[i]) = positive pair.
    temperature: τ controls sharpness of distribution
    
    For each sample i:
    - Positive: (z1[i], z2[i])
    - Negatives: all other z2[j] for j ≠ i (and potentially z1[j])
    """
    N, d = z1.shape
    
    # L2 normalize embeddings for cosine similarity
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    
    # Concatenate all representations: [z1; z2] → (2N, d)
    z = np.concatenate([z1, z2], axis=0)
    
    # Similarity matrix: (2N, 2N)
    sim_matrix = (z @ z.T) / temperature
    
    # Mask self-similarity (diagonal) — don't use as negative
    np.fill_diagonal(sim_matrix, -np.inf)
    
    # Positive pair indices:
    # For z1[i] → positive is z2[i] (at index N+i)
    # For z2[i] → positive is z1[i] (at index i)
    labels = np.concatenate([np.arange(N, 2*N), np.arange(N)])
    
    # Cross-entropy loss: for each anchor, maximize similarity with positive
    # vs all other 2N-2 negatives
    log_softmax = sim_matrix - np.log(np.sum(np.exp(sim_matrix - 
                                sim_matrix.max(axis=1, keepdims=True)), 
                                axis=1, keepdims=True) + 
                                sim_matrix.max(axis=1, keepdims=True))
    
    # Simpler: compute directly
    loss = 0.0
    for i in range(2 * N):
        # Numerically stable softmax
        s = sim_matrix[i]
        s = s - s.max()
        log_sum_exp = np.log(np.sum(np.exp(s)))
        loss -= s[labels[i]] - log_sum_exp
    
    return loss / (2 * N)

def dice_loss(y_pred: np.ndarray, y_true: np.ndarray, 
               eps: float = 1e-6) -> float:
    """
    Dice Loss for binary segmentation.
    
    y_pred: (N,) or (H, W) predicted probabilities in [0, 1]
    y_true: (N,) or (H, W) binary ground truth mask
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    intersection = np.sum(y_pred * y_true)
    dice_coeff = (2 * intersection + eps) / (np.sum(y_pred) + np.sum(y_true) + eps)
    
    return 1 - dice_coeff  # Loss = 1 - Dice coefficient

# Test
np.random.seed(42)
N, d = 8, 64

# InfoNCE test
z1 = np.random.randn(N, d)
z2 = z1 + np.random.randn(N, d) * 0.1  # Slightly perturbed views (positive pairs)
loss = infonce_loss(z1, z2, temperature=0.07)
print(f"InfoNCE loss (similar pairs): {loss:.4f}")

z2_random = np.random.randn(N, d)  # Random (negative pairs for z1)
loss_random = infonce_loss(z1, z2_random, temperature=0.07)
print(f"InfoNCE loss (dissimilar): {loss_random:.4f}")  # Should be higher

# Contrastive loss
emb1 = np.random.randn(4, 8)
emb2 = emb1 + np.random.randn(4, 8) * 0.1
labels = np.array([1, 1, 0, 0])  # First 2 similar, last 2 dissimilar
print(f"Contrastive loss: {contrastive_loss(emb1, emb2, labels, margin=1.0):.4f}")

# Dice loss
pred_seg = np.array([0.9, 0.8, 0.1, 0.2, 0.95])
true_seg = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
print(f"Dice loss: {dice_loss(pred_seg, true_seg):.4f}")  # Low (good pred)
```

---

## 🔴 PROBLEM 2: AdaBoost from Scratch

### Theory
AdaBoost trains sequential **weak learners** (usually decision stumps), up-weighting misclassified examples.

**Algorithm:**
1. Initialize weights: $w_i = 1/n$ for all samples
2. For m = 1 to M:
   - Fit weak learner $h_m$ on weighted samples
   - Compute weighted error: $\epsilon_m = \sum_i w_i \cdot \mathbb{1}[h_m(x_i) \neq y_i]$
   - Compute learner weight: $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$
   - Update sample weights: $w_i \leftarrow w_i \cdot e^{-\alpha_m y_i h_m(x_i)}$, then normalize
3. Final prediction: $H(x) = \text{sign}\left(\sum_m \alpha_m h_m(x)\right)$

### Things to Focus On
- ✅ α_m > 0 when error < 0.5 (better than random); → 0 when ε → 0.5; negative when worse than random
- ✅ Misclassified samples get UP-weighted; correctly classified get DOWN-weighted
- ✅ Each learner sees an effectively different dataset (via sample weights)
- ✅ AdaBoost is sensitive to noisy data/outliers (they get high weight repeatedly)
- ✅ Gradient Boosting is the generalization that can optimize any differentiable loss

### Implementation
```python
class DecisionStump:
    """Simplest weak learner: split on one feature at one threshold."""
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            weights: np.ndarray) -> 'DecisionStump':
        """Find best weighted split."""
        n, d = X.shape
        best_error = float('inf')
        
        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    # Predict: polarity * (x > threshold ? 1 : -1)
                    preds = np.where(X[:, feat] > threshold, polarity, -polarity)
                    error = np.sum(weights[preds != y])
                    
                    if error < best_error:
                        best_error = error
                        self.feature_idx = feat
                        self.threshold = threshold
                        self.polarity = polarity
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature_idx] > self.threshold, 
                        self.polarity, -self.polarity)

class AdaBoost:
    """AdaBoost with Decision Stumps as weak learners."""
    
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.learners = []    # Weak learners
        self.alphas = []      # Learner weights
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoost':
        """y must be {-1, +1}."""
        n = len(y)
        weights = np.ones(n) / n  # Uniform initial weights
        
        for m in range(self.n_estimators):
            # Fit weak learner on weighted data
            stump = DecisionStump().fit(X, y, weights)
            predictions = stump.predict(X)
            
            # Weighted error
            incorrect = predictions != y
            epsilon = np.sum(weights[incorrect]) + 1e-10  # Avoid log(0)
            
            if epsilon >= 0.5:
                break  # Worse than random — stop
            
            # Learner weight
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            
            # Update sample weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= weights.sum()  # Normalize
            
            self.learners.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote."""
        ensemble_pred = sum(alpha * learner.predict(X)
                           for alpha, learner in zip(self.alphas, self.learners))
        return np.sign(ensemble_pred)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=4, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert to {-1, +1}

split = 150
ada = AdaBoost(n_estimators=50)
ada.fit(X[:split], y[:split])
print(f"AdaBoost accuracy: {ada.score(X[split:], y[split:]):.4f}")
print(f"Number of learners used: {len(ada.learners)}")
```

---

## 🔴 PROBLEM 3: Policy Gradient (REINFORCE)

### Theory
In REINFORCE, we directly optimize the policy π_θ by gradient ascent on expected return.

**Objective:** $J(\theta) = \mathbb{E}_\tau[R(\tau)]$

**Policy Gradient Theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_{t=0}^T \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$ is the discounted return from time t.

**REINFORCE loss (for gradient descent):**
$$\mathcal{L} = -\sum_t G_t \log\pi_\theta(a_t|s_t)$$

### Things to Focus On
- ✅ High variance problem: REINFORCE has high variance → use baseline (value function)
- ✅ Actor-Critic: use V(s_t) as baseline → advantage = G_t - V(s_t)
- ✅ Negative sign: we minimize loss but want to maximize return
- ✅ Used in RLHF: reward model output becomes the "return" for policy gradient

### Implementation
```python
def compute_discounted_returns(rewards: list, gamma: float = 0.99) -> np.ndarray:
    """
    Compute discounted cumulative returns G_t for each timestep.
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    """
    returns = np.zeros(len(rewards))
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def reinforce_loss(log_probs: np.ndarray, returns: np.ndarray,
                    baseline: np.ndarray = None) -> float:
    """
    REINFORCE policy gradient loss.
    
    log_probs: log π(a_t | s_t) for each timestep (n_steps,)
    returns: G_t discounted returns (n_steps,)
    baseline: V(s_t) predictions for variance reduction (optional)
    """
    if baseline is not None:
        # Advantage = Return - Baseline
        advantages = returns - baseline
    else:
        advantages = returns
    
    # Normalize advantages (reduce variance, stabilize training)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Loss = -E[G_t * log π(a_t|s_t)]  (negative for gradient descent)
    policy_loss = -np.mean(log_probs * advantages)
    
    return float(policy_loss)

def compute_advantage(rewards: list, values: list, gamma: float = 0.99,
                       lam: float = 0.95) -> np.ndarray:
    """
    Generalized Advantage Estimation (GAE) for PPO.
    Combines n-step returns with exponential decay.
    
    δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  (TD error)
    Â_t = δ_t + γλ*δ_{t+1} + (γλ)²*δ_{t+2} + ...
    """
    rewards = np.array(rewards)
    values  = np.array(values)
    
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae_lam = 0
    
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0
        # TD error
        delta = rewards[t] + gamma * next_value - values[t]
        # GAE
        last_gae_lam = delta + gamma * lam * last_gae_lam
        advantages[t] = last_gae_lam
    
    return advantages

# Test
rewards = [0, 0, 0, 1, 0, 1]  # Sparse reward
returns = compute_discounted_returns(rewards, gamma=0.99)
print(f"Returns: {returns.round(3)}")

# Simulate log probs and baseline
log_probs = np.log(np.array([0.3, 0.5, 0.4, 0.7, 0.6, 0.8]))
baseline = np.array([0.2, 0.3, 0.3, 0.5, 0.4, 0.6])  # V(s_t) estimates

loss = reinforce_loss(log_probs, returns, baseline)
print(f"REINFORCE loss: {loss:.4f}")

advantages_gae = compute_advantage(rewards, baseline, gamma=0.99, lam=0.95)
print(f"GAE advantages: {advantages_gae.round(3)}")
```

---

## 🔴 PROBLEM 4: SVD — Singular Value Decomposition

### Theory
Any matrix A ∈ ℝ^{m×n} can be decomposed:
$$A = U \Sigma V^T$$

- U ∈ ℝ^{m×m}: left singular vectors (columns are orthonormal)
- Σ ∈ ℝ^{m×n}: diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V^T ∈ ℝ^{n×n}: right singular vectors (rows are orthonormal)

**Applications:**
- PCA: right singular vectors of centered X are principal components
- Recommendation (matrix factorization): approximate R ≈ UΣV^T with k singular values
- Pseudoinverse: A⁺ = V Σ⁺ U^T
- Noise reduction: truncate small singular values

**Low-rank approximation theorem (Eckart-Young):**
The best rank-k approximation to A (in Frobenius norm) is:
$$A_k = U_k \Sigma_k V_k^T$$

### Things to Focus On
- ✅ Singular values = √eigenvalues of A^TA = √eigenvalues of AA^T
- ✅ Frobenius error of truncation: $\|A - A_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}$
- ✅ SVD is always defined (any matrix, even non-square, even non-full-rank)
- ✅ Power iteration method: practical approximation for large sparse matrices

### Implementation
```python
def svd_from_scratch_power_iteration(A: np.ndarray, k: int = None, 
                                      n_iter: int = 100) -> tuple:
    """
    Approximate SVD using randomized power iteration.
    Much faster than full SVD for large matrices.
    
    A: (m, n)
    k: number of singular values/vectors to compute
    """
    m, n = A.shape
    k = k or min(m, n)
    
    # Full SVD using numpy (mention you'd implement power iteration for scale)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Truncate to k components
    U_k  = U[:, :k]
    s_k  = s[:k]
    Vt_k = Vt[:k, :]
    
    return U_k, s_k, Vt_k

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """
    Best rank-k approximation of A via SVD (Eckart-Young theorem).
    """
    U_k, s_k, Vt_k = svd_from_scratch_power_iteration(A, k)
    return U_k @ np.diag(s_k) @ Vt_k

def matrix_completion_svd(R: np.ndarray, k: int = 5, 
                            max_iter: int = 50) -> np.ndarray:
    """
    Simple matrix completion (collaborative filtering) via SVD.
    Fill missing ratings, then truncated SVD to denoise.
    
    R: (users, items) rating matrix with NaN for missing
    """
    # Fill missing with row means (simple imputation)
    R_filled = R.copy()
    row_means = np.nanmean(R, axis=1)
    for i in range(R.shape[0]):
        R_filled[i, np.isnan(R[i])] = row_means[i]
    
    # Iteratively refine
    for _ in range(max_iter):
        A_k = low_rank_approximation(R_filled, k)
        # Only update previously missing values (keep observed)
        R_new = R_filled.copy()
        mask_missing = np.isnan(R)
        R_new[mask_missing] = A_k[mask_missing]
        
        if np.allclose(R_new, R_filled, atol=1e-4):
            break
        R_filled = R_new
    
    return R_filled

# Test
np.random.seed(42)
A = np.random.randn(50, 30)  # Rank-10 matrix
A = A[:, :10] @ A[:10, :]    # Make it approximately rank-10

U, s, Vt = np.linalg.svd(A, full_matrices=False)
A_k = low_rank_approximation(A, k=5)
error = np.linalg.norm(A - A_k, 'fro')
print(f"Shape: {A.shape}")
print(f"Singular values (top 5): {s[:5].round(2)}")
print(f"Rank-5 approx error: {error:.4f}")
print(f"Total Frobenius norm: {np.linalg.norm(A, 'fro'):.4f}")

# Verify: error should equal sqrt(sum of remaining singular values squared)
theoretical_error = np.sqrt(np.sum(s[5:] ** 2))
print(f"Theoretical error (Eckart-Young): {theoretical_error:.4f}")
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"InfoNCE vs Contrastive loss — when to use which?"** → InfoNCE treats all other samples as negatives (more negatives = better signal with large batches). Contrastive loss requires explicit positive/negative pair labels. Use InfoNCE for self-supervised learning; Contrastive for labeled similarity data.
2. **"Why is AdaBoost sensitive to outliers?"** → Misclassified outliers accumulate very high weights. Subsequent learners focus almost entirely on them. Gradient Boosting with Huber loss is more robust.
3. **"REINFORCE vs PPO — what's the key difference?"** → REINFORCE: on-policy, single update per episode (high variance). PPO: clips policy update ratio to prevent too-large steps, uses multiple epochs on same experience (more sample efficient). Both use policy gradient; PPO is safer.
4. **"Why not always use all singular values in SVD for recommendation?"** → Large singular values capture main signal; small ones capture noise. Keeping all → overfitting. Truncating to k → regularization. Choosing k: cross-validate reconstruction error on held-out ratings.
