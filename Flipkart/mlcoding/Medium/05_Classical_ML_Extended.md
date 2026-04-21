# 🟡 Medium: Extended Classical ML — Naive Bayes, Ridge/Lasso, IoU, BLEU, Layer Norm, Data Drift
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Medium | **Gap-fill from universe**

---

## 🟡 PROBLEM 1: Gaussian Naive Bayes

### Theory
Naïve Bayes applies Bayes' theorem assuming features are **conditionally independent** given class:

$$P(y | x_1,...,x_d) \propto P(y) \prod_{j=1}^d P(x_j | y)$$

**Gaussian Naive Bayes:** Each feature is normally distributed given class.
$$P(x_j | y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

**Training:** Estimate μ_{jc} and σ²_{jc} from training data per class.
**Prediction:** Compute log-posterior for each class, take argmax.

### Things to Focus On
- ✅ Use LOG probabilities to avoid numerical underflow (product of many small numbers)
- ✅ Gaussian NB, Bernoulli NB (binary), Multinomial NB (counts) — know which to use when
- ✅ Laplace smoothing for zero-probability features
- ✅ Works well in high-dimensional spaces (text, genomics) despite naive assumption
- ✅ Fast: O(n×d) training, O(C×d) per prediction

### Implementation
```python
import numpy as np
from typing import List

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes from scratch.
    Assumes each feature ~ N(mu, sigma²) per class.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing  # Prevents zero variance
        self.classes_ = None
        self.class_prior_ = None  # P(y=c)
        self.theta_ = None        # Mean of each feature per class
        self.sigma_ = None        # Variance of each feature per class
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        n, d = X.shape
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        
        self.class_prior_ = np.zeros(C)
        self.theta_ = np.zeros((C, d))   # Class means
        self.sigma_ = np.zeros((C, d))   # Class variances
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[i] = len(X_c) / n
            self.theta_[i] = X_c.mean(axis=0)
            self.sigma_[i] = X_c.var(axis=0) + self.var_smoothing
        
        return self
    
    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(x_j | y=c) for each class."""
        n = len(X)
        C = len(self.classes_)
        log_likelihoods = np.zeros((n, C))
        
        for i in range(C):
            mu  = self.theta_[i]    # (d,)
            var = self.sigma_[i]    # (d,)
            
            # Log Gaussian: -0.5 * log(2π*σ²) - (x-μ)²/(2σ²)
            log_ll = -0.5 * np.log(2 * np.pi * var) - \
                     0.5 * ((X - mu) ** 2) / var
            
            log_likelihoods[:, i] = log_ll.sum(axis=1)  # Sum over features
        
        return log_likelihoods
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Log posterior: log P(y|x) ∝ log P(y) + sum log P(x_j|y)"""
        log_prior = np.log(self.class_prior_)
        log_ll = self._log_likelihood(X)
        return log_prior + log_ll  # (n, C)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalized posterior probabilities."""
        log_posterior = self.predict_log_proba(X)
        # Log-sum-exp for stability
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum(axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_log_proba(X).argmax(axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

gnb = GaussianNaiveBayes()
gnb.fit(X, y)
print(f"Accuracy: {gnb.score(X, y):.4f}")

# Compare with sklearn
from sklearn.naive_bayes import GaussianNB
sklearn_gnb = GaussianNB().fit(X, y)
print(f"sklearn accuracy: {sklearn_gnb.score(X, y):.4f}")
```

---

## 🟡 PROBLEM 2: Ridge (L2) and Lasso (L1) Regression

### Theory
**Ridge (L2 regularization):**
$$\hat{w}_{\text{ridge}} = \arg\min_w \|Xw - y\|_2^2 + \lambda\|w\|_2^2$$

Closed form: $\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$

Shrinks all coefficients toward zero. Never sets them exactly to zero.

**Lasso (L1 regularization):**
$$\hat{w}_{\text{lasso}} = \arg\min_w \|Xw - y\|_2^2 + \lambda\|w\|_1$$

No closed form. Solved via coordinate descent or subgradient methods.
Produces **sparse solutions** — sets some coefficients exactly to zero → feature selection!

**Elastic Net:** $\alpha\|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2$ — combines both.

### Things to Focus On
- ✅ Ridge = can still use all features (never zero); Lasso = automatic feature selection
- ✅ Ridge has closed form; Lasso does not (use coordinate descent)
- ✅ Multicollinearity: Ridge handles better (distributes weight)
- ✅ High-dimensional (d >> n): both help; Lasso better for sparse ground truth
- ✅ λ tuned via cross-validation

### Implementation
```python
class RidgeRegression:
    """Ridge Regression: OLS + L2 penalty. Has closed-form solution."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.b = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        n, d = X.shape
        # Augment X with bias column
        X_aug = np.hstack([X, np.ones((n, 1))])
        
        # Ridge: (X^TX + αI)^{-1} X^T y
        # Don't regularize the bias term (last diagonal element)
        A = X_aug.T @ X_aug
        A[:d, :d] += self.alpha * np.eye(d)
        
        w_aug = np.linalg.solve(A, X_aug.T @ y)
        self.w = w_aug[:d]
        self.b = w_aug[d]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

class LassoRegression:
    """
    Lasso Regression: OLS + L1 penalty.
    Solved via coordinate descent (no closed form due to |w| non-differentiability).
    """
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
    
    def _soft_threshold(self, z: float, threshold: float) -> float:
        """
        Soft thresholding operator: solution to argmin_w 0.5(w-z)² + λ|w|
        S(z, λ) = sign(z) * max(|z| - λ, 0)
        """
        if z > threshold:
            return z - threshold
        elif z < -threshold:
            return z + threshold
        else:
            return 0.0  # L1 penalty sets small coefficients EXACTLY to zero
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        
        for iteration in range(self.max_iter):
            w_old = self.w.copy()
            
            # Coordinate descent: optimize one weight at a time
            for j in range(d):
                # Partial residual: residual without feature j's contribution
                y_pred_without_j = X @ self.w + self.b - X[:, j] * self.w[j]
                residual_j = y - y_pred_without_j
                
                # OLS coefficient for feature j (no penalty)
                z_j = (X[:, j] @ residual_j) / (X[:, j] @ X[:, j] + 1e-8)
                
                # Apply soft thresholding (L1 penalty)
                threshold = self.alpha / (X[:, j] @ X[:, j] + 1e-8)
                self.w[j] = self._soft_threshold(z_j, threshold)
            
            # Update bias (unpenalized)
            self.b = (y - X @ self.w).mean()
            
            # Convergence check
            if np.max(np.abs(self.w - w_old)) < self.tol:
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

# Test — feature selection demo
np.random.seed(42)
n, d = 100, 20
X = np.random.randn(n, d)
# Only features 0, 3, 7 are truly predictive
y = 2*X[:,0] - 1.5*X[:,3] + 0.8*X[:,7] + np.random.randn(n) * 0.5

ridge = RidgeRegression(alpha=1.0).fit(X, y)
lasso = LassoRegression(alpha=0.1).fit(X, y)

print("Ridge weights (non-zero):", np.sum(np.abs(ridge.w) > 0.01))  # All ~20
print("Lasso weights (non-zero):", np.sum(np.abs(lasso.w) > 0.01))  # ~3 (sparse!)
print("True features: 0, 3, 7")
print("Lasso top features:", np.where(np.abs(lasso.w) > 0.01)[0])
```

---

## 🟡 PROBLEM 3: Layer Normalization

### Theory
Layer Normalization normalizes across the **feature dimension** for each sample (vs BatchNorm which normalizes across the batch dimension for each feature).

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where μ and σ² are computed over the feature dimension for each sample independently.

**BN vs LN:**
| | Batch Norm | Layer Norm |
|---|---|---|
| Normalize over | Batch dim (N) | Feature dim (C) |
| Works with batch_size=1 | ❌ | ✅ |
| Used in | CNNs | Transformers, RNNs |
| Running statistics | Yes (for inference) | No (same formula train/test) |

### Things to Focus On
- ✅ LN computes statistics PER SAMPLE (not across batch) → works for any batch size
- ✅ No running statistics needed → same computation in training and inference
- ✅ Used in every Transformer layer (BERT, GPT, T5)
- ✅ Applied BEFORE attention and FFN in Pre-LN transformers (more stable training)

### Implementation
```python
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization: normalize per sample over feature dimension.
    
    x:     (batch, features) or (batch, seq_len, features)
    gamma: (features,) learnable scale
    beta:  (features,) learnable shift
    """
    # Mean and variance computed per sample (last dimension)
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta

class LayerNorm:
    """Layer Norm module with learnable parameters."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)   # Scale (learnable)
        self.beta  = np.zeros(normalized_shape)  # Shift (learnable)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return layer_norm(x, self.gamma, self.beta, self.eps)
    
    def __call__(self, x):
        return self.forward(x)

# Test — compare BN and LN
np.random.seed(42)
x = np.random.randn(4, 8)  # batch=4, features=8

ln = LayerNorm(8)
out_ln = ln(x)

print(f"LN output per-sample mean: {out_ln.mean(axis=-1).round(4)}")  # ~[0,0,0,0]
print(f"LN output per-sample std:  {out_ln.std(axis=-1).round(4)}")   # ~[1,1,1,1]

# For a sequence (transformer use case)
x_seq = np.random.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
ln_seq = LayerNorm(64)
out_seq = ln_seq(x_seq)
print(f"Seq LN shape: {out_seq.shape}")  # (2, 10, 64) ✓
```

---

## 🟡 PROBLEM 4: IoU — Intersection over Union (Bounding Boxes)

### Theory
$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Intersection area}}{\text{Union area}}$$

**Box format:** [x_min, y_min, x_max, y_max] or [x_center, y_center, width, height]

**Applications:**
- Object detection: NMS (Non-Maximum Suppression) uses IoU to merge overlapping boxes
- Evaluation: mAP@0.5 means "count a detection as TP if IoU > 0.5 with any GT box"
- Segmentation: Dice coefficient = 2×IoU/(1+IoU) (roughly)

### Things to Focus On
- ✅ IoU = 0: no overlap; 1: perfect overlap
- ✅ GIoU, DIoU, CIoU: improved variants that handle non-overlapping boxes better
- ✅ NMS algorithm: sort by score → keep highest → suppress overlapping boxes with IoU > threshold

### Implementation
```python
def iou_bounding_boxes(box1: list, box2: list) -> float:
    """
    Compute IoU between two bounding boxes.
    Format: [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Intersection area (clamp to 0 if no overlap)
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    # Individual areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Union area
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def non_maximum_suppression(boxes: list, scores: list, 
                              iou_threshold: float = 0.5) -> list:
    """
    NMS: suppress redundant bounding boxes.
    
    boxes: list of [x_min, y_min, x_max, y_max]
    scores: confidence score for each box
    Returns: indices of kept boxes
    """
    if not boxes:
        return []
    
    # Sort by score descending
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while sorted_idx:
        # Take highest-scoring remaining box
        best = sorted_idx.pop(0)
        keep.append(best)
        
        # Remove boxes with IoU > threshold (suppress overlapping)
        sorted_idx = [
            i for i in sorted_idx
            if iou_bounding_boxes(boxes[best], boxes[i]) <= iou_threshold
        ]
    
    return keep

# Test
boxes = [
    [10, 10, 50, 50],  # Box A
    [15, 15, 55, 55],  # Box B (overlaps A strongly)
    [100, 100, 150, 150],  # Box C (separate region)
]
scores = [0.9, 0.8, 0.95]

iou_ab = iou_bounding_boxes(boxes[0], boxes[1])
print(f"IoU(A, B) = {iou_ab:.4f}")  # High (overlapping)
print(f"IoU(A, C) = {iou_bounding_boxes(boxes[0], boxes[2]):.4f}")  # 0 (no overlap)

kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
print(f"NMS kept boxes: {kept}")  # Should keep C (0.95) and A (0.9), suppress B
```

---

## 🟡 PROBLEM 5: BLEU Score

### Theory
BLEU (Bilingual Evaluation Understudy) — standard metric for machine translation and text generation.

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

Where:
- $p_n$ = modified n-gram precision (clipped by reference count)
- $w_n$ = weight (usually 1/N for each n-gram size)
- BP = brevity penalty (penalizes short translations)

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1-r/c} & \text{if } c \leq r \end{cases}$$

### Things to Focus On
- ✅ "Clipped" n-gram precision: can't count a word more times than it appears in reference
- ✅ Brevity penalty prevents trivially short translations scoring high
- ✅ BLEU is corpus-level metric; sentence-level BLEU is unreliable
- ✅ Alternatives: ROUGE (recall-focused, used for summarization), BERTScore (semantic)

### Implementation
```python
from collections import Counter
import math

def get_ngrams(tokens: list, n: int) -> Counter:
    """Extract n-grams as Counter."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def clipped_precision(hypothesis: list, references: list, n: int) -> float:
    """
    Modified n-gram precision with clipping.
    Each n-gram credited at most as many times as it appears in any reference.
    """
    hyp_ngrams = get_ngrams(hypothesis, n)
    
    # Maximum count from any reference
    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
    
    # Clipped counts
    clipped_count = sum(min(count, max_ref_counts.get(ngram, 0)) 
                        for ngram, count in hyp_ngrams.items())
    total_count = sum(hyp_ngrams.values())
    
    return clipped_count / total_count if total_count > 0 else 0.0

def bleu_score(hypothesis: str, references: list, max_n: int = 4) -> float:
    """
    Compute BLEU score.
    
    hypothesis: model-generated text (string)
    references: list of reference texts (list of strings)
    max_n: max n-gram size (usually 4 for BLEU-4)
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    c = len(hyp_tokens)
    r = min(len(ref), key=lambda ref: abs(len(ref) - c) 
            for ref in ref_tokens_list).__len__() if ref_tokens_list else 0
    # Simplified: pick reference closest in length
    r = min((abs(len(ref) - c), len(ref)) for ref in ref_tokens_list)[1]
    bp = 1.0 if c > r else math.exp(1 - r/c)
    
    # Geometric mean of n-gram precisions (using log to avoid underflow)
    log_avg = 0.0
    for n in range(1, max_n + 1):
        p_n = clipped_precision(hyp_tokens, ref_tokens_list, n)
        if p_n == 0:
            return 0.0  # Any zero precision → BLEU = 0
        log_avg += (1.0 / max_n) * math.log(p_n)
    
    return bp * math.exp(log_avg)

# Test
hypothesis = "the cat sat on the mat"
references = ["the cat is on the mat", "there is a cat on the mat"]

score = bleu_score(hypothesis, references, max_n=4)
print(f"BLEU-4: {score:.4f}")

# Perfect match
perfect_score = bleu_score(references[0], references, max_n=4)
print(f"Perfect: {perfect_score:.4f}")  # Should be 1.0
```

---

## 🟡 PROBLEM 6: Data Drift Detection — PSI and KL Divergence

### Theory
**Population Stability Index (PSI):**
$$\text{PSI} = \sum_i (P_i - Q_i) \ln\frac{P_i}{Q_i}$$

PSI < 0.1: No drift | 0.1–0.2: Minor | > 0.2: Significant drift (retrain!)

**KL Divergence (Kullback-Leibler):**
$$\text{KL}(P \| Q) = \sum_i P_i \log\frac{P_i}{Q_i}$$

Asymmetric: measures "extra bits to encode P using Q's distribution."

**Jensen-Shannon Divergence (symmetric):**
$$\text{JSD}(P, Q) = \frac{1}{2}\text{KL}(P\|M) + \frac{1}{2}\text{KL}(Q\|M), \quad M = \frac{P+Q}{2}$$

### Things to Focus On
- ✅ PSI: industry standard for model monitoring (especially in banking/fintech)
- ✅ KL: asymmetric — order matters! KL(train||prod) ≠ KL(prod||train)
- ✅ JSD: symmetric, bounded in [0, log(2)] — better for comparison
- ✅ Chi-squared test: statistical significance test for drift
- ✅ Binning continuous features before PSI (typically 10 bins)

### Implementation
```python
def compute_psi(expected: np.ndarray, actual: np.ndarray, 
                n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    
    expected: distribution from training/baseline period
    actual:   distribution from current/production period
    n_bins:   number of bins for discretization
    
    PSI < 0.1:  No significant drift
    PSI 0.1-0.2: Minor drift, monitor closely
    PSI > 0.2:  Major drift, investigate/retrain
    """
    # Create bins from combined data range
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    
    def bin_distribution(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Bin data and get proportions."""
        counts, _ = np.histogram(data, bins=bins)
        proportions = counts / len(data)
        # Avoid zero proportions (PSI undefined)
        return np.clip(proportions, 1e-4, None)
    
    expected_dist = bin_distribution(expected, bins)
    actual_dist   = bin_distribution(actual, bins)
    
    # Normalize to valid distributions
    expected_dist /= expected_dist.sum()
    actual_dist   /= actual_dist.sum()
    
    # PSI = sum((actual - expected) * ln(actual/expected))
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    return float(psi)

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL Divergence: KL(P||Q) = sum(P * log(P/Q))
    Measures how much P differs from Q.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Clip to avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    
    return float(np.sum(p * np.log(p / q)))

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD: symmetric version of KL. Range [0, log(2)]."""
    p = np.asarray(p, dtype=float) / np.sum(p)
    q = np.asarray(q, dtype=float) / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Test — simulate scoring drift
np.random.seed(42)
train_scores = np.random.beta(2, 5, 10000)      # Training distribution (right-skewed)
prod_scores_ok = np.random.beta(2, 5, 10000)    # Same distribution (no drift)
prod_scores_drift = np.random.beta(3, 3, 10000) # Different distribution (drift!)

psi_ok    = compute_psi(train_scores, prod_scores_ok)
psi_drift = compute_psi(train_scores, prod_scores_drift)

print(f"PSI (no drift):   {psi_ok:.4f}   → {'OK' if psi_ok < 0.1 else 'DRIFT'}")
print(f"PSI (with drift): {psi_drift:.4f} → {'OK' if psi_drift < 0.1 else 'DRIFT!'}")
```

---

## 🟡 PROBLEM 7: ETL DAG Dependency Ordering (Topological Sort)

### Theory
In ML pipelines (Airflow, Kubeflow), tasks have dependencies. We need to find a valid execution order — this is **topological sort** on a directed acyclic graph (DAG).

**Kahn's Algorithm:**
1. Find all nodes with in-degree 0 (no dependencies) → add to queue
2. Process each: remove from graph, decrement neighbors' in-degree
3. If neighbor's in-degree becomes 0 → add to queue
4. If all nodes processed → valid order; otherwise cycle exists

### Things to Focus On
- ✅ Cycle detection: if topological sort doesn't include all nodes → cycle exists → pipeline error
- ✅ Multiple valid orderings are possible
- ✅ For parallelism: process all in-degree-0 nodes simultaneously (each "level")
- ✅ Used in: Airflow DAG validation, Spark query planning, dbt model ordering

### Implementation
```python
from collections import deque, defaultdict
from typing import Dict, List, Optional

def topological_sort(tasks: List[str], dependencies: Dict[str, List[str]]) -> Optional[List[str]]:
    """
    Kahn's topological sort for ETL DAG ordering.
    
    tasks: list of all task names
    dependencies: {task: [list of tasks that must run BEFORE this task]}
    
    Returns: execution order, or None if cycle detected
    """
    # Build adjacency list and in-degree count
    in_degree = {task: 0 for task in tasks}
    adj = defaultdict(list)
    
    for task, deps in dependencies.items():
        for dep in deps:
            adj[dep].append(task)   # dep must run before task
            in_degree[task] += 1
    
    # Start with tasks that have no dependencies
    queue = deque([task for task in tasks if in_degree[task] == 0])
    order = []
    
    while queue:
        task = queue.popleft()
        order.append(task)
        
        # "Remove" this task: decrement in-degree of dependents
        for dependent in adj[task]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    # If not all tasks processed → cycle exists
    if len(order) != len(tasks):
        return None  # Cycle detected!
    
    return order

def get_parallel_stages(tasks: List[str], dependencies: Dict[str, List[str]]) -> List[List[str]]:
    """
    Group tasks into parallel execution stages (level-by-level).
    Tasks in same stage can run concurrently.
    """
    in_degree = {task: 0 for task in tasks}
    adj = defaultdict(list)
    
    for task, deps in dependencies.items():
        for dep in deps:
            adj[dep].append(task)
            in_degree[task] += 1
    
    stages = []
    queue = deque([task for task in tasks if in_degree[task] == 0])
    
    while queue:
        stage = list(queue)  # All current zero-in-degree tasks = parallel stage
        queue.clear()
        stages.append(stage)
        
        for task in stage:
            for dep in adj[task]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
    
    return stages

# Test — ML pipeline example
tasks = ['raw_data', 'feature_a', 'feature_b', 'feature_c', 'model', 'evaluation']
dependencies = {
    'feature_a': ['raw_data'],
    'feature_b': ['raw_data'],
    'feature_c': ['feature_a', 'feature_b'],   # Depends on both
    'model':     ['feature_c'],
    'evaluation': ['model'],
}

order = topological_sort(tasks, dependencies)
print(f"Execution order: {order}")

stages = get_parallel_stages(tasks, dependencies)
print("Parallel execution stages:")
for i, stage in enumerate(stages):
    print(f"  Stage {i+1}: {stage}")
# Stage 1: [raw_data]
# Stage 2: [feature_a, feature_b]  ← parallel!
# Stage 3: [feature_c]
# Stage 4: [model]
# Stage 5: [evaluation]
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Why does Lasso produce sparse solutions but Ridge doesn't?"** → L1 ball has corners at axes; L2 ball is smooth. The OLS solution touches the L1 ball at a corner (where some w=0) but the L2 ball can be touched anywhere. Geometrically: L1 constraint induces sparsity.
2. **"When would Layer Norm fail?"** → When batch size is large and you want consistent normalization across the batch (use BN instead). Also worse for CNNs where spatial correlation is important (use Group Norm there).
3. **"How do you choose between PSI and KL for drift detection?"** → PSI is industry standard for reporting (0.1/0.2 thresholds are well-understood). KL(P||Q) for measuring "how different is production from training" — undefined when Q has zero-probability regions. JSD is better when you need symmetry.
