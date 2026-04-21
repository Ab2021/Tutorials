# 🟡 Medium: ML Algorithms from Scratch
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Medium | **⭐⭐ Core Interview Content**

---

## 🟡 PROBLEM 1: Logistic Regression

### Theory
Binary classification. Models $P(y=1|x) = \sigma(w^Tx + b)$ where σ is sigmoid.

**Loss (Binary Cross-Entropy):**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

**Gradient descent updates:**
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n}X^T(\hat{y} - y)$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n}\sum(\hat{y}_i - y_i)$$

The gradient $(\hat{y} - y)$ is beautifully simple — result of combining sigmoid + BCE.

### Things to Focus On
- ✅ Gradient derivation: the $\hat{y} - y$ result is non-obvious; be ready to explain it
- ✅ Use L2 regularization (`lambda * w`) in the gradient to prevent overfitting
- ✅ Feature scaling is critical for gradient descent convergence
- ✅ Decision boundary is a hyperplane: $w^Tx + b = 0$
- ✅ Converges when log-likelihood is concave → global optimum guaranteed

### Implementation
```python
import numpy as np

class LogisticRegression:
    """
    Logistic Regression from scratch using gradient descent.
    Covers: forward pass, BCE loss, gradient computation, weight update.
    """
    
    def __init__(self, lr: float = 0.01, n_epochs: int = 1000, 
                 lambda_reg: float = 0.0, random_state: int = 42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg  # L2 regularization strength
        self.random_state = random_state
        self.w = None
        self.b = None
        self.losses = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train with gradient descent.
        
        X: (n_samples, n_features)
        y: (n_samples,) binary {0, 1}
        """
        np.random.seed(self.random_state)
        n, d = X.shape
        
        # Initialize weights (small random, not zero — zero gives symmetric gradients)
        self.w = np.zeros(d)
        self.b = 0.0
        
        for epoch in range(self.n_epochs):
            # FORWARD PASS
            z = X @ self.w + self.b       # Linear: (n,)
            y_hat = self._sigmoid(z)      # Probabilities: (n,)
            
            # LOSS (BCE + L2 regularization)
            eps = 1e-8
            bce = -np.mean(y * np.log(y_hat + eps) + (1-y) * np.log(1 - y_hat + eps))
            l2_penalty = 0.5 * self.lambda_reg * np.sum(self.w ** 2)
            loss = bce + l2_penalty
            self.losses.append(loss)
            
            # BACKWARD PASS (gradients)
            error = y_hat - y             # (n,) — the clean gradient
            dw = (X.T @ error) / n + self.lambda_reg * self.w  # L2 reg on weights
            db = error.mean()
            
            # WEIGHT UPDATE
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities."""
        return self._sigmoid(X @ self.w + self.b)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=4, random_state=42)
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)  # ALWAYS scale first!

model = LogisticRegression(lr=0.1, n_epochs=500)
model.fit(X, y)
print(f"Accuracy: {model.score(X, y):.3f}")
```

---

## 🟡 PROBLEM 2: Linear Regression (Gradient Descent + Normal Equation)

### Theory
Models $\hat{y} = w^Tx + b$ with MSE loss. Two solution methods:

**Method 1 — Normal Equation (closed form):**
$$w^* = (X^TX)^{-1}X^Ty$$

- Exact solution in O(d³ + nd²) — expensive for large d
- No hyperparameters (learning rate, epochs)

**Method 2 — Gradient Descent:**
$$w \leftarrow w - \alpha \frac{1}{n}X^T(\hat{y} - y)$$

- Scalable to large datasets
- Requires tuning α and epochs

**Ridge Regression (L2):**
$$w^* = (X^TX + \lambda I)^{-1}X^Ty$$
- The λI addition makes the matrix invertible (even if X^TX is singular)
- Shrinks weights toward zero

### Things to Focus On
- ✅ Know both methods; mention trade-offs (Normal Eq exact but O(d³); GD scalable)
- ✅ Normal equation adds λI for Ridge: makes matrix always invertible
- ✅ Intercept trick: add column of 1s to X, then solve for w (includes bias)
- ✅ Feature scaling speeds up gradient descent convergence dramatically
- ✅ For d > 10,000: prefer SGD; for d < 1000: Normal Equation is fine

### Implementation
```python
class LinearRegression:
    """Linear Regression: both closed-form and gradient descent."""
    
    def __init__(self, method: str = 'gd', lr: float = 0.01, 
                 n_epochs: int = 1000, lambda_reg: float = 0.0):
        """
        method: 'gd' for gradient descent, 'normal' for closed-form
        lambda_reg: L2 regularization (Ridge)
        """
        self.method = method
        self.lr = lr
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.w = None
        self.b = None
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Closed-form solution: w = (X^TX + λI)^{-1} X^T y
        
        Adds bias by augmenting X with a column of 1s.
        Ridge: lambda_reg > 0 prevents singular matrix, shrinks weights.
        """
        n, d = X.shape
        # Augment X with bias column: [X | 1]
        X_aug = np.hstack([X, np.ones((n, 1))])
        
        # Ridge adds λI to X^TX (except bias term — don't regularize bias)
        reg_matrix = self.lambda_reg * np.eye(d + 1)
        reg_matrix[-1, -1] = 0  # Don't regularize bias
        
        # w_aug = (X^TX + λI)^{-1} X^T y
        w_aug = np.linalg.solve(X_aug.T @ X_aug + reg_matrix, X_aug.T @ y)
        
        self.w = w_aug[:-1]
        self.b = w_aug[-1]
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Iterative gradient descent solution."""
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        
        for epoch in range(self.n_epochs):
            y_hat = X @ self.w + self.b
            error = y_hat - y               # Residuals
            
            dw = (X.T @ error) / n + self.lambda_reg * self.w
            db = error.mean()
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.method == 'normal':
            self.fit_normal_equation(X, y)
        else:
            self.fit_gradient_descent(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b
    
    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

# Test — compare both methods
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Normal equation
lr_normal = LinearRegression(method='normal', lambda_reg=0.1)
lr_normal.fit(X_scaled, y)
print(f"Normal Eq R²: {lr_normal.r2_score(X_scaled, y):.4f}")

# Gradient descent
lr_gd = LinearRegression(method='gd', lr=0.1, n_epochs=2000)
lr_gd.fit(X_scaled, y)
print(f"GD R²: {lr_gd.r2_score(X_scaled, y):.4f}")
# Both should give similar R² ✓
```

---

## 🟡 PROBLEM 3: K-Means Clustering

### Theory
Iterative algorithm:
1. Initialize k centroids (random or K-Means++)
2. **Assignment:** Assign each point to nearest centroid
3. **Update:** Move centroids to mean of assigned points
4. Repeat until convergence

**Objective:** Minimize within-cluster sum of squares (WCSS):
$$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$

**K-Means++:** Initialize centroids with probability proportional to distance from existing centroids → better initialization, faster convergence, avoids bad local optima.

### Things to Focus On
- ✅ K-Means can get stuck in local optima → run multiple times with different seeds
- ✅ K-Means assumes spherical, equal-size clusters — fails on elongated, non-convex clusters
- ✅ Elbow method / Silhouette score to choose K
- ✅ K-Means++ initialization is the standard (sklearn default)
- ✅ Sensitive to outliers → consider K-Medoids for robustness

### Implementation
```python
class KMeans:
    """
    K-Means from scratch with:
    - Random or K-Means++ initialization
    - Convergence check
    - WCSS computation
    """
    
    def __init__(self, k: int = 3, max_iter: int = 300, 
                 init: str = 'kmeans++', random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using random or K-Means++."""
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        
        if self.init == 'random':
            indices = rng.choice(n, self.k, replace=False)
            return X[indices].copy()
        
        # K-Means++ initialization
        centroids = [X[rng.randint(n)]]
        
        for _ in range(self.k - 1):
            # Distance from each point to nearest existing centroid
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], 
                axis=0
            )
            # Sample next centroid proportional to distance squared
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        # Pairwise distances: (n_samples, k)
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids]).T
        return np.argmin(dists, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Move each centroid to mean of its assigned points."""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            mask = labels == k
            if mask.sum() == 0:
                # Empty cluster: reinitialize to random point
                new_centroids[k] = X[np.random.randint(len(X))]
            else:
                new_centroids[k] = X[mask].mean(axis=0)
        return new_centroids
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Fit K-Means to data."""
        self.centroids = self._init_centroids(X)
        
        for iteration in range(self.max_iter):
            labels = self._assign(X)
            new_centroids = self._update_centroids(X, labels)
            
            # Convergence check: centroids stopped moving
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                print(f"Converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = self._compute_wcss(X, labels)
        return self
    
    def _compute_wcss(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Within-cluster sum of squares (inertia)."""
        return sum(
            np.sum((X[labels == k] - self.centroids[k]) ** 2)
            for k in range(self.k)
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to nearest centroid."""
        return self._assign(X)

# Test
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

km = KMeans(k=3, init='kmeans++', random_state=42)
km.fit(X)
print(f"Inertia: {km.inertia_:.2f}")
print(f"Centroids:\n{km.centroids}")
```

---

## 🟡 PROBLEM 4: Decision Tree Split (Information Gain & Gini)

### Theory
Decision tree splits maximize the "information gain" from a split.

**Entropy of node S:**
$$H(S) = -\sum_{c} p_c \log_2 p_c$$

**Information Gain of split on feature X at threshold t:**
$$\text{IG}(S, X, t) = H(S) - \sum_{v \in \{L, R\}} \frac{|S_v|}{|S|} H(S_v)$$

**Gini Impurity:**
$$\text{Gini}(S) = 1 - \sum_c p_c^2$$

**Gini Gain:**
$$\text{GiniGain}(S, X, t) = \text{Gini}(S) - \sum_{v \in \{L, R\}} \frac{|S_v|}{|S|} \text{Gini}(S_v)$$

Gini ≈ 2 × Entropy (computationally cheaper, no log). **sklearn uses Gini by default.**

### Things to Focus On
- ✅ Both Entropy and Gini give similar splits in practice
- ✅ Finding best split: loop over all features and thresholds → O(d × n log n)
- ✅ Pure node: entropy = 0 (only one class) → stop splitting
- ✅ Stopping criteria: max_depth, min_samples, min_impurity_decrease
- ✅ Decision trees overfit without pruning — random forests aggregate many trees

### Implementation
```python
def entropy(y: np.ndarray) -> float:
    """Shannon entropy of a label set."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Avoid log(0): 0 * log(0) = 0 by convention
    return -np.sum(probs * np.log2(probs + 1e-10))

def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: 1 - sum(p_c²)"""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_gain(y: np.ndarray, y_left: np.ndarray, 
                     y_right: np.ndarray, criterion: str = 'entropy') -> float:
    """
    Compute information gain from a split.
    criterion: 'entropy' or 'gini'
    """
    impurity_fn = entropy if criterion == 'entropy' else gini_impurity
    
    n = len(y)
    n_l, n_r = len(y_left), len(y_right)
    
    parent_impurity = impurity_fn(y)
    weighted_child = (n_l / n) * impurity_fn(y_left) + (n_r / n) * impurity_fn(y_right)
    
    return parent_impurity - weighted_child

def find_best_split(X: np.ndarray, y: np.ndarray, 
                    criterion: str = 'gini') -> dict:
    """
    Find the best feature and threshold to split on.
    Brute-force over all features × all unique thresholds.
    """
    n, d = X.shape
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(d):
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            left_mask  = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # Skip if split creates empty child
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            
            gain = information_gain(
                y, y[left_mask], y[right_mask], criterion=criterion
            )
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'gain': best_gain
    }

# Test
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data[:100], iris.target[:100]  # Binary: only first 2 classes

print("Entropy of full node:", entropy(y))  # Should be 1.0 (50/50 split)

best = find_best_split(X, y, criterion='gini')
print(f"Best split: feature {best['feature']}, threshold {best['threshold']:.2f}, gain {best['gain']:.4f}")
```

---

## 🟡 PROBLEM 5: Batch Normalization

### Theory
Normalizes layer inputs to reduce internal covariate shift.

For a mini-batch {x₁, ..., xₘ} in layer:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad y_i = \gamma \hat{x}_i + \beta$$

**Parameters:** γ (scale) and β (shift) are learnable. ε prevents division by zero.

**Training vs Inference:** 
- Training: use batch statistics
- Inference: use running mean/variance (exponential moving average of batch stats)

### Things to Focus On
- ✅ Gamma and beta are LEARNED (not just normalized — the network can undo normalization if needed)
- ✅ During inference: use running mean/var, NOT current batch stats
- ✅ Layer Normalization (LN): normalize across features for each sample — used in Transformers
- ✅ BN works poorly for small batches (use LN or GN instead)
- ✅ BN is applied BEFORE activation typically (though some papers say after)

### Implementation
```python
class BatchNormalization:
    """
    Batch Normalization for fully-connected layers.
    
    Training: normalize over batch dimension
    Inference: use running statistics
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters (initialized to identity transform)
        self.gamma = np.ones(num_features)   # Scale
        self.beta  = np.zeros(num_features)  # Shift
        
        # Running statistics (updated during training, used during inference)
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)
        
        # Cache for backprop
        self._cache = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        x: (batch_size, num_features)
        """
        if training:
            # Compute batch statistics
            mu  = x.mean(axis=0)                              # (num_features,)
            var = x.var(axis=0)                               # (num_features,)
            
            # Normalize
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
            
            # Cache for backprop
            self._cache = (x, x_norm, mu, var)
        
        else:
            # Inference: use running statistics (fixed)
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        
        return out

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization: normalize across FEATURE dimension for each sample.
    Used in Transformers (BERT, GPT) instead of BatchNorm.
    
    x: (batch_size, num_features)
    """
    mu  = x.mean(axis=-1, keepdims=True)   # Mean per sample
    var = x.var(axis=-1, keepdims=True)     # Variance per sample
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# Test
np.random.seed(42)
x = np.random.randn(32, 64)  # Batch of 32 samples, 64 features

bn = BatchNormalization(num_features=64)
out_train = bn.forward(x, training=True)
print(f"Out mean: {out_train.mean(axis=0)[:3]}")  # Should be ~0 (beta=0)
print(f"Out std:  {out_train.std(axis=0)[:3]}")   # Should be ~1 (gamma=1)
```

---

## 🟡 PROBLEM 6: Dropout

### Theory
Randomly zeroes neurons during training with probability p (drop rate).

$$\text{Training: } \tilde{h}_i = \frac{h_i \cdot \text{Bernoulli}(1-p)}{1-p}$$

The **inverted dropout** scaling by $\frac{1}{1-p}$ ensures expected values are unchanged between training and inference — this is what PyTorch/TensorFlow use.

**Why dropout works:**
- Forces network to learn redundant representations
- Equivalent to ensemble averaging over exponentially many sub-networks
- Acts as strong regularizer (dropout rate 0.5 ≈ average of 2^n models)

### Things to Focus On
- ✅ INVERTED dropout: scale by 1/(1-p) during training (not during inference)
- ✅ Different dropout masks for each sample in each forward pass
- ✅ Set to eval mode for inference: dropout is disabled
- ✅ Typical rates: 0.5 for FC layers, 0.1-0.3 for convolutional layers
- ✅ Dropout AFTER activation, BEFORE next layer

### Implementation
```python
class Dropout:
    """
    Dropout with inverted scaling.
    
    Training: randomly zero activations, scale up surviving ones
    Inference: pass through unchanged (scaling already baked in)
    """
    
    def __init__(self, p: float = 0.5):
        """p: probability of DROPPING a unit (not keeping)"""
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self._mask = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return x  # No dropout at inference (scaling already in)
        
        # Inverted dropout: keep with probability (1-p), scale by 1/(1-p)
        self._mask = (np.random.rand(*x.shape) > self.p).astype(float)
        return x * self._mask / (1 - self.p)

def apply_dropout(x: np.ndarray, p: float = 0.5, training: bool = True, 
                  seed: int = None) -> np.ndarray:
    """Functional version of dropout."""
    if not training or p == 0:
        return x
    if seed is not None:
        np.random.seed(seed)
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    return x * mask

# Test
np.random.seed(42)
x = np.ones((4, 8))
do = Dropout(p=0.5)

# Training: ~50% of values should be 0, rest scaled by 2
out_train = do.forward(x, training=True)
print(f"Training output:\n{out_train}")
print(f"Non-zero fraction: {(out_train != 0).mean():.2f}")  # ~0.5

# Inference: all pass through unchanged
out_eval = do.forward(x, training=False)
print(f"Eval output:\n{out_eval}")  # All ones ✓
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Why can't we use the Normal Equation for 1M samples?"** → O(d³) matrix inversion of X^TX becomes too expensive. Use mini-batch SGD instead.
2. **"What's the difference between K-Means and Gaussian Mixture Models?"** → K-Means uses hard assignments (each point belongs to exactly one cluster); GMM uses soft probabilistic assignments and can model elliptical clusters.
3. **"Why does Batch Norm use running statistics at inference?"** → At inference, batch size may be 1 (impossible to compute meaningful batch stats). Running stats accumulated during training give stable estimates.
4. **"Why is inverted dropout important?"** → Without scaling (1/(1-p)), training-time expectations differ from inference-time. The network learns to compensate for dropped units by inflating remaining activations. Inverted dropout corrects this without requiring separate inference scaling.
