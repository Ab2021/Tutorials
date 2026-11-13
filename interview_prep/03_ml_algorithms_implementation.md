
---

## 2. Section 1: Classical Machine Learning Algorithms from Scratch (NumPy/OOP)

### Q1: Ridge Linear Regression with Gradient Descent & Closed-Form [Medium]

#### 1. Explain the Concept & Mathematical Intuition
Linear Regression models the relationship between dependent continuous variable $y \in \mathbb{R}^n$ and independent feature matrix $X \in \mathbb{R}^{n \times d}$ as:
$$\hat{y} = Xw + b$$

In Senior interviews, you are expected to handle **Ridge Regularization ($L_2$ penalty)** to prevent overfitting and handle multicollinearity (frequently occurring in FMCG price elasticity models and macroeconomic financial datasets).

**The Objective Function (Loss):**
$$J(w, b) = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \frac{\lambda}{2n} \|w\|_2^2 = \frac{1}{2n} (Xw + b\mathbf{1} - y)^T (Xw + b\mathbf{1} - y) + \frac{\lambda}{2n} w^T w$$

**Analytical Closed-Form Solution (Normal Equation with $L_2$):**
When including the intercept by appending a column of 1s to $X$ (making it $X \in \mathbb{R}^{n \times (d+1)}$), the gradient with respect to $w$ set to zero yields:
$$w^* = (X^T X + \lambda I')^{-1} X^T y$$
*(Note: $I'$ is the identity matrix with $I'_{0,0} = 0$ so the intercept $b$ is never regularized!)*

**Gradient Descent Formulation:**
$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (Xw + b - y) + \frac{\lambda}{n} w$$
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n (Xw + b - y)$$

Parameters update rule:
$$w \leftarrow w - \eta \frac{\partial J}{\partial w}, \quad b \leftarrow b - \eta \frac{\partial J}{\partial b}$$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class RidgeRegressionScratch:
    """
    Ridge (L2 Regularized) Linear Regression implemented from scratch using NumPy.
    Supports both Batch Gradient Descent and Analytical Normal Equation.
    """
    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_iters: int = 1000, 
                 solver: str = "gd", tol: float = 1e-6):
        """
        Args:
            alpha: L2 regularization strength (lambda).
            lr: Learning rate for gradient descent.
            n_iters: Maximum gradient descent iterations.
            solver: 'gd' (Batch Gradient Descent) or 'closed_form' (Normal Equation).
            tol: Convergence tolerance for gradient descent.
        """
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.solver = solver.lower()
        self.tol = float(tol)
        
        self.weights = None  # shape: (d,)
        self.bias = 0.0      # scalar
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if self.solver == "closed_form":
            # Augment X with column of 1s for the bias term
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            # Construct regularization matrix where bias (index 0) is not regularized
            I_reg = np.eye(n_features + 1)
            I_reg[0, 0] = 0.0
            
            # w* = (X^T X + alpha * I)^(-1) X^T y
            # Using np.linalg.solve for superior numerical stability over np.linalg.inv
            A = X_aug.T @ X_aug + self.alpha * I_reg
            b_vec = X_aug.T @ y
            theta = np.linalg.solve(A, b_vec)
            
            self.bias = theta[0]
            self.weights = theta[1:]
        elif self.solver == "gd":
            self.weights = np.zeros(n_features, dtype=np.float64)
            self.bias = 0.0
            self.loss_history = []

            for iteration in range(self.n_iters):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                
                # Compute gradients
                dw = (1.0 / n_samples) * (X.T @ error) + (self.alpha / n_samples) * self.weights
                db = (1.0 / n_samples) * np.sum(error)

                # Track MSE + L2 loss
                loss = (0.5 / n_samples) * np.sum(error ** 2) + (0.5 * self.alpha / n_samples) * np.sum(self.weights ** 2)
                self.loss_history.append(loss)

                # Parameter update
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Check convergence
                if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                    break
        else:
            raise ValueError(f"Unknown solver: {self.solver}. Use 'gd' or 'closed_form'.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 (Coefficient of Determination)."""
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    X_sim = np.random.randn(200, 5)
    true_w = np.array([3.5, -1.2, 0.0, 2.1, -4.0])
    y_sim = X_sim @ true_w + 5.0 + np.random.randn(200) * 0.5

    # Test Gradient Descent solver
    model_gd = RidgeRegressionScratch(alpha=0.5, lr=0.05, n_iters=2000, solver="gd")
    model_gd.fit(X_sim, y_sim)
    
    # Test Closed Form solver
    model_cf = RidgeRegressionScratch(alpha=0.5, solver="closed_form")
    model_cf.fit(X_sim, y_sim)

    print(f"[Q1] GD R^2: {model_gd.score(X_sim, y_sim):.4f} | Bias: {model_gd.bias:.3f}")
    print(f"[Q1] CF R^2: {model_cf.score(X_sim, y_sim):.4f} | Bias: {model_cf.bias:.3f}")
    print(f"Weights match close: {np.allclose(model_gd.weights, model_cf.weights, atol=1e-2)}")
```

#### 3. Complexity & Production Trade-Offs
- **Time Complexity:**
  - Gradient Descent: $\mathcal{O}(N \cdot D \cdot \text{iters})$. Ideal when $N$ or $D > 10,000$.
  - Closed-Form: $\mathcal{O}(D^3 + N \cdot D^2)$ due to the matrix inversion/solve. When $D > 20,000$, $(X^T X)^{-1}$ becomes intractable in memory and runtime.
- **Why do we never regularize the bias $b$?** Regularizing the bias penalizes the baseline level of the outcome independent of features, forcing predictions toward zero even if the true mean of $y$ is $10^6$.
- **LSEG / General Mills Relevance:** High collinearity among promotional flags (General Mills) or macroeconomic indicators (LSEG) makes $X^T X$ near-singular. Adding $\lambda I$ guarantees strict positive definiteness: $\det(X^T X + \lambda I) > 0$.

---

### Q2: Logistic Regression with Numerically Stable Sigmoid & L1/L2 Regularization [Hard]

#### 1. Explain the Concept & Mathematical Intuition
Logistic Regression models the posterior probability of a binary label $y \in \{0, 1\}$ using the logistic sigmoid function:
$$p = P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = Xw + b$$

**Numerical Instability Trap:**
Computing $e^{-z}$ naively causes runtime floating-point overflow (`RuntimeWarning: overflow encountered in exp`) when $z < -709$ in 64-bit IEEE floating-point arithmetic.
To guarantee complete stability:
$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$
Or equivalently by clipping $z \in [-500, 500]$.

**Log-Loss Objective with ElasticNet ($L_1$ and $L_2$):**
$$J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i + \epsilon) + (1 - y_i) \ln(1 - p_i + \epsilon) \right] + \frac{\lambda_2}{2n} \|w\|_2^2 + \frac{\lambda_1}{n} \|w\|_1$$

**Gradients:**
$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (p - y) + \frac{\lambda_2}{n} w + \frac{\lambda_1}{n} \text{sign}(w)$$
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)$$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class LogisticRegressionScratch:
    """
    Numerically Stable Binary Logistic Regression with L1/L2 Regularization
    and configurable classification threshold.
    """
    def __init__(self, lr: float = 0.05, n_iters: int = 1500, l1_ratio: float = 0.0, 
                 l2_reg: float = 0.01, threshold: float = 0.5):
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.l1_ratio = float(l1_ratio)  # 0: pure L2, 1: pure L1
        self.l2_reg = float(l2_reg)
        self.threshold = float(threshold)
        
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Branching vectorized computation to eliminate overflow
        z = np.clip(z, -500.0, 500.0)
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        eps = 1e-15
        for _ in range(self.n_iters):
            linear_model = X @ self.weights + self.bias
            probs = self._sigmoid(linear_model)

            # Compute cross-entropy loss with numerical guards
            loss_ce = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
            reg_penalty = 0.5 * self.l2_reg * np.sum(self.weights ** 2) + self.l1_ratio * np.sum(np.abs(self.weights))
            self.loss_history.append(loss_ce + reg_penalty)

            # Gradients
            error = probs - y
            dw = (1.0 / n_samples) * (X.T @ error) + (self.l2_reg / n_samples) * self.weights
            if self.l1_ratio > 0:
                dw += (self.l1_ratio / n_samples) * np.sign(self.weights)
            
            db = (1.0 / n_samples) * np.sum(error)

            # Gradient update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        linear_model = X @ self.weights + self.bias
        p1 = self._sigmoid(linear_model)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    X_cls = np.random.randn(300, 4)
    # True decision boundary
    logits = X_cls @ np.array([2.0, -1.5, 3.0, -0.5]) + 0.5
    y_cls = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)

    clf = LogisticRegressionScratch(lr=0.1, n_iters=1000, l2_reg=0.05, threshold=0.5)
    clf.fit(X_cls, y_cls)
    preds = clf.predict(X_cls)
    acc = np.mean(preds == y_cls)
    print(f"[Q2] Logistic Regression Accuracy: {acc * 100:.2f}% | Final Loss: {clf.loss_history[-1]:.4f}")
```

#### 3. Interview Follow-Ups & Edge Cases
- **Why not use MSE for Logistic Regression?** MSE loss with sigmoid yields a non-convex optimization surface filled with local minima and saddle points, causing gradient vanishing when $p \approx 0$ or $1$. Cross-entropy guarantees global convexity.
- **How does class imbalance shift the bias term?** When class 1 is rare (e.g. 1% fraud at LSEG), initializing $b_0 = \ln(\pi / (1 - \pi)) = \ln(0.01 / 0.99) \approx -4.6$ dramatically accelerates convergence and stabilizes early iterations.

---

### Q3: Decision Tree Classifier from Scratch (CART & Gini Impurity) [Hard]

#### 1. Explain the Concept & Mathematical Intuition
Classification and Regression Trees (CART) construct a binary tree by greedily searching for the feature $j$ and split threshold $t$ that maximize the reduction in node impurity.

**Gini Impurity Definition:**
For a node $m$ with class distribution $p_k = \frac{1}{N_m} \sum_{i \in m} \mathbb{I}(y_i = k)$:
$$G(m) = 1 - \sum_{k=1}^K p_k^2$$

**Split Criterion (Impurity Reduction / Information Gain):**
A proposed split divides node $m$ into left child $L$ ($x_j \le t$) and right child $R$ ($x_j > t$):
$$\Delta G(m, j, t) = G(m) - \left( \frac{N_L}{N_m} G(L) + \frac{N_R}{N_m} G(R) \right)$$
The tree algorithm selects:
$$(j^*, t^*) = \arg\max_{j, t} \Delta G(m, j, t)$$

**Stopping Criteria (Pruning):**
1. Maximum depth reached (`max_depth`).
2. Node sample count less than threshold (`min_samples_split`).
3. Pure node ($G(m) = 0$).
4. Best impurity gain less than threshold (`min_impurity_decrease`).

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from typing import Optional

class TreeNode:
    """Internal binary tree node representing a decision split or leaf."""
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None,
                 value: Optional[int] = None, proba: Optional[dict] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value        # Leaf class prediction
        self.proba = proba        # Leaf class probability dictionary

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTreeClassifierScratch:
    """
    CART Binary Decision Tree Classifier using Gini Impurity.
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
                 min_impurity_decrease: float = 1e-4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[TreeNode] = None
        self.classes_ = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        current_gini = self._gini(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            
            # Test split thresholds at midpoints between unique sorted values
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_l, y_r = y[left_mask], y[right_mask]
                gain = current_gini - (len(y_l)/n_samples * self._gini(y_l) + len(y_r)/n_samples * self._gini(y_r))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        classes, counts = np.unique(y, return_counts=True)
        leaf_val = classes[np.argmax(counts)]
        proba_dict = {cls: count / len(y) for cls, count in zip(classes, counts)}

        # Check stopping conditions
        if (depth >= self.max_depth or 
            len(classes) == 1 or 
            len(y) < self.min_samples_split):
            return TreeNode(value=leaf_val, proba=proba_dict)

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain < self.min_impurity_decrease:
            return TreeNode(value=leaf_val, proba=proba_dict)

        left_mask = X[:, feat] <= thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return TreeNode(feature_idx=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _traverse_one(self, x: np.ndarray, node: TreeNode) -> int:
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_one(x, node.left)
        return self._traverse_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse_one(x, self.root) for x in X])

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    X_tree = np.random.randn(250, 4)
    # XOR-like synthetic non-linear boundary
    y_tree = ((X_tree[:, 0] > 0) ^ (X_tree[:, 1] > 0)).astype(int)

    dt = DecisionTreeClassifierScratch(max_depth=4, min_samples_split=5)
    dt.fit(X_tree, y_tree)
    preds = dt.predict(X_tree)
    print(f"[Q3] Decision Tree Accuracy: {np.mean(preds == y_tree) * 100:.2f}%")
```

#### 3. Complexity & Production Trade-Offs
- **Time Complexity:**
  - Training: $\mathcal{O}(D \cdot N \log N \cdot \text{depth})$ due to sorting features at each node.
  - Inference: $\mathcal{O}(\text{depth})$, extremely fast (sub-microsecond) for real-time low-latency financial trading decision systems at LSEG.
- **Why Gini over Entropy?** Gini avoids evaluating computationally expensive logarithms ($\log_2$), speeding up tree induction by 20-30% with negligible difference in split quality.

---

### Q4: K-Means Clustering with K-Means++ & Vectorized Broadcasting [Medium]

#### 1. Explain the Concept & Mathematical Intuition
K-Means partitions $N$ observations into $K$ disjoint clusters $S = \{S_1, S_2, \dots, S_K\}$ by minimizing Within-Cluster Sum of Squares (WCSS / Inertia):
$$J = \sum_{k=1}^K \sum_{x_i \in S_k} \|x_i - \mu_k\|_2^2$$

**Vectorized Distance Computation via NumPy Broadcasting:**
Avoid slow explicit loops over samples and centroids. Expand dimensions to compute the full $N \times K$ squared Euclidean distance matrix in one step:
$$\text{dist}(x_i, \mu_j)^2 = \|X - C\|^2 \implies \text{shape: } (N, 1, D) - (1, K, D) \to (N, K, D) \xrightarrow{\sum \text{axis}=2} (N, K)$$

**K-Means++ Initialization (Arthur & Vassilvitskii, 2007):**
Standard random initialization frequently converges to suboptimal local minima. K-Means++ guarantees an $\mathcal{O}(\log K)$ approximation bound:
1. Choose the first centroid $\mu_1$ uniformly at random from dataset $X$.
2. For each point $x$, compute $D(x) = \min_{j} \|x - \mu_j\|_2$.
3. Sample next centroid $\mu_{new}$ from $X$ with probability proportional to $D(x)^2$:
   $$P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$$
4. Repeat until $K$ centroids are chosen.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class KMeansScratch:
    """
    K-Means Clustering with K-Means++ Initialization and Vectorized Broadcasting.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, 
                 init: str = "k-means++", random_state: int = 42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init.lower()
        self.random_state = random_state
        
        self.centroids = None      # shape: (k, d)
        self.labels_ = None        # shape: (n,)
        self.inertia_ = None       # Total WCSS

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.init == "random":
            idx = rng.choice(n_samples, self.k, replace=False)
            return X[idx].copy()
        
        elif self.init == "k-means++":
            centroids = [X[rng.randint(n_samples)]]
            for _ in range(1, self.k):
                # Distances from each sample to closest existing centroid
                cent_arr = np.array(centroids) # (m, d)
                # (n, 1, d) - (1, m, d) -> (n, m)
                dist_sq = np.min(np.sum((X[:, np.newaxis, :] - cent_arr[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                
                # Prevent zero-sum probability if duplicate points exist
                prob = dist_sq / (np.sum(dist_sq) + 1e-12)
                next_idx = rng.choice(n_samples, p=prob)
                centroids.append(X[next_idx])
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, _ = X.shape
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # Vectorized distance computation: (n, 1, d) - (1, k, d) -> (n, k)
            dist_matrix = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            cluster_labels = np.argmin(dist_matrix, axis=1)

            # Update centroids with empty cluster resilience
            new_centroids = np.zeros_like(self.centroids)
            for c in range(self.k):
                members = X[cluster_labels == c]
                if len(members) > 0:
                    new_centroids[c] = np.mean(members, axis=0)
                else:
                    # Fallback: re-seed empty cluster to point furthest from any centroid
                    furthest_idx = np.argmax(np.min(dist_matrix, axis=1))
                    new_centroids[c] = X[furthest_idx]

            # Check convergence
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        # Compute final inertia
        final_dist = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        self.labels_ = np.argmin(final_dist, axis=1)
        self.inertia_ = float(np.sum(np.min(final_dist, axis=1)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dist_matrix = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dist_matrix, axis=1)

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    c1 = np.random.randn(50, 2) + np.array([5, 5])
    c2 = np.random.randn(50, 2) + np.array([-5, -5])
    c3 = np.random.randn(50, 2) + np.array([5, -5])
    X_km = np.vstack([c1, c2, c3])

    km = KMeansScratch(n_clusters=3, init="k-means++")
    km.fit(X_km)
    print(f"[Q4] KMeans Inertia: {km.inertia_:.2f} | Centroids shape: {km.centroids.shape}")
```

#### 3. Complexity & Practical Nuance
- **Time Complexity:** $\mathcal{O}(N \cdot K \cdot D \cdot I)$. Memory: $\mathcal{O}(N \cdot K \cdot D)$ temporary buffer during broadcasting (in production for $N > 10^6$, chunking across $N$ avoids RAM exhaustion).
- **Domain Application:** At General Mills, K-Means clusters retail stores into tiers (Hypermarket vs Convenience) by sales volume and SKU velocity. At LSEG, K-Means clusters order book liquidity profiles.

---

### Q5: Principal Component Analysis (PCA) via SVD & Eigendecomposition [Medium]

#### 1. Explain the Concept & Mathematical Intuition
PCA finds an orthogonal linear transformation that maps high-dimensional data $X \in \mathbb{R}^{n \times d}$ into lower-dimensional coordinates $Z \in \mathbb{R}^{n \times k}$ while maximizing explained variance.

**Mathematical Formulation:**
1. Center feature matrix: $\tilde{X} = X - \mu$, where $\mu = \frac{1}{n} \mathbf{1}^T X$.
2. **Method A (Covariance Eigendecomposition):**
   $$\Sigma = \frac{1}{n - 1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$
   Compute eigenvalues $\lambda_j$ and eigenvectors $v_j$ such that $\Sigma v_j = \lambda_j v_j$. Sort by descending eigenvalues.
3. **Method B (Singular Value Decomposition - SVD):**
   $$\tilde{X} = U \Sigma V^T$$
   The right singular vectors $V \in \mathbb{R}^{d \times d}$ are the principal axes!
   The singular values $s_j$ relate directly to covariance eigenvalues:
   $$\lambda_j = \frac{s_j^2}{n - 1}$$
   SVD is numerically superior because it avoids explicitly computing $\tilde{X}^T \tilde{X}$, which squares the condition number ($\kappa(\tilde{X}^T \tilde{X}) = \kappa(\tilde{X})^2$) and causes precision loss.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class PCAScratch:
    """
    Principal Component Analysis (PCA) implemented via SVD and Eigendecomposition.
    """
    def __init__(self, n_components: int = 2, method: str = "svd"):
        self.n_components = n_components
        self.method = method.lower()
        self.mean_ = None
        self.components_ = None            # (n_components, n_features)
        self.explained_variance_ = None    # (n_components,)
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.method == "svd":
            # X_centered = U @ S @ Vt
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigenvals = (S ** 2) / (n_samples - 1)
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ = eigenvals[:self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components] / np.sum(eigenvals)

        elif self.method == "eigen":
            # Covariance matrix: (d, d)
            cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # eigh returns in ascending order -> reverse for descending
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            self.components_ = eigenvecs[:, :self.n_components].T
            self.explained_variance_ = eigenvals[:self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components] / np.sum(eigenvals)
        else:
            raise ValueError(f"Unknown method {self.method}. Choose 'svd' or 'eigen'.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct original high-dimensional features from projection."""
        Z = np.asarray(Z, dtype=np.float64)
        return (Z @ self.components_) + self.mean_

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    # Generate correlated 3D data with high variance along 1st axis
    X_orig = np.random.randn(300, 3) @ np.diag([10.0, 2.0, 0.2])
    
    pca = PCAScratch(n_components=2, method="svd")
    pca.fit(X_orig)
    Z = pca.transform(X_orig)
    print(f"[Q5] Explained Variance Ratios: {pca.explained_variance_ratio_}")
    print(f"Total Explained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
```

#### 3. Interview Questions & Gotchas
- **Why must data be standardized (StandardScaler) before PCA?** If feature $A$ is in meters and feature $B$ is in millimeters, feature $B$ has $10^6$ times greater variance simply due to units, artificially dominating the first principal component without real informational priority.
- **Can PCA be used for non-linear dimensionality reduction?** No; PCA is strictly a linear projection. For non-linear manifolds, discuss Kernel PCA, t-SNE, or UMAP.

---

### Q6: K-Nearest Neighbors (KNN) Classifier from Scratch [Easy]

#### 1. Explain the Concept & Mathematical Intuition
KNN is an instance-based non-parametric "lazy learner". It performs zero optimization during training ($\mathcal{O}(1)$), deferring all computation to inference time.

**Distance Matrix Trick:**
Given test matrix $X_{test} \in \mathbb{R}^{m \times d}$ and training matrix $X_{train} \in \mathbb{R}^{n \times d}$, computing pairwise Euclidean distance naively with double loops is $\mathcal{O}(m \cdot n \cdot d)$ in pure Python.
Using the algebraic expansion:
$$\|A - B\|^2 = \|A\|^2 + \|B\|^2 - 2 A B^T$$
We vectorize pairwise distances into pure matrix operations:
```python
dists = np.sum(X_test**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2 * (X_test @ X_train.T)
```

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from collections import Counter

class KNNClassifierScratch:
    """
    Vectorized K-Nearest Neighbors Classifier with Euclidean Distance and Distance Weighting.
    """
    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = int(k)
        self.weights = weights.lower()
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.int64).ravel()
        return self

    def _compute_distances_vectorized(self, X_test: np.ndarray) -> np.ndarray:
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A @ B.T
        test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)       # (m, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)               # (n,)
        cross_term = 2.0 * (X_test @ self.X_train.T)               # (m, n)
        
        dist_sq = test_sq + train_sq - cross_term
        return np.sqrt(np.maximum(dist_sq, 0.0))                   # (m, n)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=np.float64)
        distances = self._compute_distances_vectorized(X_test)
        
        preds = []
        for i in range(len(X_test)):
            row_dists = distances[i]
            nearest_indices = np.argsort(row_dists)[:self.k]
            nearest_labels = self.y_train[nearest_indices]

            if self.weights == "uniform":
                vote = Counter(nearest_labels).most_common(1)[0][0]
            elif self.weights == "distance":
                weights = 1.0 / (row_dists[nearest_indices] + 1e-8)
                class_scores = {}
                for lbl, w in zip(nearest_labels, weights):
                    class_scores[lbl] = class_scores.get(lbl, 0.0) + w
                vote = max(class_scores.items(), key=lambda kv: kv[1])[0]
            else:
                raise ValueError("Weights must be 'uniform' or 'distance'")
            preds.append(vote)

        return np.array(preds)

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    X_tr = np.random.randn(150, 4)
    y_tr = (X_tr[:, 0] + X_tr[:, 1] > 0).astype(int)
    X_te = np.random.randn(50, 4)
    y_te = (X_te[:, 0] + X_te[:, 1] > 0).astype(int)

    knn = KNNClassifierScratch(k=5, weights="distance")
    knn.fit(X_tr, y_tr)
    pred_te = knn.predict(X_te)
    print(f"[Q6] KNN Test Accuracy: {np.mean(pred_te == y_te) * 100:.2f}%")
```

---

### Q7: Advanced Optimization: Momentum and Adam Optimizers from Scratch [Hard]

#### 1. Explain the Concept & Mathematical Intuition
Gradient descent in deep or ill-conditioned parameter spaces oscillates heavily across steep ravines while making agonizingly slow progress along flatter dimensions.

**Momentum:**
Accumulates an exponentially decaying moving average of past gradients:
$$v_t = \beta v_{t-1} + (1 - \beta) g_t, \quad \theta_t = \theta_{t-1} - \eta v_t$$

**Adam (Adaptive Moment Estimation - Kingma & Ba, 2014):**
Combines Momentum (1st moment, mean) and RMSProp (2nd raw moment, uncentered variance):
1. Compute gradient: $g_t = \nabla_\theta L(\theta)$
2. Update 1st moment vector: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. Update 2nd moment vector: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. **Bias Correction:** Because $m_0 = 0$ and $v_0 = 0$, both estimates are biased toward zero at early steps ($t < 50$). Correct via:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
5. Parameter update:
   $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class AdamOptimizer:
    """Adam (Adaptive Moment Estimation) Optimizer implemented from scratch."""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        # Biased moments
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)
        
        # Bias-corrected moments
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        # Parameter step
        return params - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

# Verification: Minimize non-convex Rosenbrock function f(x, y) = (1-x)^2 + 100*(y - x^2)^2
if __name__ == "__main__":
    def rosenbrock_grad(xy):
        x, y = xy[0], xy[1]
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])

    xy = np.array([-1.5, 2.0]) # Starting point
    opt = AdamOptimizer(lr=0.05)
    for _ in range(500):
        g = rosenbrock_grad(xy)
        xy = opt.update(xy, g)

    print(f"[Q7] Rosenbrock Adam Solution (Target [1, 1]): {xy.round(3)}")
```

---

### Q8: Multi-Class Softmax Regression with Log-Sum-Exp Trick [Hard]

#### 1. Explain the Concept & Mathematical Intuition
For $K > 2$ classes, Multinomial Logistic Regression maps input $x \in \mathbb{R}^d$ to probability distribution $p \in \mathbb{R}^K$ via the Softmax activation:
$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \text{where } z = Wx + b$$

**Numerical Stability (Log-Sum-Exp Trick):**
If $z_k > 700$, $e^{z_k}$ evaluates to `inf`. If $z_k < -700$, all exponentiated terms evaluate to $0.0$, producing `0 / 0 = nan`.
By subtracting the row maximum $M = \max_j z_j$:
$$\frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{-M} e^{z_k}}{e^{-M} \sum_j e^{z_j}} = \frac{e^{z_k - M}}{\sum_j e^{z_j - M}}$$
The maximum exponent is now $0$, guaranteeing $e^0 = 1$ and completely eliminating numerical overflow and underflow!

**Categorical Cross-Entropy Gradients:**
$$J(W, b) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K Y_{ik} \ln(\hat{Y}_{ik})$$
$$\nabla_W J = \frac{1}{n} X^T (\hat{Y} - Y), \quad \nabla_b J = \frac{1}{n} \sum_{i=1}^n (\hat{Y}_{i\cdot} - Y_{i\cdot})$$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class SoftmaxRegressionScratch:
    """Multi-Class Softmax Regression with Log-Sum-Exp Stability."""
    def __init__(self, lr: float = 0.05, n_iters: int = 1000, l2_reg: float = 0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.W = None  # (n_features, n_classes)
        self.b = None  # (n_classes,)
        self.classes_ = None

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (Log-Sum-Exp trick)
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # One-hot encode targets
        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y_onehot[np.arange(n_samples), y] = 1.0

        self.W = np.zeros((n_features, n_classes), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

        for _ in range(self.n_iters):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)

            error = probs - Y_onehot
            dW = (1.0 / n_samples) * (X.T @ error) + (self.l2_reg / n_samples) * self.W
            db = (1.0 / n_samples) * np.sum(error, axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    X_mult = np.random.randn(300, 5)
    # Generate 3-class target
    y_mult = np.argmax(X_mult @ np.random.randn(5, 3), axis=1)

    sm = SoftmaxRegressionScratch(lr=0.1, n_iters=800)
    sm.fit(X_mult, y_mult)
    print(f"[Q8] Softmax Multi-Class Accuracy: {np.mean(sm.predict(X_mult) == y_mult) * 100:.2f}%")
```

---

## 3. Section 2: Evaluation Metrics & Cross-Validation from Scratch

### Q9: ROC-AUC and Precision-Recall AUC (PR-AUC) from Scratch [Hard]

#### 1. Explain the Concept & Mathematical Intuition
The Receiver Operating Characteristic Area Under the Curve (ROC-AUC) evaluates the ranking ability of a binary classifier across all possible classification thresholds.

**Two Equivalent Mathematical Interpretations:**
1. **Geometric / Trapezoidal Integration:** Plot True Positive Rate ($TPR = \frac{TP}{P}$) vs False Positive Rate ($FPR = \frac{FP}{N}$) at every unique prediction threshold and integrate the area under the curve using the trapezoid rule:
   $$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$
2. **Probabilistic Interpretation (Wilcoxon-Mann-Whitney U Statistic):**
   ROC-AUC equals the exact probability that a randomly chosen positive instance is assigned a higher predicted score than a randomly chosen negative instance:
   $$\text{AUC} = P(\hat{y}_{pos} > \hat{y}_{neg}) = \frac{\sum_{i \in \text{Pos}} \text{Rank}(y_i) - \frac{n_{pos}(n_{pos} + 1)}{2}}{n_{pos} \cdot n_{neg}}$$
   *(This rank-sum formula is $\mathcal{O}(N \log N)$ and significantly faster and cleaner to implement in interviews!)*

**Precision-Recall AUC (PR-AUC / Average Precision):**
In fraud detection (LSEG) or machine component failure, negative instances outnumber positive instances 1,000 to 1. ROC-AUC is dangerously deceitful because $FPR = \frac{FP}{FP + TN}$ remains tiny even with thousands of False Positives.
PR-AUC evaluates Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{P}$), directly penalizing False Positives without dilution from True Negatives:
$$\text{PR-AUC} = \sum_{k=1}^m (R_k - R_{k-1}) \cdot P_k$$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

def roc_auc_rank_sum(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Wilcoxon-Mann-Whitney Rank-Sum implementation of ROC-AUC.
    Time Complexity: O(N log N) dominated by argsort.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined when only one class is present.")

    # Rank with average rank tie-breaking
    sort_indices = np.argsort(y_prob)
    ranks = np.empty_like(sort_indices, dtype=np.float64)
    ranks[sort_indices] = np.arange(1, len(y_prob) + 1)
    
    # Handle ties: identical probabilities receive average rank
    sorted_prob = y_prob[sort_indices]
    unique_vals, start_indices, counts = np.unique(sorted_prob, return_index=True, return_counts=True)
    for start, count in zip(start_indices, counts):
        if count > 1:
            tie_rank = np.mean(np.arange(start + 1, start + count + 1))
            tie_mask = (y_prob == sorted_prob[start])
            ranks[tie_mask] = tie_rank

    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1.0)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Precision-Recall Area Under Curve via Trapezoidal Numerical Integration.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Sort thresholds in descending order
    desc_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[desc_idx]
    
    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return 0.0

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    recalls = tps / total_pos
    precisions = tps / (tps + fps)

    # Prepend baseline (Recall=0, Precision=first calculated precision)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0]], precisions])

    # Trapezoidal integration along recall axis
    return float(np.trapz(precisions, recalls))

# Verification block
if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score as sk_roc, average_precision_score as sk_pr
    np.random.seed(42)
    y_true_sim = np.random.binomial(1, 0.15, size=500) # Imbalanced
    y_scores_sim = y_true_sim * 0.4 + np.random.rand(500) * 0.6

    my_roc = roc_auc_rank_sum(y_true_sim, y_scores_sim)
    sk_r = sk_roc(y_true_sim, y_scores_sim)
    my_pr = pr_auc_score(y_true_sim, y_scores_sim)
    
    print(f"[Q9] Scratch ROC-AUC: {my_roc:.4f} | Sklearn ROC-AUC: {sk_r:.4f}")
    print(f"[Q9] Scratch PR-AUC:  {my_pr:.4f}  | Match Sklearn close: {np.isclose(my_roc, sk_r)}")
```

---

### Q10: Classification Metrics Suite: Confusion Matrix, Specificity, F1, Macro/Micro [Easy]

#### 1. Explain the Concept & Mathematical Intuition
Given binary predictions $\hat{y} \in \{0, 1\}$ and true targets $y \in \{0, 1\}$:
- $\text{True Positives (TP)} = \sum y_i \cdot \hat{y}_i$
- $\text{False Positives (FP)} = \sum (1 - y_i) \cdot \hat{y}_i$
- $\text{True Negatives (TN)} = \sum (1 - y_i) \cdot (1 - \hat{y}_i)$
- $\text{False Negatives (FN)} = \sum y_i \cdot (1 - \hat{y}_i)$

**Core Metrics:**
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$
$$\text{Specificity} = \frac{TN}{TN + FP}, \quad \text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \text{Precision} + \text{Recall}}$$
*(At LSEG fraud detection, $\beta = 2$ is used because missing a fraudster ($FN$) is significantly more damaging than flagging an innocent user ($FP$)).*

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class ClassificationMetrics:
    """Comprehensive classification metrics engine from scratch."""
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()
        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self):
        self.tp = int(np.sum((self.y_true == 1) & (self.y_pred == 1)))
        self.fp = int(np.sum((self.y_true == 0) & (self.y_pred == 1)))
        self.tn = int(np.sum((self.y_true == 0) & (self.y_pred == 0)))
        self.fn = int(np.sum((self.y_true == 1) & (self.y_pred == 0)))

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    def f_beta(self, beta: float = 1.0) -> float:
        p, r = self.precision, self.recall
        denom = (beta ** 2) * p + r
        if denom == 0:
            return 0.0
        return (1.0 + beta ** 2) * (p * r) / denom

    def summary(self) -> dict:
        return {
            "TP": self.tp, "FP": self.fp, "TN": self.tn, "FN": self.fn,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "Specificity": round(self.specificity, 4),
            "F1-Score": round(self.f_beta(1.0), 4),
            "F2-Score": round(self.f_beta(2.0), 4),
            "Balanced_Accuracy": round(self.balanced_accuracy, 4)
        }

if __name__ == "__main__":
    y_t = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_p = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    metrics = ClassificationMetrics(y_t, y_p)
    print(f"[Q10] Summary:\n{metrics.summary()}")
```

---

### Q11: Regression Metrics for FMCG & Finance: RMSE, MAE, MAPE, WAPE, $R^2$ [Medium]

#### 1. Explain the Concept & Mathematical Intuition
In enterprise demand planning at **General Mills** (forecasting Cheerios, Pillsbury, Häagen-Dazs shipments across retail distribution centers), standard regression metrics frequently fail:

**The Intermittent Demand / Zero-Division Trap:**
When a SKU sells zero units on a given day/week ($y_i = 0$):
$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \implies \text{Division by Zero!}$$
Even adding $\epsilon$ creates distorted, explosive percentages (e.g. $y=1, \hat{y}=5 \to 400\%$ error vs $y=100, \hat{y}=104 \to 4\%$).

**The Supply Chain Standard: WAPE (Weighted Absolute Percentage Error):**
WAPE normalizes total absolute unit error by total actual volume:
$$\text{WAPE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n y_i}$$
Properties of WAPE:
- Never divides by zero unless total portfolio demand is zero.
- Treats high-volume blockbuster SKUs proportionally higher than slow-moving long-tail products.
- Expressed directly as volume error percentage (e.g., $12.4\%$ forecast error).

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class RegressionMetrics:
    """
    Production regression metrics engine tailored for FMCG supply chain and quantitative finance.
    """
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
        """MAPE with masked zeros to prevent infinite explosion."""
        non_zero_mask = np.abs(y_true) > eps
        if np.sum(non_zero_mask) == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))) * 100.0

    @staticmethod
    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error (General Mills Gold Standard)."""
        tot_actual = np.sum(np.abs(y_true))
        if tot_actual == 0:
            return 0.0
        return float(np.sum(np.abs(y_true - y_pred)) / tot_actual) * 100.0

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))

    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return {
            "RMSE": round(cls.rmse(y_true, y_pred), 3),
            "MAE": round(cls.mae(y_true, y_pred), 3),
            "MAPE(%)": round(cls.mape(y_true, y_pred), 2),
            "WAPE(%)": round(cls.wape(y_true, y_pred), 2),
            "R2": round(cls.r2_score(y_true, y_pred), 4),
            "Bias_Units": round(float(np.mean(y_pred - y_true)), 3) # Over-forecasting indicator
        }

if __name__ == "__main__":
    # Simulating General Mills retail demand with occasional zero sales
    actual_demand = np.array([120, 0, 85, 310, 0, 95, 450, 200], dtype=float)
    predicted_demand = np.array([110, 5, 90, 290, 2, 105, 420, 215], dtype=float)
    
    res = RegressionMetrics.evaluate_all(actual_demand, predicted_demand)
    print(f"[Q11] Demand Forecasting Evaluation:\n{res}")
```

---

### Q12: Stratified K-Fold and TimeSeriesSplit Cross-Validation from Scratch [Medium]

#### 1. Explain the Concept & Mathematical Intuition

**1. Stratified K-Fold:**
Standard random K-Fold randomly assigns indices across folds. In datasets with severe class imbalance (e.g. 2% fraud), random sampling produces folds with 0.5% or 4.0% minority instances, producing catastrophic variance in fold validation metrics.
Stratified K-Fold partitions the samples such that **each fold contains the exact same percentage of each target class** as the complete dataset.

**2. TimeSeriesSplit (Walk-Forward Validation):**
In financial pricing (LSEG) and retail demand (General Mills), time is strictly forward-directed. Using random K-Fold uses future information $t+k$ to train a model predicting past events $t-j$, causing catastrophic **look-ahead bias** and artificially inflated backtest accuracy.
TimeSeriesSplit uses an expanding window:
- Fold 1: Train $[0 \dots T_1]$, Test $[T_1 \dots T_2]$
- Fold 2: Train $[0 \dots T_2]$, Test $[T_2 \dots T_3]$
- Fold $K$: Train $[0 \dots T_K]$, Test $[T_K \dots T_{K+1}]$

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from typing import Generator, Tuple

class StratifiedKFoldScratch:
    """Stratified K-Fold cross-validator implemented from scratch."""
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        y = np.asarray(y).ravel()
        n_samples = len(y)
        rng = np.random.RandomState(self.random_state)
        
        # Group indices by class label
        unique_classes = np.unique(y)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        fold_buckets = [[] for _ in range(self.n_splits)]
        
        for cls, idxs in class_indices.items():
            if self.shuffle:
                rng.shuffle(idxs)
            # Distribute class indices round-robin into fold buckets
            for i, idx in enumerate(idxs):
                fold_buckets[i % self.n_splits].append(idx)

        all_indices = np.arange(n_samples)
        for fold_idx in range(self.n_splits):
            test_idx = np.array(fold_buckets[fold_idx], dtype=np.int64)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]
            yield train_idx, test_idx

class TimeSeriesSplitScratch:
    """Expanding-window Time Series Cross-Validator (Zero Look-Ahead Bias)."""
    def __init__(self, n_splits: int = 4, test_size: int = 50):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        total_test = self.n_splits * self.test_size
        min_train_size = n_samples - total_test
        if min_train_size <= 0:
            raise ValueError("Test folds exceed dataset length.")

        for i in range(self.n_splits):
            train_end = min_train_size + i * self.test_size
            test_end = train_end + self.test_size
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            yield train_idx, test_idx

if __name__ == "__main__":
    np.random.seed(42)
    # Validate Stratified K-Fold maintains exact class ratio
    y_dummy = np.array([1]*20 + [0]*80) # 20% positive
    skf = StratifiedKFoldScratch(n_splits=5)
    for fold, (tr, te) in enumerate(skf.split(np.zeros((100, 2)), y_dummy)):
        pos_ratio = np.mean(y_dummy[te] == 1)
        print(f"[Q12] Fold {fold} Test Size: {len(te)} | Positive Class Ratio: {pos_ratio:.2%}")
```

---

## 4. Section 3: Imbalanced Learning & Feature Importance

### Q13: SMOTE (Synthetic Minority Over-sampling Technique) from Scratch + Focal Loss [Hard]

#### 1. Explain the Concept & Mathematical Intuition

**1. SMOTE (Chawla et al., 2002):**
Simple random oversampling duplicates minority instances, leading tree models to build narrow, overfitted leaves around identical points.
SMOTE generates *synthetic* points along the line segment connecting minority instances and their $k$-nearest minority neighbors:
$$x_{synthetic} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)$$
where $x_{zi}$ is a randomly chosen neighbor from the $k$-nearest neighbors of $x_i$ within the minority class.

> **CRITICAL INTERVIEW TRAP:** Never apply SMOTE on the full dataset before cross-validation! Doing so synthesizes points across the train-test boundary, leaking test fold distribution into training folds and yielding artificially inflated validation scores. SMOTE must be performed **strictly inside each training fold**.

**2. Focal Loss (Lin et al., 2017):**
When class imbalance is extreme (e.g., $1:1000$ in financial trading anomalies or credit fraud at LSEG), standard Cross-Entropy is overwhelmed by the cumulative loss of easily classified negative samples ($p \approx 0.001$).
Focal Loss adds a modulating factor $(1 - p_t)^\gamma$ to down-weight easy examples:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$
where $p_t = p$ if $y=1$ else $1-p$.
- When $\gamma = 0$, Focal Loss equals standard Cross-Entropy.
- When $\gamma = 2$ and an easy negative has $p_t = 0.99$, the modulating factor is $(1 - 0.99)^2 = 0.0001$, scaling down its gradient by $10,000\times$!

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class SMOTEScratch:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implemented from scratch.
    """
    def __init__(self, k_neighbors: int = 5, sampling_strategy: float = 1.0, random_state: int = 42):
        self.k = k_neighbors
        self.sampling_strategy = sampling_strategy  # Ratio of minority/majority desired
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        rng = np.random.RandomState(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        X_min = X[y == minority_class]
        n_min = len(X_min)
        n_maj = int(np.max(counts))

        target_n_min = int(n_maj * self.sampling_strategy)
        n_synthetic_needed = target_n_min - n_min
        if n_synthetic_needed <= 0:
            return X.copy(), y.copy()

        # Compute pairwise distance within minority samples
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A @ B.T
        sq_norms = np.sum(X_min ** 2, axis=1, keepdims=True)
        dists = sq_norms + sq_norms.T - 2.0 * (X_min @ X_min.T)
        np.fill_diagonal(dists, np.inf) # Exclude self
        
        # Get k-nearest neighbors indices for each minority sample
        k_nearest_indices = np.argsort(dists, axis=1)[:, :self.k]

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            # Pick random minority sample
            base_idx = rng.randint(n_min)
            # Pick random neighbor among its k nearest
            neighbor_idx = rng.choice(k_nearest_indices[base_idx])
            
            # Interpolate
            lam = rng.uniform(0.0, 1.0)
            diff = X_min[neighbor_idx] - X_min[base_idx]
            syn_point = X_min[base_idx] + lam * diff
            synthetic_samples.append(syn_point)

        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.concatenate([y, np.full(n_synthetic_needed, minority_class, dtype=np.int64)])
        return X_resampled, y_resampled

def focal_loss(y_true: np.ndarray, y_prob: np.ndarray, gamma: float = 2.0, alpha: float = 0.25) -> float:
    """Vectorized Binary Focal Loss."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    p_t = np.where(y_true == 1, y_prob, 1.0 - y_prob)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    
    loss = -alpha_t * ((1.0 - p_t) ** gamma) * np.log(p_t)
    return float(np.mean(loss))

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    # Severe 1:20 imbalance dataset
    X_imb = np.random.randn(210, 4)
    y_imb = np.array([0]*200 + [1]*10)

    smote = SMOTEScratch(k_neighbors=3, sampling_strategy=1.0)
    X_res, y_res = smote.fit_resample(X_imb, y_imb)
    print(f"[Q13] Original shape: {X_imb.shape} (Pos: {np.sum(y_imb==1)})")
    print(f"[Q13] Resampled shape: {X_res.shape} (Pos: {np.sum(y_res==1)}, Neg: {np.sum(y_res==0)})")
```

---

### Q14: Cost-Sensitive Threshold Tuning & G-Mean / F-Beta Optimization [Medium]

#### 1. Explain the Concept & Mathematical Intuition
In production risk systems at **LSEG**, predicting a default or fraud carries vastly different business consequences than a false alarm:
- **Cost of False Negative ($C_{FN}$):** An undetected fraudulent wire transfer of \$50,000 $\implies \text{Cost} = \$50,000$.
- **Cost of False Positive ($C_{FP}$):** Flagging a legitimate transaction for 2-minute manual review $\implies \text{Cost} = \$10$.

Using standard threshold $\tau = 0.5$ maximizes generic accuracy but hemorrhages enterprise capital.
We define the **Expected Cost Function**:
$$\text{Total Cost}(\tau) = C_{FP} \cdot FP(\tau) + C_{FN} \cdot FN(\tau)$$
The goal is to find $\tau^* = \arg\min_\tau \text{Total Cost}(\tau)$.

For metric-based threshold tuning without explicit financial dollars:
- **Geometric Mean (G-Mean):** $\sqrt{\text{Sensitivity} \times \text{Specificity}}$ (balances both classes simultaneously).
- **$F_\beta$ Maximization:** Maximizes recall weighted $\beta\times$ higher than precision.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

class ThresholdOptimizer:
    """Finds optimal classification threshold given an asymmetric cost matrix or target metric."""
    def __init__(self, cost_fp: float = 10.0, cost_fn: float = 500.0):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

    def find_cost_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        thresholds = np.linspace(0.01, 0.99, 100)
        best_cost = float("inf")
        best_tau = 0.5

        for tau in thresholds:
            y_pred = (y_prob >= tau).astype(int)
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            cost = (self.cost_fp * fp) + (self.cost_fn * fn)

            if cost < best_cost:
                best_cost = cost
                best_tau = tau

        return {"optimal_threshold": round(best_tau, 4), "minimal_cost": round(best_cost, 2)}

    def find_gmean_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        thresholds = np.linspace(0.01, 0.99, 100)
        best_gmean = -1.0
        best_tau = 0.5

        for tau in thresholds:
            y_pred = (y_prob >= tau).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            gmean = np.sqrt(sens * spec)

            if gmean > best_gmean:
                best_gmean = gmean
                best_tau = tau

        return {"optimal_threshold": round(best_tau, 4), "max_gmean": round(best_gmean, 4)}

if __name__ == "__main__":
    np.random.seed(42)
    y_true_sim = np.random.binomial(1, 0.05, size=1000)
    y_prob_sim = y_true_sim * 0.4 + np.random.beta(1, 10, size=1000)

    opt = ThresholdOptimizer(cost_fp=15.0, cost_fn=500.0)
    res_cost = opt.find_cost_optimal_threshold(y_true_sim, y_prob_sim)
    res_gmean = opt.find_gmean_optimal_threshold(y_true_sim, y_prob_sim)

    print(f"[Q14] Cost-Optimal Result: {res_cost}")
    print(f"[Q14] G-Mean Optimal Result: {res_gmean}")
```

---

### Q15: Permutation Feature Importance from Scratch vs Tree MDI Bias [Hard]

#### 1. Explain the Concept & Mathematical Intuition

**1. Tree-based Mean Decrease in Impurity (MDI / Gini Importance) Trap:**
In Random Forests and XGBoost, standard `feature_importances_` measures the total reduction of Gini impurity brought by that feature across all tree splits.
**Severe Flaws of MDI:**
- Heavily biased toward **high-cardinality features** (e.g., continuous IDs or timestamp seconds), because the tree can create many arbitrary splits that reduce training impurity without true generalization value.
- Computed on training data, reflecting training fit rather than out-of-fold generalization.

**2. Permutation Feature Importance (Breiman, 2001):**
Permutation importance measures how much the model's out-of-fold evaluation metric (e.g. PR-AUC or RMSE) degrades when a single feature's values are randomly shuffled (breaking its relationship with the target $y$):
$$\text{Importance}(j) = \text{Metric}(X_{val}, y) - \text{Metric}(X_{val}^{\text{perm}(j)}, y)$$

**Properties:**
- Model-agnostic (works for LightGBM, neural nets, linear models).
- Evaluated on hold-out test/validation data.
- Does not favor high-cardinality features.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

def permutation_feature_importance_scratch(model, X_val: np.ndarray, y_val: np.ndarray, 
                                           metric_fn, n_repeats: int = 5, random_state: int = 42) -> dict:
    """
    Computes permutation feature importance from scratch.
    
    Args:
        model: Trained estimator implementing predict() or predict_proba().
        X_val: Validation feature matrix.
        y_val: Validation true targets.
        metric_fn: Callable metric_fn(y_true, y_pred) where higher is better.
        n_repeats: Number of times to permute each feature.
    """
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    n_samples, n_features = X_val.shape
    rng = np.random.RandomState(random_state)

    # Compute baseline score on pristine validation set
    baseline_score = metric_fn(y_val, model.predict(X_val))
    importances_mean = np.zeros(n_features)
    importances_std = np.zeros(n_features)

    for col in range(n_features):
        scores_perm = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            # Shuffle the single column
            shuffled_idx = rng.permutation(n_samples)
            X_perm[:, col] = X_perm[shuffled_idx, col]

            # Evaluate model degradation
            score = metric_fn(y_val, model.predict(X_perm))
            scores_perm.append(baseline_score - score)

        importances_mean[col] = float(np.mean(scores_perm))
        importances_std[col] = float(np.std(scores_perm))

    return {
        "importances_mean": importances_mean,
        "importances_std": importances_std,
        "ranking": np.argsort(importances_mean)[::-1]
    }

# Verification block with Sklearn Random Forest
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    np.random.seed(42)

    # X0 is primary signal, X1 is noise, X2 is high-cardinality pure random float
    X_mat = np.random.randn(200, 3)
    y_vec = 4.0 * X_mat[:, 0] + np.random.randn(200) * 0.2

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_mat, y_vec)

    res = permutation_feature_importance_scratch(rf, X_mat, y_vec, metric_fn=r2_score, n_repeats=5)
    print(f"[Q15] Permutation Importances: {res['importances_mean'].round(4)}")
    print(f"[Q15] Tree MDI Importances:   {rf.feature_importances_.round(4)}")
    print("Feature 0 correctly dominates in Permutation Importance!")
```

---

## 5. Section 4: Production Scikit-learn Pipelines & Feature Engineering

### Q16: Production Pipeline with ColumnTransformer, Outlier Capper & Target Encoding [Medium]

#### 1. Explain the Concept & Mathematical Intuition
In production machine learning systems, data preprocessing steps (imputation, scaling, outlier handling, categorical encoding) must be strictly bundled with the estimator into an atomic scikit-learn `Pipeline`.

**Why Target Encoding with Smoothing?**
One-Hot Encoding breaks down when categorical features have high cardinality (e.g. 5,000 retail store IDs at General Mills or 10,000 security tickers at LSEG), exploding matrix sparsity and memory.
Target encoding replaces each categorical level $c$ with the expected value of target $y$:
$$S_c = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{global}}{n_c + m}$$
where $m$ is the empirical smoothing parameter (e.g. $m=10$).
- When category count $n_c$ is tiny ($n_c = 1$), $S_c \approx \bar{y}_{global}$ (prevents overfitting rare categories).
- When $n_c$ is large ($n_c = 10,000$), $S_c \approx \bar{y}_c$.

**Custom Transformer Contract:**
To seamlessly integrate with `Pipeline` and `GridSearchCV`, custom transformers must inherit from `BaseEstimator` and `TransformerMixin`, implement `fit(X, y)` and `transform(X)`, and **never use test data statistics during `fit`**.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    """Caps numeric features at empirical percentiles to eliminate extreme outliers."""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.lower_bounds_ = np.percentile(X_arr, self.lower_quantile * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_arr, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoder for high-cardinality categorical columns.
    S_c = (n_c * mean_c + m * global_mean) / (n_c + m)
    """
    def __init__(self, m_smoothing: float = 10.0):
        self.m_smoothing = m_smoothing
        self.global_mean_ = 0.0
        self.mapping_ = {}

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        self.global_mean_ = float(y_s.mean())
        self.mapping_ = {}

        for col in X_df.columns:
            stats = y_s.groupby(X_df[col]).agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            smoothed = (counts * means + self.m_smoothing * self.global_mean_) / (counts + self.m_smoothing)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_out = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            col_map = self.mapping_.get(col, {})
            # Map unseen test categories to global training mean
            X_out[col] = X_df[col].map(col_map).fillna(self.global_mean_)
        return X_out.values

def build_production_pipeline(numeric_cols, categorical_cols):
    """Builds a complete, leak-proof Scikit-learn Pipeline."""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', OutlierWinsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ('scaler', RobustScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_encoder', SmoothTargetEncoder(m_smoothing=10.0))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])
    return full_pipeline

# Verification block
if __name__ == "__main__":
    df_sample = pd.DataFrame({
        'store_id': ['Store_A', 'Store_B', 'Store_A', 'Store_C', 'Store_B', 'Store_D'],
        'sku_volume': [120.0, 95.0, 10000.0, 45.0, np.nan, 80.0], # Notice 10,000 outlier
        'price_promo': [1.99, 2.49, 1.49, 3.00, 2.10, 2.40],
        'target': [1, 0, 1, 0, 0, 1]
    })

    pipe = build_production_pipeline(numeric_cols=['sku_volume', 'price_promo'], 
                                     categorical_cols=['store_id'])
    
    pipe.fit(df_sample[['store_id', 'sku_volume', 'price_promo']], df_sample['target'])
    preds = pipe.predict(df_sample[['store_id', 'sku_volume', 'price_promo']])
    print(f"[Q16] Pipeline Fitted Successfully! Predictions: {preds}")
```

---

### Q17: Leak-Free Feature Selection Pipeline: VIF & Mutual Information [Hard]

#### 1. Explain the Concept & Mathematical Intuition
In quantitative factor modeling (LSEG) and marketing mix modeling (General Mills), features are frequently highly correlated. Multicollinearity inflates standard error of coefficients, rendering them unstable and uninterpretable.

**Variance Inflation Factor (VIF):**
Measures how much the variance of an estimated regression coefficient increases due to collinearity with other features:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination when regressing feature $x_j$ on all remaining $d-1$ features.
- $\text{VIF} = 1$: Completely orthogonal.
- $\text{VIF} > 5$: High multicollinearity; standard industry threshold to drop or consolidate features.

**Mutual Information (MI):**
Measures non-linear dependency between feature $X$ and target $Y$:
$$I(X; Y) = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} \, dx \, dy$$
Unlike Pearson correlation (which detects only linear relationships), MI captures complex non-linear relationships without parametric assumptions.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression

class VIFSelector(BaseEstimator, TransformerMixin):
    """
    Iteratively eliminates collinear features exceeding a VIF threshold.
    Strictly calculates collinearity within training folds.
    """
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.kept_indices_ = None

    @staticmethod
    def _calculate_vif(X: np.ndarray, col_idx: int) -> float:
        y_target = X[:, col_idx]
        X_other = np.delete(X, col_idx, axis=1)
        
        # Fit ordinary least squares to compute R^2
        # Augment with intercept
        X_aug = np.hstack([np.ones((len(X), 1)), X_other])
        theta, residuals, rank, s = np.linalg.lstsq(X_aug, y_target, rcond=None)
        
        y_pred = X_aug @ theta
        ss_res = np.sum((y_target - y_pred) ** 2)
        ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        # Clip R^2 to prevent division by exact zero
        r2 = min(r2, 0.99999)
        return float(1.0 / (1.0 - r2))

    def fit(self, X, y=None):
        X_mat = np.asarray(X, dtype=np.float64)
        n_features = X_mat.shape[1]
        active_indices = list(range(n_features))

        while len(active_indices) > 1:
            current_X = X_mat[:, active_indices]
            vifs = [self._calculate_vif(current_X, i) for i in range(len(active_indices))]
            max_vif = max(vifs)
            max_idx = int(np.argmax(vifs))

            if max_vif > self.threshold:
                dropped_orig_idx = active_indices.pop(max_idx)
            else:
                break

        self.kept_indices_ = active_indices
        return self

    def transform(self, X):
        X_mat = np.asarray(X)
        return X_mat[:, self.kept_indices_]

# Verification block
if __name__ == "__main__":
    np.random.seed(42)
    x1 = np.random.randn(200)
    x2 = x1 + np.random.randn(200) * 0.01  # Extreme collinearity (VIF > 1000)
    x3 = np.random.randn(200)               # Independent signal

    X_collinear = np.column_stack([x1, x2, x3])
    selector = VIFSelector(threshold=5.0)
    selector.fit(X_collinear)

    print(f"[Q17] Original Features: 3 | Selected Non-Collinear Indices: {selector.kept_indices_}")
```

---

## 6. Section 5: Advanced Hyperparameter Optimization

### Q18: Optuna Bayesian Optimization with MedianPruning & LightGBM [Medium]

#### 1. Explain the Concept & Mathematical Intuition

**Why Bayesian Optimization over Random/Grid Search?**
- **Grid Search:** Evaluates all parameter combinations naively ($\mathcal{O}(\prod K_i)$), wasting compute on unpromising sub-spaces.
- **Random Search:** Explores independently without learning from past trials.
- **Optuna (TPE - Tree-structured Parzen Estimator):**
  Models the probability distribution of hyperparameters conditioned on past evaluation scores using Bayes' rule:
  $$P(\theta | y) = \frac{P(y | \theta) P(\theta)}{P(y)}$$
  TPE splits past trials into two density groups: good trials $\ell(\theta)$ ($y < y^*$) and poor trials $g(\theta)$ ($y \ge y^*$).
  It then samples new hyperparameters that maximize the Expected Improvement (EI):
  $$\text{EI}(\theta) \propto \frac{\ell(\theta)}{g(\theta)}$$

**Pruning (MedianPruner / Asynchronous Successive Halving):**
Evaluates intermediate validation loss at step $t$. If a trial's metric is below the median of previous trials at step $t$, Optuna terminates the trial immediately (`optuna.TrialPruned`), saving up to 80% of GPU/CPU cloud compute.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

def run_optuna_hpo():
    """
    Production Optuna study with Tree-structured Parzen Estimator (TPE)
    and MedianPruning for Gradient Boosting.
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print("[Optuna not installed] Displaying production implementation code structure.")
        return

    # Synthetic financial risk dataset
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.binomial(1, 0.2, size=500)

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search spaces
        params = {
            'max_iter': trial.suggest_int('max_iter', 50, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True)
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]

            clf = HistGradientBoostingClassifier(**params, random_state=42)
            clf.fit(X_tr, y_tr)
            
            y_pred_prob = clf.predict_proba(X_va)[:, 1]
            fold_auc = roc_auc_score(y_val, y_pred_prob)
            val_scores.append(fold_auc)

            # Report intermediate score for pruning
            trial.report(fold_auc, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(val_scores))

    # Optuna study configuration
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=15, timeout=60)

    print(f"[Q18] Optuna Best Trial ROC-AUC: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")

if __name__ == "__main__":
    run_optuna_hpo()
```

---

### Q19: Nested Cross-Validation (Double K-Fold) for Unbiased Evaluation [Medium]

#### 1. Explain the Concept & Mathematical Intuition

**The Hyperparameter Selection Bias Trap:**
When standard K-Fold cross-validation is used to optimize hyperparameters (e.g. searching 100 parameter combinations via `GridSearchCV`), the reported best cross-validation score is **optimistically biased**. The hyperparameters were tuned specifically to maximize performance on those particular validation folds.

**Nested Cross-Validation (5x3 Double K-Fold Architecture):**
Completely separates hyperparameter tuning from model performance estimation:
- **Outer Loop (e.g., 5 folds):** Evaluates true generalization error of the entire model-building process. Data is split into Outer-Train and Outer-Test.
- **Inner Loop (e.g., 3 folds):** Operates *strictly inside Outer-Train* to tune hyperparameters via Grid/Random Search.
- The best hyperparameters from the inner loop are refitted on the full Outer-Train and tested on Outer-Test.

```
Full Dataset
├── Outer Fold 1: [ Outer-Test (20%) ] | [ Outer-Train (80%) ]
│                                           ├── Inner Fold 1 (Tune)
│                                           ├── Inner Fold 2 (Tune)
│                                           └── Inner Fold 3 (Tune) -> Pick Best Params -> Eval Outer-Test
├── Outer Fold 2 ...
```

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def nested_cross_validation(X: np.ndarray, y: np.ndarray, outer_splits: int = 5, inner_splits: int = 3):
    """
    Executes Nested Cross-Validation (Double K-Fold) to guarantee zero optimistic bias.
    """
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    outer_test_scores = []
    best_params_per_fold = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_tr, y_outer_tr = X[train_idx], y[train_idx]
        X_outer_te, y_outer_te = X[test_idx], y[test_idx]

        # Inner loop: tune hyperparameters strictly inside outer train
        inner_search = GridSearchCV(
            estimator=Ridge(),
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error'
        )
        inner_search.fit(X_outer_tr, y_outer_tr)
        
        best_model = inner_search.best_estimator_
        best_params_per_fold.append(inner_search.best_params_)

        # Outer loop evaluation: test on completely untouched outer test set
        y_pred = best_model.predict(X_outer_te)
        outer_rmse = np.sqrt(mean_squared_error(y_outer_te, y_pred))
        outer_test_scores.append(outer_rmse)

    return {
        "outer_rmse_mean": float(np.mean(outer_test_scores)),
        "outer_rmse_std": float(np.std(outer_test_scores)),
        "outer_scores": outer_test_scores,
        "selected_params": best_params_per_fold
    }

if __name__ == "__main__":
    np.random.seed(42)
    X_reg = np.random.randn(200, 8)
    y_reg = X_reg @ np.random.randn(8) + np.random.randn(200) * 0.5
    
    res = nested_cross_validation(X_reg, y_reg, outer_splits=5, inner_splits=3)
    print(f"[Q19] Nested CV Unbiased RMSE: {res['outer_rmse_mean']:.4f} +/- {res['outer_rmse_std']:.4f}")
    print(f"Selected alphas across outer folds: {[p['alpha'] for p in res['selected_params']]}")
```

---

## 7. Section 6: Domain-Specific Practical Machine Learning (General Mills & LSEG)

### Q20: [General Mills / FMCG] Hierarchical SKU Demand Forecasting with Quantile LightGBM [Hard]

#### 1. Explain the Concept & Mathematical Intuition
In consumer packaged goods (CPG) supply chains at **General Mills**, point forecasts (e.g. predicting a single expected sales number via MSE) are inadequate for inventory planning. An under-forecast leads to retail stockouts (lost revenue and retailer penalties), while an over-forecast leads to warehouse spoil and holding costs.

**Quantile Regression & The Pinball Loss:**
Rather than estimating the conditional mean $\mathbb{E}[y|X]$, Quantile Regression estimates specific conditional quantiles $q \in (0, 1)$:
- $q = 0.50$: Median forecast (baseline production schedule).
- $q = 0.90$: 90th percentile pessimistic demand surge (safety stock buffer).
- $q = 0.10$: 10th percentile conservative floor (prevents over-ordering raw ingredients like flour or sugar).

**The Asymmetric Pinball (Tilted Absolute) Loss Function:**
$$L_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y)) = \begin{cases} q (y - \hat{y}) & \text{if } y \ge \hat{y} \\ (1 - q)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$
When $q = 0.90$, under-predicting ($y \ge \hat{y}$) is penalized $9\times$ more severely than over-predicting ($y < \hat{y}$), forcing the model to output the 90th percentile.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

class FMCGDemandForecaster:
    """
    End-to-End Hierarchical Retail SKU Demand Forecasting Pipeline with
    Feature Engineering and Quantile Regression (P10, P50, P90).
    """
    def __init__(self, quantiles=[0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        self.models = {}

    @staticmethod
    def engineer_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers leak-free historical lags, rolling statistics, calendar seasonality,
        and promotional price indices.
        """
        df = df.sort_values(['sku_id', 'date']).copy()

        # 1. Historical Lagged Demand (Lagged by at least 7 days to match supply chain lead time)
        for lag in [7, 14, 28]:
            df[f'lag_demand_{lag}'] = df.groupby('sku_id')['sales_units'].shift(lag)

        # 2. Rolling Window Aggregations (Computed strictly on lagged values to avoid look-ahead)
        df['rolling_mean_7d'] = df.groupby('sku_id')['lag_demand_7'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['rolling_std_7d'] = df.groupby('sku_id')['lag_demand_7'].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)
        df['rolling_mean_28d'] = df.groupby('sku_id')['lag_demand_7'].transform(lambda x: x.rolling(28, min_periods=1).mean())

        # 3. Calendar & Promotional Seasonality Features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # 4. Promotional Elasticity Index
        df['price_discount_pct'] = (df['base_price'] - df['promo_price']) / (df['base_price'] + 1e-5)
        return df.dropna()

    def fit(self, df_train: pd.DataFrame, feature_cols: list):
        X = df_train[feature_cols].values
        y = df_train['sales_units'].values

        for q in self.quantiles:
            # HistGradientBoostingRegressor supports pinball quantile loss natively
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=150,
                learning_rate=0.08,
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict_intervals(self, df_test: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        X = df_test[feature_cols].values
        preds = pd.DataFrame(index=df_test.index)
        for q in self.quantiles:
            preds[f'pred_P{int(q*100)}'] = self.models[q].predict(X)
        return preds

# Verification block
if __name__ == "__main__":
    # Create 90-day synthetic daily sales data for General Mills SKUs
    dates = pd.date_range("2025-01-01", periods=100)
    data = []
    for sku in ['Cheerios_500g', 'Pillsbury_Atta_5kg']:
        for d in dates:
            base_sales = 200 if sku == 'Cheerios_500g' else 500
            promo = 1 if d.dayofweek in [4, 5] else 0 # Weekend promo
            noise = np.random.randn() * 30
            sales = int(base_sales * (1 + 0.4 * promo) + noise)
            data.append({
                'date': d, 'sku_id': sku, 'sales_units': max(0, sales),
                'base_price': 4.5, 'promo_price': 3.5 if promo else 4.5
            })

    raw_df = pd.DataFrame(data)
    feat_df = FMCGDemandForecaster.engineer_time_series_features(raw_df)
    
    features = ['lag_demand_7', 'lag_demand_14', 'lag_demand_28', 'rolling_mean_7d', 
                'rolling_std_7d', 'day_of_week', 'month', 'is_weekend', 'price_discount_pct']

    train_df = feat_df[feat_df['date'] < '2025-03-15']
    test_df = feat_df[feat_df['date'] >= '2025-03-15']

    forecaster = FMCGDemandForecaster(quantiles=[0.10, 0.50, 0.90])
    forecaster.fit(train_df, features)
    intervals = forecaster.predict_intervals(test_df, features)

    print("[Q20] FMCG Demand Quantile Forecast (P10, P50, P90) Sample:")
    print(intervals.head())
```

---

### Q21: [LSEG / Fintech] High-Frequency Financial Fraud Detection with 1:1000 Imbalance & Probability Calibration [Hard]

#### 1. Explain the Concept & Mathematical Intuition

**1. The Extreme Imbalance & Asymmetric Loss:**
In financial market trading and electronic payment networks at **LSEG**, fraudulent transactions represent fewer than $0.1\%$ ($1:1000$) of all events.
Standard tree algorithms optimize accuracy or generic log-loss, predicting near-zero probabilities for all events.
We tune `scale_pos_weight` in XGBoost/LightGBM:
$$\text{scale\_pos\_weight} = \frac{N_{negative}}{N_{positive}} = \frac{999}{1} \approx 1000$$
This multiplies the loss gradient for positive instances by $1000\times$, forcing the tree splits to isolate fraud clusters.

**2. Probability Calibration (Platt Scaling vs Isotonic Regression):**
Artificially boosting positive weights severely distorts the raw predicted probabilities ($\hat{p} \approx 0.6$ when true likelihood is $0.02$).
For regulatory capital requirements and downstream risk limits, we require **calibrated probabilities** where:
$$P(Y = 1 | \hat{P} = p) = p$$
- **Platt Scaling:** Fits a logistic regression model on top of uncalibrated model scores: $P(Y=1|s) = \frac{1}{1 + \exp(A \cdot s + B)}$. Parametric and effective for small validation sets.
- **Isotonic Regression:** Non-parametric piecewise constant isotonic fit. Effective when validation set is large ($N > 2000$).
- **Calibration Metric (Brier Score):** Mean squared error of calibrated probabilities: $\text{Brier} = \frac{1}{n} \sum (\hat{p}_i - y_i)^2$.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score

class LSEGFraudDetectionEngine:
    """
    High-Frequency Financial Transaction Fraud Detection Pipeline
    with Class Weighting and Platt/Isotonic Probability Calibration.
    """
    def __init__(self, pos_weight: float = 100.0, calibration_method: str = "sigmoid"):
        self.pos_weight = pos_weight
        self.calibration_method = calibration_method # 'sigmoid' (Platt) or 'isotonic'
        self.calibrated_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Calculate sample weights: scale positive instances by pos_weight
        sample_weights = np.where(y_train == 1, self.pos_weight, 1.0)

        base_estimator = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        # CalibratedClassifierCV wraps the estimator using out-of-fold calibration
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.calibration_method,
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
        return self

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrated_model.predict_proba(X)[:, 1]

    def evaluate_calibration(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        probs = self.predict_calibrated_proba(X_test)
        brier = brier_score_loss(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        # Empirical calibration curve (fraction of positives vs mean predicted value)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=5)
        return {
            "Brier_Score": round(brier, 5),
            "PR_AUC": round(pr_auc, 4),
            "Calibration_Prob_Pred": prob_pred.round(4).tolist(),
            "Calibration_Prob_True": prob_true.round(4).tolist()
        }

if __name__ == "__main__":
    np.random.seed(42)
    # Severe 1:500 class imbalance
    n_samples = 2000
    X_fin = np.random.randn(n_samples, 6)
    # Fraud trigger: high velocity + extreme amount
    fraud_logits = 3.0 * X_fin[:, 0] + 2.5 * X_fin[:, 1] - 5.0
    y_fin = (1.0 / (1.0 + np.exp(-fraud_logits)) > 0.8).astype(int)
    print(f"[Q21] Total Fraud Cases in Dataset: {np.sum(y_fin)} / {n_samples} ({np.mean(y_fin):.2%})")

    engine = LSEGFraudDetectionEngine(pos_weight=50.0, calibration_method="sigmoid")
    engine.fit(X_fin[:1500], y_fin[:1500])
    eval_res = engine.evaluate_calibration(X_fin[1500:], y_fin[1500:])
    print(f"[Q21] Evaluation: {eval_res}")
```

---

### Q22: [LSEG / Credit Risk] Credit Risk Scorecard: Weight of Evidence (WoE) & Information Value (IV) [Hard]

#### 1. Explain the Concept & Mathematical Intuition
Under banking regulations (**Basel III & IFRS 9**), credit models cannot be opaque black-box neural networks. Models approving corporate loans or credit limits must be fully auditable and monotonic. The global industry standard is the **WoE-Logistic Scorecard**.

**1. Weight of Evidence (WoE):**
Bins continuous or categorical feature into discrete buckets and measures the log ratio of good borrowers to bad borrowers:
$$\text{WoE}_i = \ln \left( \frac{\% \text{Good}_i}{\% \text{Bad}_i} \right) = \ln \left( \frac{N_{good, i} / N_{good, total}}{N_{bad, i} / N_{bad, total}} \right)$$
- If $\text{WoE}_i > 0$: Bin has higher proportion of good borrowers than bad borrowers (low risk).
- If $\text{WoE}_i < 0$: Bin has higher proportion of bad borrowers (high risk).

**2. Information Value (IV) - Feature Predictive Power:**
$$\text{IV} = \sum_{i=1}^B (\% \text{Good}_i - \% \text{Bad}_i) \times \text{WoE}_i$$
**Regulatory IV Heuristic Table:**
- $\text{IV} < 0.02$: Useless / Unpredictive
- $0.02 \le \text{IV} < 0.10$: Weak predictive power
- $0.10 \le \text{IV} < 0.30$: Medium predictive power (ideal for inclusion)
- $0.30 \le \text{IV} < 0.50$: Strong predictive power
- $\text{IV} \ge 0.50$: Suspicious / Likely target leakage!

**3. Credit Scorecard Scaling (300 to 850 range):**
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \quad \text{Offset} = \text{Target Score} - \text{Factor} \times \ln(\text{Target Odds})$$
where $\text{PDO}$ is Points to Double the Odds (typically $PDO = 20$).

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class CreditScorecardEngine:
    """
    IFRS-9 / Basel III Compliant Weight of Evidence (WoE) and 
    Information Value (IV) Regulatory Scorecard Engine.
    """
    def __init__(self, target_score: int = 600, target_odds: float = 50.0, pdo: int = 20):
        self.factor = pdo / np.log(2.0)
        self.offset = target_score - self.factor * np.log(target_odds)
        self.woe_dict = {}
        self.iv_dict = {}
        self.model = None

    def compute_woe_iv(self, df: pd.DataFrame, feature: str, target: str, bins: int = 5):
        df_feat = df[[feature, target]].copy()
        df_feat['bin'] = pd.qcut(df_feat[feature], q=bins, duplicates='drop')

        total_good = np.sum(df_feat[target] == 0) # 0: Non-default (Good)
        total_bad = np.sum(df_feat[target] == 1)  # 1: Default (Bad)

        grouped = df_feat.groupby('bin', observed=False)[target].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']

        # Avoid zero division
        grouped['dist_good'] = (grouped['good'] + 0.5) / total_good
        grouped['dist_bad'] = (grouped['bad'] + 0.5) / total_bad

        grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
        grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

        iv_total = float(grouped['iv'].sum())
        self.woe_dict[feature] = grouped['woe'].to_dict()
        self.iv_dict[feature] = iv_total
        return self.woe_dict[feature], iv_total

    def build_scorecard(self, df_woe: pd.DataFrame, target_col: str):
        X = df_woe.drop(columns=[target_col]).values
        y = df_woe[target_col].values

        self.model = LogisticRegression(solver='lbfgs', random_state=42)
        self.model.fit(X, y)
        return self

    def predict_score(self, df_woe: pd.DataFrame) -> np.ndarray:
        # Score = Offset - Factor * (w @ WoE + b)
        # Note: log(odds of good) = -(w @ WoE + b) when model predicts probability of default
        logits = self.model.decision_function(df_woe.values)
        scores = self.offset - self.factor * logits
        return np.clip(np.round(scores), 300, 850)

if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    # Features: Debt-to-Income (DTI), Revolving Utilization
    dti = np.random.uniform(0.1, 0.6, n)
    revolving_util = np.random.uniform(0.05, 0.95, n)
    default_prob = 1.0 / (1.0 + np.exp(-(5.0 * dti + 4.0 * revolving_util - 4.0)))
    defaults = (np.random.rand(n) < default_prob).astype(int)

    df_credit = pd.DataFrame({'dti': dti, 'util': revolving_util, 'default': defaults})

    engine = CreditScorecardEngine()
    woe_dti, iv_dti = engine.compute_woe_iv(df_credit, 'dti', 'default')
    woe_util, iv_util = engine.compute_woe_iv(df_credit, 'util', 'default')

    print(f"[Q22] DTI Information Value (IV): {iv_dti:.4f} (Strong Predictor)")
    print(f"[Q22] Utilization IV:            {iv_util:.4f} (Strong Predictor)")
```

---

### Q23: [General Mills / Supply Chain] Safety Stock & Reorder Point Optimization with Probabilistic ML [Hard]

#### 1. Explain the Concept & Mathematical Intuition
In consumer packaged goods operations at **General Mills**, stockouts create devastating empty shelf penalties from major retailers (Walmart, Target, Tesco).
Classical inventory control formulas assume constant demand $\mu$ and fixed lead time $L$. In practice, both demand $D$ and replenishment lead time $L$ are stochastic variables with uncertainty predicted by machine learning models.

**Mathematical Formulation:**
Let predicted daily demand be $D \sim \mathcal{N}(\mu_D, \sigma_D^2)$ and supplier transit lead time be $L \sim \mathcal{N}(\mu_L, \sigma_L^2)$.
By the law of total variance, demand during lead time has variance:
$$\sigma_{DDLT}^2 = \mu_L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2$$

**1. Safety Stock (SS):**
$$\text{SS} = Z_\alpha \cdot \sqrt{\mu_L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2}$$
where $Z_\alpha$ is the standard normal inverse CDF corresponding to the desired Service Level $\alpha$ (e.g., $95\% \to Z = 1.645$, $99\% \to Z = 2.326$).

**2. Reorder Point (ROP):**
$$\text{ROP} = (\mu_D \cdot \mu_L) + \text{SS}$$
When inventory on hand + on order drops to ROP, an automated purchase order is dispatched.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np
from scipy.stats import norm

class InventoryOptimizationEngine:
    """
    Probabilistic Safety Stock & Reorder Point (ROP) Calculator
    for Retail Distribution Centers (General Mills Supply Chain).
    """
    @staticmethod
    def calculate_reorder_policy(pred_demand_mean: float, pred_demand_std: float,
                                  lead_time_mean_days: float, lead_time_std_days: float,
                                  service_level: float = 0.95) -> dict:
        """
        Calculates optimal safety stock and reorder point under dual demand & lead time uncertainty.
        """
        z_score = float(norm.ppf(service_level))
        
        # Total variance during lead time (Law of Total Variance)
        variance_ddlt = (lead_time_mean_days * (pred_demand_std ** 2)) + \
                        ((pred_demand_mean ** 2) * (lead_time_std_days ** 2))
        sigma_ddlt = np.sqrt(variance_ddlt)

        safety_stock = z_score * sigma_ddlt
        expected_demand_during_lead_time = pred_demand_mean * lead_time_mean_days
        reorder_point = expected_demand_during_lead_time + safety_stock

        return {
            "Service_Level": f"{service_level * 100:.1f}%",
            "Z_Score": round(z_score, 3),
            "Expected_Lead_Time_Demand": round(expected_demand_during_lead_time, 1),
            "Safety_Stock_Units": int(np.ceil(safety_stock)),
            "Reorder_Point_Units": int(np.ceil(reorder_point))
        }

if __name__ == "__main__":
    # Cheerios Distribution Center Parameters
    # Daily sales mean = 500 cases, daily volatility = 120 cases
    # Lead time = 7 days, lead time shipping volatility = 1.5 days
    res_95 = InventoryOptimizationEngine.calculate_reorder_policy(
        pred_demand_mean=500.0, pred_demand_std=120.0,
        lead_time_mean_days=7.0, lead_time_std_days=1.5,
        service_level=0.95
    )
    res_99 = InventoryOptimizationEngine.calculate_reorder_policy(
        pred_demand_mean=500.0, pred_demand_std=120.0,
        lead_time_mean_days=7.0, lead_time_std_days=1.5,
        service_level=0.99
    )
    print(f"[Q23] General Mills Safety Stock at 95% Service Level: {res_95}")
    print(f"[Q23] General Mills Safety Stock at 99% Service Level: {res_99}")
```

---

### Q24: [Production ML / MLOps] Population Stability Index (PSI) & Characteristic Stability Index (CSI) [Medium]

#### 1. Explain the Concept & Mathematical Intuition
Once a model is deployed into production at **General Mills** or **LSEG**, real-world data distributions drift due to economic regime shifts, inflation, or seasonal shocks (Data Drift / Concept Drift).

**Population Stability Index (PSI):**
Measures the divergence between the baseline distribution (reference training set $E$) and the production serving distribution (actual production batch $A$):
$$\text{PSI} = \sum_{b=1}^B (A_b - E_b) \times \ln\left( \frac{A_b}{E_b} \right)$$
where $E_b$ and $A_b$ are the percentage of samples in bin $b$ for Expected and Actual distributions.
- **Symmetric & Non-negative:** $\text{PSI} \ge 0$, equals 0 if and only if distributions are identical.
- **Enterprise Industry Standard Thresholds:**
  - $\text{PSI} < 0.10$: No significant shift. The model remains healthy; continue normal serving.
  - $0.10 \le \text{PSI} \le 0.25$: Moderate shift. Issue warnings; monitor performance weekly.
  - $\text{PSI} > 0.25$: Significant population drift! Trigger automated model retraining and notify on-call MLOps.

#### 2. Complete, Runnable Python Implementation
```python
import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> dict:
    """
    Computes Population Stability Index (PSI) between baseline and production distributions.
    
    Args:
        expected: Baseline reference array (e.g. training set scores or feature values).
        actual: Production serving array (e.g. current month serving scores).
        n_bins: Number of quantiles.
        eps: Epsilon smoothing to prevent division by zero or log(0).
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    # Define bins strictly based on the baseline (expected) distribution quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    # Ensure strictly increasing edges to prevent duplicate boundaries
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Count occurrences in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions with epsilon smoothing
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    # PSI Formula: sum( (Actual - Expected) * ln(Actual / Expected) )
    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_components))

    # Actionable status
    if total_psi < 0.10:
        action = "STABLE: No drift detected."
    elif total_psi <= 0.25:
        action = "WARNING: Moderate drift. Monitor metrics closely."
    else:
        action = "CRITICAL ALERT: Severe population shift! Retrain model immediately."

    return {
        "PSI": round(total_psi, 4),
        "Action": action,
        "Bin_Contributions": psi_components.round(4).tolist()
    }

if __name__ == "__main__":
    np.random.seed(42)
    # Baseline credit scores: Normal(650, 50)
    baseline_scores = np.random.normal(650, 50, size=5000)
    
    # Stable production month: Normal(651, 51)
    stable_production = np.random.normal(651, 51, size=2000)
    
    # Economic recession month: Severe distribution shift to Normal(590, 65)
    drifted_production = np.random.normal(590, 65, size=2000)

    print("[Q24] Stable Month:", calculate_psi(baseline_scores, stable_production)['Action'])
    print("[Q24] Recession Month PSI:", calculate_psi(baseline_scores, drifted_production))
```

---

## 8. Section 7: Senior Data Scientist Quick Reference Cheat Sheet

### Big-O Algorithmic Complexity Table

| Algorithm | Training Time Complexity | Prediction Time Complexity | Memory (Space) Complexity | Hyperparameter Sensitivity | Primary Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear / Ridge Regression** | $\mathcal{O}(N \cdot D \cdot \text{iters})$ (GD) or $\mathcal{O}(D^3 + N \cdot D^2)$ (Normal Eq) | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | Low (tune $\lambda$) | Multicollinearity, non-linear relationships |
| **Logistic Regression** | $\mathcal{O}(N \cdot D \cdot \text{iters})$ | $\mathcal{O}(D)$ | $\mathcal{O}(D)$ | Low (tune $C$ or $\lambda$) | Non-linear boundaries, extreme class imbalance |
| **Decision Tree (CART)** | $\mathcal{O}(D \cdot N \log N \cdot \text{depth})$ | $\mathcal{O}(\text{depth})$ | $\mathcal{O}(\text{Nodes} \cdot D)$ | High (`max_depth`, `min_samples_split`) | High variance, overfitting without pruning |
| **Random Forest** | $\mathcal{O}(B \cdot D' \cdot N \log N \cdot \text{depth})$ | $\mathcal{O}(B \cdot \text{depth})$ | $\mathcal{O}(B \cdot \text{Nodes})$ | Low (robust to tuning) | Cannot extrapolate trends, large memory footprint |
| **Gradient Boosting (XGB/LGBM)** | $\mathcal{O}(M \cdot D' \cdot N \log N)$ (histogram: $\mathcal{O}(M \cdot K_{bins} \cdot D)$) | $\mathcal{O}(M \cdot \text{depth})$ | $\mathcal{O}(M \cdot \text{Nodes})$ | High (`learning_rate`, `num_leaves`, `subsample`) | Sensitive to noisy outliers, overfitting on small data |
| **K-Means Clustering** | $\mathcal{O}(N \cdot K \cdot D \cdot I)$ | $\mathcal{O}(K \cdot D)$ | $\mathcal{O}(K \cdot D + N)$ | Medium ($K$, initialization) | Fails on non-convex/unequal cluster sizes |
| **K-Nearest Neighbors (KNN)** | $\mathcal{O}(1)$ (Lazy learner) | $\mathcal{O}(N \cdot D)$ (brute) or $\mathcal{O}(D \log N)$ (KD-Tree) | $\mathcal{O}(N \cdot D)$ | High ($K$, distance metric) | Curse of dimensionality, agonizingly slow inference |
| **PCA** | $\mathcal{O}(\min(N^2 D, N D^2))$ | $\mathcal{O}(k \cdot D)$ | $\mathcal{O}(k \cdot D)$ | Low ($k$ components) | Captures linear variance only, sensitive to scale |

---

### General Mills & LSEG Interview Question Bank & Behavioral Scenarios

#### General Mills (GBS Mumbai / Bengaluru) Core Interview Themes

1. **Trade Promotion & Cannibalization Modeling:**
   - *Question:* "When we run a 30% discount promotion on Pillsbury 5kg flour, sales spike by 300%. However, 1kg flour sales drop by 40%, and post-promotion sales slump for two weeks. How do you design an ML framework to isolate true net incremental volume from cannibalization and pantry-loading?"
   - *Senior Answer Strategy:* Formulate a structural causal model or uplift model. Engineer cross-elasticity price features (ratio of promo price to substitute SKU prices), forward-looking promotional lag variables to capture forward-buying pantry depletion, and evaluate net contribution margin rather than raw gross units.

2. **Intermittent / Zero-Inflated Retail Demand:**
   - *Question:* "For slow-moving specialty snack SKUs, 70% of days have zero sales, with occasional bursts of 50 units. Why does standard RMSE / MSE fail, and how would you architect the forecasting engine?"
   - *Senior Answer Strategy:* Standard MSE predicts small fractional numbers (e.g. 0.4 units) every day, which is physically impossible and leads to constant stockouts during burst days. Propose a Two-Stage Hurdle / Tweedie compound Poisson-gamma model:
     - Stage 1: Binary classifier (e.g., LightGBM) predicting $P(\text{Demand} > 0)$.
     - Stage 2: Truncated regressor predicting demand volume conditional on non-zero sales.
     - Evaluate using WAPE and Quantile Pinball loss ($P_{90}$) to dimension safety buffers.

3. **Communicating ML to Non-Technical Brand Managers:**
   - *Question:* "The supply chain director refuses to trust your LightGBM forecast because it predicts a demand slump during a festival week where historical averages showed growth. How do you handle this disagreement?"
   - *Senior Answer Strategy:* Use SHAP (SHapley Additive exPlanations) force plots to decompose the prediction into transparent additive contributions: demonstrate that rising inflation (CPI feature) combined with a 25% price increase pushed the SKU past consumer price elasticity thresholds, overcoming the seasonal baseline. Walk through past historical instances where price elasticity counteracted seasonality.

---

#### LSEG (London Stock Exchange Group) Core Interview Themes

1. **Look-Ahead Bias & Non-Stationarity in Financial Time Series:**
   - *Question:* "You build an XGBoost model predicting intraday asset volatility that achieves a Sharpe ratio of 3.8 in backtesting, but collapses to negative returns in production paper trading. Walk me through your debugging checklist."
   - *Senior Answer Strategy:*
     - Check 1 (Target Leakage): Were technical indicators (e.g., 20-day rolling moving average) calculated using centered windows rather than backward-looking windows?
     - Check 2 (Cross-Validation Structure): Was random K-Fold used instead of expanding-window TimeSeriesSplit / PurgedGroupTimeSeriesSplit?
     - Check 3 (Information Embargo): In high-frequency order books, trades executed at $t$ cannot observe order book state at $t$ due to network latency and exchange matching engine delays (5-50ms). Did the backtest enforce an embargo period?
     - Check 4 (Execution Slippage & Market Impact): Did the backtest assume trades execute at midpoint without crossing the bid-ask spread or moving the market?

2. **Model Governance & Explainability for Regulatory Compliance (PRA / FCA / SEC):**
   - *Question:* "Why does LSEG still rely on Weight of Evidence (WoE) Logistic Regression scorecards for credit risk underwriting rather than deep neural networks or 500-tree ensembles?"
   - *Senior Answer Strategy:* Under financial regulations (FCRA, Basel III, IFRS 9), lenders must legally provide Adverse Action notices detailing the top 4 exact reasons why an applicant was rejected. In a WoE scorecard, each feature has an explicit additive point allocation. Neural networks and complex ensembles suffer from non-monotonicity (where slightly increasing an applicant's salary could paradoxically decrease their score due to high-dimensional interaction ripples), creating legal and regulatory exposure.

3. **High-Frequency Anomaly Detection Under Extreme Class Imbalance:**
   - *Question:* "We process 50,000 FIX protocol messages per second. Only 1 in 100,000 is an abusive market manipulation (e.g., spoofing or layering). How do you architect the streaming ML scoring engine?"
   - *Senior Answer Strategy:*
     - Offline: Train an Isolation Forest / Autoencoder for unsupervised novelty detection alongside an XGBoost model tuned with Focal Loss or `scale_pos_weight = 100000`.
     - Online Inference Architecture: A two-tier screening pipeline. Tier 1 executes in C++ / Rust using compiled ONNX / Treelite decision trees under 500 microseconds to filter out 99.9% of benign orders. Tier 2 applies full calibrated scoring on flagged suspicious transactions, queueing them into an event-driven Kafka broker for human compliance analyst review.

---

### Top 10 Live Coding Gotchas in Senior DS Technical Rounds

1. **Regularizing the Intercept:** Never include the bias $b$ or the constant column of 1s in $L_1$ or $L_2$ regularization penalties. Regularizing the bias forces the baseline prediction to zero regardless of the dataset scale.
2. **Data Leakage in Scalers/Imputers:** Always call `.fit()` or `.fit_transform()` on `X_train`, but strictly `.transform()` on `X_val` and `X_test`. Using test set statistics (e.g., test mean/std) is an immediate disqualification in live coding rounds.
3. **Applying SMOTE Before Cross-Validation:** Oversampling before splitting synthesizes points that span across fold boundaries. SMOTE must be contained strictly inside each CV training fold (e.g. using `imblearn.pipeline.Pipeline`).
4. **Sigmoid Floating-Point Overflow:** Directly calling `1.0 / (1.0 + np.exp(-z))` without `np.clip(z, -500, 500)` will trigger floating-point overflow warnings for negative inputs ($z < -709$).
5. **Evaluating Intermittent Demand with MAPE:** Using MAPE on sales data containing zero units results in division by zero (`inf`). Always use WAPE ($\frac{\sum |y - \hat{y}|}{\sum y}$) or MAE for CPG / retail demand forecasting.
6. **Inappropriate Cross-Validation on Temporal Data:** Never use standard shuffled K-Fold on time series, sales sequences, or financial ticks. Use expanding-window `TimeSeriesSplit` with zero shuffle.
7. **Tree MDI Feature Importance Bias:** Never rely on default `feature_importances_` in Random Forests / XGBoost when features have vastly different cardinalities. Use Permutation Importance or SHAP values.
8. **Inverting $X^T X$ Directly:** Avoid `np.linalg.inv(X.T @ X)` which is numerically unstable for ill-conditioned matrices. Use `np.linalg.solve(X.T @ X + lambda*I, X.T @ y)` or SVD.
9. **Uncalibrated Model Scores as Real Probabilities:** Tree ensembles output uncalibrated scores that do not reflect true statistical odds. Always calibrate via Platt Scaling (`CalibratedClassifierCV(method='sigmoid')`) or Isotonic Regression before making risk decisions.
10. **Optimistic Bias from Standard GridSearchCV:** Reporting the best score from a single CV hyperparameter search overestimates generalization accuracy. Use Nested Cross-Validation (Double K-Fold) for unbiased reporting.
