# 🔴 Hard: Advanced ML — PCA, Focal Loss, Conv2D, AUC from scratch
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Hard | **⭐ Senior DS**

---

## 🔴 PROBLEM 1: PCA from Scratch (SVD Method)

### Theory
PCA finds orthogonal directions of maximum variance in data.

**Algorithm:**
1. Center data: $X_c = X - \bar{X}$
2. Compute covariance matrix: $\Sigma = \frac{1}{n-1} X_c^T X_c$
3. Eigen decomposition: $\Sigma = V \Lambda V^T$
4. Sort eigenvectors by eigenvalues (descending)
5. Project: $X_{\text{pca}} = X_c V_k$ where $V_k$ = top-k eigenvectors

**Alternative via SVD:** $X_c = U \Sigma V^T$. PCA components = rows of $V^T$ (right singular vectors).

**Explained variance:**
$$\text{EVR}_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

### Things to Focus On
- ✅ ALWAYS center first (subtract mean)
- ✅ Scale features before PCA when units differ (or use correlation matrix)
- ✅ SVD is numerically more stable than eigendecomposition of covariance
- ✅ Choose k: 95% explained variance rule
- ✅ Limitations: linear only, sensitive to scale; use Kernel PCA for nonlinear

### Implementation
```python
import numpy as np

class PCA:
    """
    PCA via SVD (more numerically stable than eigendecomposition).
    """
    
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.components_ = None       # Principal components (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """Learn PCA components from data."""
        n, d = X.shape
        
        # Step 1: Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # Step 2: SVD of centered data
        # X_c = U @ S @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Step 3: Explained variance (S² / (n-1) = eigenvalues)
        explained_var = S ** 2 / (n - 1)
        total_var = explained_var.sum()
        
        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = explained_var / total_var
        
        # Step 4: Components are rows of Vt (top-k right singular vectors)
        k = self.n_components or d
        self.components_ = Vt[:k]   # (k, d)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # (n, k)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """Reconstruct original space from PCA projection."""
        return X_pca @ self.components_ + self.mean_
    
    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Find minimum components to explain threshold% of variance."""
        cumulative = np.cumsum(self.explained_variance_ratio_)
        return np.argmax(cumulative >= threshold) + 1

# Test
np.random.seed(42)
X = np.random.randn(100, 10)
# Add correlation structure
X[:, 1] = X[:, 0] + np.random.randn(100) * 0.1  # Correlated with dim 0

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")      # (100, 10)
print(f"PCA shape: {X_pca.shape}")       # (100, 3)
print(f"Explained variance ratio: {pca.explained_variance_ratio_[:3].round(3)}")
print(f"Cumulative variance (3 comps): {pca.explained_variance_ratio_[:3].sum():.3f}")
print(f"Components for 95% variance: {pca.n_components_for_variance(0.95)}")
```

---

## 🔴 PROBLEM 2: Focal Loss (Imbalanced Classification)

### Theory
Standard BCE treats all examples equally. **Focal Loss** down-weights easy examples:

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

Where:
- $p_t = p$ if y=1, else $p_t = 1-p$
- $(1-p_t)^\gamma$ = **modulating factor** — reduces weight of easy examples
- $\alpha_t$ = class weighting (handles class imbalance separately from γ)
- γ=0: reduces to BCE; γ=2 (default in paper)

**Example:** A well-classified negative (p=0.9 for class 0) has $(1-p_t)^\gamma = (0.1)^2 = 0.01$. Its contribution is ×100 smaller than in BCE.

### Things to Focus On
- ✅ γ (gamma) controls focus: γ=0 → BCE, γ=2 → strong focus on hard examples
- ✅ α vs γ: α controls class imbalance; γ reduces easy sample weight
- ✅ Better than oversampling in e-commerce fraud (no data augmentation needed)
- ✅ Original paper: RetinaNet object detection (1-stage detector with class imbalance)
- ✅ For multi-class: per-class focal factor, then sum over classes

### Implementation
```python
def focal_loss_binary(y_true: np.ndarray, y_pred: np.ndarray, 
                       gamma: float = 2.0, alpha: float = 0.25, 
                       eps: float = 1e-7) -> float:
    """
    Focal loss for binary classification.
    
    y_true: binary labels {0, 1}
    y_pred: predicted probabilities in (0, 1)
    gamma: focusing parameter (0 = BCE, 2 = default)
    alpha: class weight for positives
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # p_t: probability of correct class
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    
    # Class-specific alpha
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    # BCE for each sample
    bce = -np.log(pt)
    
    # Focal modulation
    focal_weight = (1 - pt) ** gamma
    
    return np.mean(alpha_t * focal_weight * bce)

def focal_loss_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                           gamma: float = 2.0, eps: float = 1e-7) -> float:
    """
    Focal loss for multi-class (uses softmax probabilities).
    y_true: (batch,) integer class labels
    y_pred: (batch, n_classes) softmax probabilities
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = len(y_true)
    
    # Probability of true class
    pt = y_pred[np.arange(n), y_true]   # (batch,)
    
    # Focal loss
    return np.mean(-(1 - pt) ** gamma * np.log(pt))

# Test — high imbalance scenario
np.random.seed(42)
n = 1000
y_true = np.zeros(n)
y_true[:50] = 1  # 5% positive (fraud-like)

# Model that outputs high confidence on negatives (easy examples)
y_pred = np.random.beta(1, 5, n)  # Mostly low probabilities
y_pred[:50] *= 2  # Slightly higher for positives

bce_l = float(-np.mean(y_true * np.log(np.clip(y_pred, 1e-7, 1)) + 
              (1-y_true) * np.log(np.clip(1-y_pred, 1e-7, 1))))
fl = focal_loss_binary(y_true, y_pred, gamma=2.0, alpha=0.75)

print(f"BCE:         {bce_l:.4f}")  # Dominated by easy negatives
print(f"Focal Loss:  {fl:.4f}")     # Better balanced
```

---

## 🔴 PROBLEM 3: Conv2D from Scratch

### Theory
Convolution (cross-correlation in practice) slides a kernel over input:

$$\text{Output}[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} \text{Input}[i \cdot s + m, j \cdot s + n] \cdot \text{Kernel}[m,n]$$

**Output size formula:**
$$H_{out} = \lfloor\frac{H_{in} + 2P - K}{S}\rfloor + 1$$

Where P=padding, K=kernel size, S=stride.

**Multiple channels:** For C_in input channels and C_out filters:
- Each filter: (C_in, K, K)
- Slide over input (C_in, H, W)
- Output: (C_out, H_out, W_out)

### Things to Focus On
- ✅ Cross-correlation NOT true convolution (flipped kernel) — PyTorch does cross-correlation
- ✅ "Same" padding: P = (K-1)//2 → maintains spatial size
- ✅ Stride > 1: downsamples spatially
- ✅ Number of parameters: C_out × C_in × K × K + C_out (biases)
- ✅ Pooling vs stride 2: both downsample; max pooling is non-learnable but translation invariant

### Implementation
```python
def conv2d(image: np.ndarray, kernel: np.ndarray, 
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D convolution (cross-correlation) for single input channel.
    
    image:  (H, W) — single channel
    kernel: (kH, kW)
    Returns: (H_out, W_out)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    
    # Pad image
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch of same size as kernel
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+kH, w_start:w_start+kW]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output

def conv2d_multichannel(X: np.ndarray, filters: np.ndarray, 
                         bias: np.ndarray = None,
                         stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Multi-channel conv2d.
    
    X:       (C_in, H, W)
    filters: (C_out, C_in, kH, kW)
    bias:    (C_out,)
    Returns: (C_out, H_out, W_out)
    """
    C_out, C_in, kH, kW = filters.shape
    C_in_x, H, W = X.shape
    assert C_in == C_in_x
    
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    output = np.zeros((C_out, H_out, W_out))
    
    # Pad each input channel
    if padding > 0:
        X = np.pad(X, ((0,0),(padding,padding),(padding,padding)), constant_values=0)
    
    for out_c in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * stride, j * stride
                patch = X[:, h_s:h_s+kH, w_s:w_s+kW]  # (C_in, kH, kW)
                output[out_c, i, j] = np.sum(patch * filters[out_c])
        if bias is not None:
            output[out_c] += bias[out_c]
    
    return output

# Test
# Edge detection kernel
image = np.array([
    [1,1,1,0,0],
    [1,1,1,0,0],
    [1,1,1,0,0],
    [0,0,0,1,1],
    [0,0,0,1,1],
], dtype=float)

# Sobel horizontal edge detector
kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
output = conv2d(image, kernel, padding=1)
print(f"Input shape:  {image.shape}")   # (5, 5)
print(f"Output shape: {output.shape}")  # (5, 5) — same padding
print(f"Edge map:\n{output.round(1)}")
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"When is Focal Loss better than weighted BCE?"** → When you have both hard and easy examples in the majority class. Weighted BCE down-weights ALL negatives. Focal Loss down-weights EASY negatives specifically, keeping hard negative examples at full weight.
2. **"PCA vs t-SNE — which for feature compression in production?"** → PCA (linear, fast, invertible, deterministic). t-SNE is for visualization only — non-parametric (can't transform new data without rerunning), non-invertible.
3. **"What does stride 2 do in convolution?"** → Halves spatial dimensions (downsampling). Alternative to max pooling for downsampling. Stride-2 conv is differentiable end-to-end; pooling is not (same issue as argmax).
4. **"How many parameters in Conv2D(64 filters, 3×3 kernel, 32 input channels)?"** → 64 × 32 × 3 × 3 + 64 = 18,496. Formula: C_out × C_in × kH × kW + C_out.
