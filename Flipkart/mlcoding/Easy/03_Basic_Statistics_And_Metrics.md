# 📊 Easy: Basic Statistics & Evaluation Metrics from Scratch
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Easy | **⭐ High Frequency**

---

## 🟢 PROBLEM 1: Mean, Median, Mode

### Theory
- **Mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ — sensitive to outliers
- **Median:** Middle value when sorted — robust to outliers
- **Mode:** Most frequent value — useful for categorical data

**When to use which:**
| Scenario | Use |
|---|---|
| Symmetric distribution, no outliers | Mean |
| Skewed distribution (income, house prices) | Median |
| Categorical or multimodal data | Mode |
| Reporting "average" salary at company | Median (CEO skews mean) |

### Things to Focus On
- ✅ Median with even number of elements: average of two middle values
- ✅ Multimodal distribution can have multiple modes
- ✅ For large datasets: use np.sort + indexing for median (O(n log n))
- ✅ Relationship: For right-skewed: Mean > Median > Mode

### Implementation
```python
import numpy as np
from collections import Counter
from typing import List, Union

def compute_mean(data: List[float]) -> float:
    """Arithmetic mean: sum / count"""
    if not data:
        raise ValueError("Empty data")
    return sum(data) / len(data)

def compute_median(data: List[float]) -> float:
    """
    Median: middle value (odd n) or average of two middle (even n).
    Sorts the data first.
    """
    if not data:
        raise ValueError("Empty data")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def compute_mode(data: List) -> List:
    """
    Mode: most frequent value(s).
    Returns list (handles multimodal case).
    """
    if not data:
        raise ValueError("Empty data")
    counts = Counter(data)
    max_count = max(counts.values())
    return [val for val, cnt in counts.items() if cnt == max_count]

def mean_median_mode(data: List[float]) -> dict:
    """Combined summary statistics."""
    return {
        'mean': compute_mean(data),
        'median': compute_median(data),
        'mode': compute_mode(data)
    }

# Test
data = [1, 2, 2, 3, 4, 5, 5, 5, 100]
result = mean_median_mode(data)
print(result)
# {'mean': 14.11,  ← pulled up by outlier 100
#  'median': 4,   ← robust to outlier
#  'mode': [5]}   ← most frequent
```

---

## 🟢 PROBLEM 2: Mean Squared Error (MSE)

### Theory
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$$

Also know:
- **RMSE:** $\sqrt{\text{MSE}}$ — same units as target
- **MAE:** $\frac{1}{n}\sum |y_i - \hat{y}_i|$ — robust to outliers
- **Huber Loss:** MSE for small errors, MAE for large (best of both)

**Gradient of MSE** w.r.t. predictions:
$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

### Things to Focus On
- ✅ MSE penalizes large errors heavily (squared) → sensitive to outliers
- ✅ RMSE is in the same units as y (interpretable)
- ✅ MAE = L1 loss; MSE = L2 loss
- ✅ For regression tasks: report both MSE and MAE alongside each other
- ✅ Know Huber loss as the practical choice for robustness

### Implementation
```python
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root MSE — in same units as target."""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — robust to outliers."""
    return np.mean(np.abs(y_pred - y_true))

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber loss: MSE for small errors, MAE for large errors.
    delta controls the transition point.
    """
    error = y_pred - y_true
    abs_error = np.abs(error)
    return np.mean(np.where(
        abs_error <= delta,
        0.5 * error ** 2,                    # MSE region
        delta * (abs_error - 0.5 * delta)    # MAE region
    ))

# Test
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0,  2.0, 8.0])
print(f"MSE:   {mse(y_true, y_pred):.4f}")    # 0.375
print(f"RMSE:  {rmse(y_true, y_pred):.4f}")   # 0.612
print(f"MAE:   {mae(y_true, y_pred):.4f}")    # 0.5
print(f"Huber: {huber_loss(y_true, y_pred):.4f}") # 0.1875
```

---

## 🟢 PROBLEM 3: R² Score (Coefficient of Determination)

### Theory
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.0: Model is as good as predicting the mean
- R² < 0: Model is worse than predicting the mean (bad model!)

**Intuition:** What fraction of the total variance in y does the model explain?

### Things to Focus On
- ✅ R² can be negative — this is valid and means a bad model
- ✅ R² is not MSE normalized by y's mean (common misconception)
- ✅ Adjusted R² accounts for number of features: $1 - (1-R^2)\frac{n-1}{n-p-1}$
- ✅ R² = 0.8 means model explains 80% of target variance — good rule of thumb

### Implementation
```python
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot
    
    SS_res = sum of squared residuals (prediction errors)
    SS_tot = total sum of squares (variance in truth)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        # All true values are the same (no variance to explain)
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - ss_res / ss_tot

# Test
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_good = np.array([2.5, 0.0, 2.0, 8.0])
y_pred_mean = np.full_like(y_true, y_true.mean())  # "dumb" model
y_pred_bad  = np.array([10.0, 10.0, 10.0, 10.0])  # Really bad

print(r2_score(y_true, y_pred_good))  # ~0.948 — good model
print(r2_score(y_true, y_pred_mean))  # 0.0    — baseline
print(r2_score(y_true, y_pred_bad))   # ~ -7   — worse than baseline!
```

---

## 🟢 PROBLEM 4: Binary Cross-Entropy Loss

### Theory
$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

This is the **negative log-likelihood** of a Bernoulli model.

**Why log?** 
- Perfect prediction (ŷ=1 when y=1): log(1) = 0 — no loss
- Wrong prediction (ŷ→0 when y=1): log(0) → ∞ — infinite loss
- Heavily penalizes confident wrong predictions

**Gradient:** $\frac{\partial \text{BCE}}{\partial \hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$

### Things to Focus On
- ✅ ALWAYS clip predictions: `np.clip(y_pred, eps, 1-eps)` — log(0) = -inf!
- ✅ Accepts probabilities (0,1), NOT raw logits
- ✅ Combined with sigmoid → numerically stable: use `log_sigmoid + log_softmax`
- ✅ For multi-class use Categorical Cross-Entropy (softmax output)

### Implementation
```python
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, 
                          eps: float = 1e-7) -> float:
    """
    Binary Cross-Entropy Loss.
    
    Args:
        y_true: Binary labels {0, 1}
        y_pred: Predicted probabilities in (0, 1)
    """
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                               eps: float = 1e-7) -> float:
    """
    Categorical Cross-Entropy (multi-class).
    y_true: one-hot encoded, shape (n, C)
    y_pred: softmax probabilities,  shape (n, C)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

# Test
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.4])
print(binary_cross_entropy(y_true, y_pred))  # ~0.299 — low loss (mostly correct)

y_pred_bad = np.array([0.1, 0.9, 0.2, 0.6])
print(binary_cross_entropy(y_true, y_pred_bad))  # ~1.97 — high loss ✓
```

---

## 🟢 PROBLEM 5: Classification Metrics (P, R, F1, Accuracy)

### Theory
From the confusion matrix:
```
                Predicted +    Predicted -
Actual +     [ TP           | FN          ]
Actual -     [ FP           | TN          ]
```

$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$
$$\text{F1} = \frac{2 \cdot P \cdot R}{P + R} = \frac{2TP}{2TP + FP + FN}$$
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**F-beta score** (generalized):
$$F_\beta = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

- β=1: F1 (equal weight)
- β=2: F2 (recall weighted more — fraud detection)
- β=0.5: F0.5 (precision weighted more — spam filter)

### Things to Focus On
- ✅ For imbalanced data: accuracy is misleading! (99% acc on 1% fraud rate by predicting all 0)
- ✅ Precision = "Of all my alarms, how many were real?"
- ✅ Recall = "Of all real fraud, how many did I catch?"
- ✅ F1 = harmonic mean (punishes extreme imbalance between P and R)
- ✅ Know when to optimize P vs R: fraud → high recall; spam → high precision

### Implementation
```python
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all binary classification metrics.
    y_true, y_pred: binary arrays {0, 1}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / len(y_true)
    
    return {
        'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'accuracy':  round(accuracy, 4)
    }

def f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
    """F-beta score. beta > 1 weights recall more; beta < 1 weights precision more."""
    m = classification_metrics(y_true, y_pred)
    p, r = m['precision'], m['recall']
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)

def confusion_matrix_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return normalized confusion matrix (row-wise = recall perspective)."""
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1
    
    # Normalize by true class counts (row sums)
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / np.maximum(row_sums, 1)  # avoid division by zero

# Test — imbalanced example
y_true = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])  # 5 positives, 10 negatives
y_pred = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,1])  # Predict most as negative

m = classification_metrics(y_true, y_pred)
print(m)
# accuracy = 0.867 — looks good!
# recall   = 0.400 — catches only 40% of fraud — BAD!
# precision = 0.667
```

---

## 🟢 PROBLEM 6: Mean/Median/Percentiles

```python
def percentile_manual(data: List[float], p: float) -> float:
    """
    Compute the p-th percentile (0 ≤ p ≤ 100).
    Linear interpolation between adjacent values.
    """
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Index (can be fractional)
    idx = (p / 100) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

def iqr(data: List[float]) -> float:
    """Interquartile range: Q3 - Q1. Robust measure of spread."""
    return percentile_manual(data, 75) - percentile_manual(data, 25)

# Test
data = list(range(1, 11))  # [1, 2, ..., 10]
print(percentile_manual(data, 50))   # 5.5 (median)
print(percentile_manual(data, 25))   # 3.25 (Q1)
print(percentile_manual(data, 75))   # 7.75 (Q3)
print(iqr(data))                     # 4.5
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"When would you use MAE vs MSE?"** → MAE when outliers shouldn't dominate (house prices, wages). MSE when you want to penalize large errors heavily (safety-critical).
2. **"What does R² = 0 mean?"** → The model is no better than predicting the mean of y every time.
3. **"Why is accuracy a bad metric for fraud?"** → Baseline "predict all-normal" gets 99%+ accuracy on 1% fraud rate, but catches 0 fraud. Use Precision/Recall/F1 or PR-AUC.
4. **"What's the difference between F1 and F2?"** → F2 weights recall twice as much as precision. Use F2 when missing a fraud (false negative) is worse than a false alarm.
