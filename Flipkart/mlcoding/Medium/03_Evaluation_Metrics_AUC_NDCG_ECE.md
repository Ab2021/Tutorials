# 🟡 Medium: Evaluation Metrics — AUC-ROC, NDCG, Precision@K, ECE
> **Platform:** Deep-ML + TensorTonic | **Difficulty:** Medium-Hard | **🔥 Flipkart Critical**

---

## WHY THESE METRICS MATTER FOR FLIPKART

Flipkart is a search + recommendation + fraud company. Every major outcome metric is:
- **AUC-ROC / PR-AUC** → fraud, classification
- **NDCG@K, Precision@K** → search ranking, recommendation
- **ECE** → model calibration for business decision-making

---

## 🟡 PROBLEM 1: AUC-ROC from Scratch

### Theory
**ROC Curve:** Plots TPR (recall) vs FPR as threshold varies from 1 → 0.

$$\text{TPR} = \frac{TP}{TP+FN} \quad \text{FPR} = \frac{FP}{FP+TN}$$

**AUC (Area Under Curve):** Probability that a random positive sample ranks higher than a random negative sample.

$$\text{AUC} = P(\hat{y}_{\text{pos}} > \hat{y}_{\text{neg}})$$

AUC = 0.5 → random classifier
AUC = 1.0 → perfect classifier

**Computation:** Trapezoidal rule on sorted thresholds.

**PR-AUC:** Better metric when classes are imbalanced. Plots precision vs recall.

### Things to Focus On
- ✅ AUC is threshold-invariant (evaluates the whole ranking)
- ✅ AUC = 0.7 for fraud means model ranks a random fraud above a random normal 70% of the time
- ✅ PR-AUC > ROC-AUC for imbalanced data (minority class focused)
- ✅ Efficient implementation: sort by score → O(n log n), compute incrementally

### Implementation
```python
import numpy as np
from typing import Tuple

def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve: FPR, TPR at each threshold.
    
    y_true: binary labels {0, 1}
    y_scores: predicted probabilities
    Returns: (fpr, tpr, thresholds) sorted by threshold descending
    """
    # Sort by score descending (highest confidence positives first)
    sort_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[sort_idx]
    y_scores_sorted = y_scores[sort_idx]
    
    P = y_true.sum()   # Total positives
    N = len(y_true) - P  # Total negatives
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    thresholds = [y_scores_sorted[0] + 1]  # Start above all scores
    
    tp = 0
    fp = 0
    
    for i, (label, score) in enumerate(zip(y_true_sorted, y_scores_sorted)):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        # Add point to curve at each unique threshold
        if i == len(y_true_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i+1]:
            tpr_list.append(tp / P)
            fpr_list.append(fp / N)
            thresholds.append(score)
    
    return np.array(fpr_list), np.array(tpr_list), np.array(thresholds)

def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    AUC-ROC using trapezoidal rule.
    
    Time: O(n log n) for sort + O(n) for integration
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Trapezoidal rule: area = sum of trapezoids
    auc = 0.0
    for i in range(1, len(fpr)):
        # Width of trapezoid * average height
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    return auc

def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Average Precision (AP) = area under PR curve.
    Better metric for imbalanced classification.
    """
    sort_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[sort_idx]
    
    P = y_true.sum()
    if P == 0:
        return 0.0
    
    tp_cumsum = np.cumsum(y_true_sorted)
    positions = np.arange(1, len(y_true) + 1)
    
    precisions = tp_cumsum / positions
    recalls = tp_cumsum / P
    
    # AP = sum of P(k) * ΔR(k) for each relevant document
    relevant_mask = y_true_sorted == 1
    return np.sum(precisions[relevant_mask] * (1 / P))

# Test
np.random.seed(42)
y_true  = np.array([1,1,1,0,0,0,0,0,1,0])
y_scores = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05])

print(f"AUC-ROC: {auc_roc(y_true, y_scores):.4f}")  # Should be high
print(f"AP:      {average_precision(y_true, y_scores):.4f}")

# Verify with sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
print(f"sklearn AUC: {roc_auc_score(y_true, y_scores):.4f}")
print(f"sklearn AP:  {average_precision_score(y_true, y_scores):.4f}")
```

---

## 🟡 PROBLEM 2: NDCG@K (Normalized Discounted Cumulative Gain)

### Theory
Measures ranking quality when items have graded (multi-level) relevance.

**DCG@K:**
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{r_i} - 1}{\log_2(i+1)}$$

where rᵢ is the relevance score of item at position i.

**NDCG@K:**
$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

IDCG@K = DCG of the ideal ranking (all highly relevant items first).
NDCG = 1.0 means perfect ranking; 0.0 means worst ranking.

**Position discount:** Position 1 has no discount (log₂(2)=1). Position 2: log₂(3)≈1.585. Items lower in list contribute less.

### Things to Focus On
- ✅ Binary relevance (0/1): NDCG reduces to simple precision-recall measure
- ✅ The 2^r - 1 factor: emphasizes highly relevant items (∆ between r=2 and r=1 >> r=1 and r=0)
- ✅ Flipkart search uses NDCG as key offline ranking metric
- ✅ NDCG handles ties by averaging? No — use different tiebreaking strategies

### Implementation
```python
def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain @ K.
    relevances: array of relevance scores in ranked order (position 0 = top)
    k: cutoff position
    """
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    
    # Position discount: log2(i + 2) for 0-indexed positions
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    
    # DCG = sum((2^r - 1) / log2(i+1))
    gains = (2 ** relevances - 1) / discounts
    return gains.sum()

def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    NDCG @ K: DCG normalized by ideal DCG.
    
    relevances: actual relevance scores in ranked order
    k: cutoff
    """
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg  = dcg_at_k(np.sort(relevances)[::-1], k)  # Best possible order
    
    if ideal_dcg == 0:
        return 0.0  # No relevant items → NDCG undefined
    
    return actual_dcg / ideal_dcg

def mean_ndcg_at_k(queries_relevances: list, k: int) -> float:
    """
    Mean NDCG@K over multiple queries (for system-level evaluation).
    queries_relevances: list of relevance arrays, one per query
    """
    scores = [ndcg_at_k(rel, k) for rel in queries_relevances]
    return np.mean(scores)

# Test — Flipkart search scenario
# Query: "iPhone case" — system returns 5 products
# Relevance: 0=irrelevant, 1=somewhat relevant, 2=highly relevant

# Scenario 1: Perfect ranking
perfect = [2, 2, 1, 1, 0]
print(f"Perfect NDCG@5: {ndcg_at_k(perfect, k=5):.4f}")  # 1.0

# Scenario 2: Worst ranking (reversed)
worst = [0, 1, 1, 2, 2]
print(f"Worst NDCG@5:   {ndcg_at_k(worst, k=5):.4f}")    # < 1.0

# Scenario 3: Typical model ranking
typical = [2, 1, 0, 1, 2]
print(f"Typical NDCG@5: {ndcg_at_k(typical, k=5):.4f}")  # Between 0 and 1

# System-level evaluation
multi_query = [
    [2, 2, 1, 0, 0],
    [1, 0, 2, 1, 0],
    [0, 0, 0, 1, 2],
]
print(f"Mean NDCG@5: {mean_ndcg_at_k(multi_query, k=5):.4f}")
```

---

## 🟡 PROBLEM 3: Precision@K and Recall@K

### Theory
For a ranked list of K items:

$$\text{Precision}@K = \frac{\text{# relevant items in top } K}{K}$$
$$\text{Recall}@K = \frac{\text{# relevant items in top } K}{\text{total relevant items}}$$

**Mean Average Precision (MAP):**
$$\text{AP} = \frac{1}{|R|}\sum_{k=1}^K P(k) \cdot \text{rel}(k)$$
$$\text{MAP} = \frac{1}{Q}\sum_{q=1}^Q \text{AP}_q$$

MAP averages precision at each relevant item's position → better than single P@K.

### Things to Focus On
- ✅ P@K ignores position (which of the K positions matters equally)
- ✅ Recall@K shows coverage (are all relevant items found?)
- ✅ MAP = system-level metric across queries; captures both precision and position
- ✅ Flipkart: precision@10 matters for above-the-fold results on mobile

### Implementation
```python
def precision_at_k(actual: list, predicted: list, k: int) -> float:
    """
    Precision @ K: fraction of top-K predictions that are relevant.
    
    actual: list of relevant item IDs
    predicted: ranked list of predicted item IDs (first = most relevant)
    """
    if k == 0:
        return 0.0
    top_k = set(predicted[:k])
    relevant = set(actual)
    return len(top_k & relevant) / k

def recall_at_k(actual: list, predicted: list, k: int) -> float:
    """Recall @ K: fraction of all relevant items found in top K."""
    if not actual:
        return 0.0
    top_k = set(predicted[:k])
    relevant = set(actual)
    return len(top_k & relevant) / len(relevant)

def average_precision_at_k(actual: list, predicted: list, k: int) -> float:
    """
    Average Precision (AP): averages precision at each relevant item's hit.
    Rewards systems that rank relevant items higher.
    """
    if not actual:
        return 0.0
    
    relevant = set(actual)
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(predicted[:k]):
        if item in relevant:
            hits += 1
            # Precision at position where this relevant item was found
            sum_precisions += hits / (i + 1)
    
    return sum_precisions / min(len(relevant), k)

def mean_average_precision(queries_actual: list, queries_predicted: list, k: int = None) -> float:
    """MAP: average AP across all queries."""
    aps = [
        average_precision_at_k(actual, predicted, k or len(predicted))
        for actual, predicted in zip(queries_actual, queries_predicted)
    ]
    return np.mean(aps)

# Test — E-commerce search
actual     = ['item_A', 'item_C', 'item_E']  # 3 relevant items for query
predicted  = ['item_A', 'item_B', 'item_C', 'item_D', 'item_E', 'item_F', 'item_G', 'item_H', 'item_I', 'item_J']

for k in [1, 3, 5, 10]:
    p = precision_at_k(actual, predicted, k)
    r = recall_at_k(actual, predicted, k)
    ap = average_precision_at_k(actual, predicted, k)
    print(f"K={k:2d}: P@K={p:.3f}, R@K={r:.3f}, AP@K={ap:.3f}")
```

---

## 🟡 PROBLEM 4: Expected Calibration Error (ECE)

### Theory
A **calibrated** model means: when it predicts 80% probability, it should be correct 80% of the time.

**ECE:**
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

Where:
- Bₘ = set of predictions falling in m-th bin
- acc(Bₘ) = fraction of correct predictions in bin
- conf(Bₘ) = average confidence in bin

Lower ECE = better calibration.

**Why calibration matters:**
- Fraud threshold tuning: if model says 70% fraud probability, need to know that's accurate
- Medical AI: "85% probability of disease" must mean exactly that
- Flipkart: pricing, inventory decisions depend on calibrated risk scores

**Calibration methods:**
- **Temperature scaling:** Divide logits by T (post-hoc)
- **Platt scaling:** Fit sigmoid on held-out set
- **Isotonic regression:** Non-parametric calibration

### Things to Focus On
- ✅ Typical number of bins: 10-15 (M=10 standard)
- ✅ Reliability diagram: plot conf vs acc per bin visually
- ✅ Perfect calibration: diagonal line on reliability diagram
- ✅ Overconfident model: above diagonal (says 90%, achieves 70%)
- ✅ Underconfident: below diagonal

### Implementation
```python
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    
    y_true: true binary labels {0, 1}
    y_prob: predicted probabilities in [0, 1]
    n_bins: number of equally-spaced bins
    
    Returns: ECE (lower = better)
    """
    n = len(y_true)
    bins = np.linspace(0, 1, n_bins + 1)  # Bin edges: 0, 0.1, 0.2, ..., 1.0
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        # Mask for predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        n_in_bin = in_bin.sum()
        
        if n_in_bin == 0:
            continue
        
        # Accuracy and confidence in this bin
        accuracy   = y_true[in_bin].mean()
        confidence = y_prob[in_bin].mean()
        
        # Weighted absolute difference
        ece += (n_in_bin / n) * abs(accuracy - confidence)
    
    return ece

def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, 
                              n_bins: int = 10) -> dict:
    """Compute data for reliability diagram (calibration curve)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if in_bin.sum() > 0:
            bin_accs.append(y_true[in_bin].mean())
            bin_confs.append(y_prob[in_bin].mean())
            bin_counts.append(in_bin.sum())
    
    return {'accuracies': bin_accs, 'confidences': bin_confs, 'counts': bin_counts}

# Test
np.random.seed(42)
n = 1000

# Simulate overconfident model (predicts near 1 too often)
y_true  = (np.random.rand(n) > 0.3).astype(int)  # 70% positive
y_prob_overconf = np.clip(y_true * 0.9 + 0.05 + np.random.randn(n) * 0.05, 0, 1)
y_prob_calibrated = y_true * 0.65 + 0.1 + np.random.randn(n) * 0.1

ece_overconf  = expected_calibration_error(y_true, np.clip(y_prob_overconf, 0, 1))
ece_calibrated = expected_calibration_error(y_true, np.clip(y_prob_calibrated, 0, 1))

print(f"ECE (overconfident): {ece_overconf:.4f}")  # Higher ECE (worse)
print(f"ECE (calibrated):    {ece_calibrated:.4f}") # Lower ECE (better)
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"When would you use PR-AUC vs ROC-AUC?"** → PR-AUC for severely imbalanced datasets (fraud: 0.01% base rate). ROC-AUC is optimistic because it treats all negatives equally (TN term). PR-AUC focuses on the minority (positive) class.
2. **"NDCG vs MAP — when to use which?"** → NDCG for graded relevance (highly/somewhat/not relevant). MAP for binary relevance (relevant/not). NDCG is more common in product search; MAP in information retrieval benchmarks.
3. **"Our model has ECE=0.15. How do we fix it?"** → Temperature scaling (cheapest), Platt scaling, isotonic regression. Then verify on a proper held-out calibration set to avoid overfitting the calibration.
4. **"Precision@K dropped after a model update. What could cause it?"** → Model redistributing confidence (threshold change), harder test queries, changed item catalog, feature distribution shift.
