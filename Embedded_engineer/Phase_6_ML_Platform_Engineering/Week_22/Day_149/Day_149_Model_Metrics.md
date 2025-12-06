# Day 149: Beyond Accuracy: Model Quality Metrics
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** "Accuracy is 99%." Great, but you are predicting fraud, which is 1% of traffic. So your model just predicts "Legit" every time. It is useless. We must learn the language of **Precision, Recall, and Calibration**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the correct metric for Imbalanced Classification (Precision-Recall AUC vs ROC AUC).
2.  **Calculate** Expected Calibration Error (ECE) to ensure probability scores are realistic.
3.  **Evaluate** Ranking models using NDCG (Normalized Discounted Cumulative Gain).
4.  **Visualize** Decision Threshold trade-offs.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install scikit-learn matplotlib seaborn`.

---

## 📖 Theoretical Foundation

### 1. Classification Zoo
*   **Accuracy:** $\frac{TP+TN}{Total}$. Useless for imbalance.
*   **Precision:** $\frac{TP}{TP+FP}$. "When I say it's fraud, how often am I right?" (Cost of False Alarm).
*   **Recall:** $\frac{TP}{TP+FN}$. "How much fraud did I catch?" (Cost of Missed Detection).
*   **F1:** Harmonic mean.

### 2. Calibration
A model predicts 0.8 conf for 100 images. If it is **Well Calibrated**, exactly 80 of those images should be Positive.
Deep Learning models are notoriously *Overconfident* (predicting 0.99 for everything).

### 3. Ranking (Search/RecSys)
*   **MRR (Mean Reciprocal Rank):** Where is the first relevant item?
*   **NDCG:** Does the relevant item appear at the top? (Weighted by position).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Imbalanced Evaluation

#### 📁 `src/03_metrics.py`
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score, 
    calibration_curve, brier_score_loss
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# 1. Create Imbalanced Data (1% positive)
X, y = make_classification(
    n_samples=10000, n_features=20, n_classes=2, weights=[0.99, 0.01], random_state=42
)

# 2. Train Model
clf = LogisticRegression()
clf.fit(X, y)
probs = clf.predict_proba(X)[:, 1]

# 3. Basic Metrics
# AUROC is often misleadingly high for imbalance
auroc = roc_auc_score(y, probs)
# AUPRC is the gold standard for imbalance
auprc = average_precision_score(y, probs)

print(f"AUROC: {auroc:.3f} (Looks great)")
print(f"AUPRC: {auprc:.3f} (Truth revealed)")

# 4. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y, probs)

plt.figure()
plt.plot(recall, precision, label=f'AP={auprc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("pr_curve.png")

# 5. Calibration Plot (Reliability Diagram)
frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)

plt.figure()
plt.plot(mean_pred, frac_pos, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.savefig("calibration.png")

print(f"Brier Score (Lower is better): {brier_score_loss(y, probs):.4f}")
```

### 👨‍💻 Core Implementation: Ranking Metrics (NDCG)

```python
from sklearn.metrics import ndcg_score

# True relevance scores of documents (scale 0-5)
# Shape (n_queries, n_docs)
y_true = np.asarray([[5, 0, 2, 1, 0]])

# Model Scores (Predicted relevance)
y_score = np.asarray([[0.9, 0.1, 0.4, 0.2, 0.05]])

# NDCG @ K=3 (Only care about top 3 results)
ndcg = ndcg_score(y_true, y_score, k=3)
print(f"NDCG@3: {ndcg:.3f}")
```

---

## 🔬 Lab Exercise: "Threshold Tuning"

### Task
Optimize for Business Value.
1.  Scenario: Caught Fraud saves \$100. False Alarm costs \$10 (Manual Review). Missed Fraud costs \$0 (Lost opportunity... wait, usually Missed Fraud costs \$100).
2.  Let's say Missed Fraud = -\$100. False Alarm = -\$10. True Positive = \$0 (Prevention).
3.  **Profit Function:** $Profit = TP * 0 + TN * 0 + FP * (-10) + FN * (-100)$.
4.  Loop through prediction thresholds $t \in [0, 1]$.
5.  Calculate Confusion Matrix at $t$. Calculate Profit.
6.  **Find:** Optimal $t$ that maximizes Profit. It is rarely 0.5.

---

## 📖 Advanced Theory: Slice-Based Metrics
Aggregate AUC might be 0.9. But what is the AUC for "Users in Japan"?
*   **Slicing:** Evaluating model performance on subsets of data.
*   **Fairness:** Is the False Positive Rate same for Group A and Group B?
*   **Tools:** `Fairlearn`, `AIF360`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Imbalance:** If you have < 5% positives, ignore Accuracy/AUROC. Use Precision-Recall AUC.
2.  **Calibration:** If you use the probability for risk scoring (e.g., Loan Approval), the model MUST be calibrated. Use `CalibratedClassifierCV` (Isotonic Regression) to fix it.
3.  **Threshold:** The decision threshold (0.5) is a business decision, not a math decision.

### API Summary
```python
precision_recall_curve(y_true, y_scores)
calibration_curve(y_true, y_scores)
```

---

**Day 149 Complete** ✅

*Next: Day 150 - Drift Detection - When the world changes.*
