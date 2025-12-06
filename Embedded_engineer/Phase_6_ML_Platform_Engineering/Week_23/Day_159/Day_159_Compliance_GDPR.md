# Day 159: Example, Explanation, Exclusion: AI Compliance
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** You trained your model on `user_123`'s data. `user_123` emails you demanding deletion (GDPR Article 17). Deleting the database row is easy. But the model still remembers. What now? **Machine Unlearning**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a SISA (Sharded, Isolated, Sliced, Aggregated) training architecture for efficient unlearning.
2.  **Generate** Explanations for individual predictions using SHAP (Right to Explanation).
3.  **Audit** a dataset for Demographic Parity (Fairness).
4.  **Tag** artifacts with Data Provenance info for Regulatory Audits.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install shap fairlearn`.

---

## 📖 Theoretical Foundation

### 1. The Right to be Forgotten (RTBF)
If an EU citizen requests deletion, you must remove their data from *all* systems.
*   **Naive Approach:** Retrain from scratch. (Cost: $50k/request$).
*   **SISA Approach:**
    *   Split data into 10 shards. Train 10 sub-models. Ensemble them.
    *   If user in Shard 1 requests deletion, retrain *only* Shard 1's model. (10x faster).

### 2. The Right to Explanation
"Why was my loan denied?" (GDPR Article 22).
You cannot say "Black box said so". You must say "Because your Debt-to-Income ratio was > 40%".
*   **SHAP (Shapley Additive Explanations):** Game-theoretic approach to attribute payout to features.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: SISA Architecture

Simulate unlearning.

#### 📁 `src/sisa_trainer.py`
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

class SISAManager:
    def __init__(self, dataset, n_shards=5):
        self.shards = []
        self.models = []
        self.n_shards = n_shards
        
        # 1. Shard Data
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        shard_size = len(dataset) // n_shards
        
        for i in range(n_shards):
            start = i * shard_size
            end = (i+1) * shard_size
            subset = Subset(dataset, indices[start:end])
            self.shards.append(subset)
            
            # Initial Training
            print(f"Training Shard {i}...")
            model = self.train_shard(subset)
            self.models.append(model)
            
    def train_shard(self, dataset):
        model = nn.Linear(10, 2) # Dummy Model
        # ... training loop ...
        return model

    def predict(self, x):
        # Ensemble Vote
        votes = []
        for model in self.models:
            votes.append(model(x))
        return torch.stack(votes).mean(dim=0)
        
    def unlearn_request(self, user_id_index):
        # 2. Find which shard contains the user
        shard_idx = self.find_shard(user_id_index)
        
        print(f"Unlearning User {user_id_index} from Shard {shard_idx}")
        # Remove user from that shard's dataset
        new_shard_data = self.remove_data(self.shards[shard_idx], user_id_index)
        self.shards[shard_idx] = new_shard_data
        
        # 3. Retrain ONLY that model
        self.models[shard_idx] = self.train_shard(new_shard_data)
        print("Unlearning Complete. (Cost: 1/N of full retrain)")
```

### 👨‍💻 Core Implementation: SHAP Explanations

#### 📁 `src/shap_explain.py`
```python
import shap
import xgboost
import matplotlib.pyplot as plt

# Train model
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X, y)

# 1. Create Explainer
explainer = shap.Explainer(model)

# 2. Explain one prediction (Loan Denied)
shap_values = explainer(X.iloc[0:1])

# 3. Visualization
# Shows: Age (+2.1) pushes towards Deny. Income (-1.5) pushes towards Approve.
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("explanation.png")
```

### 👨‍💻 Core Implementation: Fairness Audit

#### 📁 `src/fairness_check.py`
```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score

# X has 'sex' column (sensitive attribute)
sensitive_feature = X['sex'] 

# Measure Accuracy for Men vs Women
metric_frame = MetricFrame(
    metrics=accuracy_score,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print(metric_frame.by_group)
# Female: 0.82
# Male:   0.95

# Audit Failure: Disparity > 10%
if abs(metric_frame.by_group['Female'] - metric_frame.by_group['Male']) > 0.1:
    print("FAIL: Model is biased.")
```

---

## 🔬 Lab Exercise: "The Auditor"

### Task
You are the Compliance Officer.
1.  **Scenario:** Engineering wants to deploy "Resume Ranker 3000".
2.  **Audit:** Run `fairness_check.py`.
3.  **Result:** Selection Rate for "Zip Codes in wealthy area" is 50%. Selection Rate for "Zip Codes in poor area" is 5%.
4.  **Verdict:** **Reject Deployment**. This violates "Disparate Impact" laws (Redlining), even if Zip Code isn't protected, it's a proxy for Race/Income.
5.  **Remediation:** Reweighting algorithms (Fairlearn) to balance the impact.

---

## 📖 Advanced Theory: Reproducibility Packet
For FDA (Medical AI), you must save:
1.  Original Container Image (SHA256).
2.  Exact Training Data (Hash).
3.  Random Seed.
4.  Hardware Driver Version (CUDA 11.2).
If you cannot reproduce the weight file bit-for-bit 5 years later, you are non-compliant.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Explainability:** Start with simple models (Trees) if explainability is a legal requirement. Deep Learning explanations (Integrated Gradients) are often unstable definitions.
2.  **Deletion:** "Soft Delete" (flag `is_deleted=True`) is fine for DBs. For ML, the model *is* the data. SISA or Linear layers are required.
3.  **Bias:** Bias usually comes from the data (historical racism/sexism), not the math. The algorithm just amplifies it.

### API Summary
```python
shap.Explainer(model)(input)
MetricFrame(metrics=..., sensitive_features=...)
```

---

**Day 159 Complete** ✅

*Next: Day 160 - Secure Deployment Ecosystems.*
