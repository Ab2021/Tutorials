# Day 150: The Silent Killer: Model Drift Detection
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** Your model predicts housing prices. Interest rates triple. The model (trained on 2020 data) still predicts high prices. This is **Concept Drift**. If you don't detect it, you lose millions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Covariate Shift (Input P(X) changes) and Prior Probability Shift (Target P(Y) changes).
2.  **Calculate** PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) Statistics.
3.  **Implement** an online Drift Detector using `alibi-detect`.
4.  **Visualize** drift using overlapping histograms of Reference vs Current window.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install alibi-detect scikit-learn matplotlib scipy`.

---

## 📖 Theoretical Foundation

### 1. Types of Drift
*   **Covariate Shift:** The input distribution changes. (e.g., Users start using mobile instead of desktop). Model *might* still be valid if it generalized well.
*   **Prior Probability Shift:** The ratio of classes changes. (e.g., Holiday season = More fraud).
*   **Concept Drift:** The fundamental relationship $P(Y|X)$ changes. (e.g., "Flu symptoms" now means "COVID", not just "Flu"). Retraining is MANDATORY.

### 2. Detection Methods
*   **Univariate:** Monitor each feature independently (e.g., Mean of `age`).
*   **Multivariate:** Monitor the joint distribution (using Autoencoders).
*   **Statistical Tests:**
    *   **KS Test:** Good for numerical features. $D = \max|F_1(x) - F_2(x)|$.
    *   **Chi-Square:** Good for categorical.
    *   **PSI:** Industry standard in Finance. $PSI > 0.2$ = Panic.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Statistical Detection (Manual)

#### 📁 `src/04_drift_stats.py`
```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# 1. Generate Data
# Reference (Training)
ref_age = np.random.normal(30, 5, 1000)
# Current (Production) - Shifted Mean
curr_age = np.random.normal(35, 5, 1000)

# 2. KS Test
stat, p_value = ks_2samp(ref_age, curr_age)
print(f"KS Stat: {stat:.4f}, P-Value: {p_value:.4e}")

if p_value < 0.05:
    print("Drift Detected! (Distributions different)")
else:
    print("No Drift.")

# 3. PSI Calculation (Manual)
def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min_val, max_val):
        input += (1e-6) # Avoid zero
        input /= input.sum()
        return input

    # Binning
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), 
                              buckets + 1)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid div by zero
    expected_percents[expected_percents == 0] = 0.0001
    actual_percents[actual_percents == 0] = 0.0001
    
    psi = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_total = np.sum(psi)
    return psi_total

psi = calculate_psi(ref_age, curr_age)
print(f"PSI: {psi:.4f}")
# Rules of thumb: < 0.1 stable, 0.1-0.2 warning, > 0.2 drift
```

### 👨‍💻 Core Implementation: Alibi Detect (Online)

Alibi Detect handles complex data types (text, images, tabular).

#### 📁 `src/05_alibi_drift.py`
```python
from alibi_detect.cd import KSDrift
import numpy as np

# 1. Initialize Detector with Reference Data
X_ref = np.random.randn(500, 2) # 2 features
cd = KSDrift(X_ref, p_val=0.05)

# 2. Test No Drift
X_h0 = np.random.randn(500, 2)
preds = cd.predict(X_h0)
print(f"Drift? {preds['data']['is_drift']} (P-vals: {preds['data']['p_val']})")

# 3. Test Drift
X_h1 = np.random.randn(500, 2) + 1.0 # Shift mean
preds = cd.predict(X_h1)
print(f"Drift? {preds['data']['is_drift']} (P-vals: {preds['data']['p_val']})")

# Output:
# Drift? 0 (False)
# Drift? 1 (True)
```

### 👨‍💻 Infrastructure: Drift Monitoring Job

Usually run as a daily Batch Job (Airflow).
1.  Query Data Warehouse (Last 24h Inference Logs).
2.  Fetch Training Data Baseline.
3.  Run `KSDrift`.
4.  If Drift -> Trigger Retraining Pipeline (Git Dispatch).

---

## 🔬 Lab Exercise: "Adversarial Drift"

### Task
Fool the detector.
1.  Create a dataset where `mean` and `std` are same as reference, but shape is different (e.g., Gaussian vs Uniform).
2.  Run generic "Mean Check". It passes.
3.  Run `KSDrift`. It Fails (Correctly detects difference).
4.  **Insight:** Simple aggregation metrics ( Avg(Age) ) hide distribution changes. You need distribution-aware tests.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Reference Window:** What do you compare against? "Training Data" is the gold standard. Comparing to "Yesterday" is safer for sudden spikes but misses slow gradual drift.
2.  **Alerting:** Don't alert on *every* drift. Some drift is seasonal (Day vs Night). Alert on Sustained Drift or High Magnitude Drift (PSI > 0.2).
3.  **Remediation:**
    *   **Auto:** Retrain on new data (mix old + new).
    *   **Manual:** If Concept Drift is massive (e.g., new regulation), you might need to engineer new features.

### API Summary
```python
ks_2samp(data1, data2)
KSDrift(x_ref).predict(x_new)
```

---

**Day 150 Complete** ✅

*Next: Day 151 - Feature Stores - Solving Training-Serving Skew.*
