# Day 152: The Gold Standard: A/B Testing Infrastructure
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** Your offline AUC increased by 0.05. Does that mean revenue goes up? Not necessarily. **A/B Testing** (Online Experimentation) is the *only* way to prove causality between Model V2 and Business Value.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Traffic Splitter mechanism (Hashing Strategy).
2.  **Calculate** the required sample size and duration for an experiment (Power Analysis).
3.  **Implement** an Experiment Assignment Service in Python.
4.  **Analyze** results using Frequentist Statistics (Z-Test) to reject the Null Hypothesis.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install statsmodels numpy`.

---

## 📖 Theoretical Foundation

### 1. Offline vs Online
*   **Offline:** Backtesting on past data. Metric: AUC/MSE. Faster. Cheaper. Proxy for success.
*   **Online:** Live traffic. Metric: CTR/Revenue/Conversion. Slower. Riskier. Real success.

### 2. Randomization Unit
Who gets assigned to Group A?
*   **User ID:** Consistent experience across devices. (Standard).
*   **Session ID:** Experience changes if user logs out. (Good for anonymous).
*   **Request ID:** Every click is random. (Bad for UX).

### 3. Hashing
To assign users deterministically without a database:
`Group = Hash(UserID + Salt) % 100`
If result < 50: Control. Else: Treatment.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Assignment Service

Stateless, deterministic assignment.

#### 📁 `src/ab_assignment.py`
```python
import hashlib

class Experiment:
    def __init__(self, name, salt, traffic_alloc=1.0, variants=["Control", "Treatment"]):
        self.name = name
        self.salt = salt
        self.traffic_alloc = traffic_alloc # fraction of Total Users to include
        self.variants = variants

    def assign(self, user_id):
        # 1. Traffic Allocation Check (Layer 1)
        # Should this user even be in the experiment?
        if not self._hash_check(user_id, "alloc", self.traffic_alloc):
             return "Excluded"
             
        # 2. Variant Assignment (Layer 2)
        # Assign to specific group
        # Assuming equal split for simplicity
        variant_idx = self._get_bucket(user_id) % len(self.variants)
        return self.variants[variant_idx]
        
    def _hash_check(self, user_id, suffix, threshold):
        val = self._get_bucket(user_id, suffix)
        return (val / 10000.0) < threshold

    def _get_bucket(self, user_id, suffix=""):
        # Deterministic Hash
        key = f"{self.salt}{user_id}{suffix}".encode("utf-8")
        hash_val = hashlib.sha1(key).hexdigest()
        # Take first 4 chars -> int -> Modulo 10000
        return int(hash_val[:4], 16) % 10000

# Usage
exp = Experiment("New_Ranking_V2", salt="exp_123", traffic_alloc=0.10)
# User 123 will ALWAYS get same assignment
print(f"User 123: {exp.assign('123')}") 
print(f"User 456: {exp.assign('456')}")
```

### 👨‍💻 Core Implementation: Power Analysis

How long do we run the test?

#### 📁 `src/power_analysis.py`
```python
import statsmodels.stats.api as sms
import math

# Baseline Conversion Rate: 10%
p1 = 0.10
# Expected Lift: +1% (Absolute) -> 11%
p2 = 0.11

# Power: 80% (Probability of detecting effect if it exists)
# Alpha: 5% (Probability of False Positive)
effect_size = sms.proportion_effectsize(p1, p2)
required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
)

required_n = math.ceil(required_n)
print(f"Required Sample Size per Group: {required_n}")
# If we have 1000 visitors/day -> We need ~2 * N / 1000 days.
```

### 👨‍💻 Core Implementation: Result Analysis

#### 📁 `src/analyze_ab.py`
```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Mock Data
# Control: 10000 users, 1000 conversions
n_con = 10000
conv_con = 1000

# Treatment: 10000 users, 1050 conversions
n_treat = 10000
conv_treat = 1050

# Z-Test
count = np.array([conv_con, conv_treat])
nobs = np.array([n_con, n_treat])

stat, p_val = proportions_ztest(count, nobs)
print(f"P-Value: {p_val:.4f}")

if p_val < 0.05:
    print("Result is Statistically Significant! Launch V2.")
else:
    print("Result is not significant. Do not launch.")
```

---

## 🔬 Lab Exercise: "SRM (Sample Ratio Mismatch)"

### Task
Debug a broken experiment.
1.  Assignment logic says 50/50 split.
2.  Observed Logs:
    *   Control: 10,000 users.
    *   Treatment: 8,000 users.
3.  **Check:** Chi-Square Goodness of Fit test on [10000, 8000] vs [9000, 9000]. P-Value ~ 0.
4.  **Cause:** Maybe "Treatment" page is crashing (500 error), so tracking logs aren't firing? Or latency is high and users bounce before assignment logs?
5.  **Action:** Invalidate experiment. Fix bug.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Peeking:** Do not check p-value every day and stop when "significant". This increases False Positive Rate. Use "Sequential Testing" frameworks if you must peek.
2.  **Interaction:** Experiment A (Blue Button) and Experiment B (New Font) running simultaneously might interact. Use "Orthogonal Layering" (separate salts) to minimize this.
3.  **Holdout:** Keep a permanent 1% Holdout group that NEVER gets new features. Helps measure long-term cumulative value of ML/AI.

### API Summary
```python
hashlib.sha1().hexdigest()
sms.NormalIndPower().solve_power()
proportions_ztest()
```

---

**Day 152 Complete** ✅

*Next: Day 153 - Shadow Deployments - The Silent Test.*
