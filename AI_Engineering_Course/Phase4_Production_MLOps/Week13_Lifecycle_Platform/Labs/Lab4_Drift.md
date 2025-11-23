# Lab 4: Data Drift Detector

## Objective
Data changes over time. Models rot.
Detect **Covariate Shift**.

## 1. The Detector (`drift.py`)

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

# 1. Reference Distribution (Training Data)
ref_dist = np.array([0.1, 0.2, 0.3, 0.4]) # Topic probabilities

# 2. Current Distribution (Production Data)
curr_dist = np.array([0.15, 0.25, 0.2, 0.4])

# 3. Calculate Drift (JS Divergence)
drift_score = jensenshannon(ref_dist, curr_dist)
print(f"Drift Score: {drift_score:.4f}")

THRESHOLD = 0.1
if drift_score > THRESHOLD:
    print("ALERT: Data Drift Detected!")
```

## 2. Analysis
If drift is high, trigger a **Retraining Pipeline**.

## 3. Submission
Submit the drift score for `curr_dist = [0.5, 0.1, 0.1, 0.3]`.
