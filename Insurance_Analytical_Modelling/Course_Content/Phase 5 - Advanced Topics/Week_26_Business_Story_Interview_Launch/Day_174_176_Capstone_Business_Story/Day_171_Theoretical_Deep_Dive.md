# Full Capstone Project Build (Part 1) - Project Scoping & Data Pipeline - Theoretical Deep Dive

## Overview
"This is it. The Final Boss."
You have learned GLMs, GBMs, Credibility, Reserving, and MLOps.
Now, you must put it all together.
Over the next 3 days, we will design an **End-to-End Insurance Pricing Engine**.
Part 1 focuses on the foundation: **Scoping the Problem** and **Building the Data Pipeline**.

---

## 1. Conceptual Foundation

### 1.1 The Business Problem

*   **Scenario:** You are the Lead Data Scientist at "SafeDrive Insurance".
*   **Issue:** The Loss Ratio in Florida has spiked to 85% (Target is 65%).
*   **Hypothesis:** Our current rating plan (GLM built in 2019) is outdated. It underprices high-risk drivers and overprices low-risk drivers (Adverse Selection).
*   **Goal:** Build a new Pricing Model (GBM) to segment risk more accurately and restore profitability.

### 1.2 The Data Ecosystem

Insurance data is messy and fragmented.
1.  **Policy System (Guidewire):** Who are they? (Age, Vehicle, Zip).
2.  **Claims System (Duck Creek):** Did they crash? (Frequency, Severity).
3.  **Billing System:** Did they pay? (Retention, Cancellations).
4.  **External Data:** Credit Score, Telematics, MVR (Motor Vehicle Report).

---

## 2. Mathematical Framework

### 2.1 The Target Variable

What are we predicting?
*   **Pure Premium:** $PP = \text{Frequency} \times \text{Severity}$.
*   **Tweedie Distribution:** Models Pure Premium directly (Compound Poisson-Gamma).
    *   $Y \sim \text{Tweedie}(\mu, p, \phi)$.
    *   $1 < p < 2$ (Point mass at zero, continuous positive tail).

### 2.2 Exposure Calculation

*   **Earned Exposure:** The fraction of the year the policy was active.
    *   If a policy starts Jan 1 and cancels Jan 31, Exposure = $31/365 \approx 0.085$.
*   **Why it matters:** If you ignore exposure, a 1-month policy looks "safer" than a 12-month policy (fewer claims), but the *rate* might be the same.

---

## 3. Theoretical Properties

### 3.1 Feature Engineering Strategy

Raw data is useless. We need *features*.
1.  **Vehicle Age:** `Current Date - Model Year`.
2.  **Driver Tenure:** `Current Date - First Licensed Date`.
3.  **Prior Claims:** Count of claims in the last 3 years (Crucial predictor).
4.  **Geo-Spatial:** Population Density of the Zip Code (Proxy for traffic).

### 3.2 One-Way Analysis (Univariate)

*   **Check:** Before modeling, plot `Average Loss Cost` vs. `Driver Age`.
*   **Expectation:** U-Shape (High for young, Low for middle, High for old).
*   **Validation:** If the data doesn't show this, your data is wrong.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The Data Pipeline (Pandas)

```python
import pandas as pd
import numpy as np

# 1. Load Data
policies = pd.read_csv("policy_data.csv")
claims = pd.read_csv("claims_data.csv")

# 2. Join (Left Join Policy to Claims)
# Note: One policy can have multiple claims.
# We aggregate claims to the policy level first.
claims_agg = claims.groupby("policy_id").agg({
    "claim_count": "sum",
    "incurred_loss": "sum"
}).reset_index()

df = pd.merge(policies, claims_agg, on="policy_id", how="left")

# 3. Fill Nulls (No claim = 0 loss)
df["claim_count"] = df["claim_count"].fillna(0)
df["incurred_loss"] = df["incurred_loss"].fillna(0)

# 4. Calculate Exposure
df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"] = pd.to_datetime(df["end_date"])
df["exposure"] = (df["end_date"] - df["start_date"]).dt.days / 365.0

# 5. Feature Engineering
df["vehicle_age"] = 2024 - df["vehicle_year"]
df["driver_age_bucket"] = pd.cut(df["driver_age"], 
                                 bins=[16, 25, 50, 75, 100], 
                                 labels=["Young", "Prime", "Senior", "Elderly"])

# 6. Filter
# Remove policies with 0 exposure or negative premium
df = df[df["exposure"] > 0]
df = df[df["written_premium"] > 0]

print(f"Modeling Dataset Shape: {df.shape}")
```

### 4.2 Data Dictionary (Metadata)

| Feature | Type | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| `policy_id` | ID | Unique Key | N/A |
| `exposure` | Numeric | Weight | (0, 1] |
| `incurred_loss` | Target | Total Loss | [0, Inf) |
| `vehicle_age` | Feature | Age of Car | [0, 50] |

---

## 5. Evaluation & Validation

### 5.1 Data Integrity Checks

*   **Duplicates:** `assert df['policy_id'].is_unique`.
*   **Leakage:** Does the dataset contain "Future Info"? (e.g., `cancellation_date` might imply a claim happened).
*   **Completeness:** Do we have coverage for all Zip Codes in Florida?

### 5.2 Train/Test/Time Split

*   **Method:** **Time-Based Split**.
    *   Train: 2018-2021.
    *   Test: 2022.
*   **Why:** Random split leaks future trends (inflation) into the past. We must test if the model can predict *the future*.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Capped Loss" Problem

*   **Issue:** One claim for \$1M ruins the model (outlier).
*   **Fix:** Cap losses at \$100k (Large Loss Loading).
    *   `df['capped_loss'] = np.minimum(df['incurred_loss'], 100000)`
    *   The excess (\$1M - \$100k) is priced separately using a "Catastrophe Load".

### 6.2 IBNR (Incurred But Not Reported)

*   **Issue:** Recent claims (Dec 2022) look cheap because they haven't developed yet.
*   **Fix:** Apply Loss Development Factors (LDFs) to the `incurred_loss` before training.

---

## 7. Advanced Topics & Extensions

### 7.1 Telematics Features

*   **Raw Data:** GPS coordinates every second.
*   **Features:**
    *   `hard_brakes_per_100_miles`
    *   `pct_driving_night`
    *   `cornering_g_force`
*   **Impact:** Highly predictive, but huge data volume.

### 7.2 External Data Enrichment

*   **Credit:** Join with TransUnion data.
*   **Weather:** Join with NOAA data (Rainfall frequency in Zip Code).

---

## 8. Regulatory & Governance Considerations

### 8.1 PII (Personally Identifiable Information)

*   **Rule:** Do not use Name, SSN, or Exact Address in the model.
*   **Action:** Hash the `policy_id`. Drop PII columns immediately after joining.

---

## 9. Practical Example

### 9.1 The "Missing VIN" Mystery

**Scenario:** 10% of policies have missing `vehicle_year`.
**Investigation:**
*   These are "Non-Owned Auto" policies (Rental coverage).
*   **Decision:** Create a separate model for them, or impute with the "Average Car" (risky).
*   **Action:** Flag them: `is_non_owned = 1`.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Garbage In, Garbage Out.** The pipeline is 80% of the work.
2.  **Exposure** is the denominator of everything.
3.  **Time-Based Split** is mandatory.

### 10.2 When to Use This Knowledge
*   **Capstone Project:** This is Step 1.
*   **Real World:** Every pricing project starts here.

### 10.3 Critical Success Factors
1.  **Domain Knowledge:** Knowing that "Coverage Symbol 01" means "Any Auto".
2.  **Reproducibility:** The pipeline must be a script, not a Jupyter Notebook.

### 10.4 Further Reading
*   **Goldburd, Khare, Tevet:** "Generalized Linear Models for Insurance Rating" (Chapter on Data).

---

## Appendix

### A. Glossary
*   **Earned Premium:** Premium attributed to the portion of the policy term that has expired.
*   **Written Premium:** Total premium for the policy term, recorded on Day 1.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Exposure** | $\frac{\text{Days Active}}{365}$ | Weighting |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
