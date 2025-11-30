# Final Capstone: Definition & Scoping (Part 1) - Project Charter - Theoretical Deep Dive

## Overview
"A goal without a plan is just a wish."
The Capstone is not just a coding exercise. It is a **Product Simulation**.
You are the Chief Data Officer (CDO). You must define a problem, scope the solution, and prove the value.

---

## 1. Conceptual Foundation

### 1.1 The "End-to-End" Mindset

*   **Student Mindset:** "I built a model with 95% accuracy."
*   **Professional Mindset:** "I built a pipeline that ingests data daily, predicts churn, and feeds the CRM, saving \$1M/year."
*   **Capstone Goal:** Build the *System*, not just the *Model*.

### 1.2 Project Archetypes

1.  **The "Profit" Project:** Pricing Optimization (Elasticity Modeling).
2.  **The "Risk" Project:** Fraud Detection (Graph Neural Networks).
3.  **The "Efficiency" Project:** Claims Automation (Computer Vision).
4.  **The "Growth" Project:** Recommender System (Cross-Sell).

---

## 2. Mathematical Framework

### 2.1 ROI Estimation (The Business Case)

$$ \text{ROI} = \frac{\text{Benefit} - \text{Cost}}{\text{Cost}} \times 100\% $$

*   **Benefit:**
    *   *Churn:* Saved Customers $\times$ LTV.
    *   *Fraud:* Detected Fraud $\times$ Avg Claim Size.
*   **Cost:**
    *   *Compute:* AWS/Azure Bill.
    *   *Labor:* Data Scientist Hours.
    *   *False Positives:* Cost of investigating a legitimate claim.

### 2.2 Feasibility Score

$$ \text{Score} = w_1(\text{DataAvailability}) + w_2(\text{Complexity}) + w_3(\text{Impact}) $$

*   *Rule of Thumb:* If you don't have the data *today*, don't pick the project.

---

## 3. Theoretical Properties

### 3.1 The "Buy vs. Build" Decision

*   **Scenario:** You need an OCR model to read receipts.
*   **Buy:** Use AWS Textract (\$1.50 per 1000 pages).
*   **Build:** Train a CNN on MNIST/Receipts.
*   **Decision:** For a Capstone, you usually *Build* to learn. In production, you might *Buy*.

### 3.2 Scope Creep (The Silent Killer)

*   **Definition:** The project grows uncontrollably.
*   **Mitigation:** **MVP (Minimum Viable Product)**.
    *   *Week 1:* Simple Logistic Regression on CSV.
    *   *Week 4:* XGBoost on SQL.
    *   *Week 6:* API Deployment.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The Project Charter (Markdown Template)

```markdown
# Project Charter: Auto Insurance Fraud Detection

## 1. Problem Statement
Fraud accounts for 10% of claims cost. Manual review is slow (3 days).

## 2. Objective
Build a real-time scoring engine to flag high-risk claims.

## 3. Metrics
*   **Primary:** Recall (Catch 80% of fraud).
*   **Constraint:** Precision (False Positive Rate < 5%).
*   **Business:** Save \$500k/year.

## 4. Data Sources
*   `claims_history.csv` (Internal).
*   `nicb_hotspots.csv` (External).

## 5. Tech Stack
*   Python (Pandas, Scikit-Learn).
*   MLflow (Tracking).
*   FastAPI (Serving).
*   Docker (Containerization).
```

### 4.2 Data Selection Checklist

*   [ ] **Volume:** Do I have > 10,000 rows? (Deep Learning needs > 100k).
*   [ ] **Target:** Is the label (Fraud/Not Fraud) reliable?
*   [ ] **History:** Do I have at least 1 year of data (Seasonality)?
*   [ ] **PII:** Have I removed Names and SSNs?

---

## 5. Evaluation & Validation

### 5.1 The "Baseline" Test

*   **Task:** Before training XGBoost, calculate the "Naive Baseline".
*   **Example (Churn):** If 5% of users churn, a model that predicts "No Churn" for everyone is 95% accurate.
*   **Goal:** Your model must beat the Baseline (and a simple Heuristic).

### 5.2 Stakeholder Sign-Off

*   **Process:** Present the Charter to the "Business Sponsor" (Instructor/Mentor).
*   **Question:** "If I build this exactly as described, will you use it?"

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Kaggle" Trap**
    *   *Mistake:* Downloading a clean CSV and spending 4 weeks tuning Hyperparameters.
    *   *Reality:* Real world data is dirty. Spend 3 weeks on Data Engineering, 1 week on Modeling.

2.  **Trap: The "Perfect" Model**
    *   *Mistake:* Trying to get 99.9% accuracy.
    *   *Reality:* 80% is often "Good Enough" for V1. Ship it.

---

## 7. Advanced Topics & Extensions

### 7.1 Synthetic Data Generation

*   **Problem:** Not enough Fraud examples (Class Imbalance).
*   **Solution:** SMOTE (Synthetic Minority Over-sampling Technique) or GANs (Generative Adversarial Networks).

### 7.2 MLOps Integration

*   **Challenge:** How do we retrain the model next month?
*   **Solution:** Build a Pipeline (Airflow/Prefect) that automates the training loop.

---

## 8. Regulatory & Governance Considerations

### 8.1 Model Risk Management (MRM)

*   **Requirement:** SR 11-7 (Federal Reserve).
*   **Documentation:** You must document *why* you chose these features and *how* the model works.

---

## 9. Practical Example

### 9.1 Worked Example: Scoping a "Telematics" Project

**Idea:** Predict accidents using Driving Data.
**Data:** 1GB of GPS logs (Lat/Lon/Speed).
**Scope Check:**
1.  **Too Big?** Processing 1GB of raw logs is hard.
2.  **Refinement:** Aggregate to "Trip Level" features (Max Speed, Hard Brakes).
3.  **Target:** Do we have "Accident" labels?
    *   *Issue:* Only 0.1% of trips have accidents.
    *   *Pivot:* Predict "Near Misses" (Hard Braking) as a proxy for risk.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Scope Small, Deliver Big.**
2.  **Data First, Model Second.**
3.  **Business Value** is the only metric that matters.

### 10.2 When to Use This Knowledge
*   **Day 1 of Capstone:** "What are we actually building?"
*   **Job Interview:** "Tell me about a project you scoped from scratch."

### 10.3 Critical Success Factors
1.  **Clarity:** Write it down. Ambiguity kills projects.
2.  **Agility:** Be ready to pivot if the data is bad.

### 10.4 Further Reading
*   **Andrew Ng:** "Machine Learning Yearning" (Chapters on Scoping).

---

## Appendix

### A. Glossary
*   **MVP:** Minimum Viable Product.
*   **ROI:** Return on Investment.
*   **SMOTE:** Synthetic Minority Over-sampling Technique.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **ROI** | $\frac{B-C}{C}$ | Business Case |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
