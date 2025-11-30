# Model Risk Management & Documentation (Part 1) - Governance Frameworks - Theoretical Deep Dive

## Overview
"A model is a tool. A bad model is a weapon."
In 2008, bad models (CDO pricing) nearly destroyed the global economy.
Since then, **Model Risk Management (MRM)** has become a critical function.
This day focuses on the **Governance Frameworks** that keep actuaries and data scientists out of jail.

---

## 1. Conceptual Foundation

### 1.1 What is Model Risk?

*   **Definition:** The potential for adverse consequences from decisions based on incorrect or misused model outputs.
*   **Two Sources:**
    1.  **Fundamental Errors:** The model is wrong (e.g., coding bug, wrong formula).
    2.  **Misuse:** The model is right, but used for the wrong purpose (e.g., using a Pricing model for Reserving).

### 1.2 The "Three Lines of Defense"

The industry standard for risk management:
1.  **First Line (The Builders):** Actuaries, Data Scientists.
    *   *Responsibility:* Build, test, document, and use the model.
2.  **Second Line (The Validators):** MRM Team.
    *   *Responsibility:* Independent review, challenge, and approval.
3.  **Third Line (The Auditors):** Internal Audit.
    *   *Responsibility:* Verify that Line 1 and Line 2 are following the policy.

---

## 2. Regulatory Framework

### 2.1 SR 11-7 (The Bible of MRM)

*   **Origin:** Federal Reserve (2011).
*   **Scope:** Banks, but adopted by large Insurers (Solvency II, NAIC).
*   **Key Principles:**
    *   **Independence:** Validators cannot report to the Model Owner.
    *   **Effective Challenge:** Validators must have the authority to say "No".
    *   **Inventory:** You must maintain a comprehensive list of all models.

### 2.2 SS1/23 (The AI Update)

*   **Origin:** Bank of England (Prudential Regulation Authority).
*   **Focus:** Model Risk Management principles for AI/ML.
*   **New Requirement:** "Drift Monitoring". You must prove the model is still valid *after* deployment.

---

## 3. Theoretical Properties

### 3.1 What is a "Model"?

*   **Definition:** "A quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates."
*   **Is Excel a Model?**
    *   *Yes:* If it calculates IBNR reserves.
    *   *No:* If it just sums up a column of numbers.
*   **EUC (End User Computing):** The gray area. Spreadsheets that act like models but fly under the radar.

### 3.2 Model Tiering

Not all models are equal.
*   **Tier 1 (Critical):** Financial Reporting, Solvency, Pricing major lines.
    *   *Validation:* Full replication, annual review.
*   **Tier 2 (Important):** Internal profitability analysis.
    *   *Validation:* Review of conceptual soundness.
*   **Tier 3 (Minor):** Ad-hoc analysis.
    *   *Validation:* Self-attestation.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The Model Inventory (JSON Schema)

```json
{
  "model_id": "MOD-2024-001",
  "name": "Auto_Frequency_GLM",
  "owner": "Jane Doe (Chief Actuary)",
  "developer": "John Smith (Data Scientist)",
  "tier": "1",
  "status": "Production",
  "description": "Predicts claim frequency for Personal Auto.",
  "technology": "Python (scikit-learn)",
  "last_validation_date": "2023-12-15",
  "next_review_date": "2024-12-15",
  "limitations": [
    "Does not account for 2024 inflation spike.",
    "Not valid for commercial vehicles."
  ]
}
```

### 4.2 The Validation Report Structure

1.  **Executive Summary:** "Approved", "Approved with Conditions", or "Rejected".
2.  **Conceptual Soundness:** Is the math right? Is the theory valid?
3.  **Data Integrity:** Is the input data clean?
4.  **Outcome Analysis:** Backtesting results.
5.  **Findings & Issues:**
    *   *High Severity:* Must fix before deployment.
    *   *Low Severity:* Fix within 6 months.

---

## 5. Evaluation & Validation

### 5.1 The "Challenger Model"

*   **Method:** The Validator builds a simpler version of the model (e.g., a Linear Regression to challenge a GBM).
*   **Goal:** If the complex model only beats the simple model by 0.1%, is it worth the risk?

### 5.2 Sensitivity Analysis

*   **Method:** Shock the inputs.
    *   "What if interest rates rise by 2%?"
    *   "What if inflation hits 10%?"
*   **Check:** Does the model output explode? (Stability check).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 "It's just a tool, not a model"

*   **Avoidance:** Developers try to classify complex tools as "calculators" to avoid MRM scrutiny.
*   **Risk:** These "calculators" often contain hidden assumptions (e.g., hardcoded tax rates) that become wrong over time.

### 6.2 Vendor Models (Black Boxes)

*   **Problem:** You buy a Cat Model (RMS). You can't see the code.
*   **Regulation:** You are still responsible for it.
*   **Solution:** "Input/Output Validation". You can't check the engine, but you can check if the car drives straight.

---

## 7. Advanced Topics & Extensions

### 7.1 AI Governance

*   **Challenge:** How do you validate a Neural Network that changes every week (Online Learning)?
*   **Answer:** You validate the *process* (the training pipeline), not just the static artifact.

### 7.2 Model Interdependency Maps

*   **Concept:** Model A feeds Model B, which feeds Model C.
*   **Risk:** Error propagation. A small error in A becomes a massive error in C.
*   **Tool:** Network Graph of data flows.

---

## 8. Regulatory & Governance Considerations

### 8.1 SOX (Sarbanes-Oxley)

*   **Impact:** If a model feeds into Financial Statements (Reserving), it is a SOX Control.
*   **Requirement:** Any change to the code must be logged, tested, and approved (Change Management).

---

## 9. Practical Example

### 9.1 The "Excel Disaster"

**Scenario:**
*   An actuary hardcodes "1.05" (5% trend) in cell Z99 of a pricing spreadsheet.
*   Next year, inflation is 2%.
*   The actuary forgets to update Z99.
*   **Result:** The company overprices by 3%, loses market share, and loses \$50M in premium.
**MRM Fix:**
*   Input parameters must be separated from calculation logic.
*   Spreadsheets must be locked.
*   Peer review is mandatory.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Three Lines of Defense.**
2.  **SR 11-7** is the standard.
3.  **Inventory** everything.

### 10.2 When to Use This Knowledge
*   **Every Day:** If you build models, you are Line 1. You must follow the policy.
*   **Job Interview:** Asking about "Model Governance" shows you are a senior-level thinker.

### 10.3 Critical Success Factors
1.  **Independence:** Validators cannot be "friends" with developers.
2.  **Documentation:** If it isn't written down, it doesn't exist.

### 10.4 Further Reading
*   **Federal Reserve:** "SR 11-7: Guidance on Model Risk Management".

---

## Appendix

### A. Glossary
*   **EUC:** End User Computing (Spreadsheets).
*   **Finding:** A deficiency identified during validation.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **N/A** | Governance is qualitative | N/A |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
