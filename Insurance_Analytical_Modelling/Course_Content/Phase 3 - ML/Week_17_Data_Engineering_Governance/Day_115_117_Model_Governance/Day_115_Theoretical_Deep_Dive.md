# Model Governance & Monitoring (Part 1) - Model Risk Management (MRM) - Theoretical Deep Dive

## Overview
"All models are wrong, but some are dangerous."
In 2008, bad models crashed the global economy. In response, regulators created **SR 11-7**.
For Actuaries and Data Scientists, **Model Risk Management (MRM)** is not just compliance; it's the safety net that prevents you from bankrupting the company.

---

## 1. Conceptual Foundation

### 1.1 What is a Model?

*   **Definition (SR 11-7):** "A quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates."
*   **Is Excel a Model?** Yes, if it contains complex logic (macros, VLOOKUP chains) used for decision making.

### 1.2 The Three Lines of Defense

1.  **First Line (Development):** You (The Actuary/Data Scientist). You build, test, and own the model.
2.  **Second Line (Validation):** The MRM Team. They independently review and "challenge" your model.
3.  **Third Line (Audit):** Internal Audit. They check if Line 1 and Line 2 are following the rules.

---

## 2. Mathematical Framework

### 2.1 Model Risk Quantification

*   **Operational Risk:** Risk of implementation errors (bugs).
*   **Conceptual Risk:** Risk that the theory is wrong (e.g., assuming Normal distribution for fat-tailed losses).
*   **Estimation Risk:** Risk of parameter uncertainty.

### 2.2 Risk Tiering

*   **Tier 1 (Critical):** High financial impact (> \$100M), high complexity. (e.g., Capital Model, Main Pricing Model).
*   **Tier 2 (Significant):** Moderate impact. (e.g., Reserving for small LOB).
*   **Tier 3 (Low):** Low impact. (e.g., Marketing segmentation).
*   *Impact:* Tier 1 requires annual validation. Tier 3 might only require self-attestation.

---

## 3. Theoretical Properties

### 3.1 Effective Challenge

*   The Validator must not just "check the boxes". They must actively try to break the model.
*   **Benchmarking:** Build a simpler "Challenger Model" (e.g., GLM vs. GBM) and compare results.
*   **Sensitivity Analysis:** What happens if inflation doubles?

### 3.2 Model Lifecycle

1.  **Inception:** "We need a new pricing model."
2.  **Development:** Coding and testing.
3.  **Validation:** Independent review.
4.  **Approval:** Governance Committee sign-off.
5.  **Implementation:** Production deployment.
6.  **Monitoring:** Ongoing checks.
7.  **Retirement:** Decommissioning.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The Model Inventory

*   A centralized database of all models.
*   **Fields:** Model ID, Owner, Validator, Tier, Status, Last Validation Date.

| Model ID | Name | Owner | Tier | Status |
| :--- | :--- | :--- | :--- | :--- |
| MOD-001 | Auto Pricing GBM | J. Doe | 1 | Production |
| MOD-002 | Marketing Churn | A. Smith | 3 | Development |

### 4.2 Model Documentation (MDD)

*   **Requirement:** If you get hit by a bus, can someone else run the model?
*   **Sections:**
    *   Assumptions & Limitations.
    *   Data Sources.
    *   Mathematical Theory.
    *   Code Structure.
    *   Testing Results.

---

## 5. Evaluation & Validation

### 5.1 Validation Tests

*   **Backtesting:** Does the model predict the past accurately?
*   **Stress Testing:** Does the model survive a 1-in-200 year event?
*   **Code Review:** Are there hard-coded variables? Is the code readable?

### 5.2 Governance Committees

*   **Model Risk Committee (MRC):** Senior executives who approve Tier 1 models.
*   *Role:* They review the Validation Report and decide: "Approve", "Approve with Conditions", or "Reject".

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: "It's AI, so it's a Black Box"**
    *   *Regulator:* "Not acceptable."
    *   *Fix:* You must explain *how* the AI works (SHAP, Feature Importance) and prove it's not discriminatory.

2.  **Trap: Vendor Models**
    *   "We bought this Catastrophe Model from RMS/AIR. We don't know how it works."
    *   *SR 11-7:* You are still responsible. You must validate the *inputs* and *outputs* and understand the *conceptual framework*.

### 6.2 Implementation Challenges

1.  **Shadow Models:**
    *   Actuaries building models on their desktops without telling MRM.
    *   *Risk:* Hidden risk. If the desktop crashes, the company stops functioning.

---

## 7. Advanced Topics & Extensions

### 7.1 AI Governance (EU AI Act)

*   Classifies AI systems by risk (Unacceptable, High, Limited, Minimal).
*   **High Risk:** Insurance Pricing and Claims. Requires strict conformity assessments.

### 7.2 Automated Governance

*   Using MLOps tools (MLflow) to automatically generate documentation and log validation metrics.
*   "Governance as Code".

---

## 8. Regulatory & Governance Considerations

### 8.1 SR 11-7 (USA)

*   The "Bible" of MRM.
*   Key Principle: "Model risk should be managed like any other risk."

### 8.2 SS1/23 (UK - PRA)

*   Specific guidance on Model Risk Management for Banks and Insurers.
*   Emphasizes Board responsibility and the "Senior Management Function" (SMF) for MRM.

---

## 9. Practical Example

### 9.1 Worked Example: Validating a Pricing Model

**Scenario:**
*   **Developer:** Builds a GBM for Auto Insurance. Claims it improves Gini by 5%.
*   **Validator:**
    *   **Review:** Checks the code. Finds a "Future Leakage" bug (using `future_claims` as a feature).
    *   **Challenge:** Builds a simple GLM. Finds that the GBM only outperforms in rural areas, but performs worse in cities.
    *   **Stress Test:** Tests the model on "Hurricane Data". The model predicts 0 losses (because it never saw a hurricane in training).
*   **Outcome:** "Approve with Conditions". The Developer must fix the bug and add a "Hurricane Overlay".

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **SR 11-7** defines the standard.
2.  **Three Lines of Defense** ensures checks and balances.
3.  **Tiering** prioritizes resources.

### 10.2 When to Use This Knowledge
*   **Every Day:** If you build models, you are the First Line of Defense.
*   **Interviews:** "How do you validate your models?" is a top interview question.

### 10.3 Critical Success Factors
1.  **Independence:** The Validator cannot report to the Developer.
2.  **Documentation:** If it isn't written down, it didn't happen.

### 10.4 Further Reading
*   **Federal Reserve:** "SR 11-7: Guidance on Model Risk Management".

---

## Appendix

### A. Glossary
*   **MRM:** Model Risk Management.
*   **EUC:** End User Computing (Excel, Access).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Risk Score** | $Impact \times Complexity$ | Tiering |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
