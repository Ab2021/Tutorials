# ### Days 37–39 – Severity GLMs

*   **Topic:** Modeling Claim Amounts
*   **Content:**
    *   **Gamma GLM:** Constant coefficient of variation. Good for attrition.
    *   **Inverse Gaussian:** Heavier tail than Gamma.
    *   **Tweedie:** Models Pure Premium directly (Compound Poisson-Gamma).
*   **Interview Pointers:**
    *   "Why not use a Normal distribution for severity?" (Severity is skewed and positive; Normal is symmetric and allows negatives).
*   **Tricky Parts:** Modeling large losses (capping/excess removal) before fitting GLM.