# ### Days 31–33 – Generalized Linear Models (GLMs) Theory (CAS MAS-I)

*   **Topic:** The Industry Standard for Pricing
*   **Content:**
    *   **Structure:** $g(E[Y]) = X\beta$. Link function $g(\cdot)$, Linear Predictor $X\beta$.
    *   **Exponential Family:** Normal, Poisson, Gamma, Inverse Gaussian.
    *   **Assumptions:** Independence, constant variance (or variance function $V(\mu)$).
*   **Interview Pointers:**
    *   "Why do we use a Log link function for frequency?" (Ensures predictions are positive; multiplicative structure).
    *   "What is the 'Offset' in a GLM?" (Accounting for exposure, e.g., $\ln(\text{Exposure})$).
*   **Tricky Parts:** Interpreting coefficients as relativities ($e^\beta$).
*   **Data Requirements:** Policy-level data with earned exposure and claim counts.