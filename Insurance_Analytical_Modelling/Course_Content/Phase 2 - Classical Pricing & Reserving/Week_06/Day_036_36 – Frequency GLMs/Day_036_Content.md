# ### Days 34–36 – Frequency GLMs

*   **Topic:** Modeling Claim Counts
*   **Content:**
    *   **Poisson GLM:** Variance = Mean. Often under-dispersed for real data.
    *   **Negative Binomial:** Variance = $\mu + \phi \mu^2$. Handles overdispersion.
    *   **Zero-Inflated Models:** For excess zeros.
*   **Interview Pointers:**
    *   "How do you test for overdispersion?" (Compare variance to mean; Chi-square test).
*   **Challenges:** Sparse classes (e.g., young drivers with high-performance cars).