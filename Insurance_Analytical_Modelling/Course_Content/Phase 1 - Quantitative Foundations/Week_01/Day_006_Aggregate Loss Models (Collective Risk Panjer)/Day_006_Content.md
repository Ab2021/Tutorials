# ### Day 6 – Aggregate Loss Models (SOA FAM-S)

*   **Topic:** Collective Risk Model
*   **Content:**
    *   Model: $S = \sum_{i=1}^{N} X_i$ where $N$ is frequency, $X$ is severity.
    *   Compound Distributions: Compound Poisson-Gamma (Tweedie).
    *   Panjer Recursion: Numerical method to compute PDF of $S$.
    *   Normal & Log-Normal Approximations for $S$.
*   **Interview Pointers:**
    *   "How do you model total loss for a portfolio?"
    *   "What is the Tweedie distribution and why is it popular in GLMs?" (Models pure premium directly).
*   **Tricky Parts:** Convolutions of distributions.
*   **Data Requirements:** Aggregated claims data (total loss per period).