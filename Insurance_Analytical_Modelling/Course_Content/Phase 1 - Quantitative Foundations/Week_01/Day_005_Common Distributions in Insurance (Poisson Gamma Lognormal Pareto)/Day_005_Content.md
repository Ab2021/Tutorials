# ### Day 5 – Common Distributions in Insurance

*   **Topic:** Frequency & Severity Distributions
*   **Content:**
    *   **Frequency (Count):** Poisson (Mean=Var), Negative Binomial (Mean < Var, overdispersion), Binomial.
    *   **Severity (Amount):** Gamma, Lognormal (Skewed), Pareto (Heavy tail/Catastrophes), Weibull.
    *   **Mixture Models:** Zero-Inflated Poisson (ZIP) for excess zeros.
*   **Interview Pointers:**
    *   "Why use Negative Binomial instead of Poisson?" (Answer: To handle overdispersion/variance > mean).
    *   "Which distribution fits large liability claims best?" (Answer: Pareto).
*   **Challenges:** Fitting tails of distributions where data is sparse.