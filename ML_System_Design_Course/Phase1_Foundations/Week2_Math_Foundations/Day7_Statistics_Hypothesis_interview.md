# Day 7: Statistics & Hypothesis Testing - Interview Questions

> **Topic**: Statistical Inference
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is a Null Hypothesis ($H_0$) and an Alternative Hypothesis ($H_1$)?
**Answer:**
*   **Null ($H_0$)**: The default assumption (e.g., "No effect", "Coin is fair").
*   **Alternative ($H_1$)**: What we want to prove (e.g., "Drug works", "Coin is biased").

### 2. Explain Type I (False Positive) and Type II (False Negative) errors.
**Answer:**
*   **Type I ($\alpha$)**: Rejecting Null when it is True. (Convicting an innocent person).
*   **Type II ($\beta$)**: Failing to reject Null when it is False. (Letting a guilty person go free).

### 3. What is a p-value? How do you interpret it?
**Answer:**
*   The probability of observing data *at least as extreme* as the current observation, assuming the Null Hypothesis is true.
*   **Interpretation**: Low p-value (< 0.05) means data is unlikely under Null -> Reject Null. It is **not** the probability that Null is true.

### 4. What is the Significance Level ($\alpha$)?
**Answer:**
*   The threshold for rejecting the Null (usually 0.05).
*   It is the probability of making a Type I error that we are willing to accept.

### 5. Explain the concept of "Statistical Power".
**Answer:**
*   Power = $1 - \beta$.
*   Probability of correctly rejecting the Null when it is indeed False.
*   "If there is a real effect, how likely are we to detect it?"

### 6. What is a Confidence Interval?
**Answer:**
*   A range derived from sample data that is likely to contain the true population parameter.
*   **95% CI**: If we repeat the experiment 100 times, 95 of the calculated intervals will contain the true mean.

### 7. What is a t-test? When do you use it vs a z-test?
**Answer:**
*   **t-test**: Used when population variance is **unknown** and sample size is **small** (< 30). Uses Student's t-distribution (fatter tails).
*   **z-test**: Used when population variance is **known** or sample size is **large**. Uses Normal distribution.

### 8. Explain A/B Testing. What are the key steps?
**Answer:**
*   Comparing two versions (A and B) to see which performs better.
*   **Steps**:
    1.  Define Metric (e.g., Conversion Rate).
    2.  Determine Sample Size (Power Analysis).
    3.  Randomly split traffic.
    4.  Run test.
    5.  Calculate p-value.

### 9. What is the difference between Correlation and Causation?
**Answer:**
*   **Correlation**: A and B move together.
*   **Causation**: A causes B.
*   **Difference**: Confounders. Ice cream sales correlate with shark attacks (Confounder: Summer/Heat). Ice cream doesn't cause shark attacks.

### 10. What is Simpson's Paradox?
**Answer:**
*   A trend appears in different groups of data but disappears or reverses when these groups are combined.
*   **Example**: Drug A is better than Drug B for men. Drug A is better for women. But Drug B is better overall (because Drug A was given mostly to severe cases).

### 11. What is Sampling Bias?
**Answer:**
*   When the sample is not representative of the population.
*   **Example**: Surveying people via landline phones (misses young people).

### 12. Explain the Chi-Square test.
**Answer:**
*   Used for **Categorical** data.
*   Tests if there is a significant difference between Expected frequencies and Observed frequencies in a contingency table.
*   Used to check independence between two categorical variables.

### 13. What is ANOVA (Analysis of Variance)?
**Answer:**
*   Generalization of t-test to **more than 2 groups**.
*   Tests if the means of 3+ groups are equal by comparing variance *between* groups vs variance *within* groups.

### 14. What is the difference between Parametric and Non-parametric tests?
**Answer:**
*   **Parametric** (t-test, ANOVA): Assume data follows a specific distribution (usually Normal). More powerful if assumptions hold.
*   **Non-parametric** (Mann-Whitney, Kruskal-Wallis): Make no assumptions about distribution. Use ranks. Robust to outliers.

### 15. What is Bootstrapping?
**Answer:**
*   Resampling method.
*   Repeatedly sample from the dataset **with replacement** to estimate the sampling distribution of a statistic (e.g., mean, median).
*   Allows calculating Confidence Intervals without formulas.

### 16. Explain the concept of Standard Error.
**Answer:**
*   Standard Deviation of the sampling distribution of a statistic.
*   $SE = \sigma / \sqrt{n}$.
*   Measures how precise our estimate of the mean is.

### 17. What is the Bonferroni Correction? Why is it used?
**Answer:**
*   Used when testing multiple hypotheses (Multiple Comparisons Problem).
*   If you test 20 hypotheses at $\alpha=0.05$, you expect 1 false positive by chance.
*   **Fix**: Divide $\alpha$ by number of tests ($0.05 / 20$). Very conservative.

### 18. What is a Confounding Variable?
**Answer:**
*   An outside influence that changes the effect of a dependent and independent variable.
*   See Simpson's Paradox example.

### 19. How do you determine the sample size needed for an A/B test?
**Answer:**
*   **Power Analysis**.
*   Depends on:
    1.  **Baseline Rate** (Current conversion).
    2.  **MDE** (Minimum Detectable Effect - e.g., 1% lift).
    3.  **Power** (usually 0.8).
    4.  **Significance Level** (0.05).

### 20. What is the Kolmogorov-Smirnov (KS) test?
**Answer:**
*   Non-parametric test to check if two samples come from the same distribution (or if one sample matches a reference distribution).
*   Measures the max distance between their Cumulative Distribution Functions (CDFs).
