# Day 10.3: Statistical Significance Testing - A Practical Guide

## Introduction: Is Your Improvement Real, or Just Luck?

So, you've trained two models. Model A gets 88.2% accuracy on the test set, and Model B gets 88.9%. Is Model B actually better? Or could that 0.7% difference just be due to random chance in the data split or the model's random initialization? This is a critical question that is often overlooked.

**Statistical significance testing** provides a formal framework for answering this question. It helps us determine whether the observed difference in performance between two models is likely a real improvement or simply a result of statistical noise.

This guide will provide a practical, high-level introduction to the concepts and implementation of statistical tests for comparing machine learning models, focusing on the widely used **McNemar's Test**.

**Disclaimer:** This is a deep topic rooted in statistics. This guide aims to provide the practical intuition and implementation for a data scientist, not a rigorous statistical proof.

**Today's Learning Objectives:**

1.  **Understand the Need for Statistical Testing:** Grasp why a simple comparison of accuracy scores can be misleading.
2.  **Learn the Core Concepts: Null Hypothesis and p-value:** Understand the basic terminology of hypothesis testing.
3.  **Explore McNemar's Test:** Learn about this simple and effective test for comparing two classification models.
4.  **Implement McNemar's Test:** Write the code to perform the test and interpret its results.
5.  **Understand the Limitations and Best Practices:** Know when statistical tests are appropriate and what they can (and cannot) tell you.

--- 

## Part 1: The Null Hypothesis and the p-value

Hypothesis testing is the core of statistical inference.

1.  **The Null Hypothesis (H₀):** This is the default assumption, the "status quo." In our case, the null hypothesis is that **there is no significant difference in performance between Model A and Model B**. The observed difference is just due to random chance.

2.  **The Alternative Hypothesis (H₁):** This is what we are trying to prove. In our case, it's that **there *is* a significant difference in performance between the two models**.

3.  **The p-value:** This is the key result of a statistical test. The p-value is the probability of observing our results (or something more extreme) **if the null hypothesis were true**.
    *   A **small p-value** (typically ≤ 0.05) means: "It is very unlikely we would see this result if the models were actually the same." This gives us evidence to **reject the null hypothesis** and conclude that the difference is statistically significant.
    *   A **large p-value** (> 0.05) means: "This result is quite plausible even if the models have the same underlying performance." We **fail to reject the null hypothesis** and conclude that we don't have enough evidence to say the difference is real.

The threshold (0.05) is called the **significance level (alpha)**.

--- 

## Part 2: McNemar's Test - Comparing Classifiers

McNemar's test is a non-parametric statistical test used on paired nominal data. It's particularly well-suited for comparing the performance of two binary classification models.

**How it works:**
The test focuses only on the samples where the two models **disagree**. It's based on a 2x2 **contingency table** of their prediction outcomes.

|                    | **Model B: Correct** | **Model B: Incorrect** |
|--------------------|----------------------|------------------------|
| **Model A: Correct** | `n_cc` (both correct) | `n_ci` (A correct, B incorrect) |
| **Model A: Incorrect** | `n_ic` (A incorrect, B correct) | `n_ii` (both incorrect) |

*   The test ignores the cases where both models were correct (`n_cc`) or both were incorrect (`n_ii`).
*   It focuses on the disagreements: `n_ci` (cases where Model A won) and `n_ic` (cases where Model B won).
*   The null hypothesis is that the two models have the same error rate, which means we would expect `n_ci` to be roughly equal to `n_ic`.
*   The test uses the chi-squared statistic to determine if the difference between `n_ci` and `n_ic` is statistically significant.

### 2.1. Implementing McNemar's Test

We can use the `mcnemar` function from the `statsmodels` library for a convenient implementation.

```python
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

print("--- Part 2: Implementing McNemar's Test ---")

# --- 1. Generate Dummy Prediction Data ---
# Let's imagine we have a test set of 1000 samples.
# 0 = Incorrect, 1 = Correct

# Scenario 1: Models have similar performance
# Model A gets 900 correct, Model B gets 905 correct.
# Is B significantly better?

# Let's create the predictions. We need to define the disagreements.
# n_ci (A correct, B incorrect) = 15
# n_ic (A incorrect, B correct) = 20
# The rest are agreements.

# We can represent this as a contingency table.
# [[n_cc, n_ci], [n_ic, n_ii]]

# Total correct for A = n_cc + n_ci = 900
# Total correct for B = n_cc + n_ic = 905
# Total incorrect for A = n_ic + n_ii = 100
# Total incorrect for B = n_ci + n_ii = 95
# Solving this system gives: n_cc = 885, n_ii = 80

contingency_table_1 = [[885, 15],
                       [20,  80]]

# --- 2. Perform the Test ---
result_1 = mcnemar(contingency_table_1, exact=False) # Use exact=False for chi-squared approximation

print("--- Scenario 1: Similar Performance ---")
print(f"Contingency Table:\n{np.array(contingency_table_1)}")
print(f"Statistic (chi-squared): {result_1.statistic:.4f}")
print(f"p-value: {result_1.pvalue:.4f}")

# --- 3. Interpret the Result ---
alpha = 0.05
if result_1.pvalue <= alpha:
    print("Result: The difference is statistically significant (p <= 0.05). We reject the null hypothesis.")
else:
    print("Result: The difference is NOT statistically significant (p > 0.05). We fail to reject the null hypothesis.")
print("--> We cannot conclude that Model B is genuinely better than Model A.")


# --- Scenario 2: Models have a larger difference ---
# Model A gets 900 correct, Model B gets 920 correct.
# n_ci (A correct, B incorrect) = 10
# n_ic (A incorrect, B correct) = 30
# Solving gives: n_cc = 890, n_ii = 70

contingency_table_2 = [[890, 10],
                       [30,  70]]

result_2 = mcnemar(contingency_table_2, exact=False)

print("\n--- Scenario 2: Larger Performance Difference ---")
print(f"Contingency Table:\n{np.array(contingency_table_2)}")
print(f"Statistic (chi-squared): {result_2.statistic:.4f}")
print(f"p-value: {result_2.pvalue:.4f}")

if result_2.pvalue <= alpha:
    print("Result: The difference is statistically significant (p <= 0.05). We reject the null hypothesis.")
else:
    print("Result: The difference is NOT statistically significant (p > 0.05). We fail to reject the null hypothesis.")
print("--> We have statistical evidence that Model B's performance is genuinely different from Model A's.")
```

--- 

## Part 3: Best Practices and Limitations

### When to Use Statistical Tests?
*   When comparing two or more final models on a fixed test set.
*   When publishing results in an academic paper or a formal report to add credibility.
*   When making a critical business decision based on a model's performance improvement (e.g., "Should we spend $100,000 to deploy Model B?").

### What Tests Can't Tell You
*   **Practical Significance:** A result can be statistically significant but practically meaningless. On a massive test set, a 0.01% accuracy improvement might be statistically significant, but it's unlikely to have any real-world impact.
*   **Which Model is "Better":** The test tells you if there is a *difference*, but not necessarily which model is better in a practical sense. You must combine the test results with the actual performance metrics and the business context.
*   **The "Why":** The test doesn't explain *why* one model is better. That requires further analysis of the models' errors.

### Other Tests to Be Aware Of
*   **Paired t-test on cross-validation scores:** A common method where you perform K-fold cross-validation for two models on the same folds and then run a paired t-test on the K performance scores for each model.
*   **Wilcoxon signed-rank test:** A non-parametric alternative to the paired t-test, which doesn't assume the scores are normally distributed.

## Conclusion

Moving beyond a simple comparison of metric scores is a sign of maturity as a machine learning practitioner. Statistical significance testing provides a principled way to assess whether the improvements you are seeing are real or just a product of randomness.

**Key Takeaways:**

1.  **Don't Trust Small Differences Blindly:** A small improvement in a metric on a single test set is not conclusive evidence that one model is better.
2.  **Hypothesis Testing is the Framework:** We assume the models are the same (the null hypothesis) and then calculate the probability (p-value) of seeing our result if that assumption were true.
3.  **p-value is Your Guide:** A small p-value (e.g., ≤ 0.05) allows you to reject the null hypothesis and claim a statistically significant difference.
4.  **McNemar's Test is a Simple, Powerful Tool:** For comparing two classifiers, it provides an easy way to check if the difference in their error rates is significant by focusing on the instances where they disagree.
5.  **Context is King:** Statistical significance must always be interpreted alongside practical significance. A tiny, statistically significant improvement may not be worth the cost of deploying a new model.

By incorporating statistical tests into your evaluation toolkit, you add a layer of rigor and credibility to your work, ensuring that the conclusions you draw are well-supported by evidence.

## Self-Assessment Questions

1.  **Null Hypothesis:** When comparing two models, what is the null hypothesis?
2.  **p-value:** If you run a test and get a p-value of 0.3, what do you conclude about the difference between your two models?
3.  **McNemar's Test:** What specific set of predictions does McNemar's test focus on when comparing two models?
4.  **Contingency Table:** In the McNemar's test contingency table, what does the cell `n_ic` represent?
5.  **Practical vs. Statistical Significance:** You test two models on a dataset of 10 million users. Model A has an accuracy of 99.998%, and Model B has an accuracy of 99.999%. A statistical test gives a p-value of 0.0001. Is this result statistically significant? Is it practically significant? Explain.
