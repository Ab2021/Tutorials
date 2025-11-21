# Day 6: Probability Theory - Interview Questions

> **Topic**: Math Foundations
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Explain the difference between Frequentist and Bayesian statistics.
**Answer:**
*   **Frequentist**: Probability is the limit of relative frequency over infinite trials. Parameters are **fixed** but unknown constants. Data is random. (e.g., "The coin has a fixed bias $\theta$").
*   **Bayesian**: Probability is a measure of **belief**. Parameters are **random variables** with distributions. Data is fixed. We update our belief (Posterior) based on data.

### 2. What is Bayes' Theorem? Write the formula and explain the terms.
**Answer:**
*   $$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$
*   **Prior $P(A)$**: Belief before seeing evidence.
*   **Likelihood $P(B|A)$**: Probability of evidence given the hypothesis.
*   **Posterior $P(A|B)$**: Updated belief after seeing evidence.
*   **Evidence $P(B)$**: Normalizing constant.

### 3. What is a Random Variable?
**Answer:**
*   A function that maps outcomes of a random process to numerical values.
*   Example: Coin flip (Heads, Tails) -> Random Variable X (1, 0).

### 4. Explain the Law of Large Numbers.
**Answer:**
*   As the sample size $N$ grows, the sample mean $\bar{X}$ converges to the true population mean $\mu$.
*   This justifies using data averages to estimate theoretical probabilities.

### 5. What is the Central Limit Theorem? Why is it important for Machine Learning?
**Answer:**
*   The sum (or average) of many independent random variables tends toward a **Normal Distribution**, regardless of the original distribution.
*   **Importance**: Justifies the assumption of Gaussian noise in many models (Linear Regression) and explains why initialization schemes work.

### 6. Difference between Probability Mass Function (PMF) and Probability Density Function (PDF).
**Answer:**
*   **PMF**: For **Discrete** variables. Gives exact probability $P(X=k)$.
*   **PDF**: For **Continuous** variables. Value represents density. Probability is the **integral** (area under curve). $P(X=x) = 0$ for any specific point.

### 7. What is Expectation and Variance?
**Answer:**
*   **Expectation ($E[X]$)**: The weighted average (center of mass). $\sum x P(x)$.
*   **Variance ($Var(X)$)**: Measure of spread. Expected squared deviation from mean. $E[(X - \mu)^2]$.

### 8. Explain the difference between Covariance and Correlation.
**Answer:**
*   **Covariance**: Measures direction of linear relationship. Unbounded scale (depends on units).
*   **Correlation**: Normalized Covariance (between -1 and 1). Unitless. $Corr(X,Y) = Cov(X,Y) / (\sigma_X \sigma_Y)$.

### 9. What is a Bernoulli distribution?
**Answer:**
*   Discrete distribution for a single trial with two outcomes (Success/Failure).
*   Parameter $p$.
*   $P(X=1) = p$, $P(X=0) = 1-p$.

### 10. What is a Binomial distribution?
**Answer:**
*   Sum of $n$ independent Bernoulli trials.
*   Number of successes in $n$ flips.
*   Parameters: $n, p$.

### 11. What is a Poisson distribution? Give a real-world example.
**Answer:**
*   Models the number of events occurring in a fixed interval of time/space.
*   Parameter $\lambda$ (rate).
*   **Example**: Number of emails arriving in an hour.

### 12. What is a Gaussian (Normal) distribution? Why is it so common in nature?
**Answer:**
*   Bell curve defined by Mean $\mu$ and Variance $\sigma^2$.
*   **Commonality**: Due to Central Limit Theorem. Many natural processes are sums of small independent factors.

### 13. Explain Conditional Probability.
**Answer:**
*   Probability of A happening given that B has occurred.
*   $P(A|B) = P(A \cap B) / P(B)$.

### 14. What does it mean for two events to be Independent?
**Answer:**
*   Knowing B gives no information about A.
*   $P(A|B) = P(A)$.
*   Equivalently: $P(A \cap B) = P(A)P(B)$.

### 15. What is Maximum Likelihood Estimation (MLE)?
**Answer:**
*   Method to estimate parameters $\theta$ by maximizing the Likelihood function $L(\theta) = P(Data | \theta)$.
*   "Which parameters make the observed data most probable?"

### 16. What is Maximum A Posteriori (MAP) estimation? How does it differ from MLE?
**Answer:**
*   Maximizes the **Posterior** $P(\theta | Data)$.
*   $MAP \approx Likelihood \times Prior$.
*   Differs from MLE by including a **Prior**. Acts as regularization.

### 17. Explain the concept of "Marginalization".
**Answer:**
*   Summing (or integrating) out variables from a joint distribution to get the distribution of a subset.
*   $P(X) = \sum_Y P(X, Y)$.

### 18. What is the Chain Rule of Probability?
**Answer:**
*   Factorizes a joint distribution into conditional probabilities.
*   $P(A, B, C) = P(A) P(B|A) P(C|A, B)$.
*   Fundamental for Bayesian Networks and RNNs.

### 19. What is a Joint Probability Distribution?
**Answer:**
*   Probability of multiple events happening together. $P(X=x, Y=y)$.

### 20. Explain the "Monty Hall Problem" and the intuition behind the solution.
**Answer:**
*   3 doors. 1 Car, 2 Goats. You pick Door 1. Host opens Door 3 (Goat). Should you switch?
*   **Yes**.
*   **Intuition**: Initial probability of Car is 1/3. Probability of "Not Door 1" is 2/3. Host reveals Door 3 is empty, so that entire 2/3 probability shifts to Door 2. Switching doubles your chance.
