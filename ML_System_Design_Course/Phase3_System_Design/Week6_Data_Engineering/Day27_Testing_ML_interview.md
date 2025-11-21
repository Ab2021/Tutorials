# Day 27: Testing ML Systems - Interview Questions

> **Topic**: QA for AI
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. How is testing ML different from testing Software?
**Answer:**
*   **Software**: Logic is fixed. Input + Logic = Output. Deterministic.
*   **ML**: Logic is learned. Probabilistic. Fails silently. Data is part of the code.

### 2. What are Unit Tests for ML?
**Answer:**
*   Testing individual functions (preprocessing, loss).
*   Example: `assert normalize([10, 10]) == [0, 0]`.
*   Example: `assert output_shape == (batch, classes)`.

### 3. What are Integration Tests for ML?
**Answer:**
*   Testing the full pipeline (Data -> Train -> Save).
*   Run on a small dummy dataset to ensure no crashes.

### 4. What is a "Smoke Test" (Sanity Check)?
**Answer:**
*   Quick check before expensive training.
*   **Overfit a single batch**: Loss should go to 0. If not, bug in model/data.

### 5. What are "Invariance Tests" (Metamorphic Testing)?
**Answer:**
*   Perturb input in a way that shouldn't change output.
*   NLP: "I hate this" -> Negative. "I really hate this" -> Negative.
*   CV: Rotate image -> Prediction stays same.

### 6. What are "Directional Expectation Tests"?
**Answer:**
*   Perturb input in a way that *should* change output predictably.
*   Housing: Increase square footage -> Price should increase.

### 7. What is "Minimum Functionality Test" (Behavioral Test)?
**Answer:**
*   Specific examples the model *must* get right.
*   "Golden Set".
*   NLP: "This is great" must be Positive.

### 8. How do you test for Data Leakage?
**Answer:**
*   Check overlap between Train and Test set IDs.
*   Check if features contain future information (Time travel).
*   Feature Importance: If one feature has 99% importance, suspect leakage.

### 9. What is "Differential Testing" (Regression Testing)?
**Answer:**
*   Compare New Model vs Old Model.
*   Predictions shouldn't change drastically for most inputs.
*   Diff should be explained by improvement.

### 10. How do you test a Data Pipeline?
**Answer:**
*   **Schema Validation**: Check types.
*   **Value Validation**: Check ranges, nulls.
*   **Count Validation**: Row count consistency.

### 11. What is "Adversarial Testing"?
**Answer:**
*   Intentionally trying to fool the model.
*   Adding noise to image, changing one word in text.
*   Ensures robustness.

### 12. What is a "Golden Dataset"?
**Answer:**
*   Curated set of high-quality, verified examples.
*   Used for final sign-off before deployment.

### 13. How do you validate a model offline?
**Answer:**
*   **Hold-out set**: Test set.
*   **Cross-Validation**: K-Fold.
*   **Backtesting**: For time-series (Train on past, test on "future" past).

### 14. What is "Slice-based Evaluation"?
**Answer:**
*   Evaluating metrics on specific subgroups (Slices).
*   Overall Accuracy 90%. Accuracy on "Mobile Users" 50%.
*   Finds hidden biases/failures.

### 15. How do you test for Reproducibility?
**Answer:**
*   Fix random seeds.
*   Run training twice.
*   Assert weights are identical.

### 16. What is "Stress Testing" for serving?
**Answer:**
*   Load testing (Locust/JMeter).
*   Send high QPS. Measure latency and error rate.
*   Find breaking point.

### 17. What is "Calibration Testing"?
**Answer:**
*   Check if predicted probabilities match observed frequencies.
*   Reliability Diagram.

### 18. How do you test for Bias/Fairness?
**Answer:**
*   Compare False Positive Rates across groups (e.g., Male vs Female).
*   Equal Opportunity Difference.

### 19. What is "Model Assertions"?
**Answer:**
*   Runtime checks.
*   `assert prob_sum == 1.0`.
*   `assert age < 120`.

### 20. Why is "Silent Failure" dangerous in ML?
**Answer:**
*   Model predicts garbage, but system doesn't crash.
*   Downstream business decisions are made on bad data.
*   Requires monitoring distributions, not just exceptions.
