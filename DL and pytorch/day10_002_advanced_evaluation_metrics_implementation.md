# Day 10.2: Advanced Evaluation Metrics - A Practical Guide

## Introduction: Beyond Accuracy

While standard metrics like Accuracy, Precision, Recall, and F1-Score provide a good overview of model performance, they don't always tell the whole story, especially for complex tasks or when the business impact of errors is nuanced. Advanced evaluation metrics provide a deeper, more specialized view of a model's behavior.

This guide provides a practical exploration of several advanced metrics and concepts that are crucial for a thorough evaluation of classification and regression models.

**Today's Learning Objectives:**

1.  **Understand Multi-Class Metrics:** Learn how to calculate and interpret metrics like Macro and Weighted F1-Score for problems with more than two classes.
2.  **Explore Regression Metrics for Business Context:** Go beyond MAE and RMSE to understand Mean Absolute Percentage Error (MAPE) and R-squared (Coefficient of Determination).
3.  **Grasp Ranking and Probability-based Metrics:** Understand the intuition and use cases for Average Precision (AP), mean Average Precision (mAP), and Log Loss.
4.  **Implement Metrics in PyTorch and Scikit-learn:** See how to easily compute these advanced metrics using standard libraries.

---

## Part 1: Metrics for Multi-Class Classification

When we move from binary to multi-class classification, we can't just use a single Precision or Recall score. We need to average the metrics calculated for each class. There are several ways to do this:

*   **Macro Average:** Calculate the metric independently for each class and then take the unweighted average. This treats **every class as equally important**, regardless of how many samples it has. This is a good metric to use if you want to know how the model performs on infrequent classes.

*   **Weighted Average:** Calculate the metric for each class, but when averaging, weight each class's score by its **support** (the number of true instances for that class). This is useful when class imbalance is significant and you care more about the performance on the more common classes.

*   **Micro Average:** Calculate the metrics globally by counting the total true positives, false negatives, and false positives across all classes. For a multi-class problem, the micro-averaged F1-score is equivalent to the overall accuracy.

### 1.1. Implementing Multi-Class Averaging

```python
import torch
from sklearn.metrics import f1_score, classification_report

print("--- Part 1: Multi-Class Metrics ---")

# --- Dummy Multi-Class Data ---
# 3 classes (0, 1, 2). Class 0 is common, class 2 is rare.
y_true = torch.tensor([0, 1, 0, 0, 0, 1, 2, 0, 1, 0])
y_pred = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 1, 2])

# --- Using Scikit-learn --- #
# Scikit-learn makes this very easy with the `average` parameter.

# Macro F1-Score
f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"Macro F1-Score: {f1_macro:.4f} (Treats all classes equally)")

# Weighted F1-Score
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1-Score: {f1_weighted:.4f} (Accounts for class imbalance)")

# Micro F1-Score (same as accuracy)
f1_micro = f1_score(y_true, y_pred, average='micro')
print(f"Micro F1-Score: {f1_micro:.4f} (Equivalent to overall accuracy)")

# The classification report provides a comprehensive breakdown
print("\n--- Full Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Class 0 (Common)', 'Class 1', 'Class 2 (Rare)']))
```

**Interpretation of the Report:**
Notice how the F1-score for the rare `Class 2` is 0.0 because the model failed to predict it correctly. The `macro avg` is pulled down significantly by this poor performance on the rare class. The `weighted avg` is higher because it gives more weight to the good performance on the common `Class 0`.

---

## Part 2: Advanced Regression Metrics

### 2.1. Mean Absolute Percentage Error (MAPE)

*   **What it is:** The average of the absolute percentage errors. It measures the error as a percentage of the true value.
*   **Formula:** `(1/n) * sum(|(y_true - y_pred) / y_true|)`
*   **When to use it:** When you want to understand the error relative to the magnitude of the true values. An error of 10 is very different when the true value is 100 (10% error) versus when the true value is 1000 (1% error). MAPE is great for business contexts and explaining model error to non-technical stakeholders.
*   **Pitfall:** It is undefined if any `y_true` value is zero. It can also be biased if `y_true` values are very close to zero.

### 2.2. R-squared (R²) - Coefficient of Determination

*   **What it is:** Measures the **proportion of the variance** in the dependent variable (the target) that is predictable from the independent variables (the features). It compares your model's performance to a baseline model that simply predicts the mean of the target values.
*   **Range:** Can be from -∞ to 1.
    *   **R² = 1:** The model perfectly predicts the target values.
    *   **R² = 0:** The model performs no better than the baseline mean-predicting model.
    *   **R² < 0:** The model is actively worse than just predicting the mean.
*   **When to use it:** To get a quick sense of the overall "goodness of fit" of your model.

### 2.3. Implementation

```python
from sklearn.metrics import mean_absolute_percentage_error, r2_score

print("\n--- Part 2: Advanced Regression Metrics ---")

# --- Dummy Regression Data ---
y_true_reg = torch.tensor([100., 250., 150., 400., 650.])
y_pred_reg = torch.tensor([110., 245., 165., 420., 630.])

# --- MAPE ---
mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
print(f"--> On average, our predictions are off by {mape*100:.2f}% of the true value.")

# --- R-squared ---
r2 = r2_score(y_true_reg, y_pred_reg)
print(f"\nR-squared (R²): {r2:.4f}")
print(f"--> Our model explains {r2*100:.2f}% of the variance in the target data.")
```

---

## Part 3: Ranking and Probability Metrics

These metrics are crucial for tasks where the model outputs a ranked list of predictions or a probability score.

### 3.1. Average Precision (AP) and mean Average Precision (mAP)

*   **What it is:** AP summarizes the Precision-Recall curve into a single number. It's the weighted average of precisions achieved at each threshold, with the increase in recall from the previous threshold as the weight. It is the primary metric for **object detection** and information retrieval tasks.
*   **mAP:** In object detection, AP is calculated for each class, and then averaged across all classes to get the **mean Average Precision (mAP)**.
*   **Interpretation:** A higher mAP means the model is better at both correctly classifying objects and localizing them accurately.

### 3.2. Log Loss (Binary Cross-Entropy)

*   **What it is:** This is the same as the Binary Cross-Entropy loss function we use for training, but it can also be used as an evaluation metric. It evaluates the performance of a classifier that outputs a **probability value**.
*   **Formula:** `-(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))`
*   **Interpretation:** It heavily penalizes predictions that are confident and wrong. A model that predicts a probability of 0.9 for a sample that is actually negative will have a much higher log loss than a model that predicts 0.6. It measures how well-calibrated your model's probabilities are.

### 3.3. Implementation

```python
from sklearn.metrics import average_precision_score, log_loss

print("\n--- Part 3: Ranking and Probability Metrics ---")

# --- Dummy Data for AP and Log Loss ---
# True binary labels
y_true_prob = torch.tensor([0, 1, 1, 0, 1, 1])
# Model's predicted probabilities (scores)
y_scores_prob = torch.tensor([0.1, 0.8, 0.6, 0.3, 0.7, 0.4])

# --- Average Precision ---
# This is the area under the Precision-Recall curve.
ap_score = average_precision_score(y_true_prob, y_scores_prob)
print(f"Average Precision (AP): {ap_score:.4f}")

# --- Log Loss ---
# Note: The input probabilities are clipped to avoid log(0) errors.
logloss_score = log_loss(y_true_prob, y_scores_prob)
print(f"\nLog Loss (Binary Cross-Entropy): {logloss_score:.4f}")

# --- Demonstrate Log Loss Penalty ---
# Confident and WRONG prediction
confident_wrong_preds = torch.tensor([0.99])
confident_wrong_true = torch.tensor([0])
loss_confident_wrong = log_loss(confident_wrong_true, confident_wrong_preds)

# Unconfident and WRONG prediction
unconfident_wrong_preds = torch.tensor([0.51])
unconfident_wrong_true = torch.tensor([0])
loss_unconfident_wrong = log_loss(unconfident_wrong_true, unconfident_wrong_preds)

print(f"Log Loss for a confident wrong prediction (0.99 vs 0): {loss_confident_wrong:.2f}")
print(f"Log Loss for an unconfident wrong prediction (0.51 vs 0): {loss_unconfident_wrong:.2f}")
print("--> The penalty for being confident and wrong is much higher.")
```

## Conclusion

Choosing the right evaluation metric is as important as choosing the right model architecture or loss function. A single metric like accuracy can be dangerously misleading. A thorough evaluation requires a suite of metrics that reflect the nuances of the task and the underlying business goal.

**Key Takeaways:**

1.  **For Multi-Class Problems, Average Wisely:** Use **Macro-F1** if you care about performance on rare classes. Use **Weighted-F1** if you care more about performance on common classes.
2.  **For Regression, Think in Percentages:** **MAPE** is an excellent, interpretable metric for explaining model performance to a non-technical audience.
3.  **For Object Detection, Use mAP:** Mean Average Precision is the standard metric for evaluating the performance of object detectors.
4.  **For Probabilistic Predictions, Use Log Loss:** If you need to evaluate not just the correctness of a classification but also its confidence, Log Loss is the right tool.

By expanding your toolkit of evaluation metrics, you can gain a much deeper and more honest understanding of your model's true performance.

## Self-Assessment Questions

1.  **Macro vs. Weighted Average:** You are building a 3-class classifier. The classes have a distribution of 80%, 15%, and 5%. If you want to ensure the model is performing well on the rarest class, which averaging method for the F1-score should you pay more attention to?
2.  **MAPE:** When is MAPE a more useful metric than MAE?
3.  **R-squared:** Your model achieves an R-squared score of -0.5. What does this tell you about your model's performance?
4.  **mAP:** What two tasks is mean Average Precision (mAP) simultaneously evaluating in an object detection context?
5.  **Log Loss:** Model A predicts a probability of 0.6 for a positive class sample. Model B predicts 0.9 for the same sample. Which model will have a lower (better) Log Loss score for this specific sample?

