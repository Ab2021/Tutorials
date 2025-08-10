# Day 1.5: Evaluation Metrics - A Practical Implementation Guide

## Introduction: How Good Is Your Model, Really?

After training a model, we get a set of predictions. But how do we quantify if these predictions are good, bad, or just mediocre? Simply looking at the loss function on the test set is not enough. We need evaluation metrics that are interpretable and aligned with the business problem we are trying to solve.

This guide provides a practical, code-first exploration of the most important evaluation metrics in machine learning. We will cover metrics for both classification and regression tasks. For each metric, we will:

1.  Understand its formula and intuition.
2.  Implement its calculation from scratch using PyTorch tensors.
3.  Show how to use the highly optimized and convenient implementations from `scikit-learn`.
4.  Discuss when to use it and what pitfalls to avoid.

**Today's Learning Objectives:**

1.  **Master the Confusion Matrix:** Understand and interpret True Positives, True Negatives, False Positives, and False Negatives.
2.  **Implement Classification Metrics:** Calculate and interpret Accuracy, Precision, Recall, and the F1-Score.
3.  **Handle Imbalanced Datasets:** Understand why accuracy is misleading for imbalanced data and why Precision-Recall curves are often a better choice than ROC curves.
4.  **Visualize Model Performance:** Plot and interpret the ROC Curve and the Precision-Recall Curve.
5.  **Implement Regression Metrics:** Calculate and interpret Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

---

## Part 1: The Foundation - The Confusion Matrix

For any classification problem, the **confusion matrix** is the starting point for almost every metric. It's a simple table that breaks down the performance of a classifier.

Let's consider a binary classification problem (e.g., predicting "Spam" vs. "Not Spam").

|                    | **Predicted: Not Spam** | **Predicted: Spam** |
|--------------------|-------------------------|---------------------|
| **Actual: Not Spam** | True Negative (TN)      | False Positive (FP) |
| **Actual: Spam**     | False Negative (FN)     | True Positive (TP)  |

*   **True Positive (TP):** The email was Spam, and we correctly predicted Spam.
*   **True Negative (TN):** The email was Not Spam, and we correctly predicted Not Spam.
*   **False Positive (FP):** The email was Not Spam, but we incorrectly predicted Spam. (A "false alarm" - Type I Error).
*   **False Negative (FN):** The email was Spam, but we incorrectly predicted Not Spam. (A "miss" - Type II Error).

### 1.1. Calculating the Confusion Matrix

Let's generate some dummy predictions and true labels and then calculate the confusion matrix components.

```python
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---
Let's create some dummy data ---
# True labels
y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
# Model's predictions
y_pred = torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])

# ---
Manual Calculation using PyTorch ---
def get_confusion_matrix_components(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()
    return TP, TN, FP, FN

TP, TN, FP, FN = get_confusion_matrix_components(y_true, y_pred)

print("---
Manual Confusion Matrix Calculation ---")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}\n")

# ---
Using Scikit-learn (The Easy Way) ---
# Scikit-learn's confusion_matrix returns the matrix in the format:
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_true, y_pred)

print("---
Scikit-learn confusion_matrix ---")
print(cm)

# For better visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## Part 2: Core Classification Metrics

Now that we have the TP, TN, FP, and FN counts, we can calculate the most common classification metrics.

### 2.1. Accuracy

**What it is:** The percentage of total predictions that were correct.
**Formula:** `(TP + TN) / (TP + TN + FP + FN)`
**When to use it:** When your classes are balanced and the cost of a False Positive and a False Negative are roughly equal.
**When to AVOID it:** When you have an **imbalanced dataset**. (The "Accuracy Paradox" we discussed in the previous guide).

### 2.2. Precision

**What it is:** Of all the times the model predicted the positive class, what percentage were actually positive?
**Formula:** `TP / (TP + FP)`
**When to use it:** When the cost of a **False Positive** is high. 
*   *Example:* Spam detection. You don't want to incorrectly classify an important email (a non-spam) as spam (a False Positive). This would be very costly.
*   *Example:* Recommending a very expensive product. A wrong recommendation (FP) wastes the user's time and trust.

### 2.3. Recall (Sensitivity or True Positive Rate)

**What it is:** Of all the actual positive cases, what percentage did the model correctly identify?
**Formula:** `TP / (TP + FN)`
**When to use it:** When the cost of a **False Negative** is high.
*   *Example:* Medical diagnosis for a serious disease. Failing to detect the disease (a False Negative) is catastrophic.
*   *Example:* Fraud detection. Failing to detect a fraudulent transaction (FN) costs the company money.

### 2.4. F1-Score

**What it is:** The harmonic mean of Precision and Recall. It provides a single score that balances both.
**Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
**When to use it:** When you need to balance Precision and Recall, and there's an uneven class distribution.

### 2.5. Implementation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# We'll use the same dummy data as before
y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
TP, TN, FP, FN = get_confusion_matrix_components(y_true, y_pred)

# ---
Manual Calculation ---
def calculate_metrics_manual(TP, TN, FP, FN):
    # Handle division by zero
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

acc_man, pre_man, rec_man, f1_man = calculate_metrics_manual(TP, TN, FP, FN)

print("---
Manual Metric Calculation ---")
print(f"Accuracy: {acc_man:.2f}")
print(f"Precision: {pre_man:.2f}")
print(f"Recall: {rec_man:.2f}")
print(f"F1-Score: {f1_man:.2f}\n")

# ---
Using Scikit-learn (The Easy Way) ---
acc_skl = accuracy_score(y_true, y_pred)
pre_skl = precision_score(y_true, y_pred)
rec_skl = recall_score(y_true, y_pred)
f1_skl = f1_score(y_true, y_pred)

print("---
Scikit-learn Metric Calculation ---")
print(f"Accuracy: {acc_skl:.2f}")
print(f"Precision: {pre_skl:.2f}")
print(f"Recall: {rec_skl:.2f}")
print(f"F1-Score: {f1_skl:.2f}\n")

# Scikit-learn also provides a convenient classification report
print("---
Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Not Spam', 'Spam']))
```

---

## Part 3: Visualizing Classifier Performance

Metrics like Precision and Recall are calculated based on a specific **decision threshold**. A model's output is usually a probability (e.g., 0.75). We need to decide at what threshold to classify it as positive (e.g., if probability > 0.5, classify as 1). Changing this threshold will change our TP, FP, FN, and TN counts, and therefore change our metrics.

Visual tools like the ROC curve and Precision-Recall curve help us evaluate a model across *all* possible thresholds.

### 3.1. The ROC Curve (Receiver Operating Characteristic)

**What it is:** A plot of the **True Positive Rate (Recall)** vs. the **False Positive Rate** at various threshold settings.
*   **False Positive Rate (FPR):** `FP / (FP + TN)` (What proportion of actual negatives were incorrectly classified?)

**How to interpret it:**
*   The **top-left corner** is the ideal point (FPR=0, TPR=1).
*   A model that is closer to the top-left corner is better.
*   The diagonal line represents a random classifier (no better than guessing).
*   **AUC (Area Under the Curve):** The area under the ROC curve. A single number summary of the curve.
    *   AUC = 1: Perfect classifier.
    *   AUC = 0.5: Useless (random) classifier.

**When to use it:** For balanced datasets, or when you care equally about the performance on the positive and negative classes.

### 3.2. The Precision-Recall (PR) Curve

**What it is:** A plot of **Precision** vs. **Recall (TPR)** at various threshold settings.

**How to interpret it:**
*   The **top-right corner** is the ideal point (Recall=1, Precision=1).
*   A model that is closer to the top-right corner is better.

**When to use it:** This is the preferred curve for **imbalanced datasets**. The ROC curve can be misleadingly optimistic when the number of negative samples is very large. The PR curve focuses on the performance on the minority (positive) class, which is often what we care about most.

### 3.3. Implementation

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# For these curves, we need the model's predicted probabilities, not the final 0/1 predictions.
# Let's create some dummy probability scores.
y_true = torch.tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
y_scores = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.9, 0.05, 0.6, 0.7])

# ---
ROC Curve ---
fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ---
Precision-Recall Curve ---
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
```

---

## Part 4: Regression Metrics

For regression tasks, where we predict a continuous value (like a price or a temperature), the concept of TP/FP doesn't apply. Instead, we measure the average error between the predicted values and the true values.

### 4.1. Mean Absolute Error (MAE)

**What it is:** The average of the absolute differences between predictions and true values.
**Formula:** `(1/n) * sum(|y_true - y_pred|)`
**Interpretation:** Very straightforward. An MAE of 5.0 means that, on average, our prediction is off by 5.0 units. It's in the same unit as the target variable.

### 4.2. Mean Squared Error (MSE)

**What it is:** The average of the squared differences between predictions and true values.
**Formula:** `(1/n) * sum((y_true - y_pred)^2)`
**Interpretation:** Harder to interpret directly because the units are squared. Its main advantage is that it heavily penalizes large errors. This makes it a good choice for the loss function during training, as it encourages the model to avoid making big mistakes.

### 4.3. Root Mean Squared Error (RMSE)

**What it is:** The square root of the MSE.
**Formula:** `sqrt(MSE)`
**Interpretation:** This is often the best of both worlds. Like MSE, it penalizes large errors heavily. But by taking the square root, the error is back in the same units as the target variable, making it as interpretable as MAE.

### 4.4. Implementation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---
Dummy Regression Data ---
y_true_reg = torch.tensor([2.5, 5.0, 1.5, 4.0, 6.5])
y_pred_reg = torch.tensor([3.0, 4.5, 1.8, 3.5, 7.0])

# ---
Manual Calculation ---
def calculate_regression_metrics_manual(y_true, y_pred):
    mae = torch.abs(y_true - y_pred).mean().item()
    mse = torch.mean((y_true - y_pred)**2).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
    return mae, mse, rmse

mae_man, mse_man, rmse_man = calculate_regression_metrics_manual(y_true_reg, y_pred_reg)

print("---
Manual Regression Metric Calculation ---")
print(f"Mean Absolute Error (MAE): {mae_man:.2f}")
print(f"Mean Squared Error (MSE): {mse_man:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_man:.2f}\n")

# ---
Using Scikit-learn ---
mae_skl = mean_absolute_error(y_true_reg, y_pred_reg)
mse_skl = mean_squared_error(y_true_reg, y_pred_reg)
# Scikit-learn has a `squared=False` argument for MSE to get RMSE
rmse_skl = mean_squared_error(y_true_reg, y_pred_reg, squared=False)

print("---
Scikit-learn Regression Metric Calculation ---")
print(f"Mean Absolute Error (MAE): {mae_skl:.2f}")
print(f"Mean Squared Error (MSE): {mse_skl:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_skl:.2f}\n")
```

## Conclusion: Choosing the Right Metric is Crucial

We've seen that there is no single "best" metric. The choice depends entirely on the specific problem and the business context.

**Final Checklist for Choosing a Metric:**

1.  **Is it a classification or regression problem?** This is the first and most important split.
2.  **If classification, is the dataset balanced?**
    *   **Yes:** Accuracy can be a good starting point, along with Precision and Recall.
    *   **No:** **Avoid accuracy.** Focus on Precision, Recall, F1-Score, and the Precision-Recall Curve.
3.  **What is the business cost of errors?**
    *   **High cost for False Positives?** Prioritize **Precision**.
    *   **High cost for False Negatives?** Prioritize **Recall**.
4.  **If regression, how much do you want to penalize large errors?**
    *   **Equally:** Use **MAE** for its easy interpretation.
    *   **Heavily:** Use **RMSE** to give more weight to large mistakes.

By thoughtfully selecting your evaluation metric, you ensure that you are optimizing for what truly matters and that your model's performance score accurately reflects its value in a real-world scenario.

## Self-Assessment Questions

1.  **Confusion Matrix:** A model is tested on 100 samples. It gets 90 correct. Of the 10 it got wrong, 6 were False Positives and 4 were False Negatives. What are the TP and TN counts?
2.  **Precision vs. Recall:** You are building a system to unlock a door with facial recognition. Which is more important, Precision or Recall? Why?
3.  **ROC vs. PR Curve:** You are building a model to detect a rare manufacturing defect that occurs in 0.1% of products. Which curve (ROC or PR) would give you a more realistic view of your model's performance?
4.  **Regression Metrics:** A model predicts house prices. Model A has an MAE of $10,000 and an RMSE of $30,000. Model B has an MAE of $15,000 and an RMSE of $20,000. What can you infer about the types of errors each model makes?
5.  **Thresholding:** If you want to increase the Recall of your model, should you generally increase or decrease the decision threshold? What effect will this have on Precision?
