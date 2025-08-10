# Day 10.1: Evaluation Methodology & Design - A Practical Guide

## Introduction: How Do You Know if Your Model is Actually Good?

Building and training a model is only half the battle. How do you get a trustworthy estimate of its performance on new, unseen data? A model that gets 99% accuracy on the data it was trained on is meaningless if its performance plummets to 50% the moment it sees a new example. This is the problem of **generalization**, and a robust evaluation methodology is our tool for measuring it.

Designing a proper evaluation strategy is the most important part of the scientific process in machine learning. It prevents you from fooling yourself and others, ensures your results are credible, and allows for fair comparison between different models.

This guide will provide a practical walkthrough of the principles behind designing a robust evaluation strategy, focusing on data splitting, the dangers of data leakage, and the gold standard of cross-validation.

**Today's Learning Objectives:**

1.  **Understand the Three Essential Data Splits:** Learn the distinct roles of the **Training Set**, **Validation Set**, and **Test Set**.
2.  **Beware the Peril of Data Leakage:** Understand what data leakage is and why it leads to overly optimistic and untrustworthy results.
3.  **Implement a Proper Train-Val-Test Split:** Learn how to correctly partition your data for model development and final evaluation.
4.  **Grasp the Concept of Cross-Validation:** Understand K-Fold Cross-Validation as a more robust evaluation technique, especially for smaller datasets.
5.  **Implement K-Fold Cross-Validation:** Write a complete training and evaluation loop using cross-validation to get a more reliable estimate of your model's performance.

---

## Part 1: The Three Data Splits - A Separation of Duties

A robust evaluation strategy relies on splitting your data into three independent sets:

1.  **Training Set:**
    *   **Purpose:** The data the model actually **learns from**. The model sees this data and its corresponding labels, and the optimizer adjusts the model's weights based on the loss calculated on this set.
    *   **Size:** Typically the largest portion of the data (e.g., 60-80%).

2.  **Validation (or Development) Set:**
    *   **Purpose:** This set is used for **tuning the model and its hyperparameters**. During development, you train your model on the training set and then evaluate its performance on the validation set. You use this performance to make decisions: Should I use a different learning rate? Should I add more layers? Should I use more dropout? Should I stop training now (early stopping)?
    *   **The Golden Rule:** The model **never trains** on the validation data. It only sees it for evaluation during the development phase.
    *   **Size:** Typically 10-20% of the data.

3.  **Test (or Hold-out) Set:**
    *   **Purpose:** This set is used **only once**, at the very end of your project, to get a final, unbiased estimate of your model's performance on unseen data.
    *   **The Second Golden Rule:** You must pretend the test set **does not exist** during the entire development process. You should never, ever make any decisions or tune any hyperparameters based on the test set performance. If you do, you have invalidated your results, because the test set is no longer truly "unseen."
    *   **Size:** Typically 10-20% of the data.

**Analogy:**
*   **Training Set:** The textbook and homework problems you use to study for an exam.
*   **Validation Set:** The practice exams you take to gauge your progress and decide what topics to study more. You might take many practice exams.
*   **Test Set:** The final, official exam. You only get one shot, and its score is your final grade.

---

## Part 2: The Danger of Data Leakage

**Data leakage** is the cardinal sin of machine learning evaluation. It occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates.

**Common Ways Data Leakage Occurs:**

1.  **Tuning on the Test Set:** The most common mistake. If you tweak your model until it performs well on the test set, you have simply overfit to the test set, and your reported performance is meaningless.

2.  **Preprocessing Before Splitting:** Imagine you are normalizing your data (e.g., using `StandardScaler` in scikit-learn). If you `fit` the scaler on the *entire dataset* and then split it into train and test, you have leaked information. The training set has now been scaled using information (the mean and standard deviation) from the test set. **The correct way:** Split the data first, then `fit` the scaler *only* on the training set, and use that fitted scaler to `transform` both the training and test sets.

### 2.1. Implementing a Correct Train-Val-Test Split

```python
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("---" + "-" * 20 + " Part 2: Correct Data Splitting and Preprocessing " + "-" * 20 + "---")

# --- 1. Generate some dummy data ---
# Let's imagine a dataset with 1000 samples and 10 features
X_raw = torch.randn(1000, 10)
y_raw = torch.randint(0, 2, (1000,))

# --- 2. The WRONG Way (Leakage!) ---
# scaler_leak = StandardScaler()
# X_scaled_leak = scaler_leak.fit_transform(X_raw)
# X_train, X_test, ... = train_test_split(X_scaled_leak, ...)
# Here, the training data was scaled using the mean/std of the test data.

# --- 3. The CORRECT Way ---
# First, split into training and a temporary set (which will be split into val and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42 # 30% for val+test
)

# Now, split the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42 # 50% of temp -> 15% of total
)

print(f"Data split into:")
print(f"  - Training set size: {len(X_train)}")
print(f"  - Validation set size: {len(X_val)}")
print(f"  - Test set size: {len(X_test)}")

# --- 4. Preprocess AFTER Splitting ---
# Fit the scaler ONLY on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Apply the SAME fitted scaler to all three sets
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nData correctly split and scaled without leakage.")
```

---

## Part 3: K-Fold Cross-Validation - A More Robust Approach

**The Problem with a Single Validation Set:** If you have a small dataset, how you split it can have a big impact on the validation results. By chance, you might get a particularly "easy" or "hard" validation set, leading to a misleading performance estimate.

**The Solution: K-Fold Cross-Validation**

K-Fold CV provides a more robust estimate of model performance by using the data more efficiently.

**The Process:**
1.  Split your data into a training set and a test set. **Set the test set aside.**
2.  Take the remaining training data and split it into `K` equal-sized "folds" (e.g., K=5 or K=10).
3.  Now, run `K` rounds of training:
    *   **Round 1:** Use Fold 1 as the validation set and Folds 2, 3, 4, 5 as the training set.
    *   **Round 2:** Use Fold 2 as the validation set and Folds 1, 3, 4, 5 as the training set.
    *   ...and so on, until every fold has been used as the validation set exactly once.
4.  **Aggregate the Results:** You now have `K` different validation scores. You can average these scores to get a much more robust and reliable estimate of your model's performance. The standard deviation of these scores also gives you an idea of how stable your model's performance is.

![K-Fold Cross-Validation](https://i.imgur.com/0Kz1D4M.png)

### 3.1. Implementing K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

print("\n---" + "-" * 20 + " Part 3: K-Fold Cross-Validation " + "-" * 20 + "---")

# --- 1. Get the data (we'll use the full dataset for this demo, excluding a final test set)
X_full_train = X_raw
y_full_train = y_raw

# --- 2. Define the K-Fold Splitter ---
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# --- 3. The Cross-Validation Loop ---
validation_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_train)):
    print(f"\n--- FOLD {fold + 1}/{K} ---")
    
    # --- a. Get the data for this fold ---
    X_train_fold = X_full_train[train_idx]
    y_train_fold = y_full_train[train_idx]
    X_val_fold = X_full_train[val_idx]
    y_val_fold = y_full_train[val_idx]
    
    # --- b. Create Datasets and DataLoaders ---
    train_fold_dataset = TensorDataset(X_train_fold, y_train_fold.float().view(-1, 1))
    val_fold_dataset = TensorDataset(X_val_fold, y_val_fold.float().view(-1, 1))
    train_fold_loader = DataLoader(train_fold_dataset, batch_size=32)
    val_fold_loader = DataLoader(val_fold_dataset, batch_size=32)
    
    # --- c. Initialize model and optimizer for each fold ---
    # It's crucial to re-initialize the model for each fold!
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # --- d. Train the model (simplified loop) ---
    for epoch in range(50): # Train for a fixed number of epochs
        model.train()
        for X_b, y_b in train_fold_loader:
            p = model(X_b); l = loss_fn(p, y_b)
            optimizer.zero_grad(); l.backward(); optimizer.step()
            
    # --- e. Evaluate on the validation fold ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for X_b, y_b in val_fold_loader:
            preds = model(X_b)
            val_correct += ((torch.sigmoid(preds) > 0.5) == y_b).sum().item()
            
    fold_accuracy = val_correct / len(val_fold_dataset)
    validation_accuracies.append(fold_accuracy)
    print(f"Validation Accuracy for Fold {fold + 1}: {fold_accuracy:.4f}")

# --- 4. Aggregate the Results ---
mean_accuracy = np.mean(validation_accuracies)
std_accuracy = np.std(validation_accuracies)

print("\n---" + "-" * 20 + " K-Fold Cross-Validation Results " + "-" * 20 + "---")
print(f"Validation Accuracies for each fold: {validation_accuracies}")
print(f"Mean Validation Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Validation Accuracy: {std_accuracy:.4f}")
```

## Conclusion

A disciplined evaluation methodology is the bedrock of reliable machine learning. Without it, your results are not trustworthy.

**Key Takeaways:**

1.  **Use Three Sets:** Your workflow should always involve a training set (for learning), a validation set (for tuning), and a test set (for final, unbiased evaluation).
2.  **Protect Your Test Set:** The test set must be held out and used only once at the very end. All development decisions must be made using the validation set.
3.  **Prevent Data Leakage:** Be extremely careful about your preprocessing pipeline. Any operation that learns from data (like fitting a scaler or an encoder) must be fitted *only* on the training data and then applied to the other sets.
4.  **Use Cross-Validation for Robustness:** For smaller datasets, K-Fold Cross-Validation provides a much more reliable estimate of model performance than a single train-val split. It gives you a mean performance and a measure of its variance.

By adopting these principles, you ensure that your reported results are a true reflection of your model's ability to generalize to the real world.

## Self-Assessment Questions

1.  **Data Splits:** What is the specific purpose of the validation set?
2.  **Data Leakage:** You want to fill missing values in your dataset with the mean of the column. What is the correct, leak-free way to do this in a train-test split scenario?
3.  **Test Set:** When is the appropriate time to use the test set, and how many times should you use it?
4.  **K-Fold CV:** What is the main advantage of using 5-Fold Cross-Validation over a single 80/20 train-validation split?
5.  **Model Re-initialization:** In the K-Fold CV implementation, why is it critical to re-initialize the model and optimizer inside the loop for each fold?

