# Day 9.1: Overfitting Analysis & Prevention - A Practical Guide

## Introduction: The Model That Knew Too Much

One of the most fundamental challenges in machine learning is the trade-off between **bias** and **variance**. A model with high bias is too simple and fails to capture the underlying patterns in the data (**underfitting**). A model with high variance is too complex and learns the training data *too* well—it memorizes not just the signal, but also the noise. This is called **overfitting**.

An overfit model performs exceptionally well on the data it was trained on, but fails to **generalize** to new, unseen data. This makes it useless for any real-world application. Identifying and preventing overfitting is therefore a critical skill for any machine learning practitioner.

This guide will provide a practical, hands-on demonstration of how to diagnose and prevent overfitting.

**Today's Learning Objectives:**

1.  **Induce and Visualize Overfitting:** Intentionally train a model that is too powerful for a simple dataset to see overfitting in action.
2.  **Diagnose Overfitting with Learning Curves:** Learn to plot training loss vs. validation loss and training accuracy vs. validation accuracy. Understand that a large gap between these curves is the classic sign of overfitting.
3.  **Implement Early Stopping:** Learn about this simple and highly effective regularization technique where we stop training when the validation performance stops improving.
4.  **Understand the Role of Model Capacity:** See how reducing a model's complexity (making it smaller) can be a powerful way to combat overfitting.

--- 

## Part 1: Inducing and Diagnosing Overfitting

To study overfitting, we first need to create it. The easiest way to do this is to use a model with a very high capacity (i.e., many parameters) on a dataset that is relatively small or simple.

We will create a simple synthetic dataset for a binary classification task and then train a large MLP on it.

### 1.1. Creating a Synthetic Dataset

We will use `scikit-learn` to generate a simple, non-linear "moons" dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

print("--- Part 1: Inducing and Diagnosing Overfitting ---")

# --- 1. Generate Synthetic Data ---
# n_samples: total points, noise: standard deviation of Gaussian noise added
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- 2. Create Dataset and Split into Train/Validation ---
dataset = TensorDataset(X_tensor, y_tensor)

# Split the data: 70% for training, 30% for validation
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Created a synthetic dataset with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# --- 3. Define an Overly Complex Model ---
# This model has many more parameters than necessary for this simple task.
class OverfitModel(nn.Module):
    def __init__(self):
        super(OverfitModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = OverfitModel()
```

### 1.2. Training and Plotting Learning Curves

Now, we will train the model for many epochs and record the training and validation loss and accuracy at each epoch. Plotting these values is the key to diagnosing overfitting.

```python
# --- 4. The Training Loop ---
def train_and_diagnose(model, train_loader, val_loader, num_epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store metrics for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds_logits = model(X_batch)
            loss = loss_fn(preds_logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += ((torch.sigmoid(preds_logits) > 0.5) == y_batch).sum().item()

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds_logits = model(X_batch)
                loss = loss_fn(preds_logits, y_batch)
                val_loss += loss.item()
                val_correct += ((torch.sigmoid(preds_logits) > 0.5) == y_batch).sum().item()

        # --- Record Metrics ---
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / len(train_dataset))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / len(val_dataset))
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
            
    return history

history_overfit = train_and_diagnose(model, train_loader, val_loader)

# --- 5. Plotting the Learning Curves ---
def plot_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axs[0].plot(history['train_loss'], label='Training Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Accuracy
    axs[1].plot(history['train_acc'], label='Training Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_title('Accuracy Curves')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.show()

print("\nPlotting learning curves for the overfit model...")
plot_curves(history_overfit)
```

**Interpreting the Curves:**

You will see a classic pattern:
*   **Loss Curves:** The training loss consistently decreases, approaching zero. The validation loss, however, decreases for a while and then starts to **increase**. This is the moment the model stops generalizing and starts memorizing the training data noise.
*   **Accuracy Curves:** The training accuracy approaches 100%. The validation accuracy plateaus or may even start to decrease.

The **gap** between the training curve and the validation curve is the visual representation of overfitting.

--- 

## Part 2: Prevention Technique 1 - Early Stopping

Early stopping is the simplest and one of the most effective regularization techniques.

**The Idea:** Monitor the validation loss during training. If the validation loss does not improve for a certain number of consecutive epochs (a parameter called `patience`), we stop the training process and save the model from the epoch where the validation loss was at its minimum.

This directly prevents the model from continuing to train into the overfitting region we identified in the learning curves.

### 2.1. Implementing Early Stopping

Let's modify our training loop to include early stopping.

```python
import copy

print("\n--- Part 2: Implementing Early Stopping ---")

def train_with_early_stopping(model, train_loader, val_loader, num_epochs=300, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # --- Early Stopping Variables ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds_logits = model(X_batch)
            loss = loss_fn(preds_logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation Phase with Early Stopping Logic ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds_logits = model(X_batch)
                loss = loss_fn(preds_logits, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        # (Recording other metrics would go here)

        # --- Check for improvement ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model weights
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f} | Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            # Load the best model weights before returning
            model.load_state_dict(best_model_wts)
            break
            
    return model, history

# Re-initialize the model and train with early stopping
model_es = OverfitModel()
history_es = train_with_early_stopping(model_es, train_loader, val_loader)
```

--- 

## Part 3: Prevention Technique 2 - Reducing Model Capacity

Another direct way to fight overfitting is to make the model less complex. A model with fewer parameters has less "memorization power" and is forced to learn a more general representation.

Let's define a smaller model and train it on the same data.

```python
print("\n--- Part 3: Reducing Model Capacity ---")

# --- 1. Define a Simpler Model ---
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32), # Much smaller hidden layer
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# --- 2. Train and Diagnose the Simpler Model ---
print("\nTraining the simpler model...")
simple_model = SimpleModel()
history_simple = train_and_diagnose(simple_model, train_loader, val_loader)

# --- 3. Plot the new curves ---
print("\nPlotting learning curves for the simpler model...")
plot_curves(history_simple)
```

**Interpreting the New Curves:**
When you plot the curves for the simpler model, you will notice that the gap between the training and validation curves is much smaller. The model doesn't achieve 100% accuracy on the training set, but its validation performance is more stable and often better than the overfit model's best performance.

## Conclusion

Overfitting is a constant battle in machine learning. A model that is not powerful enough will underfit, while a model that is too powerful will overfit. The key is to find the right balance for your specific dataset.

**Key Takeaways for Analysis and Prevention:**

1.  **Always Use a Validation Set:** You cannot diagnose overfitting without a separate validation set. This is the single most important practice for reliable model development.
2.  **Plot Your Learning Curves:** Visualizing your training and validation metrics over time is the primary tool for identifying when and how badly your model is overfitting.
3.  **The Gap is the Signal:** A significant and growing gap between your training and validation performance is the classic sign of overfitting.
4.  **Start with Early Stopping:** It's a simple, effective, and computationally cheap way to prevent the worst effects of overfitting.
5.  **Find the Right Capacity:** If overfitting is severe, try reducing the complexity of your model (fewer layers or fewer neurons per layer). If your model is underfitting (both training and validation scores are poor), you may need to increase its capacity.

In the next guides, we will explore more advanced regularization techniques like Dropout and Weight Decay that give us more tools to fight overfitting.

## Self-Assessment Questions

1.  **Defining Overfitting:** In your own words, what is overfitting?
2.  **Diagnosing Overfitting:** If you plot your loss curves and see both the training and validation loss decreasing and staying close together, is your model overfitting?
3.  **Early Stopping:** What metric should you monitor for early stopping: training loss or validation loss? Why?
4.  **Model Capacity:** You train a model and find that its validation accuracy is much lower than its training accuracy. What is one of the first things you could try to fix this?
5.  **Generalization:** What does it mean for a model to "generalize" well?

