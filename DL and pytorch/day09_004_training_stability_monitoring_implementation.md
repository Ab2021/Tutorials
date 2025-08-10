# Day 9.4: Training Stability & Monitoring - A Practical Guide

## Introduction: Is My Model Actually Learning?

Starting a training run is easy. But once it's running, how do you know if it's working correctly? Is the loss supposed to jump around like that? Is a loss of 0.1 good or bad? Why did my loss suddenly become `NaN`?

Monitoring the training process is a critical skill. It allows you to debug problems, make informed decisions about hyperparameters, and gain confidence that your model is learning effectively. Simply printing the loss at the end of each epoch is not enough.

This guide provides a practical overview of techniques and tools for monitoring training stability and performance, focusing on two key areas: **numerical stability** (preventing exploding or vanishing gradients) and **experiment tracking**.

**Today's Learning Objectives:**

1.  **Understand Exploding and Vanishing Gradients:** Grasp the practical consequences of these numerical instability problems.
2.  **Implement Gradient Clipping:** Learn this essential technique to prevent exploding gradients.
3.  **Learn to Interpret Learning Curves:** Go beyond simple loss plots to understand what different patterns in your training curves mean.
4.  **Understand the Importance of Learning Rate Scheduling:** See how dynamically adjusting the learning rate during training can improve stability and performance.
5.  **Introduce Experiment Tracking Tools (TensorBoard):** Learn how to use TensorBoard to log and visualize metrics, creating a powerful dashboard for monitoring your experiments.

---

## Part 1: Numerical Stability - Gradient Clipping

**The Problem: Exploding Gradients**

In deep networks, especially RNNs, the gradients calculated during backpropagation can multiply together and become astronomically large. This leads to huge updates to the weights, causing the model's parameters to become `inf` or `NaN` (Not a Number). The result is a training process that completely collapses.

**The Solution: Gradient Clipping**

Gradient clipping is a simple and effective technique to prevent this. After calculating the gradients with `loss.backward()`, but **before** calling `optimizer.step()`, we check the size (the norm) of the gradients. If the total norm of all gradients exceeds a certain threshold, we scale them down to be exactly at that threshold. This acts like a safety rail, preventing the weight updates from ever becoming too large.

### 1.1. Implementing Gradient Clipping

```python
import torch
import torch.nn as nn

print("---"" Part 1: Gradient Clipping ---")

# --- Create a simple model and dummy data ---
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# An input that might cause large gradients
input_tensor = torch.randn(1, 10) * 100 
target = torch.tensor([[1000.0]])

# --- The training step with clipping ---

# 1. Standard forward and backward pass
optimizer.zero_grad()
prediction = model(input_tensor)
loss = (prediction - target)**2
loss.backward()

# At this point, the gradients might be huge.
# We can inspect their norm before clipping.
# torch.nn.utils.clip_grad_norm_ takes an iterable of parameters and a max_norm.
# It computes the norm over all gradients together.
total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Total gradient norm before clipping: {total_norm_before:.2f}")

# 2. Apply Gradient Clipping
# We re-calculate the gradients for a fair comparison
optimizer.zero_grad()
prediction = model(input_tensor)
loss = (prediction - target)**2
loss.backward()

# Now we clip the gradients to a max value (e.g., 1.0)
max_norm = 1.0
total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

print(f"\nApplying gradient clipping with max_norm={max_norm}")
print(f"Total gradient norm after clipping: {total_norm_after:.2f}")

# 3. Optimizer step
# Now we perform the update with the clipped (safe) gradients.
optimizer.step()

print("\nOptimizer step performed with clipped gradients.")
```

---

## Part 2: Learning Rate Scheduling

Choosing a fixed learning rate is often suboptimal. 
*   If it's too high, the model can become unstable and diverge.
*   If it's too low, training will be very slow.

A **Learning Rate Scheduler** dynamically adjusts the learning rate during training according to a pre-defined schedule. This is a very common and effective technique.

**Common Strategies:**

*   **StepLR:** Decreases the learning rate by a factor (`gamma`) every `step_size` epochs.
*   **ReduceLROnPlateau:** Reduces the learning rate when a monitored metric (like validation loss) has stopped improving. This is a very popular and intuitive scheduler.
*   **CosineAnnealingLR:** Smoothly anneals the learning rate from an initial value down to a minimum value following a cosine curve.

### 2.1. Implementing `ReduceLROnPlateau`

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("\n---"" Part 2: Learning Rate Scheduling ---")

# --- Setup ---
model_sched = nn.Linear(10, 1)
optimizer_sched = torch.optim.SGD(model_sched.parameters(), lr=0.1)

# --- Create the Scheduler ---
# This scheduler will monitor the 'val_loss'.
# If the val_loss doesn't improve for `patience=3` epochs, the LR is reduced by a factor of 0.1.
scheduler = ReduceLROnPlateau(
    optimizer_sched, 
    mode='min',      # 'min' means we want the monitored quantity to decrease
    factor=0.1,      # Factor by which the learning rate will be reduced
    patience=3,      # Number of epochs with no improvement after which learning rate will be reduced
    verbose=True     # Prints a message when the learning rate is updated
)

# --- Dummy Training Loop to Demonstrate ---
print("\nDemonstrating ReduceLROnPlateau scheduler:")
for epoch in range(10):
    # In a real loop, you would calculate the actual validation loss
    dummy_val_loss = 2.0 if epoch < 5 else 2.1 # Simulate a loss that stops improving after epoch 5
    
    print(f"Epoch {epoch+1}, Current LR: {optimizer_sched.param_groups[0]['lr']:.5f}, Val Loss: {dummy_val_loss}")
    
    # The scheduler step is called after the validation phase
    scheduler.step(dummy_val_loss)
```

---

## Part 3: Experiment Tracking with TensorBoard

Printing metrics to the console is fine for simple tests, but for serious projects, you need a better way to log and visualize your experiments. **TensorBoard** is a powerful visualization toolkit that comes with TensorFlow but is easily integrated with PyTorch.

It allows you to log metrics, images, model graphs, and more, and view them in an interactive web-based dashboard.

### 3.1. Using `SummaryWriter`

PyTorch's interface to TensorBoard is the `SummaryWriter` class.

1.  **Installation:** `pip install tensorboard`
2.  **Instantiate `SummaryWriter`:** Create a writer object, pointing it to a log directory.
3.  **Log Metrics:** Inside your training loop, use methods like `writer.add_scalar()` to log your metrics.
4.  **Launch TensorBoard:** From your terminal, run `tensorboard --logdir=runs` (where `runs` is your log directory).

```python
from torch.utils.tensorboard import SummaryWriter

print("\n---"" Part 3: Experiment Tracking with TensorBoard ---")

# --- 1. Setup ---
# This will create a directory like 'runs/my_experiment_1' to store the logs.
writer = SummaryWriter('runs/my_experiment_1')

# We will reuse the training setup from the regularization guide
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader, random_split

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [350, 150])
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

model_tb = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_tb.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_tb.to(device)

# --- 2. The Training Loop with TensorBoard Logging ---
print("\nTraining with TensorBoard logging...")
print("Run `tensorboard --logdir=runs` in your terminal to see the dashboard.")
num_epochs = 100
for epoch in range(num_epochs):
    # --- Training ---
    model_tb.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model_tb(X_batch); loss = loss_fn(preds, y_batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    # --- Validation ---
    model_tb.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model_tb(X_batch); val_loss += loss_fn(preds, y_batch).item()
    avg_val_loss = val_loss / len(val_loader)
    
    # --- 3. Log the scalars ---
    # The first argument is the tag (the name of the plot).
    # The second argument is the scalar value.
    # The third argument is the global step (e.g., the epoch number).
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    
    # You can also log multiple plots on the same graph
    writer.add_scalars('Loss/Combined', {
        'train': avg_train_loss,
        'validation': avg_val_loss
    }, epoch)

# --- 4. Close the writer ---
writer.close()
print("\nFinished training. Log files saved to 'runs/my_experiment_1'.")
```

## Conclusion

Effective training is an active process of monitoring and intervention. By using tools like gradient clipping, learning rate scheduling, and experiment trackers like TensorBoard, you move from being a passive observer to an informed practitioner who can diagnose problems and guide the training process towards a better, more stable solution.

**Key Takeaways:**

1.  **Prevent Explosions:** Use `torch.nn.utils.clip_grad_norm_` in your training loop if you are working with deep networks or RNNs to prevent exploding gradients.
2.  **Don't Use a Fixed Learning Rate:** A learning rate scheduler is almost always a good idea. `ReduceLROnPlateau` is a great, intuitive starting point.
3.  **Log Everything:** Use an experiment tracker like TensorBoard from the start. Logging your metrics allows you to visualize progress, compare different experiments, and debug effectively.
4.  **Interpret the Curves:** Learn to read your loss and accuracy curves. A spiky, unstable loss might indicate a learning rate that is too high. A validation loss that is flat might mean the model is not complex enough or the learning rate is too low. A validation loss that is increasing is a clear sign of overfitting.

By incorporating these stability and monitoring techniques into your workflow, you can train more robust models and accelerate your development cycle.

## Self-Assessment Questions

1.  **Gradient Clipping:** At what point in the training step should you apply gradient clipping?
2.  **Learning Rate Schedulers:** What is the main advantage of using a learning rate scheduler over a fixed learning rate?
3.  **`ReduceLROnPlateau`:** What metric should you typically pass to the `scheduler.step()` method when using `ReduceLROnPlateau`?
4.  **TensorBoard:** What is the name of the PyTorch class used to write logs for TensorBoard?
5.  **Debugging:** You start training and your loss immediately becomes `NaN`. What is the most likely cause, and what technique from this guide could you use to fix it?

