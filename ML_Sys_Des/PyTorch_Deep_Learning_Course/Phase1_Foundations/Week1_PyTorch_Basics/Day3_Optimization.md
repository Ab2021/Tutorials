# Day 3: Optimization & Loss Landscapes - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Gradient Descent, Convexity, and Optimizers

## 1. Theoretical Foundation: The Optimization Problem

### The Loss Landscape
We want to find parameters $\theta$ that minimize the Loss Function $L(\theta)$.
Imagine a high-dimensional terrain.
*   **Convex Functions**: Bowl-shaped. One global minimum. Easy to optimize. (e.g., Linear Regression).
*   **Non-Convex Functions**: Hills and valleys. Many local minima, saddle points, and plateaus. (e.g., Deep Neural Networks).

### Gradient Descent (GD)
The "ball rolling down the hill".
$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t) $$
*   $\eta$: Learning Rate (Step size).
*   $\nabla L$: Gradient (Direction of steepest ascent). We go opposite.

### Stochastic Gradient Descent (SGD)
Computing gradients over the entire dataset (Batch GD) is too slow.
SGD estimates the gradient using a **mini-batch** $B$.
$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t; B) $$
*   **Noise**: The estimate is noisy. This noise actually helps escape shallow local minima!

## 2. Advanced Optimizers: Beyond SGD

### Momentum
SGD oscillates in ravines (steep in one direction, flat in another).
Momentum adds "velocity" $v_t$.
$$ v_{t+1} = \gamma v_t + \eta \nabla L(\theta_t) $$
$$ \theta_{t+1} = \theta_t - v_{t+1} $$
*   Accumulates speed in consistent directions.
*   Dampens oscillations.

### RMSprop & Adam (Adaptive Learning Rates)
Different parameters need different learning rates.
*   **RMSprop**: Divides gradient by a running average of its magnitude.
*   **Adam (Adaptive Moment Estimation)**: Combines Momentum (1st moment) and RMSprop (2nd moment).
    *   The gold standard for most DL tasks.

## 3. Implementation: PyTorch Optimizers

PyTorch separates the model (Weights) from the optimizer (Update Rule).

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define Model (Weights)
model = nn.Linear(10, 1)

# 2. Define Optimizer
# We pass the parameters to optimize
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# OR
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Optimization Step (Inside Training Loop)
# a. Zero Gradients (Clear previous history)
optimizer.zero_grad()

# b. Forward Pass & Loss
output = model(input)
loss = loss_fn(output, target)

# c. Backward Pass (Compute Gradients)
loss.backward()

# d. Update Weights (theta = theta - lr * grad)
optimizer.step()
```

## 4. Learning Rate Schedulers

The Learning Rate (LR) is the most important hyperparameter.
*   **High LR**: Diverges or bounces around.
*   **Low LR**: Slow convergence or gets stuck.
*   **Scheduler**: Decay LR over time.

```python
# Decay LR by factor of 0.1 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# In loop:
# train(...)
# scheduler.step()
```

## 5. Convexity vs Non-Convexity in DL

Deep Networks are highly non-convex. Why does SGD work?
*   **Saddle Points**: In high dimensions, most critical points are saddle points (min in some dims, max in others), not local minima. SGD escapes saddle points easily.
*   **Mode Connectivity**: Many local minima are connected by low-loss paths.
*   **Flat Minima**: We prefer "flat" minima (wide valleys) over "sharp" minima. Flat minima generalize better (robust to shift between train and test data).
