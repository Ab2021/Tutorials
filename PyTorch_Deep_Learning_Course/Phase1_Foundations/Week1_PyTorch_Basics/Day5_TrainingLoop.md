# Day 5: The Training Loop - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: The Optimization Cycle, Validation, and Metrics

## 1. Theoretical Foundation: The Learning Algorithm

Deep Learning training is an iterative process of **Empirical Risk Minimization**.
We want to minimize the loss over the training distribution $P_{data}$.

### The Cycle (Epoch vs Batch)
*   **Epoch**: One complete pass through the entire dataset.
*   **Batch**: A subset of data used for one gradient update.
*   **Iteration**: One update step.

### Train vs Validation vs Test
*   **Train**: Used to compute gradients and update weights.
*   **Validation (Dev)**: Used to tune hyperparameters (LR, Architecture) and Early Stopping. **Never** train on this.
*   **Test**: Used for final evaluation. Unseen data.

### Overfitting vs Underfitting
*   **Underfitting**: High Bias. Model is too simple. High Train Loss, High Val Loss.
*   **Overfitting**: High Variance. Model memorizes noise. Low Train Loss, High Val Loss.
*   **Goal**: Low Train Loss, Low Val Loss (Generalization).

## 2. The Standard PyTorch Loop

Unlike Keras (`model.fit`), PyTorch requires you to write the loop explicitly. This gives you full control.

```python
import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set to training mode (Dropout/BN)
    running_loss = 0.0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 2. Backward
        optimizer.zero_grad() # Clear old grads
        loss.backward()       # Compute new grads
        
        # 3. Update
        optimizer.step()      # Update weights
        
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval() # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    
    with torch.no_grad(): # Disable gradient calculation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            
    acc = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), acc
```

## 3. Best Practices

### 1. `model.train()` vs `model.eval()`
Always toggle this.
*   In `train()`: Dropout is active. BatchNorm updates running stats.
*   In `eval()`: Dropout is off. BatchNorm uses frozen stats.
*   Forgetting this is a common bug (Validation accuracy fluctuates wildly).

### 2. `optimizer.zero_grad()` placement
Usually placed before `.backward()`.
Alternatively, `for param in model.parameters(): param.grad = None` (faster).

### 3. Numerical Stability
*   Use `nn.CrossEntropyLoss` (which takes Logits) instead of `Softmax` + `NLLLoss`.
*   LogSoftmax is numerically more stable than Softmax.

### 4. Gradient Accumulation
If batch size 32 fits in GPU, but you want effective batch size 128:
```python
accumulation_steps = 4
for i, (inputs, targets) in enumerate(loader):
    loss = criterion(model(inputs), targets)
    loss = loss / accumulation_steps # Normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 4. Metrics
Loss is for the optimizer. Metrics are for humans.
*   **Accuracy**: Classification.
*   **F1-Score**: Imbalanced classification.
*   **IoU**: Segmentation.
*   **MSE/MAE**: Regression.

Use `torchmetrics` library for robust implementations.
