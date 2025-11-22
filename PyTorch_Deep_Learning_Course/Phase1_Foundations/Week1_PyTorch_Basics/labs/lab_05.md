# Lab 05: Linear Regression from Scratch

## Difficulty
🟡 Medium

## Problem Statement
Implement Linear Regression `y = wx + b` using PyTorch tensors and Autograd (no `nn.Linear` or `optim.SGD`).
1. Initialize weights `w` and bias `b` randomly.
2. Implement forward pass.
3. Compute loss (MSE).
4. Compute gradients.
5. Update weights manually using Gradient Descent.

## Starter Code
```python
import torch

def train_linear_regression(X, y, epochs=100, lr=0.01):
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    for epoch in range(epochs):
        # TODO: Forward, Loss, Backward, Update
        pass
    return w, b
```
