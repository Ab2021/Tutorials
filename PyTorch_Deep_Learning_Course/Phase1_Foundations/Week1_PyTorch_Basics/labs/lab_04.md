# Lab 04: Custom Loss Function

## Difficulty
🟡 Medium

## Problem Statement
Implement Mean Squared Error (MSE) loss manually as a custom `nn.Module`.
It should take `predictions` and `targets` and return the average squared difference.

## Starter Code
```python
import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # TODO: Implement MSE formula: mean((y_pred - y_true)^2)
        pass
```
