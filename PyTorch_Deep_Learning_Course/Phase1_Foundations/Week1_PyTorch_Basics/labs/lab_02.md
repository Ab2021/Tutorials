# Lab 02: Autograd Mechanics

## Difficulty
🟡 Medium

## Problem Statement
Visualize the computational graph.
1. Create tensors `a` and `b` with `requires_grad=True`.
2. Compute `c = a * b + 3`.
3. Compute `d = c.mean()`.
4. Call `d.backward()`.
5. Print gradients of `a` and `b`.

## Starter Code
```python
import torch

def autograd_demo():
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([6.0, 4.0], requires_grad=True)
    # TODO: Compute c, d, backward
    pass
```
