# Lab 4: Normalization Layers

## Objective
Implement **LayerNorm** and **RMSNorm**.
RMSNorm is used in Llama and is slightly faster/simpler than LayerNorm.

## 1. LayerNorm (`norm.py`)

$y = \frac{x - \mu}{\sigma} * \gamma + \beta$

```python
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Test against PyTorch
x = torch.randn(2, 5)
my_ln = MyLayerNorm(5)
pt_ln = nn.LayerNorm(5)

# Copy weights
pt_ln.weight.data = my_ln.gamma.data
pt_ln.bias.data = my_ln.beta.data

print("My LN:", my_ln(x))
print("PyTorch LN:", pt_ln(x))
```

## 2. RMSNorm
$y = \frac{x}{RMS(x)} * \gamma$
(No mean subtraction, no bias).

## 3. Challenge
Implement `MyRMSNorm`.
Verify that it is computationally cheaper (fewer operations).

## 4. Submission
Submit the `MyRMSNorm` code.
