# Lab 3: PEFT from Scratch (LoRA)

## Objective
Implement LoRA manually to understand the matrix algebra.
$W' = W + BA$

## 1. The Layer (`lora_layer.py`)

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen Pretrained Weight
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        
        # Trainable Adapters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Wx + (BA)x * scaling
        base_out = F.linear(x, self.weight)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling
```

## 2. Analysis
Only `lora_A` and `lora_B` have gradients. `weight` is frozen.
This reduces trainable params by >99%.

## 3. Submission
Calculate the number of parameters for a 4096->4096 layer with Rank 8 vs Full Fine-tuning.
