# Day 23: Parameter-Efficient Fine-tuning (PEFT)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. LoRA Mathematics

**Hypothesis:**
Pre-trained models have a very low "intrinsic dimension". Even though they have billions of parameters, the task-specific adaptation happens in a low-dimensional subspace.

**Update Rule:**
$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x $$
- $W_0$: Frozen pre-trained weights ($d \times k$).
- $B$: Low-rank matrix ($d \times r$). Init = 0.
- $A$: Low-rank matrix ($r \times k$). Init = Gaussian.
- $\alpha$: Scaling factor.
- $r$: Rank.

**Why Init B=0?**
To ensure that at the start of training, $\Delta W = 0$. The model behaves exactly like the pre-trained model. If we initialized both A and B randomly, the initial output would be noisy and destabilize training.

**Scaling Factor ($\alpha$):**
Used to tune the strength of the adapter.
Usually set $\alpha = r$ or $\alpha = 2r$.
During inference, we can merge weights: $W_{merged} = W_0 + \frac{\alpha}{r} BA$.

### 2. QLoRA: The 4-bit Revolution

**Normal Float 4 (NF4):**
Standard 4-bit integers are evenly spaced.
Neural network weights are normally distributed (bell curve).
NF4 is a data type where the quantization levels are spaced according to the quantiles of a normal distribution. This minimizes quantization error for weights.

**Double Quantization:**
Quantization requires storing "quantization constants" (scales).
In 65B models, even these constants take up significant memory (0.5 bits per param).
QLoRA quantizes the constants themselves (8-bit quantization of the 32-bit constants).

**Paged Optimizers:**
Uses NVIDIA Unified Memory to automatically evict optimizer states to CPU RAM if GPU VRAM spikes. Prevents OOM spikes.

### 3. DoRA (Weight-Decomposed Low-Rank Adaptation)

**Concept:**
LoRA updates the *direction* and *magnitude* of weights simultaneously.
DoRA decomposes the weight matrix into Magnitude ($m$) and Direction ($V$).
$$ W = m \frac{V}{||V||} $$
It applies LoRA only to the Direction component $V$, while training Magnitude $m$ separately.
**Result:** consistently outperforms LoRA, closer to FFT.

### Code: LoRA Linear Layer from Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        # Frozen base layer
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        
        # LoRA matrices
        self.lora_a = nn.Parameter(torch.randn(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Init A: Kaiming Uniform
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # Init B: Zero
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x):
        # Base output
        base_out = self.linear(x)
        
        # LoRA output: x @ A.T @ B.T * scaling
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        
        return base_out + lora_out
    
    def merge(self):
        # Merge weights for inference
        delta_w = (self.lora_b @ self.lora_a) * self.scaling
        self.linear.weight.data += delta_w
```
