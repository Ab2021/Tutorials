# Day 19: Memory Optimization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. ZeRO: Communication vs. Memory Trade-off

**Standard Data Parallelism:**
- **Forward:** No communication.
- **Backward:** All-Reduce gradients (Size $N$).
- **Total Comm:** $2N$ per step.

**ZeRO Stage 3 (Sharded Parameters):**
- **Forward:**
    - Each layer needs full weights.
    - **All-Gather** weights for Layer $i$. Compute. Discard weights.
    - Comm: $N$.
- **Backward:**
    - **All-Gather** weights for Layer $i$. Compute grads. Discard weights.
    - **Reduce-Scatter** gradients.
    - Comm: $N + N = 2N$.
- **Total Comm:** $3N$ per step.

**Analysis:**
ZeRO-3 increases communication volume by 1.5x compared to standard DP.
However, it enables training models that are $N_{gpus}$ times larger.
**Optimization:** Prefetching. While computing Layer $i$, start All-Gather for Layer $i+1$.

### 2. Gradient Checkpointing Implementation

**Forward Pass (No Checkpoint):**
$$ x_1 = f_1(x_0) \to \text{Save } x_1 $$
$$ x_2 = f_2(x_1) \to \text{Save } x_2 $$
$$ \dots $$
Memory: $O(L)$.

**Forward Pass (Checkpoint):**
$$ x_1 = f_1(x_0) \to \text{Drop } x_1 $$
$$ x_2 = f_2(x_1) \to \text{Drop } x_2 $$
$$ \dots $$
Only save inputs to checkpoints (e.g., every $\sqrt{L}$ layers).
Memory: $O(\sqrt{L})$.

**Backward Pass:**
Need $x_1$ to compute grad of $f_2$.
Re-compute: $x_1 = f_1(x_0)$.
Compute grad. Drop $x_1$.

### 3. CPU Offloading (ZeRO-Offload)

**Mechanism:**
- Move Optimizer States (FP32) to CPU RAM.
- Move Gradients (FP16) to CPU RAM during backward pass.
- **Update Step:** Compute $W_{t+1} = W_t - \eta \nabla W$ on **CPU**.
- Copy updated weights back to GPU.

**Bottleneck:** PCIe Bandwidth (CPU <-> GPU).
- PCIe Gen4: 64 GB/s.
- NVLink: 600 GB/s.
- Offloading is slow, but allows training 10x larger models on commodity hardware.

### 4. Flash Attention Memory Benefits

Flash Attention is often cited for speed, but its **memory** impact is equally important.
- Standard Attention: Stores $N \times N$ attention matrix for backward pass. $O(N^2)$.
- Flash Attention: Recomputes attention scores during backward pass. $O(N)$.
- This is essentially "Gradient Checkpointing applied to the Attention Matrix".

### Code: Manual Gradient Checkpointing

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class DeepNet(nn.Module):
    def __init__(self, num_layers, dim, use_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([ResBlock(dim) for _ in range(num_layers)])
        self.use_checkpointing = use_checkpointing
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing:
                # Checkpoint requires input to have requires_grad=True
                # Function to run must not have side effects
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# Usage
model = DeepNet(num_layers=100, dim=1024, use_checkpointing=True)
# Memory usage will be ~constant regardless of depth!
```
