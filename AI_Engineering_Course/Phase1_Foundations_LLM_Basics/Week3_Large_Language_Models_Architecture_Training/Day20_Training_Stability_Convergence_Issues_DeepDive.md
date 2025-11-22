# Day 20: Training Stability & Convergence Issues
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Pre-LN vs. Post-LN: The Mathematics of Stability

**Post-LN (Original Transformer):**
$$ x_{l+1} = \text{LayerNorm}(x_l + F(x_l)) $$
**Gradient Issue:**
The gradient through LayerNorm depends on the scale of the input.
As we go deeper, the magnitude of $x_l$ tends to grow (due to residual addition).
This means gradients at the bottom layers (close to input) can vanish or explode depending on initialization.
**Result:** Requires very careful warmup and small LR.

**Pre-LN (GPT-2, LLaMA):**
$$ x_{l+1} = x_l + F(\text{LayerNorm}(x_l)) $$
**Gradient Benefit:**
The gradient flows directly through the residual path ($x_l$) without passing through LayerNorm.
$$ \frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} + \dots $$
This creates a "Gradient Superhighway".
The magnitude of gradients remains stable regardless of depth.
**Result:** Can train much deeper models with higher learning rates and less warmup.

**Trade-off:** Pre-LN models sometimes have slightly worse representation capacity than Post-LN (theoretical), but the training stability benefits outweigh this massively.

### 2. QK Norm and Z-Loss (PaLM / ViT-22B)

**QK Norm:**
In large models, the attention logits ($Q \cdot K^T$) can grow very large.
Softmax($\text{large numbers}$) -> One-hot distribution -> Gradients vanish (saturation).
**Solution:** Apply LayerNorm to Query and Key vectors *before* dot product.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{LN}(Q)\text{LN}(K)^T}{\sqrt{d}}\right)V $$

**Z-Loss (Router Z-Loss in MoE):**
Encourages logit values to remain small.
$$ L_{aux} = \lambda \sum \log^2(e^{logit}) $$
Helps prevent numerical instability in exponential functions.

### 3. Spike Detection & Recovery

**Mechanism:**
1.  **Monitor Gradient Norm:** Calculate $G = ||\nabla W||_2$.
2.  **Moving Average:** Keep a running average $\mu_G$.
3.  **Threshold:** If $G > k \cdot \mu_G$ (e.g., $k=10$), it's a spike.
4.  **Action:**
    - **Skip:** Do not call `optimizer.step()`.
    - **Reset:** If spikes persist, reload last checkpoint and skip the data batch.

### 4. Weight Initialization Scaling

For deep Transformers (Pre-LN), the output variance grows with depth $L$.
To keep variance constant, we scale the initialization of the residual branch weights by $1/\sqrt{L}$ (or $1/\sqrt{2L}$).
**GPT-2 Init:**
Scale weights of the projection layers (output of Attention and FFN) by $1/\sqrt{N_{layers}}$.

### Code: Spike Detection Training Loop

```python
import torch
import numpy as np

class SpikeDetector:
    def __init__(self, window_size=100, threshold_factor=5.0):
        self.history = []
        self.window_size = window_size
        self.factor = threshold_factor
        
    def check(self, grad_norm):
        if len(self.history) < 10:
            self.history.append(grad_norm)
            return False # Not enough data
            
        avg = np.mean(self.history)
        std = np.std(self.history)
        
        # Update history
        self.history.append(grad_norm)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Check spike
        if grad_norm > avg + self.factor * std:
            return True
        return False

# Training Loop Integration
detector = SpikeDetector()

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Calculate Norm
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    if detector.check(total_norm.item()):
        print(f"Spike detected (Norm: {total_norm:.2f}). Skipping step.")
        continue # Skip update
        
    optimizer.step()
```
