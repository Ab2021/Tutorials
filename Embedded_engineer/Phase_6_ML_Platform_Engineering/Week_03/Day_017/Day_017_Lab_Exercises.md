# Days 17-21: Week 3 Continued - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Quick Reference Labs

### Day 17: cuDNN Convolution Algorithms
```python
import torch
torch.backends.cudnn.benchmark = True  # Auto-tune
```

### Day 18: Fused Operations
```python
# torch.jit.script enables fusion
@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.797 * (x + 0.044715 * x ** 3)))
```

### Day 19: cuFFT Spectral Analysis
```python
spectrum = torch.fft.rfft(signal)  # Real FFT
power = spectrum.abs() ** 2        # Power spectrum
```

### Day 20: Sparse Neural Networks
```python
# Pruning creates sparsity
mask = weight.abs() > threshold
weight.data *= mask
```

### Day 21: Custom Layer Project
```python
# Integrate with PyTorch
class CustomLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return custom_forward(x)
    
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return custom_backward(x, grad)
```

---

## 📝 Week 3 Summary
| Day | Topic | Key API |
|-----|-------|---------|
| 15 | cuBLAS | `torch.mm`, `torch.bmm` |
| 16 | Batched GEMM | `torch.bmm` |
| 17 | cuDNN Conv | `F.conv2d` |
| 18 | Fused Ops | `torch.jit.script` |
| 19 | cuFFT | `torch.fft` |
| 20 | Sparse | `torch.sparse` |
| 21 | Project | Custom activation |
