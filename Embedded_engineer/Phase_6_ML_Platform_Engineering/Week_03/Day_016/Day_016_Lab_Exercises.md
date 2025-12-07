# Day 16-21: Week 3 Lab Exercises - cuBLAS, cuDNN & Math Libraries
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 16: cuBLAS Advanced - Batched Operations

```python
import torch

def batched_gemm():
    """Batched matrix multiplication."""
    batch = 64
    M, N, K = 256, 256, 256
    
    A = torch.randn(batch, M, K, device='cuda')
    B = torch.randn(batch, K, N, device='cuda')
    
    # Batched GEMM
    C = torch.bmm(A, B)
    print(f"Batched GEMM: {A.shape} @ {B.shape} = {C.shape}")
```

---

## Day 17: cuDNN Convolution

```python
import torch
import torch.nn.functional as F

def cudnn_conv():
    """cuDNN convolution benchmark."""
    x = torch.randn(32, 64, 224, 224, device='cuda')
    w = torch.randn(128, 64, 3, 3, device='cuda')
    
    # Enable cuDNN autotuning
    torch.backends.cudnn.benchmark = True
    
    y = F.conv2d(x, w, padding=1)
    print(f"Conv2d: {x.shape} -> {y.shape}")
```

---

## Day 18: Fused Operations

```python
import torch
import torch.nn as nn

def fused_operations():
    """Fused Conv-BN-ReLU."""
    model = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
    ).cuda()
    
    x = torch.randn(32, 64, 56, 56, device='cuda')
    y = model(x)
    print(f"Fused layers: {x.shape} -> {y.shape}")
```

---

## Day 19: cuFFT

```python
import torch

def fft_demo():
    """FFT operations."""
    signal = torch.randn(1024, device='cuda')
    spectrum = torch.fft.fft(signal)
    recovered = torch.fft.ifft(spectrum)
    
    print(f"Signal: {signal.shape}")
    print(f"Spectrum: {spectrum.shape}")
    print(f"Reconstruction error: {(signal - recovered.real).abs().max():.6f}")
```

---

## Day 20: Sparse Operations

```python
import torch

def sparse_ops():
    """Sparse matrix operations."""
    # Create sparse tensor
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device='cuda')
    values = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    sparse = torch.sparse_coo_tensor(indices, values, (1000, 1000))
    
    dense = torch.randn(1000, 100, device='cuda')
    result = torch.sparse.mm(sparse, dense)
    print(f"Sparse @ Dense: {result.shape}")
```

---

## Day 21: Week 3 Project - Custom Activation

```python
import torch
import torch.nn as nn

class GELU(nn.Module):
    """Custom GELU implementation."""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / 3.14159)) * 
            (x + 0.044715 * x.pow(3))
        ))

# Benchmark custom vs built-in
x = torch.randn(1000, 1000, device='cuda')
custom = GELU().cuda()
builtin = nn.GELU().cuda()

# Verify correctness
print(f"Max diff: {(custom(x) - builtin(x)).abs().max():.6f}")
```

---

## 📚 Resources
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)
