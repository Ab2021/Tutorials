# Days 29-35: Week 5 - Deep Learning Compiler Stack Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 29: Compiler Comparison

```python
import torch

# Eager execution
model = torch.nn.Linear(1000, 1000).cuda()
x = torch.randn(32, 1000, device='cuda')

# JIT compiled
model_jit = torch.jit.trace(model, x)

# torch.compile (PyTorch 2.0+)
model_compiled = torch.compile(model)
```

---

## Day 30: XLA with JAX

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

# First call compiles, subsequent calls are fast
a = jnp.ones((1000, 1000))
b = jnp.ones((1000, 1000))
c = matmul(a, b)
```

---

## Day 31: TVM Compilation

```python
# TVM auto-scheduling
# import tvm
# from tvm import relay
# 
# mod, params = relay.frontend.from_onnx(onnx_model)
# target = tvm.target.cuda()
# lib = relay.build(mod, target, params=params)
```

---

## Day 32: OpenAI Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols)
    
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator
    
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, softmax, mask=col_offsets < n_cols)
```

---

## Day 33: Mixed Precision Training

```python
import torch

# Automatic Mixed Precision
scaler = torch.cuda.amp.GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Day 34: FlashAttention

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

# Uses FlashAttention under the hood (PyTorch 2.0+)
q = torch.randn(32, 8, 512, 64, device='cuda')
k = torch.randn(32, 8, 512, 64, device='cuda')
v = torch.randn(32, 8, 512, 64, device='cuda')

output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## Day 35: Week 5 Project

```python
# Benchmark same model across compilers
# 1. PyTorch eager
# 2. torch.jit.trace
# 3. torch.compile
# 4. TensorRT
# Compare latency and accuracy
```

---

## 📝 Week 5 Summary
| Day | Topic | Key Concept |
|-----|-------|-------------|
| 29 | Compilers Overview | JIT vs AOT |
| 30 | XLA | JAX, TensorFlow XLA |
| 31 | TVM | Auto-scheduling |
| 32 | Triton | Block-level kernels |
| 33 | Mixed Precision | AMP, Loss Scaling |
| 34 | FlashAttention | IO-aware kernels |
| 35 | Project | Compiler comparison |
