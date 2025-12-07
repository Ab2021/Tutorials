# Day 15: cuBLAS Fundamentals - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: cuBLAS Basics (Beginner)

```python
import torch
import time

def cublas_basics():
    """Demonstrate cuBLAS operations via PyTorch."""
    print("cuBLAS Fundamentals")
    print("=" * 40)
    
    # BLAS Level 1: Vector operations
    x = torch.randn(10000, device='cuda')
    y = torch.randn(10000, device='cuda')
    
    dot_product = torch.dot(x, y)
    print(f"L1 - Dot product: {dot_product.item():.4f}")
    
    # BLAS Level 2: Matrix-vector
    A = torch.randn(1000, 1000, device='cuda')
    v = torch.randn(1000, device='cuda')
    
    result = torch.mv(A, v)
    print(f"L2 - Matrix-vector shape: {result.shape}")
    
    # BLAS Level 3: Matrix-matrix (GEMM)
    B = torch.randn(1000, 1000, device='cuda')
    C = torch.mm(A, B)
    print(f"L3 - Matrix-matrix shape: {C.shape}")

if __name__ == "__main__":
    cublas_basics()
```

---

## 🔬 Exercise 2: GEMM Performance (Intermediate)

```python
import torch
import time

def benchmark_gemm():
    """Benchmark GEMM at different sizes."""
    sizes = [512, 1024, 2048, 4096]
    
    print("GEMM Performance Benchmark")
    print("=" * 50)
    print(f"{'Size':>6} | {'Time (ms)':>10} | {'TFLOPS':>8}")
    print("-" * 50)
    
    for N in sizes:
        A = torch.randn(N, N, device='cuda')
        B = torch.randn(N, N, device='cuda')
        
        # Warm up
        C = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            C = torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / 10
        flops = 2 * N * N * N
        tflops = flops / (time_ms / 1000) / 1e12
        
        print(f"{N:>6} | {time_ms:>10.3f} | {tflops:>8.2f}")

if __name__ == "__main__":
    benchmark_gemm()
```

---

## 🔬 Exercise 3: Mixed Precision GEMM (Advanced)

```python
import torch

def mixed_precision_gemm():
    """Compare FP32 vs FP16 performance."""
    N = 4096
    
    print("Mixed Precision GEMM")
    print("=" * 40)
    
    # FP32
    A32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        C32 = torch.mm(A32, B32)
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / 10 * 1000
    
    # FP16
    A16 = A32.half()
    B16 = B32.half()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        C16 = torch.mm(A16, B16)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"FP32: {fp32_time:.2f} ms")
    print(f"FP16: {fp16_time:.2f} ms")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")

if __name__ == "__main__":
    mixed_precision_gemm()
```

---

## 📚 Additional Resources
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
