# Day 13: Multi-GPU Programming - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Device Management (Beginner)

### Objective
Manage multiple GPU devices.

### Code

```python
import torch

def multi_gpu_basics():
    """Explore multi-GPU setup."""
    
    print("Multi-GPU Device Management")
    print("=" * 50)
    
    # Check available devices
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    if num_gpus < 2:
        print("\n⚠ Only 1 GPU available. Multi-GPU exercises will simulate.")
        return
    
    # Move tensors to specific devices
    print("\nDevice Placement:")
    a = torch.randn(1000, device='cuda:0')
    print(f"  Tensor a on: {a.device}")
    
    if num_gpus >= 2:
        b = torch.randn(1000, device='cuda:1')
        print(f"  Tensor b on: {b.device}")

if __name__ == "__main__":
    multi_gpu_basics()
```

---

## 🔬 Exercise 2: Data Parallel Processing (Intermediate)

### Objective
Process data across multiple GPUs.

### Code

```python
import torch
import torch.nn as nn
import time

def data_parallel_demo():
    """Demonstrate DataParallel."""
    
    print("Data Parallel Demo")
    print("=" * 50)
    
    num_gpus = torch.cuda.device_count()
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 4000),
        nn.ReLU(),
        nn.Linear(4000, 1000),
    )
    
    # Single GPU
    model_single = model.cuda()
    x = torch.randn(256, 1000, device='cuda')
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = model_single(x)
    torch.cuda.synchronize()
    single_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"Single GPU: {single_time:.2f} ms")
    
    if num_gpus >= 2:
        # Multi-GPU with DataParallel
        model_multi = nn.DataParallel(model.cuda())
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            y = model_multi(x)
        torch.cuda.synchronize()
        multi_time = (time.perf_counter() - start) / 100 * 1000
        
        print(f"Multi-GPU ({num_gpus} GPUs): {multi_time:.2f} ms")
        print(f"Speedup: {single_time/multi_time:.2f}x")
    else:
        print("(Multi-GPU test skipped - only 1 GPU)")

if __name__ == "__main__":
    data_parallel_demo()
```

---

## 🔬 Exercise 3: Peer-to-Peer Communication (Advanced)

### Objective
Direct GPU-to-GPU memory transfer.

### Code

```python
import torch
import time

def peer_to_peer_demo():
    """Demonstrate P2P memory access."""
    
    print("Peer-to-Peer Communication")
    print("=" * 50)
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        print("Need 2+ GPUs for P2P demo")
        return
    
    # Check P2P access capability
    print("\nP2P Access Matrix:")
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU{i} -> GPU{j}: {'✓' if can_access else '✗'}")
    
    # Test transfer speeds
    size = 100 * 1024 * 1024 // 4  # 100 MB
    src = torch.randn(size, device='cuda:0')
    dst = torch.empty(size, device='cuda:1')
    
    # Via host (slow)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        dst.copy_(src.cpu().cuda(1))
    torch.cuda.synchronize()
    via_host = (time.perf_counter() - start) / 10 * 1000
    
    # Direct P2P (fast)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        dst.copy_(src)  # Direct copy
    torch.cuda.synchronize()
    direct = (time.perf_counter() - start) / 10 * 1000
    
    print(f"\nTransfer 100MB GPU0 -> GPU1:")
    print(f"  Via Host: {via_host:.2f} ms")
    print(f"  Direct P2P: {direct:.2f} ms")
    print(f"  Speedup: {via_host/direct:.2f}x")

if __name__ == "__main__":
    peer_to_peer_demo()
```

---

## 🐛 Common Issues

### Issue: Device mismatch error
**Fix:** Ensure all tensors in operation are on same device

### Issue: P2P not working
**Cause:** GPUs not in same PCIe tree
**Fix:** Check `can_device_access_peer()`

---

## 📚 Additional Resources
- [PyTorch Multi-GPU](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [CUDA P2P](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
