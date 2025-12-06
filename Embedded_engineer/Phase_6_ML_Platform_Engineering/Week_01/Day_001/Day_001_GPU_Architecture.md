# Day 1: GPU Architecture Deep Dive
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Understanding the fundamental architecture of NVIDIA GPUs that powers modern AI/ML workloads.
> This foundation is critical for writing efficient CUDA code and optimizing ML systems.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** the architectural differences between CPUs and GPUs
2. **Identify** key GPU hardware components (SMs, Warps, Threads)
3. **Explain** the GPU memory hierarchy and its performance implications
4. **Analyze** GPU specifications and compute capabilities
5. **Use** nvidia-smi and deviceQuery for GPU inspection

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 5.0+ minimum, 7.0+ recommended)
- Minimum 4GB GPU memory (8GB+ recommended for later labs)

### Software Environment
```bash
# Verify NVIDIA driver installation
nvidia-smi

# Install CUDA Toolkit (Ubuntu)
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version

# Python environment setup
conda create -n gpu_platform python=3.10 -y
conda activate gpu_platform
pip install numpy torch pycuda cupy-cuda12x
```

### Prior Knowledge
- Basic understanding of computer architecture
- Familiarity with parallel computing concepts
- Python programming skills

---

## 📖 Theoretical Foundation

### 1. CPU vs GPU Architecture Comparison

#### The Fundamental Design Philosophy

**CPU (Central Processing Unit):**
- Optimized for **latency** - get single tasks done as fast as possible
- Complex control logic and large caches
- Branch prediction and speculative execution
- Few powerful cores (4-64 typically)
- Serial execution model

**GPU (Graphics Processing Unit):**
- Optimized for **throughput** - process many tasks simultaneously
- Simple control logic, massive parallelism
- Many weaker cores (thousands)
- Parallel execution model
- Data-parallel workloads

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPU Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │  Core 0 │ │  Core 1 │ │  Core 2 │ │  Core 3 │  ...         │
│  │  ┌───┐  │ │  ┌───┐  │ │  ┌───┐  │ │  ┌───┐  │              │
│  │  │ALU│  │ │  │ALU│  │ │  │ALU│  │ │  │ALU│  │              │
│  │  └───┘  │ │  └───┘  │ │  └───┘  │ │  └───┘  │              │
│  │ L1/L2   │ │ L1/L2   │ │ L1/L2   │ │ L1/L2   │              │
│  │ Cache   │ │ Cache   │ │ Cache   │ │ Cache   │              │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                        L3 Cache                                 │
│                     Memory Controller                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        GPU Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Streaming Multiprocessors                 │  │
│  │ ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐ ... ┌────┐┌────┐   │  │
│  │ │SM 0││SM 1││SM 2││SM 3││SM 4││SM 5│     │SM N││SM N+1│  │  │
│  │ └────┘└────┘└────┘└────┘└────┘└────┘     └────┘└────┘   │  │
│  │                                                          │  │
│  │ Each SM contains:                                        │  │
│  │ ├── 32-128 CUDA Cores                                    │  │
│  │ ├── Shared Memory                                        │  │
│  │ ├── L1 Cache                                             │  │
│  │ └── Warp Schedulers                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           L2 Cache                              │
│                        Memory Controllers                       │
│                         HBM2/GDDR6 Memory                       │
└─────────────────────────────────────────────────────────────────┘
```

#### Performance Metrics Comparison

| Metric | CPU (Intel i9-13900K) | GPU (NVIDIA H100) |
|--------|----------------------|-------------------|
| Cores | 24 (8P+16E) | 16,896 CUDA cores |
| Clock Speed | Up to 5.8 GHz | 1.8 GHz |
| Memory Bandwidth | ~90 GB/s | 3.35 TB/s |
| FP32 Performance | ~1 TFLOPS | 67 TFLOPS |
| TDP | 253W | 700W |
| Best For | Complex control flow | Massive parallelism |

### 2. NVIDIA GPU Hierarchy

#### GPU Generations (Compute Capability)

| Generation | Compute Capability | Key Features | Example GPUs |
|------------|-------------------|--------------|--------------|
| **Pascal** | 6.x | First unified memory | GTX 1080, Tesla P100 |
| **Volta** | 7.0 | Tensor Cores introduced | Tesla V100, Titan V |
| **Turing** | 7.5 | RT Cores, INT8 Tensor | RTX 2080, T4 |
| **Ampere** | 8.x | 3rd gen Tensor, TF32 | RTX 3090, A100 |
| **Ada Lovelace** | 8.9 | 4th gen Tensor Cores | RTX 4090, L40 |
| **Hopper** | 9.0 | Transformer Engine, FP8 | H100, H200 |
| **Blackwell** | 10.0 | 5th gen Tensor, FP4 | B100, B200 |

#### Streaming Multiprocessor (SM) Architecture

The SM is the fundamental building block of NVIDIA GPUs:

```
┌────────────────────────────────────────────────────────────┐
│                 Streaming Multiprocessor (SM)               │
├────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐    │
│  │              Warp Schedulers (4x)                   │    │
│  │  [Scheduler 0] [Scheduler 1] [Scheduler 2] [Scheduler 3]│
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  ┌─────────────────────┐  ┌─────────────────────┐         │
│  │   Processing Block   │  │   Processing Block   │        │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │         │
│  │  │ 16 FP32 Units │  │  │  │ 16 FP32 Units │  │         │
│  │  │ 16 INT32 Units│  │  │  │ 16 INT32 Units│  │         │
│  │  │ 8 FP64 Units  │  │  │  │ 8 FP64 Units  │  │         │
│  │  │ 4 Tensor Cores│  │  │  │ 4 Tensor Cores│  │         │
│  │  │ 1 Load/Store  │  │  │  │ 1 Load/Store  │  │         │
│  │  └───────────────┘  │  │  └───────────────┘  │         │
│  └─────────────────────┘  └─────────────────────┘         │
│                                                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Shared Memory / L1 Cache (192 KB)           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Register File (256 KB)                  │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

#### Thread Hierarchy: Grids, Blocks, and Threads

```
┌─────────────────────────────────────────────────────────────────┐
│                           GRID                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │   Block(0,0)    Block(1,0)    Block(2,0)    Block(3,0)   │  │
│  │   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐   │  │
│  │   │████████│    │████████│    │████████│    │████████│   │  │
│  │   │████████│    │████████│    │████████│    │████████│   │  │
│  │   └────────┘    └────────┘    └────────┘    └────────┘   │  │
│  │                                                           │  │
│  │   Block(0,1)    Block(1,1)    Block(2,1)    Block(3,1)   │  │
│  │   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐   │  │
│  │   │████████│    │████████│    │████████│    │████████│   │  │
│  │   │████████│    │████████│    │████████│    │████████│   │  │
│  │   └────────┘    └────────┘    └────────┘    └────────┘   │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                    Zooming into one Block:
┌────────────────────────────────────────┐
│            Block (1,1)                  │
│  ┌──────────────────────────────────┐  │
│  │  Thread Thread Thread Thread ... │  │
│  │  (0,0)  (1,0)  (2,0)  (3,0)     │  │
│  │                                  │  │
│  │  Thread Thread Thread Thread ... │  │
│  │  (0,1)  (1,1)  (2,1)  (3,1)     │  │
│  │                                  │  │
│  │  ...                             │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Executed as WARPS (32 threads each)   │
└────────────────────────────────────────┘
```

#### Warps: The Fundamental Execution Unit

- A **warp** is a group of 32 threads that execute in lockstep (SIMT)
- All threads in a warp execute the same instruction simultaneously
- Branch divergence occurs when threads in a warp take different paths

```python
# Example: Understanding warp execution
# If threads 0-15 take 'if' branch and threads 16-31 take 'else' branch:
# The warp will execute BOTH branches sequentially, masking inactive threads

# Good: All threads take same path (no divergence)
if condition_same_for_all_threads:
    do_something()

# Bad: Threads take different paths (divergence)
if thread_id % 2 == 0:
    do_something()  # Only even threads active
else:
    do_other()      # Only odd threads active
```

### 3. GPU Memory Hierarchy

Understanding memory is **critical** for GPU performance:

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Memory Hierarchy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    REGISTERS (per thread)                 │   │
│  │                    Speed: ~1 cycle                        │   │
│  │                    Size: 255 per thread max               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               SHARED MEMORY (per block)                   │   │
│  │               Speed: ~5 cycles                            │   │
│  │               Size: 48-164 KB (configurable)              │   │
│  │               Scope: Block-level sharing                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   L1 CACHE (per SM)                       │   │
│  │                   Speed: ~30 cycles                       │   │
│  │                   Size: 128-256 KB                        │   │
│  │                   Hardware managed                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   L2 CACHE (shared)                       │   │
│  │                   Speed: ~200 cycles                      │   │
│  │                   Size: 6-60 MB                           │   │
│  │                   All SMs share                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              GLOBAL MEMORY (HBM2/GDDR6)                   │   │
│  │              Speed: ~400-600 cycles                       │   │
│  │              Size: 8-80 GB                                │   │
│  │              Bandwidth: 900 GB/s - 3.35 TB/s             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Memory Types and Use Cases

| Memory Type | Scope | Lifetime | Speed | Best For |
|-------------|-------|----------|-------|----------|
| **Registers** | Thread | Thread | Fastest | Private variables |
| **Local Memory** | Thread | Thread | Slow (global) | Spilled registers |
| **Shared Memory** | Block | Block | Fast | Inter-thread communication |
| **L1 Cache** | SM | Automatic | Fast | Hardware-managed caching |
| **L2 Cache** | Device | Automatic | Medium | Reduce global memory access |
| **Global Memory** | Device | Application | Slowest | Large data structures |
| **Constant Memory** | Device | Application | Fast (cached) | Read-only constants |
| **Texture Memory** | Device | Application | Fast (cached) | Spatial locality data |

### 4. Compute Capability Deep Dive

Compute Capability determines what features are available:

```python
"""
Compute Capability Feature Matrix
"""

COMPUTE_CAPABILITIES = {
    "7.0": {  # Volta
        "tensor_cores": True,
        "independent_thread_scheduling": True,
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 96 * 1024,  # 96 KB
        "fp16_compute": True,
        "int8_tensor": False,
    },
    "8.0": {  # Ampere (A100)
        "tensor_cores": True,  # 3rd gen
        "tf32": True,
        "bf16": True,
        "int8_tensor": True,
        "sparsity": True,
        "max_shared_memory_per_block": 163 * 1024,  # 163 KB
    },
    "9.0": {  # Hopper (H100)
        "tensor_cores": True,  # 4th gen
        "transformer_engine": True,
        "fp8": True,
        "thread_block_clusters": True,
        "max_shared_memory_per_block": 228 * 1024,  # 228 KB
    },
}
```

---

## 💻 Implementation

### 🛠️ Project Structure
```text
day_001_gpu_architecture/
├── src/
│   ├── gpu_info.py
│   ├── memory_benchmark.py
│   └── device_query.cu
├── notebooks/
│   └── gpu_exploration.ipynb
└── README.md
```

### 👨‍💻 Core Implementation

#### 📁 `src/gpu_info.py`
```python
#!/usr/bin/env python3
"""
Day 1: GPU Architecture Deep Dive
Phase 6: AI/ML Platform Engineering with GPU Programming

Comprehensive GPU information and analysis tool using PyTorch and CuPy.
"""

import torch
import subprocess
import json
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class GPUInfo:
    """Data class to hold GPU information."""
    name: str
    compute_capability: tuple
    total_memory_gb: float
    cuda_cores: int
    sm_count: int
    max_threads_per_block: int
    max_threads_per_sm: int
    warp_size: int
    max_shared_memory_per_block: int
    max_shared_memory_per_sm: int
    l2_cache_size: int
    memory_bus_width: int
    memory_clock_rate_ghz: float
    
    def theoretical_bandwidth_gb_s(self) -> float:
        """Calculate theoretical memory bandwidth."""
        # Bandwidth = (memory_bus_width / 8) * memory_clock_rate * 2 (DDR)
        return (self.memory_bus_width / 8) * self.memory_clock_rate_ghz * 2
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "total_memory_gb": self.total_memory_gb,
            "cuda_cores": self.cuda_cores,
            "sm_count": self.sm_count,
            "max_threads_per_block": self.max_threads_per_block,
            "warp_size": self.warp_size,
            "theoretical_bandwidth_gb_s": self.theoretical_bandwidth_gb_s(),
        }


def get_gpu_info(device_id: int = 0) -> GPUInfo:
    """
    Retrieve comprehensive GPU information using PyTorch CUDA API.
    
    Args:
        device_id: CUDA device index
        
    Returns:
        GPUInfo dataclass with GPU specifications
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and driver installation.")
    
    props = torch.cuda.get_device_properties(device_id)
    
    # Estimate CUDA cores based on architecture
    # This is architecture-dependent
    cuda_cores_per_sm = {
        (7, 0): 64,   # Volta
        (7, 5): 64,   # Turing
        (8, 0): 64,   # Ampere (A100)
        (8, 6): 128,  # Ampere (RTX 30xx)
        (8, 9): 128,  # Ada Lovelace
        (9, 0): 128,  # Hopper
    }
    
    cc = (props.major, props.minor)
    cores_per_sm = cuda_cores_per_sm.get(cc, 64)  # Default to 64
    
    return GPUInfo(
        name=props.name,
        compute_capability=cc,
        total_memory_gb=props.total_memory / (1024**3),
        cuda_cores=props.multi_processor_count * cores_per_sm,
        sm_count=props.multi_processor_count,
        max_threads_per_block=props.max_threads_per_block,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        warp_size=props.warp_size,
        max_shared_memory_per_block=props.max_shared_memory_per_block,
        max_shared_memory_per_sm=props.max_shared_memory_per_multiprocessor,
        l2_cache_size=props.l2_cache_size,
        memory_bus_width=props.memory_bus_width if hasattr(props, 'memory_bus_width') else 0,
        memory_clock_rate_ghz=props.memory_clock_rate / 1e6 if hasattr(props, 'memory_clock_rate') else 0,
    )


def print_gpu_summary(info: GPUInfo) -> None:
    """Print formatted GPU summary."""
    print("=" * 60)
    print(f"GPU: {info.name}")
    print("=" * 60)
    print(f"  Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")
    print(f"  Total Memory: {info.total_memory_gb:.2f} GB")
    print(f"  CUDA Cores: {info.cuda_cores:,}")
    print(f"  Streaming Multiprocessors (SMs): {info.sm_count}")
    print("-" * 60)
    print("Thread Configuration:")
    print(f"  Warp Size: {info.warp_size}")
    print(f"  Max Threads per Block: {info.max_threads_per_block:,}")
    print(f"  Max Threads per SM: {info.max_threads_per_sm:,}")
    print("-" * 60)
    print("Memory Configuration:")
    print(f"  Max Shared Memory per Block: {info.max_shared_memory_per_block / 1024:.1f} KB")
    print(f"  Max Shared Memory per SM: {info.max_shared_memory_per_sm / 1024:.1f} KB")
    print(f"  L2 Cache Size: {info.l2_cache_size / (1024**2):.1f} MB")
    print("=" * 60)


def get_nvidia_smi_info() -> Dict:
    """
    Get detailed GPU information from nvidia-smi.
    
    Returns:
        Dictionary with nvidia-smi output
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        gpus = []
        
        for i, line in enumerate(lines):
            values = [v.strip() for v in line.split(',')]
            gpus.append({
                'id': i,
                'name': values[0],
                'memory_total_mb': float(values[1]),
                'memory_used_mb': float(values[2]),
                'memory_free_mb': float(values[3]),
                'gpu_utilization_percent': float(values[4]) if values[4] != '[N/A]' else 0,
                'memory_utilization_percent': float(values[5]) if values[5] != '[N/A]' else 0,
                'temperature_c': float(values[6]) if values[6] != '[N/A]' else 0,
                'power_draw_w': float(values[7]) if values[7] != '[N/A]' else 0,
                'graphics_clock_mhz': float(values[8]) if values[8] != '[N/A]' else 0,
                'memory_clock_mhz': float(values[9]) if values[9] != '[N/A]' else 0,
            })
        
        return {'gpus': gpus}
    
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return {'gpus': [], 'error': str(e)}
    except FileNotFoundError:
        print("nvidia-smi not found. Is the NVIDIA driver installed?")
        return {'gpus': [], 'error': 'nvidia-smi not found'}


def check_cuda_health() -> Dict[str, bool]:
    """
    Perform health checks on CUDA installation.
    
    Returns:
        Dictionary with health check results
    """
    checks = {}
    
    # Check CUDA availability
    checks['cuda_available'] = torch.cuda.is_available()
    
    if checks['cuda_available']:
        # Check device count
        checks['device_count'] = torch.cuda.device_count()
        
        # Check if we can allocate memory
        try:
            x = torch.zeros(1000, device='cuda')
            del x
            torch.cuda.empty_cache()
            checks['memory_allocation'] = True
        except Exception as e:
            checks['memory_allocation'] = False
            checks['memory_error'] = str(e)
        
        # Check cuDNN
        checks['cudnn_available'] = torch.backends.cudnn.is_available()
        if checks['cudnn_available']:
            checks['cudnn_version'] = torch.backends.cudnn.version()
        
        # Check Tensor Cores (by trying a tensor operation)
        try:
            if torch.cuda.get_device_capability()[0] >= 7:
                checks['tensor_cores_likely'] = True
            else:
                checks['tensor_cores_likely'] = False
        except:
            checks['tensor_cores_likely'] = False
    
    return checks


def benchmark_memory_bandwidth(size_mb: int = 100, iterations: int = 10) -> Dict[str, float]:
    """
    Quick memory bandwidth benchmark.
    
    Args:
        size_mb: Size of data to transfer in MB
        iterations: Number of iterations for averaging
        
    Returns:
        Dictionary with bandwidth measurements
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    size = size_mb * 1024 * 1024 // 4  # Number of float32 elements
    
    # Host to Device bandwidth
    h2d_times = []
    cpu_tensor = torch.randn(size, dtype=torch.float32)
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        gpu_tensor = cpu_tensor.cuda()
        end.record()
        
        torch.cuda.synchronize()
        h2d_times.append(start.elapsed_time(end))
        del gpu_tensor
        torch.cuda.empty_cache()
    
    # Device to Host bandwidth
    d2h_times = []
    gpu_tensor = torch.randn(size, dtype=torch.float32, device='cuda')
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        cpu_tensor = gpu_tensor.cpu()
        end.record()
        
        torch.cuda.synchronize()
        d2h_times.append(start.elapsed_time(end))
    
    # Device to Device bandwidth
    d2d_times = []
    src = torch.randn(size, dtype=torch.float32, device='cuda')
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        dst = src.clone()
        end.record()
        
        torch.cuda.synchronize()
        d2d_times.append(start.elapsed_time(end))
        del dst
        torch.cuda.empty_cache()
    
    # Calculate bandwidths (GB/s)
    data_size_gb = size_mb / 1024
    
    return {
        'h2d_bandwidth_gb_s': data_size_gb / (sum(h2d_times) / len(h2d_times) / 1000),
        'd2h_bandwidth_gb_s': data_size_gb / (sum(d2h_times) / len(d2h_times) / 1000),
        'd2d_bandwidth_gb_s': data_size_gb / (sum(d2d_times) / len(d2d_times) / 1000),
        'test_size_mb': size_mb,
        'iterations': iterations,
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DAY 1: GPU Architecture Deep Dive")
    print("Phase 6: AI/ML Platform Engineering")
    print("="*60 + "\n")
    
    # 1. Check CUDA health
    print("1. CUDA Health Check")
    print("-" * 40)
    health = check_cuda_health()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    if not health.get('cuda_available', False):
        print("\nCUDA is not available. Exiting.")
        exit(1)
    
    # 2. Get GPU information
    print("\n2. GPU Information (via PyTorch)")
    print("-" * 40)
    for i in range(torch.cuda.device_count()):
        info = get_gpu_info(i)
        print_gpu_summary(info)
    
    # 3. Get nvidia-smi information
    print("\n3. Real-time GPU Status (nvidia-smi)")
    print("-" * 40)
    smi_info = get_nvidia_smi_info()
    for gpu in smi_info.get('gpus', []):
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB ({gpu['memory_used_mb']/gpu['memory_total_mb']*100:.1f}%)")
        print(f"    GPU Utilization: {gpu['gpu_utilization_percent']:.1f}%")
        print(f"    Temperature: {gpu['temperature_c']:.0f}°C")
        print(f"    Power: {gpu['power_draw_w']:.1f}W")
    
    # 4. Quick bandwidth benchmark
    print("\n4. Memory Bandwidth Benchmark")
    print("-" * 40)
    bw = benchmark_memory_bandwidth(size_mb=100, iterations=5)
    if 'error' not in bw:
        print(f"  Host→Device: {bw['h2d_bandwidth_gb_s']:.2f} GB/s")
        print(f"  Device→Host: {bw['d2h_bandwidth_gb_s']:.2f} GB/s")
        print(f"  Device→Device: {bw['d2d_bandwidth_gb_s']:.2f} GB/s")
    
    print("\n" + "="*60)
    print("GPU Architecture Analysis Complete!")
    print("="*60)
```

---

## 🔬 Lab Exercise: "GPU Inspection and Analysis"

### Lab Objectives
- Use nvidia-smi to monitor GPU status
- Query GPU properties programmatically
- Understand relationship between hardware and software limits

### Step-by-Step Instructions

#### Step 1: Basic nvidia-smi Commands
```bash
# View GPU summary
nvidia-smi

# Continuous monitoring (every 1 second)
nvidia-smi -l 1

# Query specific attributes
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# View processes using GPU
nvidia-smi pmon -s um -d 1

# View detailed GPU topology
nvidia-smi topo -m
```

#### Step 2: Run the GPU Info Script
```bash
cd day_001_gpu_architecture
python src/gpu_info.py
```

#### Step 3: Analyze Your GPU
```python
# Interactive exploration
import torch

# Your GPU information
device = torch.device('cuda:0')
props = torch.cuda.get_device_properties(device)

# Calculate occupancy limits
max_blocks_per_sm = 32  # Architecture dependent
max_threads = props.max_threads_per_multi_processor
threads_per_block = 256

theoretical_blocks = max_threads // threads_per_block
print(f"With {threads_per_block} threads/block:")
print(f"  Theoretical blocks per SM: {theoretical_blocks}")
print(f"  Total potential concurrent threads: {theoretical_blocks * threads_per_block * props.multi_processor_count:,}")
```

### Expected Output
```
============================================================
DAY 1: GPU Architecture Deep Dive
Phase 6: AI/ML Platform Engineering
============================================================

1. CUDA Health Check
----------------------------------------
  cuda_available: True
  device_count: 1
  memory_allocation: True
  cudnn_available: True
  cudnn_version: 8902
  tensor_cores_likely: True

2. GPU Information (via PyTorch)
----------------------------------------
============================================================
GPU: NVIDIA GeForce RTX 3090
============================================================
  Compute Capability: 8.6
  Total Memory: 24.00 GB
  CUDA Cores: 10,496
  Streaming Multiprocessors (SMs): 82
------------------------------------------------------------
Thread Configuration:
  Warp Size: 32
  Max Threads per Block: 1,024
  Max Threads per SM: 1,536
------------------------------------------------------------
Memory Configuration:
  Max Shared Memory per Block: 48.0 KB
  Max Shared Memory per SM: 100.0 KB
  L2 Cache Size: 6.0 MB
============================================================

4. Memory Bandwidth Benchmark
----------------------------------------
  Host→Device: 12.45 GB/s
  Device→Host: 13.21 GB/s
  Device→Device: 756.32 GB/s
```

### Challenge Extensions
1. **Beginner:** Add GPU temperature monitoring to the script
2. **Intermediate:** Calculate and verify theoretical vs measured memory bandwidth
3. **Advanced:** Write a tool that recommends optimal block sizes based on GPU specs

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### Issue 1: "CUDA not available"
**Symptom:** `torch.cuda.is_available()` returns `False`
**Causes:**
- NVIDIA driver not installed
- CUDA toolkit version mismatch
- PyTorch installed without CUDA support

**Solution:**
```bash
# Check driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 2: "CUDA out of memory"
**Symptom:** `RuntimeError: CUDA out of memory`
**Causes:**
- Too many tensors on GPU
- Memory fragmentation
- Other processes using GPU

**Solution:**
```python
# Check current memory usage
print(torch.cuda.memory_summary())

# Clear cache
torch.cuda.empty_cache()

# Move tensors to CPU when not needed
tensor = tensor.cpu()

# Use context manager for temporary tensors
with torch.no_grad():
    # Operations here won't track gradients (saves memory)
    pass
```

#### Issue 3: Driver/CUDA Version Mismatch
**Symptom:** Various CUDA initialization errors
**Solution:**
```bash
# Check compatibility
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit CUDA version

# Driver CUDA version must be >= Toolkit version
```

---

## ⚡ Performance Optimization

### Key Takeaways for GPU Performance

1. **Maximize Occupancy:** Use enough threads to hide memory latency
2. **Coalesce Memory Access:** Adjacent threads should access adjacent memory
3. **Use Shared Memory:** For frequently accessed data within a block
4. **Minimize Host-Device Transfers:** Keep data on GPU as long as possible
5. **Avoid Branch Divergence:** Threads in a warp should take same path

### Quick Optimization Checklist
```python
# ❌ Bad: Too few threads
kernel<<<1, 1>>>(data)

# ✅ Good: Many threads (occupancy)
kernel<<<grid_size, 256>>>(data)

# ❌ Bad: Non-coalesced access
data[threadIdx.x * stride]

# ✅ Good: Coalesced access
data[blockIdx.x * blockDim.x + threadIdx.x]

# ❌ Bad: Repeated global memory access
for i in range(100):
    result += global_array[tid]

# ✅ Good: Load to shared memory once
__shared__ float shared_array[256];
shared_array[tid] = global_array[tid];
__syncthreads();
for i in range(100):
    result += shared_array[tid];
```

---

## 📚 Further Reading

### Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA GPU Architecture Whitepapers](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/compare/)

### Papers
- "Volta: A GPU Architecture for Deep Learning" - NVIDIA
- "Hopper Architecture In-Depth" - NVIDIA

### Video Resources
- [CUDA Programming Tutorial Series](https://www.youtube.com/results?search_query=cuda+programming+tutorial)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/dli)

---

## 🔗 Connections

### Previous Day
- This is Day 1 - the beginning of Phase 6!

### Next Day
- **Day 2:** CUDA Programming Model - Writing your first kernels

### Related Topics
- Week 3: cuBLAS and cuDNN leverage this GPU architecture
- Week 4: TensorRT optimization depends on understanding GPU memory

---

## 📝 Daily Summary

### Key Takeaways
1. GPUs are optimized for **throughput** with thousands of simple cores, unlike CPUs optimized for **latency**
2. The **SM (Streaming Multiprocessor)** is the fundamental GPU building block containing CUDA cores, shared memory, and schedulers
3. **Warps** (32 threads) are the basic execution unit - understand this for avoiding divergence
4. **Memory hierarchy** (registers → shared → L1 → L2 → global) dramatically affects performance
5. **Compute Capability** determines available features like Tensor Cores and precision modes

### Commands Reference
```bash
# Check GPU status
nvidia-smi

# Query specific GPU properties
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Monitor continuously
nvidia-smi -l 1

# Check CUDA version
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

**Day 1 Complete** ✅

*Next: Day 2 - CUDA Programming Model - Time to write your first GPU kernel!*
