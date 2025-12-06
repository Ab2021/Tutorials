# Day 13: Multi-GPU Programming
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Scale beyond single GPU limitations - master peer-to-peer communication, multi-GPU memory management, and workload distribution.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Enumerate** and select GPUs in multi-GPU systems
2. **Implement** peer-to-peer memory access and transfers
3. **Design** workload distribution strategies for multiple GPUs
4. **Synchronize** operations across multiple devices
5. **Handle** topology and NUMA considerations
6. **Build** scalable multi-GPU applications for ML workloads

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- System with 2+ NVIDIA GPUs (or NVIDIA DGX/HGX)
- NVLink for optimal peer-to-peer (optional but recommended)

### Software Environment
```bash
# Check available GPUs
nvidia-smi -L

# Check topology
nvidia-smi topo -m

# Python environment
pip install cupy-cuda12x numpy

# Verify multi-GPU access
python -c "import cupy as cp; print(f'GPUs: {cp.cuda.runtime.getDeviceCount()}')"
```

### Prior Knowledge
- Day 8: CUDA Streams
- Day 10: Atomic Operations

---

## 📖 Theoretical Foundation

### 1. GPU Topology and Communication

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-GPU SYSTEM TOPOLOGY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PCIe-Based System:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                              CPU                                     │    │
│  │                               │                                      │    │
│  │                          PCIe Switch                                 │    │
│  │                         ╱    │    ╲                                  │    │
│  │                    ┌───┐  ┌───┐  ┌───┐                              │    │
│  │                    │GPU│  │GPU│  │GPU│                              │    │
│  │                    │ 0 │  │ 1 │  │ 2 │                              │    │
│  │                    └───┘  └───┘  └───┘                              │    │
│  │                                                                      │    │
│  │  GPU-to-GPU: Through PCIe (~32 GB/s per direction)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  NVLink-Based System (DGX/HGX):                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                              CPU                                     │    │
│  │                               │                                      │    │
│  │                          PCIe Switch                                 │    │
│  │                              │                                       │    │
│  │                    ┌───┬═══════┬───┐                                │    │
│  │                    │GPU│ NVLink │GPU│                                │    │
│  │                    │ 0 │◄═════►│ 1 │ (~600 GB/s bidirectional)     │    │
│  │                    └───┘       └───┘                                │    │
│  │                      ║           ║                                   │    │
│  │                    ┌───┐       ┌───┐                                │    │
│  │                    │GPU│◄═════►│GPU│                                │    │
│  │                    │ 2 │ NVLink │ 3 │                               │    │
│  │                    └───┘       └───┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Peer-to-Peer (P2P) Access

P2P allows one GPU to directly access another GPU's memory:

```cpp
// Enable peer access
cudaDeviceCanAccessPeer(&canAccess, gpuA, gpuB);
if (canAccess) {
    cudaSetDevice(gpuA);
    cudaDeviceEnablePeerAccess(gpuB, 0);
    
    cudaSetDevice(gpuB);
    cudaDeviceEnablePeerAccess(gpuA, 0);
}

// Now GPU A can directly read/write GPU B's memory
// kernel_on_gpuA<<<...>>>(ptr_on_gpuB);  // Direct access!
```

**P2P Transfer Methods:**
1. **cudaMemcpyPeer** - Explicit copy between devices
2. **Direct Access** - Kernel reads/writes remote memory
3. **GPUDirect RDMA** - For network cards and storage

### 3. NUMA Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NUMA TOPOLOGY IMPACT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Multi-Socket System:                                                        │
│  ┌───────────────────────┐    ┌───────────────────────┐                     │
│  │        Socket 0       │    │        Socket 1       │                     │
│  │  ┌─────┐   ┌─────┐   │    │   ┌─────┐   ┌─────┐  │                     │
│  │  │CPU 0│   │ RAM │   │◄──►│   │ RAM │   │CPU 1│  │                     │
│  │  └──┬──┘   └─────┘   │ QPI│   └─────┘   └──┬──┘  │                     │
│  │     │                 │    │                 │     │                     │
│  │  ┌──┴──┐   ┌─────┐   │    │   ┌─────┐   ┌──┴──┐  │                     │
│  │  │GPU 0│   │GPU 1│   │    │   │GPU 2│   │GPU 3│  │                     │
│  │  └─────┘   └─────┘   │    │   └─────┘   └─────┘  │                     │
│  └───────────────────────┘    └───────────────────────┘                     │
│                                                                              │
│  Best Practice:                                                              │
│  - Pin CPU threads to socket with local GPUs                                │
│  - Allocate host memory on local NUMA node                                  │
│  - Minimize cross-socket GPU communication                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Workload Distribution Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WORKLOAD DISTRIBUTION PATTERNS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DATA PARALLELISM (Same operation, different data)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Data = [A | B | C | D]                                             │    │
│  │            │   │   │   │                                             │    │
│  │         GPU0 GPU1 GPU2 GPU3                                          │    │
│  │          ▼   ▼   ▼   ▼                                               │    │
│  │        f(A) f(B) f(C) f(D)  ◄── Same function on different chunks   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  2. MODEL PARALLELISM (Different operations, same/related data)              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Neural Network Layers:                                              │    │
│  │  Input ─▶ [Layer1] ─▶ [Layer2] ─▶ [Layer3] ─▶ Output                │    │
│  │            GPU 0        GPU 1        GPU 2                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  3. PIPELINE PARALLELISM (Stream of work through stages)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Time:     t0      t1      t2      t3      t4                       │    │
│  │  GPU 0: [Batch1] [Batch2] [Batch3] [Batch4] [Batch5]                │    │
│  │  GPU 1:         [Batch1] [Batch2] [Batch3] [Batch4]                 │    │
│  │  GPU 2:                 [Batch1] [Batch2] [Batch3]                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Implementation

### 👨‍💻 Core Implementation

#### 📁 `src/multi_gpu.py` - Multi-GPU Operations
```python
#!/usr/bin/env python3
"""
Day 13: Multi-GPU Programming
Phase 6: AI/ML Platform Engineering with GPU Programming

This module demonstrates multi-GPU programming patterns including
peer-to-peer access, workload distribution, and synchronization.
"""

import cupy as cp
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import time


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory_gb: float
    compute_capability: Tuple[int, int]
    pci_bus_id: str


class MultiGPUManager:
    """
    Manager for multi-GPU operations.
    
    This class provides utilities for discovering GPUs, managing
    peer-to-peer access, and distributing workloads.
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize the multi-GPU manager.
        
        Args:
            device_ids: Specific devices to use. If None, use all available.
        """
        self.num_devices = cp.cuda.runtime.getDeviceCount()
        
        if device_ids is None:
            self.device_ids = list(range(self.num_devices))
        else:
            self.device_ids = device_ids
        
        self.gpu_info = self._collect_gpu_info()
        self.p2p_matrix = self._check_p2p_access()
        self.streams = self._create_streams()
    
    def _collect_gpu_info(self) -> Dict[int, GPUInfo]:
        """Collect information about all GPUs."""
        info = {}
        
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                device = cp.cuda.Device(device_id)
                props = device.attributes
                
                info[device_id] = GPUInfo(
                    device_id=device_id,
                    name=device.name,
                    total_memory_gb=device.mem_info[1] / (1024**3),
                    compute_capability=device.compute_capability,
                    pci_bus_id=f"{props.get('PCI_BUS_ID', 'Unknown')}"
                )
        
        return info
    
    def _check_p2p_access(self) -> Dict[Tuple[int, int], bool]:
        """Check peer-to-peer access between all GPU pairs."""
        p2p = {}
        
        for src in self.device_ids:
            for dst in self.device_ids:
                if src != dst:
                    can_access = cp.cuda.runtime.deviceCanAccessPeer(src, dst)
                    p2p[(src, dst)] = can_access
        
        return p2p
    
    def _create_streams(self) -> Dict[int, cp.cuda.Stream]:
        """Create a stream for each GPU."""
        streams = {}
        
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                streams[device_id] = cp.cuda.Stream(non_blocking=True)
        
        return streams
    
    def enable_p2p_access(self) -> None:
        """Enable peer-to-peer access between all capable GPU pairs."""
        for (src, dst), can_access in self.p2p_matrix.items():
            if can_access:
                with cp.cuda.Device(src):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(dst)
                    except cp.cuda.runtime.CUDARuntimeError:
                        pass  # Already enabled
    
    def print_topology(self) -> None:
        """Print GPU topology information."""
        print("\n" + "="*60)
        print("GPU TOPOLOGY")
        print("="*60)
        
        print(f"\nNumber of GPUs: {len(self.device_ids)}")
        
        print("\nDevice Information:")
        for device_id, info in self.gpu_info.items():
            print(f"  GPU {device_id}: {info.name}")
            print(f"    Memory: {info.total_memory_gb:.1f} GB")
            print(f"    Compute: {info.compute_capability[0]}.{info.compute_capability[1]}")
        
        print("\nPeer-to-Peer Access Matrix:")
        print("     ", end="")
        for d in self.device_ids:
            print(f" GPU{d}", end="")
        print()
        
        for src in self.device_ids:
            print(f"GPU{src}", end="")
            for dst in self.device_ids:
                if src == dst:
                    print("   - ", end="")
                else:
                    p2p = self.p2p_matrix.get((src, dst), False)
                    print("   ✓ " if p2p else "   ✗ ", end="")
            print()
    
    def allocate_on_device(self, device_id: int, shape: Tuple, dtype=cp.float32) -> cp.ndarray:
        """
        Allocate array on specific device.
        
        Args:
            device_id: Target GPU
            shape: Array shape
            dtype: Data type
            
        Returns:
            CuPy array on specified device
        """
        with cp.cuda.Device(device_id):
            return cp.zeros(shape, dtype=dtype)
    
    def copy_to_device(self, data: np.ndarray, device_id: int) -> cp.ndarray:
        """
        Copy numpy array to specific GPU.
        
        Args:
            data: NumPy array to copy
            device_id: Target GPU
            
        Returns:
            CuPy array on target device
        """
        with cp.cuda.Device(device_id):
            return cp.asarray(data)
    
    def copy_between_devices(self, src_array: cp.ndarray, 
                             dst_device: int) -> cp.ndarray:
        """
        Copy array from one GPU to another.
        
        Args:
            src_array: Source array on some GPU
            dst_device: Target GPU ID
            
        Returns:
            Copy of array on target device
        """
        # Get to host first (works without P2P)
        host_data = src_array.get()
        
        with cp.cuda.Device(dst_device):
            return cp.asarray(host_data)
    
    def synchronize_all(self) -> None:
        """Synchronize all GPUs."""
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                cp.cuda.Stream.null.synchronize()


# ============================================================================
# DATA PARALLEL OPERATIONS
# ============================================================================

class DataParallelExecutor:
    """
    Execute operations in data-parallel fashion across multiple GPUs.
    
    This class splits input data across GPUs and combines results.
    """
    
    def __init__(self, manager: MultiGPUManager):
        """
        Initialize with a multi-GPU manager.
        
        Args:
            manager: MultiGPUManager instance
        """
        self.manager = manager
        self.num_gpus = len(manager.device_ids)
    
    def parallel_map(self, func, data: np.ndarray) -> np.ndarray:
        """
        Apply function to data in parallel across GPUs.
        
        Args:
            func: Function to apply (takes and returns CuPy array)
            data: Input data (will be split across GPUs)
            
        Returns:
            Combined result
        """
        # Split data
        chunks = np.array_split(data, self.num_gpus)
        results = [None] * self.num_gpus
        
        def process_chunk(idx, chunk):
            device_id = self.manager.device_ids[idx]
            with cp.cuda.Device(device_id):
                d_chunk = cp.asarray(chunk)
                d_result = func(d_chunk)
                results[idx] = d_result.get()
        
        # Execute in parallel using threads
        threads = []
        for i, chunk in enumerate(chunks):
            t = threading.Thread(target=process_chunk, args=(i, chunk))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Combine results
        return np.concatenate(results)
    
    def parallel_reduce(self, data: np.ndarray, reduce_op='sum') -> float:
        """
        Reduce data across multiple GPUs.
        
        Args:
            data: Input data
            reduce_op: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Reduced value
        """
        chunks = np.array_split(data, self.num_gpus)
        partial_results = [None] * self.num_gpus
        
        def reduce_chunk(idx, chunk):
            device_id = self.manager.device_ids[idx]
            with cp.cuda.Device(device_id):
                d_chunk = cp.asarray(chunk)
                
                if reduce_op == 'sum':
                    result = float(cp.sum(d_chunk))
                elif reduce_op == 'mean':
                    result = (float(cp.sum(d_chunk)), len(chunk))
                elif reduce_op == 'max':
                    result = float(cp.max(d_chunk))
                elif reduce_op == 'min':
                    result = float(cp.min(d_chunk))
                
                partial_results[idx] = result
        
        # Execute
        threads = []
        for i, chunk in enumerate(chunks):
            t = threading.Thread(target=reduce_chunk, args=(i, chunk))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Final reduction
        if reduce_op == 'sum':
            return sum(partial_results)
        elif reduce_op == 'mean':
            total_sum = sum(r[0] for r in partial_results)
            total_count = sum(r[1] for r in partial_results)
            return total_sum / total_count
        elif reduce_op == 'max':
            return max(partial_results)
        elif reduce_op == 'min':
            return min(partial_results)


# ============================================================================
# MULTI-GPU MATRIX MULTIPLICATION
# ============================================================================

def multi_gpu_matmul(A: np.ndarray, B: np.ndarray, 
                     manager: MultiGPUManager) -> np.ndarray:
    """
    Perform matrix multiplication split across multiple GPUs.
    
    Strategy: Split rows of A across GPUs, each GPU computes
    partial result rows of C = A @ B.
    
    Args:
        A: Matrix A (M, K)
        B: Matrix B (K, N)
        manager: Multi-GPU manager
        
    Returns:
        C = A @ B (M, N)
    """
    M, K = A.shape
    _, N = B.shape
    num_gpus = len(manager.device_ids)
    
    # Split A by rows
    rows_per_gpu = M // num_gpus
    results = [None] * num_gpus
    
    def compute_partial(idx):
        device_id = manager.device_ids[idx]
        start_row = idx * rows_per_gpu
        end_row = start_row + rows_per_gpu if idx < num_gpus - 1 else M
        
        with cp.cuda.Device(device_id):
            # Copy portions to this GPU
            d_A_chunk = cp.asarray(A[start_row:end_row])
            d_B = cp.asarray(B)  # Each GPU needs full B
            
            # Compute partial result
            d_C_chunk = d_A_chunk @ d_B
            
            results[idx] = d_C_chunk.get()
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        list(executor.map(compute_partial, range(num_gpus)))
    
    # Combine results
    return np.vstack(results)


# ============================================================================
# GRADIENT AGGREGATION (for distributed training)
# ============================================================================

class AllReduceAggregator:
    """
    Ring AllReduce implementation for gradient aggregation.
    
    This simulates the communication pattern used in distributed
    deep learning frameworks.
    """
    
    def __init__(self, manager: MultiGPUManager):
        self.manager = manager
        self.num_gpus = len(manager.device_ids)
    
    def allreduce_sum(self, arrays: List[cp.ndarray]) -> List[cp.ndarray]:
        """
        Sum arrays across all GPUs (each GPU gets the sum).
        
        This is a simple reduce-broadcast implementation.
        Full ring allreduce would be more efficient.
        
        Args:
            arrays: List of arrays, one per GPU
            
        Returns:
            List of arrays, each containing the sum
        """
        # Gather to host and sum
        host_arrays = []
        for i, arr in enumerate(arrays):
            with cp.cuda.Device(self.manager.device_ids[i]):
                host_arrays.append(arr.get())
        
        # Sum on CPU
        total = sum(host_arrays)
        
        # Broadcast back to all GPUs
        results = []
        for i in range(self.num_gpus):
            with cp.cuda.Device(self.manager.device_ids[i]):
                results.append(cp.asarray(total))
        
        return results


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demonstrate_multi_gpu_basics():
    """Demonstrate basic multi-GPU operations."""
    print("\n" + "="*60)
    print("MULTI-GPU BASICS DEMONSTRATION")
    print("="*60)
    
    manager = MultiGPUManager()
    manager.print_topology()
    
    if len(manager.device_ids) < 2:
        print("\nNote: Single GPU detected. Multi-GPU examples will run sequentially.")
    
    # Enable P2P if available
    manager.enable_p2p_access()
    
    # Allocate data on different GPUs
    print("\n--- Allocating data on different GPUs ---")
    
    arrays = {}
    for device_id in manager.device_ids:
        arrays[device_id] = manager.allocate_on_device(device_id, (1000, 1000))
        print(f"  Allocated 1000x1000 float32 on GPU {device_id}")


def demonstrate_data_parallel():
    """Demonstrate data parallel execution."""
    print("\n" + "="*60)
    print("DATA PARALLEL EXECUTION")
    print("="*60)
    
    manager = MultiGPUManager()
    executor = DataParallelExecutor(manager)
    
    # Create large dataset
    n = 10_000_000
    data = np.random.randn(n).astype(np.float32)
    
    print(f"\nData size: {n:,} elements ({n * 4 / 1e6:.1f} MB)")
    print(f"Number of GPUs: {len(manager.device_ids)}")
    
    # Parallel map
    print("\n1. Parallel Map (sin²(x) + cos²(x)):")
    
    def compute_func(x):
        return cp.sin(x) ** 2 + cp.cos(x) ** 2
    
    start = time.perf_counter()
    result = executor.parallel_map(compute_func, data)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"   Time: {elapsed:.2f} ms")
    print(f"   Mean result: {np.mean(result):.6f} (expected: 1.0)")
    
    # Parallel reduce
    print("\n2. Parallel Reduce (sum):")
    
    start = time.perf_counter()
    total = executor.parallel_reduce(data, 'sum')
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"   Time: {elapsed:.2f} ms")
    print(f"   Sum: {total:.6f}")
    print(f"   Expected: {np.sum(data):.6f}")


def demonstrate_multi_gpu_matmul():
    """Demonstrate multi-GPU matrix multiplication."""
    print("\n" + "="*60)
    print("MULTI-GPU MATRIX MULTIPLICATION")
    print("="*60)
    
    manager = MultiGPUManager()
    
    # Matrix dimensions
    M, K, N = 4096, 4096, 4096
    
    print(f"\nMatrix sizes: A({M}x{K}) @ B({K}x{N})")
    print(f"Number of GPUs: {len(manager.device_ids)}")
    
    # Create matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Multi-GPU matmul
    start = time.perf_counter()
    C_multi = multi_gpu_matmul(A, B, manager)
    elapsed_multi = (time.perf_counter() - start) * 1000
    
    # Single GPU comparison
    with cp.cuda.Device(0):
        d_A = cp.asarray(A)
        d_B = cp.asarray(B)
        
        # Warmup
        _ = d_A @ d_B
        cp.cuda.Stream.null.synchronize()
        
        start = time.perf_counter()
        d_C_single = d_A @ d_B
        cp.cuda.Stream.null.synchronize()
        elapsed_single = (time.perf_counter() - start) * 1000
        
        C_single = d_C_single.get()
    
    print(f"\nResults:")
    print(f"  Single GPU: {elapsed_single:.2f} ms")
    print(f"  Multi-GPU:  {elapsed_multi:.2f} ms")
    
    if len(manager.device_ids) > 1:
        speedup = elapsed_single / elapsed_multi
        print(f"  Speedup: {speedup:.2f}x")
    
    # Verify
    max_error = np.max(np.abs(C_multi - C_single))
    print(f"  Max Error: {max_error:.2e}")


def demonstrate_gradient_sync():
    """Demonstrate gradient synchronization pattern."""
    print("\n" + "="*60)
    print("GRADIENT SYNCHRONIZATION (AllReduce)")
    print("="*60)
    
    manager = MultiGPUManager()
    aggregator = AllReduceAggregator(manager)
    
    # Simulate gradients on each GPU
    gradient_size = 1_000_000
    gradients = []
    
    for i, device_id in enumerate(manager.device_ids):
        with cp.cuda.Device(device_id):
            # Each GPU has slightly different "gradients"
            grad = cp.random.randn(gradient_size, dtype=cp.float32) * (i + 1)
            gradients.append(grad)
    
    print(f"\nGradient size: {gradient_size:,} parameters")
    print(f"Number of GPUs: {len(manager.device_ids)}")
    
    # AllReduce
    start = time.perf_counter()
    synced_gradients = aggregator.allreduce_sum(gradients)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"\nAllReduce time: {elapsed:.2f} ms")
    
    # Verify all GPUs have same gradients
    first = synced_gradients[0].get()
    all_same = all(np.allclose(first, g.get()) for g in synced_gradients)
    print(f"All GPUs have same gradient: {all_same}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("DAY 13: MULTI-GPU PROGRAMMING")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*60)
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"\nDetected {num_gpus} GPU(s)")
    
    demonstrate_multi_gpu_basics()
    demonstrate_data_parallel()
    demonstrate_multi_gpu_matmul()
    demonstrate_gradient_sync()
    
    print("\n" + "="*60)
    print("Day 13 Complete: Multi-GPU Programming")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Building a Multi-GPU Training Simulator"

### Lab Objectives
1. Implement data parallel forward pass
2. Simulate gradient computation and aggregation
3. Measure scaling efficiency

### Implementation

```python
#!/usr/bin/env python3
"""
Lab: Multi-GPU Training Simulator
Day 13: Multi-GPU Programming
"""

import cupy as cp
import numpy as np
from typing import List, Dict
import time


class SimpleLinearLayer:
    """A simple linear layer for demonstration."""
    
    def __init__(self, in_features: int, out_features: int, device_id: int):
        self.device_id = device_id
        
        with cp.cuda.Device(device_id):
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.weight = cp.random.randn(in_features, out_features, dtype=cp.float32) * scale
            self.bias = cp.zeros(out_features, dtype=cp.float32)
            
            # Gradient storage
            self.weight_grad = None
            self.bias_grad = None
            
            # Cache for backward
            self.input_cache = None
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Forward pass."""
        self.input_cache = x
        return x @ self.weight + self.bias
    
    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """Backward pass."""
        # Gradient w.r.t. weight: x.T @ grad_output
        self.weight_grad = self.input_cache.T @ grad_output
        
        # Gradient w.r.t. bias: sum across batch
        self.bias_grad = cp.sum(grad_output, axis=0)
        
        # Gradient w.r.t. input
        return grad_output @ self.weight.T


class DataParallelTrainer:
    """
    Simulates data-parallel distributed training.
    """
    
    def __init__(self, model_factory, device_ids: List[int]):
        """
        Initialize trainer with a model replicated across GPUs.
        
        Args:
            model_factory: Function that creates a model given device_id
            device_ids: List of GPU IDs to use
        """
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        
        # Create model replica on each GPU
        self.models = {
            device_id: model_factory(device_id)
            for device_id in device_ids
        }
    
    def forward(self, x_chunks: List[np.ndarray]) -> List[cp.ndarray]:
        """
        Forward pass on each GPU.
        
        Args:
            x_chunks: Input chunks, one per GPU
            
        Returns:
            Output chunks from each GPU
        """
        outputs = []
        
        for i, device_id in enumerate(self.device_ids):
            with cp.cuda.Device(device_id):
                x = cp.asarray(x_chunks[i])
                output = self.models[device_id].forward(x)
                outputs.append(output)
        
        return outputs
    
    def backward(self, grad_chunks: List[cp.ndarray]) -> None:
        """
        Backward pass on each GPU.
        
        Args:
            grad_chunks: Gradient chunks from loss
        """
        for i, device_id in enumerate(self.device_ids):
            with cp.cuda.Device(device_id):
                self.models[device_id].backward(grad_chunks[i])
    
    def allreduce_gradients(self) -> None:
        """Average gradients across all GPUs."""
        # Collect gradients
        weight_grads = []
        bias_grads = []
        
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                weight_grads.append(self.models[device_id].weight_grad.get())
                bias_grads.append(self.models[device_id].bias_grad.get())
        
        # Average
        avg_weight_grad = sum(weight_grads) / self.num_gpus
        avg_bias_grad = sum(bias_grads) / self.num_gpus
        
        # Distribute back
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                self.models[device_id].weight_grad = cp.asarray(avg_weight_grad)
                self.models[device_id].bias_grad = cp.asarray(avg_bias_grad)
    
    def update(self, lr: float = 0.01) -> None:
        """Update weights using SGD."""
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                model = self.models[device_id]
                model.weight -= lr * model.weight_grad
                model.bias -= lr * model.bias_grad


def run_lab():
    """Run the multi-GPU training lab."""
    print("="*60)
    print("LAB: Multi-GPU Training Simulator")
    print("Day 13: Multi-GPU Programming")
    print("="*60)
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    device_ids = list(range(num_gpus))
    
    print(f"\nUsing {num_gpus} GPU(s)")
    
    # Configuration
    batch_size = 4096  # Total batch size
    in_features = 1024
    out_features = 512
    num_iterations = 10
    
    batch_per_gpu = batch_size // num_gpus
    
    print(f"Batch size: {batch_size} (total), {batch_per_gpu} (per GPU)")
    print(f"Model: Linear({in_features} → {out_features})")
    
    # Model factory
    def create_model(device_id):
        return SimpleLinearLayer(in_features, out_features, device_id)
    
    # Create trainer
    trainer = DataParallelTrainer(create_model, device_ids)
    
    # Training loop
    print(f"\nRunning {num_iterations} iterations...")
    
    times = {
        'forward': [],
        'backward': [],
        'allreduce': [],
        'update': []
    }
    
    for iteration in range(num_iterations):
        # Generate data
        x_chunks = [np.random.randn(batch_per_gpu, in_features).astype(np.float32)
                   for _ in range(num_gpus)]
        
        # Forward
        start = time.perf_counter()
        outputs = trainer.forward(x_chunks)
        for d in device_ids:
            with cp.cuda.Device(d):
                cp.cuda.Stream.null.synchronize()
        times['forward'].append((time.perf_counter() - start) * 1000)
        
        # Fake gradient (would come from loss)
        grad_chunks = [cp.ones((batch_per_gpu, out_features), dtype=cp.float32) 
                       for _ in range(num_gpus)]
        
        # Backward
        start = time.perf_counter()
        trainer.backward(grad_chunks)
        times['backward'].append((time.perf_counter() - start) * 1000)
        
        # AllReduce
        start = time.perf_counter()
        trainer.allreduce_gradients()
        times['allreduce'].append((time.perf_counter() - start) * 1000)
        
        # Update
        start = time.perf_counter()
        trainer.update(lr=0.01)
        times['update'].append((time.perf_counter() - start) * 1000)
    
    # Report
    print("\nAverage timing (ms):")
    total = 0
    for name, t in times.items():
        avg = np.mean(t[2:])  # Skip warmup
        total += avg
        print(f"  {name:<12}: {avg:.3f}")
    print(f"  {'TOTAL':<12}: {total:.3f}")
    
    # Calculate throughput
    samples_per_sec = batch_size / (total / 1000)
    print(f"\nThroughput: {samples_per_sec:.0f} samples/second")
    
    print("\nLab complete!")


if __name__ == "__main__":
    run_lab()
```

---

## 📝 Daily Summary

### Key Takeaways
1. **Multi-GPU requires topology awareness** - NVLink vs PCIe matters
2. **P2P access enables direct GPU-to-GPU communication**
3. **Data parallelism** splits data across GPUs
4. **Synchronization is critical** - AllReduce for gradients
5. **NUMA considerations** affect multi-socket systems
6. **Scaling efficiency** depends on communication overhead

### API Summary
```python
# CuPy multi-GPU
with cp.cuda.Device(device_id):
    # All operations happen on this GPU
    array = cp.zeros(...)

# Check P2P
cp.cuda.runtime.deviceCanAccessPeer(src, dst)
cp.cuda.runtime.deviceEnablePeerAccess(peer_device)

# Device count
num_gpus = cp.cuda.runtime.getDeviceCount()
```

### Performance Tips
- Overlap communication with computation
- Use NVLink for large gradient transfers
- Pin CPU threads to NUMA nodes
- Batch small transfers

---

**Day 13 Complete** ✅

*Next: Day 14 - Week 2 Review & Multi-GPU Project!*
