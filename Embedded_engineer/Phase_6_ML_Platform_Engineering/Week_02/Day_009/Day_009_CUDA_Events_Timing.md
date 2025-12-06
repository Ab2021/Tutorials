# Day 9: CUDA Events and Timing
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Master accurate GPU timing, inter-stream dependencies, and profiling techniques for performance optimization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Create** and manage CUDA events for synchronization
2. **Measure** GPU execution time with microsecond precision
3. **Implement** inter-stream dependencies using events
4. **Debug** timing issues and synchronization problems
5. **Build** custom profiling infrastructure for GPU code
6. **Analyze** performance bottlenecks using event-based timing

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Display driver 450.x or newer recommended

### Software Environment
```bash
# Python environment
pip install cupy-cuda12x numpy pandas matplotlib seaborn

# Verify CUDA events support
python -c "import cupy as cp; e = cp.cuda.Event(); print('Events OK')"
```

### Prior Knowledge
- Day 8: CUDA Streams and Concurrency
- Understanding of asynchronous GPU execution

---

## 📖 Theoretical Foundation

### 1. What Are CUDA Events?

CUDA events are **synchronization markers** that can be placed in a stream. They serve two primary purposes:

1. **Timing:** Measure elapsed time between two points in GPU execution
2. **Synchronization:** Create dependencies between streams

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CUDA Events Concept                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stream Timeline:                                                            │
│  ───────────────────────────────────────────────────────────────────▶       │
│       │                    │                    │                    │       │
│       ▼                    ▼                    ▼                    ▼       │
│   ┌───────┐            ┌───────┐            ┌───────┐            ┌───────┐  │
│   │Event A│            │Kernel │            │Event B│            │Copy   │  │
│   │Record │            │Execute│            │Record │            │D2H    │  │
│   └───────┘            └───────┘            └───────┘            └───────┘  │
│       │                                          │                          │
│       └──────── Elapsed Time (B - A) ───────────┘                          │
│                                                                              │
│  Event Operations:                                                           │
│  • cudaEventRecord(event, stream)  - Place event in stream                  │
│  • cudaEventSynchronize(event)     - Wait for event on host                 │
│  • cudaEventQuery(event)           - Check if event completed               │
│  • cudaEventElapsedTime(&ms, A, B) - Get time between events                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Event Timing Precision

CUDA events provide GPU hardware-based timing with key characteristics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Timing Precision Comparison                         │
├───────────────────────┬────────────────┬────────────────────────────────────┤
│ Method                │ Precision      │ Measures                           │
├───────────────────────┼────────────────┼────────────────────────────────────┤
│ CPU time.time()       │ ~1 ms          │ Wall clock (includes sync waits)   │
│ CPU time.perf_counter │ ~100 ns        │ Wall clock (includes sync waits)   │
│ CUDA Events           │ ~0.5 μs        │ GPU execution only (no sync wait)  │
│ Nsight Compute        │ ~10 ns         │ Hardware counters, detailed        │
└───────────────────────┴────────────────┴────────────────────────────────────┘
```

**Why CUDA Events are More Accurate:**
- They are recorded directly by the GPU
- They don't include CPU-side synchronization overhead
- They measure actual GPU work, not waiting time

### 3. Event Creation Flags

```cpp
// Event creation with flags
cudaEvent_t event;

// Default: No special flags
cudaEventCreate(&event);

// Blocking sync: cudaEventSynchronize will block CPU completely
cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

// Disable timing: Slightly faster events when timing not needed
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

// Interprocess: Can be used across processes with IPC
cudaEventCreateWithFlags(&event, cudaEventInterprocess);
```

### 4. Inter-Stream Synchronization with Events

Events enable fine-grained dependency management between streams:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Inter-Stream Dependencies                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stream 0:  ─────┬─────────────────────────────────────────────▶            │
│                  │ A (H2D)     B (Compute)    C (Event)                      │
│                  │ ████████    █████████████      │                          │
│                  │                                │                          │
│                  │                                │ cudaEventRecord()        │
│                  │                                ▼                          │
│  Stream 1:  ─────┴────────────────────────────────┬─────────────▶           │
│                               cudaStreamWaitEvent()│                         │
│                               (waits for Event C)  │ D (Compute using        │
│                                                    │    result from B)       │
│                                                    █████████████████         │
│                                                                              │
│  Code:                                                                       │
│    cudaEventRecord(eventC, stream0);                                         │
│    cudaStreamWaitEvent(stream1, eventC, 0);  // stream1 waits for eventC    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Event-Based Synchronization Patterns

#### Pattern 1: Simple Timing
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(data);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
```

#### Pattern 2: Producer-Consumer
```cpp
// Producer stream produces data
cudaMemcpyAsync(d_data, h_data, size, H2D, producerStream);
producer_kernel<<<..., producerStream>>>(d_data);
cudaEventRecord(dataReady, producerStream);

// Consumer stream waits for data
cudaStreamWaitEvent(consumerStream, dataReady, 0);
consumer_kernel<<<..., consumerStream>>>(d_data);
```

#### Pattern 3: Fork-Join
```cpp
// Fork: single producer, multiple consumers
cudaEventRecord(forkEvent, stream0);

// Multiple consumers wait
cudaStreamWaitEvent(stream1, forkEvent, 0);
cudaStreamWaitEvent(stream2, forkEvent, 0);
cudaStreamWaitEvent(stream3, forkEvent, 0);

// Each stream processes independently
kernel1<<<..., stream1>>>();
kernel2<<<..., stream2>>>();
kernel3<<<..., stream3>>>();

// Join: single stream waits for all
cudaEventRecord(done1, stream1);
cudaEventRecord(done2, stream2);
cudaEventRecord(done3, stream3);

cudaStreamWaitEvent(stream0, done1, 0);
cudaStreamWaitEvent(stream0, done2, 0);
cudaStreamWaitEvent(stream0, done3, 0);

// Continue after all complete
final_kernel<<<..., stream0>>>();
```

---

## 💻 Implementation

### 🛠️ Project Structure
```text
day_009_cuda_events/
├── src/
│   ├── event_timing.py
│   ├── profiler.py
│   ├── dependency_manager.py
│   └── benchmark_framework.py
├── tests/
│   └── test_events.py
├── examples/
│   └── producer_consumer.py
└── README.md
```

### 👨‍💻 Core Implementation

#### 📁 `src/event_timing.py` - Event Timing Fundamentals
```python
#!/usr/bin/env python3
"""
Day 9: CUDA Events and Timing
Phase 6: AI/ML Platform Engineering with GPU Programming

This module provides comprehensive event timing utilities for
accurate GPU performance measurement.
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any
import statistics
from contextlib import contextmanager


@dataclass
class TimingResult:
    """Results from a timing measurement."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples: List[float] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
                f"(min: {self.min_ms:.3f}, max: {self.max_ms:.3f})")
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'samples': self.samples
        }


class CUDATimer:
    """
    High-precision GPU timer using CUDA events.
    
    This class provides accurate timing of GPU operations by using
    hardware events that are recorded directly on the GPU timeline.
    
    Usage:
        timer = CUDATimer()
        
        timer.start()
        # GPU operations here
        timer.stop()
        
        elapsed = timer.elapsed()
    """
    
    def __init__(self, stream: Optional[cp.cuda.Stream] = None):
        """
        Initialize the timer.
        
        Args:
            stream: Optional CUDA stream. If None, uses default stream.
        """
        self.stream = stream
        self.start_event = cp.cuda.Event()
        self.stop_event = cp.cuda.Event()
        self._is_recording = False
    
    def start(self) -> 'CUDATimer':
        """Record start event."""
        if self._is_recording:
            raise RuntimeError("Timer is already recording. Call stop() first.")
        
        if self.stream:
            self.start_event.record(self.stream)
        else:
            self.start_event.record()
        
        self._is_recording = True
        return self
    
    def stop(self) -> 'CUDATimer':
        """Record stop event."""
        if not self._is_recording:
            raise RuntimeError("Timer is not recording. Call start() first.")
        
        if self.stream:
            self.stop_event.record(self.stream)
        else:
            self.stop_event.record()
        
        self._is_recording = False
        return self
    
    def elapsed(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        This will synchronize with the stop event if necessary.
        """
        self.stop_event.synchronize()
        return cp.cuda.get_elapsed_time(self.start_event, self.stop_event)
    
    @contextmanager
    def measure(self):
        """
        Context manager for timing a code block.
        
        Usage:
            timer = CUDATimer()
            with timer.measure():
                # GPU operations
            print(f"Elapsed: {timer.elapsed()} ms")
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()


class GPUProfiler:
    """
    Comprehensive GPU profiler with named regions and statistics.
    
    This profiler allows timing multiple operations and collecting
    statistics over multiple runs.
    
    Usage:
        profiler = GPUProfiler()
        
        for _ in range(10):
            with profiler.region("kernel_1"):
                kernel1<<<grid, block>>>()
            
            with profiler.region("kernel_2"):
                kernel2<<<grid, block>>>()
        
        profiler.print_summary()
    """
    
    def __init__(self, warmup_runs: int = 2):
        """
        Initialize the profiler.
        
        Args:
            warmup_runs: Number of initial runs to discard
        """
        self.warmup_runs = warmup_runs
        self.regions: Dict[str, List[float]] = {}
        self.run_counts: Dict[str, int] = {}
        self.current_stream: Optional[cp.cuda.Stream] = None
    
    def set_stream(self, stream: cp.cuda.Stream) -> None:
        """Set the stream to use for timing."""
        self.current_stream = stream
    
    @contextmanager
    def region(self, name: str):
        """
        Context manager for timing a named region.
        
        Args:
            name: Name of the region for reporting
        """
        if name not in self.regions:
            self.regions[name] = []
            self.run_counts[name] = 0
        
        timer = CUDATimer(self.current_stream)
        
        timer.start()
        try:
            yield
        finally:
            timer.stop()
            elapsed = timer.elapsed()
            
            self.run_counts[name] += 1
            
            # Skip warmup runs
            if self.run_counts[name] > self.warmup_runs:
                self.regions[name].append(elapsed)
    
    def get_result(self, name: str) -> Optional[TimingResult]:
        """Get timing results for a named region."""
        if name not in self.regions or len(self.regions[name]) == 0:
            return None
        
        samples = self.regions[name]
        return TimingResult(
            name=name,
            mean_ms=statistics.mean(samples),
            std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            min_ms=min(samples),
            max_ms=max(samples),
            samples=samples.copy()
        )
    
    def get_all_results(self) -> Dict[str, TimingResult]:
        """Get timing results for all regions."""
        return {name: self.get_result(name) 
                for name in self.regions 
                if self.get_result(name) is not None}
    
    def print_summary(self) -> None:
        """Print a formatted summary of all timing results."""
        results = self.get_all_results()
        
        if not results:
            print("No timing data collected.")
            return
        
        print("\n" + "="*70)
        print("GPU PROFILER SUMMARY")
        print("="*70)
        print(f"{'Region':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min':>10} {'Max':>10}")
        print("-"*70)
        
        total = 0.0
        for name, result in sorted(results.items()):
            print(f"{name:<30} {result.mean_ms:>12.3f} {result.std_ms:>12.3f} "
                  f"{result.min_ms:>10.3f} {result.max_ms:>10.3f}")
            total += result.mean_ms
        
        print("-"*70)
        print(f"{'TOTAL':<30} {total:>12.3f}")
        print("="*70)
    
    def reset(self) -> None:
        """Reset all timing data."""
        self.regions.clear()
        self.run_counts.clear()


class EventTimingAnalyzer:
    """
    Analyzer for comparing different timing approaches and validating
    event-based timing accuracy.
    """
    
    @staticmethod
    def compare_timing_methods(operation: Callable, iterations: int = 100) -> Dict:
        """
        Compare CPU timing vs CUDA event timing.
        
        Args:
            operation: A callable that performs GPU operations
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing comparison results
        """
        import time
        
        cpu_times = []
        gpu_times = []
        
        # Warmup
        for _ in range(5):
            operation()
            cp.cuda.Stream.null.synchronize()
        
        timer = CUDATimer()
        
        for _ in range(iterations):
            # CPU timing (includes sync overhead)
            cp.cuda.Stream.null.synchronize()
            cpu_start = time.perf_counter()
            
            # GPU timing
            timer.start()
            operation()
            timer.stop()
            
            # Wait and record CPU time
            cp.cuda.Stream.null.synchronize()
            cpu_end = time.perf_counter()
            
            cpu_times.append((cpu_end - cpu_start) * 1000)  # Convert to ms
            gpu_times.append(timer.elapsed())
        
        return {
            'cpu_timing': {
                'mean_ms': statistics.mean(cpu_times),
                'std_ms': statistics.stdev(cpu_times),
                'min_ms': min(cpu_times),
                'max_ms': max(cpu_times)
            },
            'gpu_timing': {
                'mean_ms': statistics.mean(gpu_times),
                'std_ms': statistics.stdev(gpu_times),
                'min_ms': min(gpu_times),
                'max_ms': max(gpu_times)
            },
            'overhead_ms': statistics.mean(cpu_times) - statistics.mean(gpu_times),
            'overhead_percent': ((statistics.mean(cpu_times) / statistics.mean(gpu_times)) - 1) * 100
        }


def demonstrate_event_basics():
    """Demonstrate basic event timing operations."""
    print("\n" + "="*70)
    print("EVENT BASICS DEMONSTRATION")
    print("="*70)
    
    # Simple timing
    print("\n1. Basic Event Timing:")
    
    size = 10_000_000
    data = cp.random.randn(size, dtype=cp.float32)
    
    timer = CUDATimer()
    
    timer.start()
    result = cp.sin(data) ** 2 + cp.cos(data) ** 2
    timer.stop()
    
    print(f"   Operation: sin²(x) + cos²(x) on {size:,} elements")
    print(f"   GPU Time: {timer.elapsed():.3f} ms")
    
    # Using context manager
    print("\n2. Context Manager Timing:")
    
    with timer.measure():
        result = cp.fft.fft(data)
        cp.cuda.Stream.null.synchronize()
    
    print(f"   Operation: FFT on {size:,} elements")
    print(f"   GPU Time: {timer.elapsed():.3f} ms")
    
    # Compare with CPU timing
    print("\n3. CPU vs GPU Timing Comparison:")
    
    def test_operation():
        return cp.linalg.norm(cp.random.randn(1000, 1000))
    
    comparison = EventTimingAnalyzer.compare_timing_methods(test_operation, 50)
    
    print(f"   CPU Timing: {comparison['cpu_timing']['mean_ms']:.3f} ± "
          f"{comparison['cpu_timing']['std_ms']:.3f} ms")
    print(f"   GPU Timing: {comparison['gpu_timing']['mean_ms']:.3f} ± "
          f"{comparison['gpu_timing']['std_ms']:.3f} ms")
    print(f"   Sync Overhead: {comparison['overhead_ms']:.3f} ms "
          f"({comparison['overhead_percent']:.1f}%)")


def demonstrate_profiler():
    """Demonstrate the GPU profiler."""
    print("\n" + "="*70)
    print("GPU PROFILER DEMONSTRATION")
    print("="*70)
    
    profiler = GPUProfiler(warmup_runs=2)
    
    size = 5_000_000
    
    for i in range(10):
        # Generate data
        with profiler.region("generate_data"):
            data = cp.random.randn(size, dtype=cp.float32)
        
        # Transform 1
        with profiler.region("transform_sin"):
            result1 = cp.sin(data)
        
        # Transform 2
        with profiler.region("transform_exp"):
            result2 = cp.exp(-cp.abs(result1))
        
        # Reduction
        with profiler.region("reduction"):
            final = cp.sum(result2)
        
        # Force sync for accurate timing
        cp.cuda.Stream.null.synchronize()
    
    profiler.print_summary()


def demonstrate_inter_stream_events():
    """Demonstrate inter-stream synchronization with events."""
    print("\n" + "="*70)
    print("INTER-STREAM EVENT SYNCHRONIZATION")
    print("="*70)
    
    size = 2_000_000
    
    # Create streams
    stream_producer = cp.cuda.Stream(non_blocking=True)
    stream_consumer1 = cp.cuda.Stream(non_blocking=True)
    stream_consumer2 = cp.cuda.Stream(non_blocking=True)
    
    # Create events
    data_ready_event = cp.cuda.Event()
    consumer1_done = cp.cuda.Event()
    consumer2_done = cp.cuda.Event()
    
    # Overall timing
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    
    # Producer: Generate data
    with stream_producer:
        data = cp.random.randn(size, dtype=cp.float32)
        data_ready_event.record(stream_producer)
    
    # Consumer 1: Wait for data, then process
    stream_consumer1.wait_event(data_ready_event)
    with stream_consumer1:
        result1 = cp.sin(data)
        consumer1_done.record(stream_consumer1)
    
    # Consumer 2: Wait for data, then process differently
    stream_consumer2.wait_event(data_ready_event)
    with stream_consumer2:
        result2 = cp.cos(data)
        consumer2_done.record(stream_consumer2)
    
    # Wait for both consumers
    consumer1_done.synchronize()
    consumer2_done.synchronize()
    
    # Combine results (could be in another stream)
    final_result = result1 ** 2 + result2 ** 2
    
    end.record()
    end.synchronize()
    
    total_time = cp.cuda.get_elapsed_time(start, end)
    
    print(f"\n   Producer-Consumer Pattern with Events:")
    print(f"   Data size: {size:,} elements")
    print(f"   Total time: {total_time:.3f} ms")
    print(f"   Result verification: max value = {float(cp.max(final_result)):.6f}")
    print(f"   (Should be ~1.0 since sin²(x) + cos²(x) = 1)")


def benchmark_event_overhead():
    """Measure the overhead of CUDA events themselves."""
    print("\n" + "="*70)
    print("EVENT OVERHEAD MEASUREMENT")
    print("="*70)
    
    iterations = 1000
    
    # Measure overhead of event creation
    import time
    
    start = time.perf_counter()
    events = [cp.cuda.Event() for _ in range(iterations)]
    creation_time = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n   Event creation overhead: {creation_time * 1000:.3f} μs per event")
    
    # Measure overhead of record/query
    stream = cp.cuda.Stream()
    event = cp.cuda.Event()
    
    record_times = []
    for _ in range(iterations):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        event.record(stream)
        record_times.append((time.perf_counter() - start) * 1000)
    
    print(f"   Event record overhead: {statistics.mean(record_times) * 1000:.3f} μs")
    
    # Measure timing call overhead
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    
    timing_times = []
    for _ in range(iterations):
        start_event.record()
        stop_event.record()
        stop_event.synchronize()
        
        t_start = time.perf_counter()
        _ = cp.cuda.get_elapsed_time(start_event, stop_event)
        timing_times.append((time.perf_counter() - t_start) * 1000)
    
    print(f"   Elapsed time call overhead: {statistics.mean(timing_times) * 1000:.3f} μs")


def main():
    """Main function running all demonstrations."""
    print("="*70)
    print("DAY 9: CUDA EVENTS AND TIMING")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*70)
    
    print(f"\nGPU: {cp.cuda.Device().name}")
    
    demonstrate_event_basics()
    demonstrate_profiler()
    demonstrate_inter_stream_events()
    benchmark_event_overhead()
    
    print("\n" + "="*70)
    print("Day 9 Complete: CUDA Events and Timing")
    print("="*70)


if __name__ == "__main__":
    main()
```

#### 📁 `src/benchmark_framework.py` - Production Benchmarking
```python
#!/usr/bin/env python3
"""
Day 9: Benchmarking Framework
Phase 6: AI/ML Platform Engineering with GPU Programming

A production-quality benchmarking framework using CUDA events
for accurate GPU performance measurement.
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Optional, Tuple
import statistics
import json
from datetime import datetime
import platform


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    warmup_iterations: int = 5
    timed_iterations: int = 100
    sync_before_timing: bool = True
    sync_after_each: bool = False
    clear_cache: bool = False


@dataclass
class BenchmarkResult:
    """Complete results from a benchmark run."""
    name: str
    config: BenchmarkConfig
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    samples: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'median_ms': self.median_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'iterations': len(self.samples),
            'warmup': self.config.warmup_iterations,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
                f"(p95: {self.p95_ms:.3f}, p99: {self.p99_ms:.3f})")


class GPUBenchmark:
    """
    Production-quality GPU benchmarking framework.
    
    Features:
    - Warmup iterations to handle JIT compilation and cache warming
    - Statistical analysis with percentiles
    - Automatic GPU synchronization
    - Metadata collection (GPU info, driver version, etc.)
    - JSON export for CI/CD integration
    
    Usage:
        benchmark = GPUBenchmark()
        
        @benchmark.register("matmul_1024")
        def matmul_benchmark():
            A = cp.random.randn(1024, 1024, dtype=cp.float32)
            B = cp.random.randn(1024, 1024, dtype=cp.float32)
            return A @ B
        
        results = benchmark.run_all()
        benchmark.save_results("benchmark_results.json")
    """
    
    def __init__(self, default_config: Optional[BenchmarkConfig] = None):
        """
        Initialize the benchmark framework.
        
        Args:
            default_config: Default configuration for all benchmarks
        """
        self.default_config = default_config or BenchmarkConfig(
            name="default",
            warmup_iterations=5,
            timed_iterations=100
        )
        
        self.benchmarks: Dict[str, Tuple[Callable, BenchmarkConfig]] = {}
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Collect system metadata
        self.metadata = self._collect_metadata()
    
    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect system and GPU metadata."""
        device = cp.cuda.Device()
        props = device.attributes
        
        return {
            'gpu_name': device.name,
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            'total_memory_gb': device.mem_info[1] / (1024**3),
            'cuda_version': cp.cuda.runtime.runtimeGetVersion(),
            'cupy_version': cp.__version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        }
    
    def register(self, name: str, config: Optional[BenchmarkConfig] = None):
        """
        Decorator to register a benchmark function.
        
        Args:
            name: Name of the benchmark
            config: Optional custom configuration
        """
        def decorator(func: Callable):
            cfg = config or BenchmarkConfig(name=name, 
                warmup_iterations=self.default_config.warmup_iterations,
                timed_iterations=self.default_config.timed_iterations)
            self.benchmarks[name] = (func, cfg)
            return func
        return decorator
    
    def add_benchmark(self, name: str, func: Callable, 
                      config: Optional[BenchmarkConfig] = None) -> None:
        """
        Add a benchmark programmatically.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            config: Optional configuration
        """
        cfg = config or BenchmarkConfig(name=name,
            warmup_iterations=self.default_config.warmup_iterations,
            timed_iterations=self.default_config.timed_iterations)
        self.benchmarks[name] = (func, cfg)
    
    def run(self, name: str) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Name of the benchmark to run
            
        Returns:
            BenchmarkResult with timing statistics
        """
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        func, config = self.benchmarks[name]
        
        print(f"Running benchmark: {name}")
        print(f"  Warmup: {config.warmup_iterations} iterations")
        print(f"  Timed: {config.timed_iterations} iterations")
        
        # Warmup
        for _ in range(config.warmup_iterations):
            if config.sync_before_timing:
                cp.cuda.Stream.null.synchronize()
            _ = func()
            if config.sync_after_each:
                cp.cuda.Stream.null.synchronize()
        
        # Clear memory if requested
        if config.clear_cache:
            cp.get_default_memory_pool().free_all_blocks()
        
        # Timed iterations
        samples = []
        start_event = cp.cuda.Event()
        stop_event = cp.cuda.Event()
        
        for _ in range(config.timed_iterations):
            if config.sync_before_timing:
                cp.cuda.Stream.null.synchronize()
            
            start_event.record()
            _ = func()
            stop_event.record()
            
            stop_event.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start_event, stop_event))
        
        # Calculate statistics
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        result = BenchmarkResult(
            name=name,
            config=config,
            mean_ms=statistics.mean(samples),
            std_ms=statistics.stdev(samples) if n > 1 else 0.0,
            min_ms=min(samples),
            max_ms=max(samples),
            median_ms=statistics.median(samples),
            p95_ms=sorted_samples[int(n * 0.95)],
            p99_ms=sorted_samples[int(n * 0.99)],
            samples=samples,
            metadata=self.metadata
        )
        
        self.results[name] = result
        print(f"  Result: {result.summary()}")
        
        return result
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run all registered benchmarks."""
        print("\n" + "="*70)
        print("RUNNING ALL BENCHMARKS")
        print("="*70)
        print(f"GPU: {self.metadata['gpu_name']}")
        print(f"CUDA: {self.metadata['cuda_version']}")
        print("")
        
        for name in self.benchmarks:
            self.run(name)
            print("")
        
        return self.results
    
    def compare(self, baseline_name: str, 
                comparison_names: Optional[List[str]] = None) -> None:
        """
        Compare benchmark results against a baseline.
        
        Args:
            baseline_name: Name of the baseline benchmark
            comparison_names: Names to compare (all if None)
        """
        if baseline_name not in self.results:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")
        
        baseline = self.results[baseline_name]
        comparisons = comparison_names or [n for n in self.results if n != baseline_name]
        
        print("\n" + "="*70)
        print(f"COMPARISON (baseline: {baseline_name})")
        print("="*70)
        print(f"{'Benchmark':<30} {'Time (ms)':>12} {'vs Baseline':>12} {'Speedup':>10}")
        print("-"*70)
        
        print(f"{baseline_name:<30} {baseline.mean_ms:>12.3f} {'(baseline)':>12} {1.0:>10.2f}x")
        
        for name in comparisons:
            if name in self.results:
                result = self.results[name]
                diff = result.mean_ms - baseline.mean_ms
                speedup = baseline.mean_ms / result.mean_ms
                sign = '+' if diff > 0 else ''
                print(f"{name:<30} {result.mean_ms:>12.3f} {sign}{diff:>11.3f} {speedup:>10.2f}x")
        
        print("="*70)
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        output = {
            'metadata': self.metadata,
            'results': {name: result.to_dict() for name, result in self.results.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def print_summary(self) -> None:
        """Print summary of all results."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Benchmark':<35} {'Mean (ms)':>10} {'Std':>10} {'P95':>10} {'P99':>10}")
        print("-"*70)
        
        for name, result in sorted(self.results.items()):
            print(f"{name:<35} {result.mean_ms:>10.3f} {result.std_ms:>10.3f} "
                  f"{result.p95_ms:>10.3f} {result.p99_ms:>10.3f}")
        
        print("="*70)


def demo_benchmark_framework():
    """Demonstrate the benchmarking framework."""
    print("\n" + "="*70)
    print("BENCHMARK FRAMEWORK DEMONSTRATION")
    print("="*70)
    
    benchmark = GPUBenchmark()
    
    # Register benchmarks
    @benchmark.register("vector_add_1M")
    def vector_add_small():
        a = cp.random.randn(1_000_000, dtype=cp.float32)
        b = cp.random.randn(1_000_000, dtype=cp.float32)
        return a + b
    
    @benchmark.register("vector_add_10M")
    def vector_add_large():
        a = cp.random.randn(10_000_000, dtype=cp.float32)
        b = cp.random.randn(10_000_000, dtype=cp.float32)
        return a + b
    
    @benchmark.register("matmul_512")
    def matmul_small():
        a = cp.random.randn(512, 512, dtype=cp.float32)
        b = cp.random.randn(512, 512, dtype=cp.float32)
        return a @ b
    
    @benchmark.register("matmul_2048")
    def matmul_large():
        a = cp.random.randn(2048, 2048, dtype=cp.float32)
        b = cp.random.randn(2048, 2048, dtype=cp.float32)
        return a @ b
    
    # Run all benchmarks
    benchmark.run_all()
    
    # Print summary
    benchmark.print_summary()
    
    # Compare matrix multiplication sizes
    benchmark.compare("matmul_512")


if __name__ == "__main__":
    demo_benchmark_framework()
```

---

## 🔬 Lab Exercise: "Building a Multi-Kernel Profiler"

### Lab Objectives
1. Create a profiler that tracks multiple kernels across streams
2. Visualize kernel execution timeline
3. Detect and measure overlap between operations

### Step-by-Step Instructions

#### Step 1: Setup
```bash
cd day_009_cuda_events
mkdir -p lab
pip install matplotlib pandas
```

#### Step 2: Implement Timeline Profiler

```python
#!/usr/bin/env python3
"""
Lab: Multi-Kernel Timeline Profiler
Day 9: CUDA Events and Timing
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class KernelEvent:
    """Record of a kernel execution event."""
    name: str
    stream_id: int
    start_ms: float
    end_ms: float
    color: str = "blue"
    
    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class TimelineProfiler:
    """
    Profiler that records kernel execution timelines across streams.
    
    TODO: Students implement this class
    """
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        self.events: List[KernelEvent] = []
        
        # Global start event for timeline reference
        self.global_start = cp.cuda.Event()
        self.global_start.record()
        
        # Track events per stream
        self.stream_events: Dict[int, List[Tuple[cp.cuda.Event, cp.cuda.Event, str]]] = {
            i: [] for i in range(num_streams)
        }
        
        # Colors for different kernel types
        self.colors = {
            'transfer_h2d': '#3498db',  # Blue
            'transfer_d2h': '#e74c3c',  # Red
            'compute': '#2ecc71',       # Green
            'reduction': '#9b59b6',     # Purple
        }
    
    def record_kernel(self, name: str, stream_id: int, kernel_type: str = 'compute'):
        """
        Context manager to record a kernel execution.
        
        Args:
            name: Display name for the kernel
            stream_id: Which stream this runs on
            kernel_type: Type for coloring (transfer_h2d, transfer_d2h, compute, reduction)
        """
        class KernelRecorder:
            def __init__(recorder_self, profiler):
                recorder_self.profiler = profiler
                recorder_self.start_event = cp.cuda.Event()
                recorder_self.end_event = cp.cuda.Event()
                recorder_self.name = name
                recorder_self.stream_id = stream_id
                recorder_self.kernel_type = kernel_type
            
            def __enter__(recorder_self):
                stream = recorder_self.profiler.streams[stream_id]
                recorder_self.start_event.record(stream)
                return recorder_self
            
            def __exit__(recorder_self, *args):
                stream = recorder_self.profiler.streams[stream_id]
                recorder_self.end_event.record(stream)
                
                # Store for later processing
                recorder_self.profiler.stream_events[stream_id].append(
                    (recorder_self.start_event, recorder_self.end_event, 
                     name, kernel_type)
                )
        
        return KernelRecorder(self)
    
    def synchronize_and_process(self) -> List[KernelEvent]:
        """
        Synchronize all streams and process recorded events.
        
        Returns:
            List of KernelEvent objects with resolved timings
        """
        # Sync all streams
        for stream in self.streams:
            stream.synchronize()
        
        self.events = []
        
        for stream_id, event_list in self.stream_events.items():
            for start, end, name, kernel_type in event_list:
                end.synchronize()
                
                start_ms = cp.cuda.get_elapsed_time(self.global_start, start)
                end_ms = cp.cuda.get_elapsed_time(self.global_start, end)
                
                self.events.append(KernelEvent(
                    name=name,
                    stream_id=stream_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    color=self.colors.get(kernel_type, '#95a5a6')
                ))
        
        return self.events
    
    def plot_timeline(self, save_path: Optional[str] = None) -> None:
        """
        Plot the kernel execution timeline.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.events:
            self.synchronize_and_process()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot each event as a horizontal bar
        for event in self.events:
            ax.barh(
                y=event.stream_id,
                width=event.duration_ms,
                left=event.start_ms,
                height=0.6,
                color=event.color,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add label if bar is wide enough
            if event.duration_ms > 1:
                ax.text(
                    event.start_ms + event.duration_ms / 2,
                    event.stream_id,
                    event.name,
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold'
                )
        
        # Formatting
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Stream')
        ax.set_yticks(range(self.num_streams))
        ax.set_yticklabels([f'Stream {i}' for i in range(self.num_streams)])
        ax.set_title('GPU Kernel Execution Timeline')
        
        # Legend
        legend_patches = [
            mpatches.Patch(color=color, label=name.replace('_', ' ').title())
            for name, color in self.colors.items()
        ]
        ax.legend(handles=legend_patches, loc='upper right')
        
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Timeline saved to: {save_path}")
        else:
            plt.show()
    
    def print_statistics(self) -> None:
        """Print timing statistics."""
        if not self.events:
            self.synchronize_and_process()
        
        print("\n" + "="*60)
        print("TIMELINE STATISTICS")
        print("="*60)
        
        # Total duration
        if self.events:
            total_duration = max(e.end_ms for e in self.events)
            print(f"Total duration: {total_duration:.3f} ms")
            
            # Per-stream statistics
            for stream_id in range(self.num_streams):
                stream_events = [e for e in self.events if e.stream_id == stream_id]
                if stream_events:
                    stream_time = sum(e.duration_ms for e in stream_events)
                    print(f"  Stream {stream_id}: {len(stream_events)} events, "
                          f"{stream_time:.3f} ms total")
        
        print("="*60)


def run_lab():
    """Lab demonstration."""
    print("="*60)
    print("LAB: Multi-Kernel Timeline Profiler")
    print("Day 9: CUDA Events and Timing")
    print("="*60)
    
    profiler = TimelineProfiler(num_streams=3)
    
    size = 2_000_000
    
    # Simulate a pipeline with multiple operations
    print("\nRecording pipeline execution...")
    
    # Stage 1: Data loading (Stream 0)
    with profiler.record_kernel("H2D_batch1", 0, 'transfer_h2d'):
        data1 = cp.random.randn(size, dtype=cp.float32)
    
    with profiler.record_kernel("H2D_batch2", 1, 'transfer_h2d'):
        data2 = cp.random.randn(size, dtype=cp.float32)
    
    # Stage 2: Processing
    with profiler.record_kernel("Compute_A", 0, 'compute'):
        result1 = cp.sin(data1) ** 2
    
    with profiler.record_kernel("Compute_B", 1, 'compute'):
        result2 = cp.cos(data2) ** 2
    
    # Stage 3: Reduction
    with profiler.record_kernel("Reduce", 2, 'reduction'):
        final = cp.sum(result1) + cp.sum(result2)
    
    # Process and visualize
    profiler.synchronize_and_process()
    profiler.print_statistics()
    profiler.plot_timeline("timeline.png")
    
    print("\nLab complete!")


if __name__ == "__main__":
    run_lab()
```

### Expected Output
```
============================================================
LAB: Multi-Kernel Timeline Profiler
Day 9: CUDA Events and Timing
============================================================

Recording pipeline execution...

============================================================
TIMELINE STATISTICS
============================================================
Total duration: 15.234 ms
  Stream 0: 2 events, 8.542 ms total
  Stream 1: 2 events, 7.891 ms total
  Stream 2: 1 events, 2.156 ms total
============================================================

Timeline saved to: timeline.png

Lab complete!
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### Issue 1: Negative Elapsed Time
**Symptom:** `cudaEventElapsedTime` returns negative value
**Cause:** Events recorded on different streams without proper synchronization

**Solution:**
```python
# Wrong: Events on different streams
event1.record(stream1)
event2.record(stream2)
elapsed = cp.cuda.get_elapsed_time(event1, event2)  # May be negative!

# Right: Same stream or proper synchronization
event1.record(stream1)
stream2.wait_event(event1)  # Ensure ordering
event2.record(stream2)
```

#### Issue 2: Timing Includes Sync Overhead
**Symptom:** Event timing much longer than expected
**Cause:** Including synchronization time in measurement

**Solution:**
```python
# Wrong: Sync before stop event
start.record()
kernel<<<>>>()
cp.cuda.Stream.null.synchronize()  # This is included in timing!
stop.record()

# Right: Record before sync
start.record()
kernel<<<>>>()
stop.record()
stop.synchronize()  # Sync AFTER recording stop
elapsed = cp.cuda.get_elapsed_time(start, stop)
```

---

## 📝 Daily Summary

### Key Takeaways
1. **CUDA events provide GPU-accurate timing** - More precise than CPU timing
2. **Events enable inter-stream synchronization** - Create dependencies without full sync
3. **Always record both start and stop** - Before synchronizing
4. **Event overhead is minimal** - ~0.5 μs per event operation
5. **Use events for profiling** - Build custom profilers for detailed analysis
6. **Statistical analysis matters** - Mean, std, percentiles reveal performance characteristics

### Commands Reference
```python
# Create event
event = cp.cuda.Event()

# Record event
event.record(stream)

# Synchronize (wait for event)
event.synchronize()

# Check completion (non-blocking)
is_done = event.query()

# Get elapsed time
elapsed_ms = cp.cuda.get_elapsed_time(start, stop)

# Stream waits for event
stream.wait_event(event)
```

---

**Day 9 Complete** ✅

*Next: Day 10 - Atomic Operations and Reductions - Building robust parallel algorithms!*
