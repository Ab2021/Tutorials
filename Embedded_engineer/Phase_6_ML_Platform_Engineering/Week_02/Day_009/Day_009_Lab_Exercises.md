# Day 9: CUDA Events and Timing - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Precise Kernel Timing (Beginner)

### Objective
Use CUDA events for accurate GPU timing.

### Code

```python
import torch

def event_timing():
    """Demonstrate CUDA event timing."""
    
    print("CUDA Event Timing Demo")
    print("=" * 40)
    
    # Create events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Prepare data
    a = torch.randn(5000, 5000, device='cuda')
    b = torch.randn(5000, 5000, device='cuda')
    
    # Warm up
    c = a @ b
    torch.cuda.synchronize()
    
    # Time the operation
    start_event.record()
    c = a @ b
    end_event.record()
    
    # Wait for completion
    torch.cuda.synchronize()
    
    # Get elapsed time in milliseconds
    elapsed = start_event.elapsed_time(end_event)
    print(f"Matrix multiply (5000x5000): {elapsed:.3f} ms")
    
    # Calculate FLOPS
    flops = 2 * 5000 * 5000 * 5000  # 2*N^3 for matmul
    tflops = flops / (elapsed / 1000) / 1e12
    print(f"Performance: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    event_timing()
```

---

## 🔬 Exercise 2: Multi-Operation Timing (Intermediate)

### Objective
Time multiple operations and identify bottlenecks.

### Code

```python
import torch

class GpuTimer:
    """Reusable GPU timer using CUDA events."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.timings = {}
    
    def start(self):
        self.start_event.record()
    
    def stop(self, name):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)
        self.timings[name] = elapsed
        return elapsed
    
    def report(self):
        print("\nTiming Report")
        print("=" * 40)
        total = sum(self.timings.values())
        for name, time_ms in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = time_ms / total * 100
            print(f"{name:20s}: {time_ms:8.3f} ms ({pct:5.1f}%)")
        print("-" * 40)
        print(f"{'Total':20s}: {total:8.3f} ms")

def profile_pipeline():
    """Profile a typical ML pipeline."""
    
    timer = GpuTimer()
    batch_size = 64
    seq_len = 512
    hidden = 768
    
    # Simulate transformer layer operations
    x = torch.randn(batch_size, seq_len, hidden, device='cuda')
    
    # Linear projection
    W = torch.randn(hidden, hidden * 3, device='cuda')
    timer.start()
    qkv = x @ W
    timer.stop("Linear (QKV)")
    
    # Reshape for attention
    timer.start()
    q, k, v = qkv.chunk(3, dim=-1)
    timer.stop("Chunk")
    
    # Attention scores
    timer.start()
    scores = torch.bmm(q.view(-1, seq_len, hidden), 
                       k.view(-1, seq_len, hidden).transpose(1, 2))
    timer.stop("Attention Scores")
    
    # Softmax
    timer.start()
    attn = torch.softmax(scores, dim=-1)
    timer.stop("Softmax")
    
    # Output
    timer.start()
    out = torch.bmm(attn, v.view(-1, seq_len, hidden))
    timer.stop("Attention Output")
    
    timer.report()

if __name__ == "__main__":
    profile_pipeline()
```

---

## 🔬 Exercise 3: Inter-Stream Synchronization (Advanced)

### Objective
Use events to synchronize between streams.

### Code

```python
import torch
import time

def stream_sync_with_events():
    """Synchronize streams using events."""
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    sync_event = torch.cuda.Event()
    
    a = torch.randn(3000, 3000, device='cuda')
    
    print("Stream Synchronization Demo")
    print("=" * 40)
    
    # Stream 1: Compute
    with torch.cuda.stream(stream1):
        b = a @ a
        b = b @ a
        sync_event.record()  # Signal completion
        print("Stream1: Computation queued")
    
    # Stream 2: Wait for stream1, then continue
    with torch.cuda.stream(stream2):
        sync_event.wait()  # Wait for stream1
        c = b + 1  # Use result from stream1
        print("Stream2: Using stream1 result")
    
    torch.cuda.synchronize()
    print("All complete!")

if __name__ == "__main__":
    stream_sync_with_events()
```

---

## 🐛 Common Issues

### Issue: Incorrect timing
**Cause:** Not synchronizing before reading elapsed time
**Fix:** Always call `torch.cuda.synchronize()` before `elapsed_time()`

### Issue: Zero elapsed time
**Cause:** Kernel hasn't finished
**Fix:** Call `end_event.synchronize()` or `torch.cuda.synchronize()`

---

## 📚 Additional Resources
- [CUDA Events Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)
