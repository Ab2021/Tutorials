# Module 23: Advanced Concurrency

## Overview
Advanced parallel programming techniques including lock-free data structures, memory ordering, and high-performance concurrent algorithms.

## Learning Objectives
By the end of this module, you will be able to:
- Implement lock-free data structures
- Understand memory ordering semantics
- Write wait-free algorithms
- Use concurrent data structures
- Apply task-based parallelism
- Understand GPU programming basics

## Key Concepts

### 1. Lock-Free Programming
Data structures that don't use locks.

```cpp
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
    };
    
    std::atomic<Node*> head{nullptr};
    
public:
    void push(const T& data) {
        Node* newNode = new Node{data, head.load()};
        while (!head.compare_exchange_weak(newNode->next, newNode));
    }
    
    bool pop(T& result) {
        Node* oldHead = head.load();
        while (oldHead && !head.compare_exchange_weak(oldHead, oldHead->next));
        if (oldHead) {
            result = oldHead->data;
            delete oldHead;
            return true;
        }
        return false;
    }
};
```

### 2. Memory Ordering
Fine-grained control over memory synchronization.

```cpp
std::atomic<int> x{0}, y{0};

// Thread 1
x.store(1, std::memory_order_release);

// Thread 2
while (x.load(std::memory_order_acquire) == 0);
y.store(1, std::memory_order_relaxed);
```

### 3. Wait-Free Algorithms
Algorithms that guarantee progress for all threads.

```cpp
class WaitFreeCounter {
    std::atomic<int> count{0};
public:
    void increment() {
        count.fetch_add(1, std::memory_order_relaxed);
    }
    
    int get() const {
        return count.load(std::memory_order_relaxed);
    }
};
```

### 4. Task-Based Parallelism
Using task abstractions instead of threads.

```cpp
#include <execution>

std::vector<int> data(1000000);
std::for_each(std::execution::par, data.begin(), data.end(),
    [](int& x) { x = process(x); });
```

### 5. GPU Programming
Basics of GPU acceleration.

```cpp
// CUDA example
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

## Rust Comparison

### Atomics
**C++:**
```cpp
std::atomic<int> counter{0};
counter.fetch_add(1);
```

**Rust:**
```rust
use std::sync::atomic::{AtomicUsize, Ordering};
let counter = AtomicUsize::new(0);
counter.fetch_add(1, Ordering::SeqCst);
```

### Lock-Free Structures
**C++:**
```cpp
LockFreeStack<int> stack;
```

**Rust:**
```rust
use crossbeam::queue::SegQueue;
let queue = SegQueue::new();
```

## Labs

1. **Lab 23.1**: Lock-Free Stack
2. **Lab 23.2**: Lock-Free Queue
3. **Lab 23.3**: Memory Ordering Deep Dive
4. **Lab 23.4**: ABA Problem Solutions
5. **Lab 23.5**: Wait-Free Algorithms
6. **Lab 23.6**: Concurrent Hash Map
7. **Lab 23.7**: Task-Based Parallelism
8. **Lab 23.8**: Work Stealing
9. **Lab 23.9**: GPU Programming Basics
10. **Lab 23.10**: Parallel Framework (Capstone)

## Additional Resources
- "C++ Concurrency in Action" by Anthony Williams
- "The Art of Multiprocessor Programming"
- Intel Threading Building Blocks (TBB)

## Next Module
After completing this module, proceed to **Module 24: Networking**.
