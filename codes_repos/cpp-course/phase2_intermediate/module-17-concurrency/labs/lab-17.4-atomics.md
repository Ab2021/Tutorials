# Lab 17.4: Atomic Operations

## Objective
Use atomic operations for lock-free programming and simple synchronization.

## Instructions

### Step 1: Atomic Counter
Create `atomics.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

std::atomic<int> counter(0); // Atomic variable

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter; // Thread-safe without mutex!
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(increment);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Counter: " << counter << "\n"; // Correct: 1000000
    return 0;
}
```

### Step 2: Atomic Operations
```cpp
#include <atomic>

std::atomic<int> value(0);

void atomicOps() {
    value.store(42);           // Atomic write
    int x = value.load();      // Atomic read
    value.fetch_add(10);       // Atomic add
    value.fetch_sub(5);        // Atomic subtract
    
    int expected = 47;
    bool success = value.compare_exchange_strong(expected, 100);
    // If value == expected, set to 100; otherwise update expected
}
```

### Step 3: Memory Ordering
```cpp
#include <atomic>

std::atomic<int> x(0), y(0);

// Relaxed ordering (fastest, least synchronization)
void relaxed() {
    x.store(1, std::memory_order_relaxed);
    int r = y.load(std::memory_order_relaxed);
}

// Acquire-release ordering
void acquireRelease() {
    x.store(1, std::memory_order_release);
    int r = y.load(std::memory_order_acquire);
}

// Sequential consistency (default, strongest)
void sequential() {
    x.store(1, std::memory_order_seq_cst);
    int r = y.load(std::memory_order_seq_cst);
}
```

### Step 4: Atomic Flag (Spinlock)
```cpp
#include <atomic>

class Spinlock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // Spin
        }
    }
    
    void unlock() {
        flag.clear(std::memory_order_release);
    }
};
```

## Challenges

### Challenge 1: Lock-Free Stack
Implement a simple lock-free stack using atomics.

### Challenge 2: Benchmark
Compare atomic operations vs mutex for a simple counter.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <mutex>

// Challenge 1: Lock-free stack
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };
    
    std::atomic<Node*> head{nullptr};
    
public:
    void push(const T& data) {
        Node* newNode = new Node(data);
        newNode->next = head.load();
        
        while (!head.compare_exchange_weak(newNode->next, newNode)) {
            // Retry if head changed
        }
    }
    
    bool pop(T& result) {
        Node* oldHead = head.load();
        
        while (oldHead && !head.compare_exchange_weak(oldHead, oldHead->next)) {
            // Retry if head changed
        }
        
        if (oldHead) {
            result = oldHead->data;
            delete oldHead;
            return true;
        }
        return false;
    }
};

// Challenge 2: Benchmark
void benchmarkAtomic() {
    std::atomic<int> counter(0);
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 100000; ++j) {
                ++counter;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Atomic: " << duration.count() << "ms, "
              << "Counter: " << counter << "\n";
}

void benchmarkMutex() {
    int counter = 0;
    std::mutex mtx;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 100000; ++j) {
                std::lock_guard<std::mutex> lock(mtx);
                ++counter;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Mutex: " << duration.count() << "ms, "
              << "Counter: " << counter << "\n";
}

int main() {
    std::cout << "=== Lock-Free Stack ===\n";
    
    LockFreeStack<int> stack;
    
    std::thread producer([&]() {
        for (int i = 0; i < 10; ++i) {
            stack.push(i);
        }
    });
    
    std::thread consumer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int value;
        while (stack.pop(value)) {
            std::cout << "Popped: " << value << "\n";
        }
    });
    
    producer.join();
    consumer.join();
    
    std::cout << "\n=== Benchmark ===\n";
    benchmarkAtomic();
    benchmarkMutex();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used atomic variables for thread-safe operations
✅ Understood memory ordering
✅ Implemented lock-free stack (Challenge 1)
✅ Benchmarked atomic vs mutex (Challenge 2)

## Rust Comparison
```rust
use std::sync::atomic::{AtomicUsize, Ordering};

let counter = AtomicUsize::new(0);
counter.fetch_add(1, Ordering::SeqCst);
```

## Key Learnings
- Atomics provide lock-free synchronization
- Faster than mutexes for simple operations
- Memory ordering controls synchronization strength
- `compare_exchange` enables lock-free algorithms
- Use atomics for counters, flags, and simple state

## Next Steps
Proceed to **Lab 17.5: Futures and Promises**.
