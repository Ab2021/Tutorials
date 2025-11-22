# Module 17: Concurrency and Multithreading

## 🎯 Learning Objectives

- Master `std::thread` for parallel execution
- Understand mutexes and locks
- Use condition variables for synchronization
- Understand atomics and memory ordering
- Use `std::async` and futures
- Avoid data races and deadlocks
- Understand thread-safe patterns
- Use thread pools
- Master `std::jthread` (C++20)
- Understand memory models

## 📖 Key Concepts

### Threads
```cpp
std::thread t([]{ std::cout << "Hello from thread\n"; });
t.join();
```

### Mutexes
```cpp
std::mutex mtx;
std::lock_guard<std::mutex> lock(mtx);
// Critical section
```

### Atomics
```cpp
std::atomic<int> counter{0};
counter++;
```

## 🦀 Rust vs C++ Comparison

**C++:** Manual synchronization, data races possible.
**Rust:** Ownership prevents data races at compile time.

## Labs

1. Thread Basics
2. Mutexes and Locks
3. Condition Variables
4. Atomics
5. Async and Futures
6. Thread Pools
7. Lock-Free Programming
8. Memory Ordering
9. Thread-Safe Queue
10. Parallel Algorithms (Capstone)

## Next Steps
Proceed to **Module 18: Build Systems**.
