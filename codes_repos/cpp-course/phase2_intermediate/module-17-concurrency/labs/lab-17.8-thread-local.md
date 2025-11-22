# Lab 17.8: Thread-Local Storage

## Objective
Use thread-local storage for per-thread data without synchronization.

## Instructions

### Step 1: thread_local Keyword
Create `thread_local.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <vector>

thread_local int threadId = 0; // Each thread has its own copy

void worker(int id) {
    threadId = id; // Set thread-local value
    
    std::cout << "Thread " << std::this_thread::get_id() 
              << " has threadId: " << threadId << "\n";
}

int main() {
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return 0;
}
```

### Step 2: Thread-Local Objects
```cpp
#include <string>

class ThreadContext {
public:
    std::string name;
    int counter = 0;
    
    ThreadContext() {
        std::cout << "ThreadContext created\n";
    }
    
    ~ThreadContext() {
        std::cout << "ThreadContext destroyed\n";
    }
};

thread_local ThreadContext context;

void useContext(const std::string& name) {
    context.name = name;
    context.counter++;
    
    std::cout << context.name << ": " << context.counter << "\n";
}
```

### Step 3: Thread-Local Random Generator
```cpp
#include <random>

thread_local std::mt19937 rng(std::random_device{}());

int getRandomNumber() {
    std::uniform_int_distribution<int> dist(1, 100);
    return dist(rng); // Each thread has its own RNG
}
```

## Challenges

### Challenge 1: Thread-Local Cache
Implement a per-thread cache to avoid contention.

### Challenge 2: Performance Counter
Create thread-local performance counters for profiling.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <random>

// Challenge 1: Thread-local cache
template<typename K, typename V>
class ThreadLocalCache {
    thread_local static std::unordered_map<K, V> cache;
    
public:
    static bool get(const K& key, V& value) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    static void put(const K& key, const V& value) {
        cache[key] = value;
    }
    
    static size_t size() {
        return cache.size();
    }
};

template<typename K, typename V>
thread_local std::unordered_map<K, V> ThreadLocalCache<K, V>::cache;

// Challenge 2: Performance counter
class PerformanceCounter {
    struct Stats {
        size_t operations = 0;
        std::chrono::microseconds totalTime{0};
    };
    
    thread_local static Stats stats;
    
public:
    class Timer {
        std::chrono::high_resolution_clock::time_point start;
        
    public:
        Timer() : start(std::chrono::high_resolution_clock::now()) {}
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            stats.totalTime += duration;
            ++stats.operations;
        }
    };
    
    static void report() {
        std::cout << "Thread " << std::this_thread::get_id() << ": "
                  << stats.operations << " operations, "
                  << stats.totalTime.count() << "μs total\n";
    }
};

thread_local PerformanceCounter::Stats PerformanceCounter::stats;

// Thread-local random number generator
thread_local std::mt19937 rng(std::random_device{}());

int getRandomNumber(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

void worker(int id) {
    // Use thread-local cache
    for (int i = 0; i < 10; ++i) {
        ThreadLocalCache<int, std::string>::put(i, "Value" + std::to_string(i));
    }
    
    std::cout << "Thread " << id << " cache size: " 
              << ThreadLocalCache<int, std::string>::size() << "\n";
    
    // Use performance counter
    for (int i = 0; i < 100; ++i) {
        PerformanceCounter::Timer timer;
        
        // Simulate work
        int sum = 0;
        for (int j = 0; j < 1000; ++j) {
            sum += getRandomNumber(1, 100);
        }
    }
    
    PerformanceCounter::report();
}

int main() {
    std::cout << "=== Thread-Local Storage ===\n";
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `thread_local` for per-thread data
✅ Created thread-local objects
✅ Implemented thread-local cache (Challenge 1)
✅ Built performance counters (Challenge 2)

## Rust Comparison
```rust
use std::cell::RefCell;

thread_local! {
    static COUNTER: RefCell<u32> = RefCell::new(0);
}

COUNTER.with(|c| {
    *c.borrow_mut() += 1;
});
```

## Key Learnings
- `thread_local` creates per-thread instances
- No synchronization needed for thread-local data
- Useful for caches, RNGs, and performance counters
- Each thread has its own copy
- Destroyed when thread exits

## Next Steps
Proceed to **Lab 17.9: Parallel Algorithms**.
