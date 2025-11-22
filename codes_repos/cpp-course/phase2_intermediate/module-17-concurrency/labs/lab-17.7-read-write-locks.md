# Lab 17.7: Read-Write Locks

## Objective
Use shared mutexes for efficient concurrent reading with exclusive writing.

## Instructions

### Step 1: The Problem
Multiple readers can safely access data simultaneously, but writers need exclusive access.

Create `read_write_locks.cpp`.

```cpp
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <vector>

class ThreadSafeCounter {
    int value = 0;
    mutable std::shared_mutex mtx;
    
public:
    // Multiple readers can hold shared lock
    int read() const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        return value;
    }
    
    // Writer needs exclusive lock
    void write(int v) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        value = v;
    }
    
    void increment() {
        std::unique_lock<std::shared_mutex> lock(mtx);
        ++value;
    }
};
```

### Step 2: Using Read-Write Locks
```cpp
int main() {
    ThreadSafeCounter counter;
    
    std::vector<std::thread> readers;
    std::vector<std::thread> writers;
    
    // Multiple readers
    for (int i = 0; i < 5; ++i) {
        readers.emplace_back([&]() {
            for (int j = 0; j < 10; ++j) {
                std::cout << "Read: " << counter.read() << "\n";
            }
        });
    }
    
    // Few writers
    for (int i = 0; i < 2; ++i) {
        writers.emplace_back([&]() {
            for (int j = 0; j < 5; ++j) {
                counter.increment();
            }
        });
    }
    
    for (auto& r : readers) r.join();
    for (auto& w : writers) w.join();
    
    return 0;
}
```

### Step 3: Upgrade Lock Pattern
```cpp
class UpgradableData {
    int value = 0;
    mutable std::shared_mutex mtx;
    
public:
    void conditionalUpdate() {
        // Start with shared lock
        std::shared_lock<std::shared_mutex> readLock(mtx);
        
        if (value < 100) {
            // Upgrade to exclusive lock
            readLock.unlock();
            std::unique_lock<std::shared_mutex> writeLock(mtx);
            
            // Recheck condition (may have changed)
            if (value < 100) {
                value = 100;
            }
        }
    }
};
```

## Challenges

### Challenge 1: Cache
Implement a thread-safe cache with read-write locks.

### Challenge 2: Benchmark
Compare performance of shared_mutex vs regular mutex for read-heavy workloads.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>

// Challenge 1: Thread-safe cache
template<typename K, typename V>
class ThreadSafeCache {
    std::unordered_map<K, V> cache;
    mutable std::shared_mutex mtx;
    
public:
    bool get(const K& key, V& value) const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        auto it = cache.find(key);
        if (it != cache.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    void put(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        cache[key] = value;
    }
    
    void remove(const K& key) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        cache.erase(key);
    }
    
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        return cache.size();
    }
};

// Challenge 2: Benchmark
void benchmarkSharedMutex() {
    ThreadSafeCache<int, std::string> cache;
    
    // Populate cache
    for (int i = 0; i < 100; ++i) {
        cache.put(i, "Value" + std::to_string(i));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> readers;
    for (int i = 0; i < 10; ++i) {
        readers.emplace_back([&]() {
            for (int j = 0; j < 10000; ++j) {
                std::string value;
                cache.get(j % 100, value);
            }
        });
    }
    
    for (auto& r : readers) {
        r.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "shared_mutex: " << duration.count() << "ms\n";
}

void benchmarkRegularMutex() {
    std::unordered_map<int, std::string> cache;
    std::mutex mtx;
    
    // Populate cache
    for (int i = 0; i < 100; ++i) {
        cache[i] = "Value" + std::to_string(i);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> readers;
    for (int i = 0; i < 10; ++i) {
        readers.emplace_back([&]() {
            for (int j = 0; j < 10000; ++j) {
                std::lock_guard<std::mutex> lock(mtx);
                auto it = cache.find(j % 100);
            }
        });
    }
    
    for (auto& r : readers) {
        r.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "regular mutex: " << duration.count() << "ms\n";
}

int main() {
    std::cout << "=== Thread-Safe Cache ===\n";
    
    ThreadSafeCache<std::string, int> cache;
    
    // Writers
    std::thread w1([&]() {
        cache.put("key1", 100);
        cache.put("key2", 200);
    });
    
    std::thread w2([&]() {
        cache.put("key3", 300);
    });
    
    w1.join();
    w2.join();
    
    // Readers
    std::vector<std::thread> readers;
    for (int i = 0; i < 5; ++i) {
        readers.emplace_back([&, id = i]() {
            int value;
            if (cache.get("key1", value)) {
                std::cout << "Reader " << id << " got: " << value << "\n";
            }
        });
    }
    
    for (auto& r : readers) {
        r.join();
    }
    
    std::cout << "\n=== Benchmark ===\n";
    benchmarkSharedMutex();
    benchmarkRegularMutex();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `shared_mutex` for concurrent reads
✅ Applied exclusive locks for writes
✅ Implemented thread-safe cache (Challenge 1)
✅ Benchmarked shared vs regular mutex (Challenge 2)

## Rust Comparison
```rust
use std::sync::RwLock;

let lock = RwLock::new(5);

// Multiple readers
let r1 = lock.read().unwrap();
let r2 = lock.read().unwrap();

// Exclusive writer
let mut w = lock.write().unwrap();
*w += 1;
```

## Key Learnings
- `shared_mutex` allows concurrent reads
- `shared_lock` for reading, `unique_lock` for writing
- Improves performance for read-heavy workloads
- Writers still need exclusive access
- C++17 feature

## Next Steps
Proceed to **Lab 17.8: Thread-Local Storage**.
