# Lab 17.5: Futures and Promises

## Objective
Use futures and promises for asynchronous result retrieval.

## Instructions

### Step 1: Basic Promise and Future
Create `futures.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <future>

void computeValue(std::promise<int> prom) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    prom.set_value(42); // Set result
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    
    std::thread t(computeValue, std::move(prom));
    
    std::cout << "Waiting for result...\n";
    int result = fut.get(); // Block until ready
    std::cout << "Result: " << result << "\n";
    
    t.join();
    return 0;
}
```

### Step 2: async for Easy Async
```cpp
#include <future>

int compute() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 42;
}

int main() {
    // Launch async task
    std::future<int> fut = std::async(std::launch::async, compute);
    
    std::cout << "Doing other work...\n";
    
    int result = fut.get(); // Get result
    std::cout << "Result: " << result << "\n";
    
    return 0;
}
```

### Step 3: Exception Handling
```cpp
#include <future>

void mayThrow(std::promise<int> prom) {
    try {
        throw std::runtime_error("Error!");
    } catch (...) {
        prom.set_exception(std::current_exception());
    }
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    
    std::thread t(mayThrow, std::move(prom));
    
    try {
        int result = fut.get(); // Rethrows exception
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
    
    t.join();
    return 0;
}
```

### Step 4: Shared Future
```cpp
#include <future>

int main() {
    std::promise<int> prom;
    std::shared_future<int> fut = prom.get_future().share();
    
    // Multiple threads can wait on shared_future
    auto waiter = [fut]() {
        std::cout << "Result: " << fut.get() << "\n";
    };
    
    std::thread t1(waiter);
    std::thread t2(waiter);
    
    prom.set_value(42);
    
    t1.join();
    t2.join();
    
    return 0;
}
```

## Challenges

### Challenge 1: Parallel Computation
Use `std::async` to compute multiple values in parallel.

### Challenge 2: Timeout
Use `wait_for` to timeout if result isn't ready.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <future>
#include <vector>
#include <numeric>
#include <chrono>

// Challenge 1: Parallel computation
int sumRange(int start, int end) {
    int sum = 0;
    for (int i = start; i < end; ++i) {
        sum += i;
    }
    return sum;
}

void parallelSum() {
    const int total = 1000000;
    const int numTasks = 4;
    const int chunkSize = total / numTasks;
    
    std::vector<std::future<int>> futures;
    
    // Launch parallel tasks
    for (int i = 0; i < numTasks; ++i) {
        int start = i * chunkSize;
        int end = (i + 1) * chunkSize;
        futures.push_back(std::async(std::launch::async, sumRange, start, end));
    }
    
    // Collect results
    int totalSum = 0;
    for (auto& fut : futures) {
        totalSum += fut.get();
    }
    
    std::cout << "Parallel sum: " << totalSum << "\n";
}

// Challenge 2: Timeout
void timeoutExample() {
    auto slowCompute = []() {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        return 42;
    };
    
    std::future<int> fut = std::async(std::launch::async, slowCompute);
    
    // Wait with timeout
    auto status = fut.wait_for(std::chrono::seconds(2));
    
    if (status == std::future_status::ready) {
        std::cout << "Result: " << fut.get() << "\n";
    } else if (status == std::future_status::timeout) {
        std::cout << "Timeout!\n";
    } else {
        std::cout << "Deferred\n";
    }
}

// Advanced: Multiple async operations
void multipleAsync() {
    auto task1 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return "Task 1";
    });
    
    auto task2 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return "Task 2";
    });
    
    auto task3 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return "Task 3";
    });
    
    std::cout << task1.get() << " done\n";
    std::cout << task2.get() << " done\n";
    std::cout << task3.get() << " done\n";
}

int main() {
    std::cout << "=== Parallel Sum ===\n";
    parallelSum();
    
    std::cout << "\n=== Timeout Example ===\n";
    timeoutExample();
    
    std::cout << "\n=== Multiple Async ===\n";
    multipleAsync();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used promises and futures for async results
✅ Launched tasks with `std::async`
✅ Handled exceptions across threads
✅ Computed values in parallel (Challenge 1)
✅ Used `wait_for` with timeout (Challenge 2)

## Rust Comparison
```rust
use std::thread;

let handle = thread::spawn(|| {
    42
});

let result = handle.join().unwrap();
```

## Key Learnings
- Futures represent future results
- Promises set values for futures
- `std::async` simplifies async programming
- `get()` blocks until result is ready
- Exceptions propagate through futures

## Next Steps
Proceed to **Lab 17.6: Thread Pools**.
