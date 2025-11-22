# Lab 17.3: Condition Variables

## Objective
Coordinate threads using condition variables for efficient waiting.

## Instructions

### Step 1: The Problem - Busy Waiting
Create `condition_variables.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;
bool ready = false;

void busyWait() {
    while (true) {
        std::lock_guard<std::mutex> lock(mtx);
        if (ready) break; // Wasteful!
    }
    std::cout << "Data ready!\n";
}
```

### Step 2: Solution - Condition Variable
```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void worker() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; }); // Efficient waiting
    std::cout << "Data ready!\n";
}

void producer() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one(); // Wake up waiting thread
}

int main() {
    std::thread t1(worker);
    std::thread t2(producer);
    
    t1.join();
    t2.join();
    
    return 0;
}
```

### Step 3: Producer-Consumer Pattern
```cpp
#include <queue>
#include <condition_variable>

std::queue<int> dataQueue;
std::mutex mtx;
std::condition_variable cv;

void producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            dataQueue.push(i);
        }
        cv.notify_one();
    }
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !dataQueue.empty(); });
        
        int value = dataQueue.front();
        dataQueue.pop();
        lock.unlock();
        
        std::cout << "Consumed: " << value << "\n";
    }
}
```

## Challenges

### Challenge 1: Multiple Consumers
Implement a queue with multiple consumer threads.

### Challenge 2: Timeout
Use `wait_for` to timeout if data isn't ready.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>

// Challenge 1: Thread-safe queue with multiple consumers
template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(std::move(value));
        }
        cv.notify_one();
    }
    
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !queue.empty() || done; });
        
        if (queue.empty()) {
            return false; // Queue is done
        }
        
        value = std::move(queue.front());
        queue.pop();
        return true;
    }
    
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all(); // Wake all waiting threads
    }
};

// Challenge 2: Timeout example
class TimeoutExample {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    
public:
    void waitWithTimeout() {
        std::unique_lock<std::mutex> lock(mtx);
        
        if (cv.wait_for(lock, std::chrono::seconds(2), [this]{ return ready; })) {
            std::cout << "Data ready!\n";
        } else {
            std::cout << "Timeout!\n";
        }
    }
    
    void setReady() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            ready = true;
        }
        cv.notify_one();
    }
};

int main() {
    std::cout << "=== Multiple Consumers ===\n";
    
    ThreadSafeQueue<int> queue;
    
    // Producer
    std::thread producer([&]() {
        for (int i = 0; i < 20; ++i) {
            queue.push(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        queue.finish();
    });
    
    // Multiple consumers
    std::vector<std::thread> consumers;
    for (int i = 0; i < 3; ++i) {
        consumers.emplace_back([&, id = i]() {
            int value;
            while (queue.pop(value)) {
                std::cout << "Consumer " << id << " got: " << value << "\n";
            }
        });
    }
    
    producer.join();
    for (auto& c : consumers) {
        c.join();
    }
    
    std::cout << "\n=== Timeout Example ===\n";
    
    TimeoutExample example;
    std::thread waiter([&]() { example.waitWithTimeout(); });
    
    // Uncomment to test success case:
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // example.setReady();
    
    waiter.join();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used condition variables for efficient waiting
✅ Implemented producer-consumer pattern
✅ Created thread-safe queue (Challenge 1)
✅ Used `wait_for` with timeout (Challenge 2)

## Rust Comparison
```rust
use std::sync::{Arc, Mutex, Condvar};

let pair = Arc::new((Mutex::new(false), Condvar::new()));
let (lock, cvar) = &*pair;

let mut started = lock.lock().unwrap();
while !*started {
    started = cvar.wait(started).unwrap();
}
```

## Key Learnings
- Condition variables avoid busy waiting
- Always use with `unique_lock`
- `notify_one()` wakes one thread
- `notify_all()` wakes all waiting threads
- Use predicate to avoid spurious wakeups

## Next Steps
Proceed to **Lab 17.4: Atomic Operations**.
