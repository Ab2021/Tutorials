# Lab 17.1: Creating Threads

## Objective
Learn to create and manage threads using `std::thread`.

## Instructions

### Step 1: Basic Thread Creation
Create `basic_threads.cpp`.

```cpp
#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(hello); // Create thread
    t.join();             // Wait for completion
    
    std::cout << "Main thread\n";
    return 0;
}
```

### Step 2: Thread with Parameters
```cpp
#include <iostream>
#include <thread>

void greet(const std::string& name, int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << "Hello, " << name << "!\n";
    }
}

int main() {
    std::thread t(greet, "Alice", 3);
    t.join();
    
    return 0;
}
```

### Step 3: Lambda Threads
```cpp
#include <thread>

int main() {
    int value = 42;
    
    std::thread t([value]() {
        std::cout << "Value: " << value << "\n";
    });
    
    t.join();
    return 0;
}
```

### Step 4: Detached Threads
```cpp
#include <thread>
#include <chrono>

int main() {
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Detached thread\n";
    });
    
    t.detach(); // Thread runs independently
    
    // Must ensure main doesn't exit before detached thread
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    return 0;
}
```

## Challenges

### Challenge 1: Multiple Threads
Create multiple threads and wait for all to complete.

### Challenge 2: Thread IDs
Print the ID of each thread.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// Challenge 1: Multiple threads
void worker(int id) {
    std::cout << "Thread " << id << " starting\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));
    std::cout << "Thread " << id << " done\n";
}

void multipleThreads() {
    std::vector<std::thread> threads;
    
    // Create threads
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for all
    for (auto& t : threads) {
        t.join();
    }
}

// Challenge 2: Thread IDs
void printThreadId(int num) {
    std::cout << "Thread " << num 
              << " ID: " << std::this_thread::get_id() << "\n";
}

void threadIds() {
    std::thread t1(printThreadId, 1);
    std::thread t2(printThreadId, 2);
    
    std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n";
    
    t1.join();
    t2.join();
}

int main() {
    std::cout << "=== Multiple Threads ===\n";
    multipleThreads();
    
    std::cout << "\n=== Thread IDs ===\n";
    threadIds();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and joined threads
✅ Passed parameters to threads
✅ Used lambda functions with threads
✅ Created multiple threads (Challenge 1)
✅ Printed thread IDs (Challenge 2)

## Rust Comparison
```rust
use std::thread;

fn main() {
    let handle = thread::spawn(|| {
        println!("Hello from thread!");
    });
    
    handle.join().unwrap();
}
```

## Key Learnings
- `std::thread` creates new threads
- `join()` waits for thread completion
- `detach()` allows independent execution
- Threads can take functions, lambdas, or functors
- Always join or detach threads before destruction

## Next Steps
Proceed to **Lab 17.2: Mutexes and Locks**.
