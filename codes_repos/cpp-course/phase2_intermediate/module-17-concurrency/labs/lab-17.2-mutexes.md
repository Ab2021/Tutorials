# Lab 17.2: Mutexes and Locks

## Objective
Protect shared data with mutexes and RAII lock guards.

## Instructions

### Step 1: Race Condition Problem
Create `mutexes.cpp`.

```cpp
#include <iostream>
#include <thread>
#include <vector>

int counter = 0; // Shared data

void increment() {
    for (int i = 0; i < 100000; ++i) {
        ++counter; // Race condition!
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
    
    std::cout << "Counter: " << counter << "\n"; // Wrong result!
    return 0;
}
```

### Step 2: Fix with Mutex
```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        mtx.lock();
        ++counter;
        mtx.unlock();
    }
}
```

### Step 3: RAII with lock_guard
```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // RAII
        ++counter;
    } // Automatically unlocked
}
```

### Step 4: unique_lock for Flexibility
```cpp
#include <mutex>

std::mutex mtx;

void flexibleLocking() {
    std::unique_lock<std::mutex> lock(mtx);
    
    // Can unlock manually
    lock.unlock();
    
    // Do work without lock
    
    // Relock
    lock.lock();
}
```

## Challenges

### Challenge 1: Scoped Locking
Implement a function that locks only a critical section.

### Challenge 2: Deadlock
Create and then fix a deadlock scenario.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

// Challenge 1: Scoped locking
class BankAccount {
    int balance;
    mutable std::mutex mtx;
    
public:
    BankAccount(int initial) : balance(initial) {}
    
    void deposit(int amount) {
        std::lock_guard<std::mutex> lock(mtx);
        balance += amount;
    }
    
    void withdraw(int amount) {
        std::lock_guard<std::mutex> lock(mtx);
        if (balance >= amount) {
            balance -= amount;
        }
    }
    
    int getBalance() const {
        std::lock_guard<std::mutex> lock(mtx);
        return balance;
    }
};

// Challenge 2: Deadlock example and fix
class DeadlockExample {
    std::mutex mtx1, mtx2;
    
public:
    // BROKEN: Can deadlock
    void brokenTransfer() {
        std::lock_guard<std::mutex> lock1(mtx1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::lock_guard<std::mutex> lock2(mtx2);
        // Transfer logic
    }
    
    // FIXED: Use std::lock
    void fixedTransfer() {
        std::lock(mtx1, mtx2); // Atomic lock
        std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
        // Transfer logic
    }
    
    // BETTER: Use scoped_lock (C++17)
    void betterTransfer() {
        std::scoped_lock lock(mtx1, mtx2);
        // Transfer logic
    }
};

int main() {
    std::cout << "=== Thread-Safe Counter ===\n";
    
    std::mutex mtx;
    int counter = 0;
    
    auto increment = [&]() {
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            ++counter;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(increment);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Counter: " << counter << "\n"; // Correct: 1000000
    
    std::cout << "\n=== Bank Account ===\n";
    BankAccount account(1000);
    
    std::thread t1([&]() { account.deposit(100); });
    std::thread t2([&]() { account.withdraw(50); });
    
    t1.join();
    t2.join();
    
    std::cout << "Balance: " << account.getBalance() << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Identified race conditions
✅ Used mutex to protect shared data
✅ Applied RAII with `lock_guard`
✅ Implemented scoped locking (Challenge 1)
✅ Fixed deadlock with `std::lock` (Challenge 2)

## Rust Comparison
```rust
use std::sync::Mutex;

let counter = Mutex::new(0);
let mut num = counter.lock().unwrap();
*num += 1;
// Automatically unlocked when num goes out of scope
```

## Key Learnings
- Mutexes prevent race conditions
- Always use RAII locks (`lock_guard`, `unique_lock`)
- `scoped_lock` prevents deadlocks (C++17)
- Lock only critical sections
- Avoid holding locks longer than necessary

## Next Steps
Proceed to **Lab 17.3: Condition Variables**.
