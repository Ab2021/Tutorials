# Lab 15.6: Make Functions (make_unique, make_shared)

## Objective
Understand why `make_unique` and `make_shared` are preferred.

## Instructions

### Step 1: Exception Safety
Create `make_functions.cpp`.
Direct `new` can leak in exception scenarios.

```cpp
#include <iostream>
#include <memory>

void process(std::unique_ptr<int> p, int value) {
    std::cout << *p + value << "\n";
}

int riskyFunction() {
    throw std::runtime_error("Error");
    return 42;
}

int main() {
    // UNSAFE: If riskyFunction throws, memory leaks!
    // process(std::unique_ptr<int>(new int(10)), riskyFunction());
    
    // SAFE: make_unique is exception-safe
    process(std::make_unique<int>(10), riskyFunction());
    
    return 0;
}
```

### Step 2: Performance (make_shared)
`make_shared` allocates control block and object together (one allocation).

```cpp
// Two allocations: object + control block
std::shared_ptr<int> p1(new int(42));

// One allocation: object and control block together
auto p2 = std::make_shared<int>(42);
```

### Step 3: Cannot Use with Custom Deleters
```cpp
// make_unique doesn't support custom deleters
// Must use constructor
auto deleter = [](int* p) { delete p; };
std::unique_ptr<int, decltype(deleter)> p(new int(42), deleter);
```

## Challenges

### Challenge 1: Measure Performance
Compare allocation time between `new` + `shared_ptr` vs `make_shared`.

### Challenge 2: Array make_unique
Use `make_unique` for arrays (C++14).
```cpp
auto arr = std::make_unique<int[]>(10);
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <chrono>

int main() {
    // Challenge 1: Performance comparison
    const int N = 1000000;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        std::shared_ptr<int> p(new int(i));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto p = std::make_shared<int>(i);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "new + shared_ptr: " << duration1.count() << "ms\n";
    std::cout << "make_shared: " << duration2.count() << "ms\n";
    
    // Challenge 2: Array
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i) arr[i] = i;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood exception safety benefit
✅ Understood performance benefit of `make_shared`
✅ Measured performance difference (Challenge 1)
✅ Used `make_unique` for arrays (Challenge 2)

## Key Learnings
- Always prefer `make_unique`/`make_shared`
- Exception-safe and more efficient
- Cannot use with custom deleters (use constructor)

## Next Steps
Proceed to **Lab 15.7: Smart Pointers and Polymorphism**.
