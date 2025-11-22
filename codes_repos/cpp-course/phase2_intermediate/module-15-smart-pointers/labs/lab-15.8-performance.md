# Lab 15.8: Performance Comparison

## Objective
Understand the performance characteristics of different smart pointers.

## Instructions

### Step 1: Size Comparison
Create `performance.cpp`.

```cpp
#include <iostream>
#include <memory>

int main() {
    std::cout << "sizeof(int*): " << sizeof(int*) << "\n";
    std::cout << "sizeof(unique_ptr<int>): " << sizeof(std::unique_ptr<int>) << "\n";
    std::cout << "sizeof(shared_ptr<int>): " << sizeof(std::shared_ptr<int>) << "\n";
    std::cout << "sizeof(weak_ptr<int>): " << sizeof(std::weak_ptr<int>) << "\n";
    
    return 0;
}
```

### Step 2: Creation Overhead
```cpp
#include <chrono>

const int N = 1000000;

auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < N; ++i) {
    int* p = new int(i);
    delete p;
}
auto end = std::chrono::high_resolution_clock::now();
// Measure time...
```

### Step 3: Copy Overhead
```cpp
auto sp = std::make_shared<int>(42);
// Copying shared_ptr is atomic (thread-safe but slower)
for (int i = 0; i < N; ++i) {
    auto copy = sp; // Atomic increment
}
```

## Challenges

### Challenge 1: Benchmark Suite
Create a comprehensive benchmark comparing:
- Raw pointer
- unique_ptr
- shared_ptr
For creation, copying (where applicable), and dereferencing.

### Challenge 2: Cache Effects
Measure performance difference between:
- `vector<unique_ptr<int>>` (pointer indirection)
- `vector<int>` (contiguous memory)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

template <typename Func>
void benchmark(const char* name, Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << name << ": " << duration.count() << "ms\n";
}

int main() {
    const int N = 1000000;
    
    // Creation benchmarks
    benchmark("Raw pointer", [N]() {
        for (int i = 0; i < N; ++i) {
            int* p = new int(i);
            delete p;
        }
    });
    
    benchmark("unique_ptr", [N]() {
        for (int i = 0; i < N; ++i) {
            auto p = std::make_unique<int>(i);
        }
    });
    
    benchmark("shared_ptr", [N]() {
        for (int i = 0; i < N; ++i) {
            auto p = std::make_shared<int>(i);
        }
    });
    
    // Cache effects
    std::vector<std::unique_ptr<int>> vec_ptr;
    std::vector<int> vec_val;
    
    for (int i = 0; i < 10000; ++i) {
        vec_ptr.push_back(std::make_unique<int>(i));
        vec_val.push_back(i);
    }
    
    benchmark("Sum via pointers", [&]() {
        long long sum = 0;
        for (const auto& p : vec_ptr) sum += *p;
    });
    
    benchmark("Sum via values", [&]() {
        long long sum = 0;
        for (int v : vec_val) sum += v;
    });
    
    return 0;
}
```
</details>

## Success Criteria
✅ Measured size overhead
✅ Benchmarked creation time
✅ Compared cache performance (Challenge 2)

## Key Learnings
- `unique_ptr` has zero overhead (same size as raw pointer)
- `shared_ptr` is 2x size (pointer + control block pointer)
- Atomic operations in `shared_ptr` have cost
- Contiguous memory (vector of values) is faster than pointers

## Next Steps
Proceed to **Lab 15.9: Common Pitfalls**.
