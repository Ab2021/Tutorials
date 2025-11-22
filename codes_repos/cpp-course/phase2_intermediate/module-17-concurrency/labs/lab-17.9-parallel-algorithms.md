# Lab 17.9: Parallel Algorithms

## Objective
Use C++17 parallel algorithms for automatic parallelization.

## Instructions

### Step 1: Execution Policies
Create `parallel_algorithms.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>

int main() {
    std::vector<int> vec(1000000);
    std::iota(vec.begin(), vec.end(), 0);
    
    // Sequential execution
    std::sort(std::execution::seq, vec.begin(), vec.end());
    
    // Parallel execution
    std::sort(std::execution::par, vec.begin(), vec.end());
    
    // Parallel unsequenced (vectorized)
    std::sort(std::execution::par_unseq, vec.begin(), vec.end());
    
    return 0;
}
```

### Step 2: Parallel Algorithms
```cpp
#include <execution>
#include <algorithm>
#include <numeric>

void parallelExamples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Parallel for_each
    std::for_each(std::execution::par, vec.begin(), vec.end(), 
        [](int& x) { x *= 2; });
    
    // Parallel transform
    std::vector<int> result(vec.size());
    std::transform(std::execution::par, vec.begin(), vec.end(), 
        result.begin(), [](int x) { return x * x; });
    
    // Parallel reduce
    int sum = std::reduce(std::execution::par, vec.begin(), vec.end());
    
    // Parallel find
    auto it = std::find(std::execution::par, vec.begin(), vec.end(), 10);
}
```

### Step 3: Parallel Numeric Algorithms
```cpp
#include <numeric>

void parallelNumeric() {
    std::vector<int> vec(1000);
    std::iota(vec.begin(), vec.end(), 1);
    
    // Parallel accumulate
    int sum = std::reduce(std::execution::par, vec.begin(), vec.end());
    
    // Parallel transform_reduce
    int sumOfSquares = std::transform_reduce(
        std::execution::par,
        vec.begin(), vec.end(),
        0,
        std::plus<>(),
        [](int x) { return x * x; }
    );
}
```

## Challenges

### Challenge 1: Benchmark
Compare sequential vs parallel execution for various algorithms.

### Challenge 2: Custom Parallel Operation
Implement a custom parallel map-reduce operation.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <chrono>
#include <cmath>

// Challenge 1: Benchmark
template<typename Func>
auto benchmark(const std::string& name, Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << name << ": " << duration.count() << "ms\n";
    return duration;
}

void benchmarkSort() {
    const size_t size = 10000000;
    
    std::cout << "=== Sort Benchmark ===\n";
    
    // Sequential
    std::vector<int> vec1(size);
    std::generate(vec1.begin(), vec1.end(), std::rand);
    benchmark("Sequential sort", [&]() {
        std::sort(std::execution::seq, vec1.begin(), vec1.end());
    });
    
    // Parallel
    std::vector<int> vec2(size);
    std::generate(vec2.begin(), vec2.end(), std::rand);
    benchmark("Parallel sort", [&]() {
        std::sort(std::execution::par, vec2.begin(), vec2.end());
    });
}

void benchmarkTransform() {
    const size_t size = 10000000;
    std::vector<double> vec(size);
    std::iota(vec.begin(), vec.end(), 1.0);
    
    std::cout << "\n=== Transform Benchmark ===\n";
    
    // Sequential
    std::vector<double> result1(size);
    benchmark("Sequential transform", [&]() {
        std::transform(std::execution::seq, vec.begin(), vec.end(), 
            result1.begin(), [](double x) { return std::sqrt(x); });
    });
    
    // Parallel
    std::vector<double> result2(size);
    benchmark("Parallel transform", [&]() {
        std::transform(std::execution::par, vec.begin(), vec.end(), 
            result2.begin(), [](double x) { return std::sqrt(x); });
    });
}

// Challenge 2: Custom parallel map-reduce
template<typename InputIt, typename MapFunc, typename ReduceFunc, typename T>
T parallelMapReduce(InputIt first, InputIt last, MapFunc map, ReduceFunc reduce, T init) {
    std::vector<T> mapped(std::distance(first, last));
    
    // Parallel map
    std::transform(std::execution::par, first, last, mapped.begin(), map);
    
    // Parallel reduce
    return std::reduce(std::execution::par, mapped.begin(), mapped.end(), init, reduce);
}

void customMapReduce() {
    std::vector<int> vec(1000000);
    std::iota(vec.begin(), vec.end(), 1);
    
    // Sum of squares using custom map-reduce
    auto sumOfSquares = parallelMapReduce(
        vec.begin(), vec.end(),
        [](int x) { return x * x; },      // Map
        std::plus<int>(),                  // Reduce
        0                                  // Initial value
    );
    
    std::cout << "\nSum of squares: " << sumOfSquares << "\n";
}

int main() {
    benchmarkSort();
    benchmarkTransform();
    customMapReduce();
    
    // Additional parallel algorithms
    std::vector<int> vec(1000000);
    std::iota(vec.begin(), vec.end(), 1);
    
    std::cout << "\n=== Parallel Algorithms ===\n";
    
    // Parallel count_if
    auto count = std::count_if(std::execution::par, vec.begin(), vec.end(),
        [](int x) { return x % 2 == 0; });
    std::cout << "Even numbers: " << count << "\n";
    
    // Parallel any_of
    bool hasLarge = std::any_of(std::execution::par, vec.begin(), vec.end(),
        [](int x) { return x > 999000; });
    std::cout << "Has large number: " << std::boolalpha << hasLarge << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used execution policies for parallelization
✅ Applied parallel algorithms
✅ Benchmarked sequential vs parallel (Challenge 1)
✅ Implemented custom map-reduce (Challenge 2)

## Compiler Support
```bash
# GCC (requires TBB)
g++ -std=c++17 -ltbb parallel_algorithms.cpp -o parallel

# MSVC
cl /std:c++17 /EHsc parallel_algorithms.cpp
```

## Key Learnings
- C++17 adds execution policies to STL algorithms
- `std::execution::seq` - sequential
- `std::execution::par` - parallel
- `std::execution::par_unseq` - parallel + vectorized
- Automatic parallelization without manual threading

## Next Steps
Proceed to **Lab 17.10: Concurrent Server (Capstone)**.
