# Lab 14.7: Lambdas with STL Algorithms

## Objective
Use lambdas to customize STL algorithms.

## Instructions

### Step 1: Sort with Custom Comparator
Create `stl_lambdas.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {5, 2, 8, 1, 9};
    
    // Sort descending
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;
    });
    
    for (int n : v) std::cout << n << " ";
    
    return 0;
}
```

### Step 2: Transform
```cpp
std::vector<int> squared;
std::transform(v.begin(), v.end(), std::back_inserter(squared),
    [](int x) { return x * x; }
);
```

### Step 3: Find_if
```cpp
auto it = std::find_if(v.begin(), v.end(), [](int x) {
    return x > 5;
});
if (it != v.end()) std::cout << "Found: " << *it << "\n";
```

### Step 4: Count_if
```cpp
int count = std::count_if(v.begin(), v.end(), [](int x) {
    return x % 2 == 0;
});
std::cout << "Even count: " << count << "\n";
```

## Challenges

### Challenge 1: Remove_if with Erase
Remove all elements greater than 5.
```cpp
v.erase(std::remove_if(v.begin(), v.end(), [](int x) {
    return x > 5;
}), v.end());
```

### Challenge 2: Accumulate with Lambda
Use `std::accumulate` with a lambda to compute product.
```cpp
int product = std::accumulate(v.begin(), v.end(), 1,
    [](int acc, int x) { return acc * x; }
);
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Challenge 1: Remove > 5
    v.erase(std::remove_if(v.begin(), v.end(), [](int x) {
        return x > 5;
    }), v.end());
    
    for (int n : v) std::cout << n << " ";
    std::cout << "\n";
    
    // Challenge 2: Product
    int product = std::accumulate(v.begin(), v.end(), 1,
        [](int acc, int x) { return acc * x; }
    );
    std::cout << "Product: " << product << "\n";
    
    // Partition
    std::vector<int> nums = {1, 2, 3, 4, 5, 6};
    auto pivot = std::partition(nums.begin(), nums.end(), [](int x) {
        return x % 2 == 0;
    });
    std::cout << "Evens first: ";
    for (int n : nums) std::cout << n << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used lambdas with `sort`, `transform`, `find_if`
✅ Used `count_if` and `remove_if`
✅ Implemented erase-remove idiom (Challenge 1)
✅ Used `accumulate` with lambda (Challenge 2)

## Key Learnings
- Lambdas make STL algorithms extremely flexible
- No need to write separate functor classes
- Capture allows parameterizing algorithms

## Next Steps
Proceed to **Lab 14.8: IIFE Pattern** for initialization.
