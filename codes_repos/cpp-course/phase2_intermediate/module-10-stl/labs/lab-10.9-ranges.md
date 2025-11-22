# Lab 10.9: Ranges and Views (C++20)

## Objective
Use C++20 Ranges to compose algorithms in a readable, functional style.

## Instructions

### Step 1: Filter and Transform
Create `ranges.cpp`.
Take a vector, filter evens, square them.

```cpp
#include <iostream>
#include <vector>
#include <ranges>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    
    auto result = v 
        | std::views::filter([](int n){ return n % 2 == 0; }) 
        | std::views::transform([](int n){ return n * n; });
        
    for(int n : result) std::cout << n << " "; // 4 16 36
    
    return 0;
}
```
*Note: This is "Lazy Evaluation". Nothing happens until you iterate.*

### Step 2: Take and Drop
Take the first 3 elements.

```cpp
auto first3 = v | std::views::take(3);
```

### Step 3: Sort (Constrained Algorithm)
`std::ranges::sort(v);`
No need for `begin(), end()`.

## Challenges

### Challenge 1: Reverse View
Pipe the result into `std::views::reverse`.

### Challenge 2: Map Values
Create a map. Use `std::views::values` to iterate only over the values.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <ranges>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    // Challenge 1
    auto rev = v | std::views::reverse;
    for(int n : rev) std::cout << n << " ";
    std::cout << "\n";
    
    // Challenge 2
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}};
    for(const auto& s : m | std::views::values) {
        std::cout << s << " ";
    }
    
    // Ranges Sort
    std::ranges::sort(v, std::greater<int>());
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used Pipe syntax `|`
✅ Used `filter` and `transform` views
✅ Used `std::ranges::sort`
✅ Used `values` view (Challenge 2)

## Key Learnings
- Ranges make C++ look like Python/Rust
- Views are lazy and non-owning
- `std::ranges` algorithms accept containers directly

## Next Steps
Proceed to **Lab 10.10: Text Frequency Analyzer** to apply STL skills.
