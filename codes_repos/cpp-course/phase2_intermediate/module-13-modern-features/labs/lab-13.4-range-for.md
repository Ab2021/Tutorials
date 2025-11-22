# Lab 13.4: Range-Based For Loops

## Objective
Master the range-based for loop syntax and understand its benefits.

## Instructions

### Step 1: Basic Syntax
Create `range_for.cpp`.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    // Old way
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
    
    // Range-based for
    for (int x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    
    return 0;
}
```

### Step 2: By Reference
Modify elements using references.

```cpp
for (int& x : v) {
    x *= 2; // Double each element
}
```

### Step 3: Const Reference
Read-only access (avoids copying).

```cpp
std::vector<std::string> names = {"Alice", "Bob"};
for (const auto& name : names) {
    std::cout << name << "\n";
}
```

## Challenges

### Challenge 1: C-Style Arrays
Range-based for works on C arrays too!
```cpp
int arr[] = {10, 20, 30};
for (int x : arr) { ... }
```

### Challenge 2: Custom Range
Make your own class iterable by implementing `begin()` and `end()`.
```cpp
class Range {
    int start, end;
public:
    Range(int s, int e) : start(s), end(e) {}
    int* begin() { return &start; }
    int* end() { return &end; }
};
```
(This is simplified; real implementation needs proper iterators).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

int main() {
    // Challenge 1: C-array
    int arr[] = {10, 20, 30, 40};
    for (const auto& x : arr) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    
    // Modify by reference
    for (auto& x : arr) {
        x += 5;
    }
    
    for (auto x : arr) std::cout << x << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used range-based for with vectors
✅ Modified elements using reference
✅ Used const reference for read-only
✅ Applied to C-style arrays (Challenge 1)

## Key Learnings
- Range-based for is safer (no off-by-one errors)
- Use `auto&` to modify, `const auto&` to read
- Works with any type that has `begin()` and `end()`

## Next Steps
Proceed to **Lab 13.5: Uniform Initialization** for consistent syntax.
