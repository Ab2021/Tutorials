# Lab 15.2: Unique Pointer with Arrays

## Objective
Use `unique_ptr` to manage dynamic arrays.

## Instructions

### Step 1: Array Syntax
Create `unique_array.cpp`.

```cpp
#include <iostream>
#include <memory>

int main() {
    // Array version uses delete[]
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * 10;
    }
    
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    
    return 0;
}
```

### Step 2: Why Not vector?
`std::vector` is usually better, but `unique_ptr<T[]>` is useful for:
- C API interop
- Fixed-size arrays
- Performance-critical code

### Step 3: Custom Size
```cpp
size_t size = 10;
auto arr2 = std::make_unique<int[]>(size);
```

## Challenges

### Challenge 1: 2D Array
Create a 2D array using `unique_ptr`.
```cpp
auto rows = std::make_unique<std::unique_ptr<int[]>[]>(3);
for (int i = 0; i < 3; ++i) {
    rows[i] = std::make_unique<int[]>(4);
}
```

### Challenge 2: Compare with vector
Measure performance difference between `unique_ptr<int[]>` and `std::vector<int>` for large arrays.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

int main() {
    // Challenge 1: 2D array
    const int rows = 3, cols = 4;
    auto matrix = std::make_unique<std::unique_ptr<int[]>[]>(rows);
    
    for (int i = 0; i < rows; ++i) {
        matrix[i] = std::make_unique<int[]>(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = i * cols + j;
        }
    }
    
    std::cout << "Matrix:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created array with `unique_ptr<T[]>`
✅ Used array subscript operator
✅ Created 2D array (Challenge 1)

## Key Learnings
- `unique_ptr<T[]>` uses `delete[]` automatically
- Prefer `std::vector` in most cases
- Useful for C API compatibility

## Next Steps
Proceed to **Lab 15.3: Shared Pointer Basics**.
