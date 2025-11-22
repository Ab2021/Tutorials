# Lab 6.3: Array Allocation and Deallocation

## Objective
Learn the specific syntax for heap-allocated arrays.

## Instructions

### Step 1: Allocate Array
Create `array_alloc.cpp`.

```cpp
#include <iostream>

int main() {
    int size;
    std::cout << "Size: ";
    std::cin >> size;
    
    int* arr = new int[size]; // Note the []
    
    for(int i=0; i<size; ++i) arr[i] = i;
    
    delete[] arr; // Note the []
    return 0;
}
```

### Step 2: Mismatched Delete
What happens if you use `delete` instead of `delete[]`?
It's Undefined Behavior. It might only free the first element, causing a leak, or corrupt the heap.

### Step 3: 2D Array (Matrix)
Allocate a 3x3 matrix.

```cpp
int** matrix = new int*[3]; // Array of pointers
for(int i=0; i<3; ++i) {
    matrix[i] = new int[3]; // Array of ints
}
```

## Challenges

### Challenge 1: Cleanup Matrix
Write the code to properly delete the 3x3 matrix from Step 3.
*Hint: Delete inner arrays first, then the outer array.*

### Challenge 2: Flattened 2D Array
Instead of `int**`, allocate a single 1D array of size `rows * cols` and access it using logic: `arr[row * cols + col]`.
This is more cache-friendly.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 1: Cleanup
    int** matrix = new int*[3];
    for(int i=0; i<3; ++i) matrix[i] = new int[3];
    
    // Delete inner
    for(int i=0; i<3; ++i) delete[] matrix[i];
    // Delete outer
    delete[] matrix;
    
    // Challenge 2: Flattened
    int rows = 3, cols = 3;
    int* flat = new int[rows * cols];
    
    // Access (1, 2)
    flat[1 * cols + 2] = 42;
    
    delete[] flat;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `new[]` and `delete[]` correctly
✅ Allocated dynamic 2D array
✅ Properly deallocated 2D array (Challenge 1)
✅ Implemented flattened array (Challenge 2)

## Key Learnings
- Always match `new[]` with `delete[]`
- Multi-dimensional dynamic arrays are complex
- Flattening arrays improves performance and simplifies cleanup

## Next Steps
Proceed to **Lab 6.4: Memory Leaks** to learn detection.
