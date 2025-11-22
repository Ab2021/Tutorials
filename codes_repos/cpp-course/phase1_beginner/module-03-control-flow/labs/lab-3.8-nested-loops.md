# Lab 3.8: Nested Loops

## Objective
Practice using loops inside loops to solve multi-dimensional problems.

## Instructions

### Step 1: Multiplication Table
Create `multiplication.cpp`. Print a 10x10 multiplication table.

```cpp
#include <iostream>
#include <iomanip> // For std::setw

int main() {
    for (int row = 1; row <= 10; ++row) {
        for (int col = 1; col <= 10; ++col) {
            std::cout << std::setw(4) << (row * col);
        }
        std::cout << std::endl;
    }
    return 0;
}
```
*Note: `std::setw(4)` ensures numbers align nicely.*

### Step 2: Coordinate Grid
Print all coordinates (x, y) where x is 0-2 and y is 0-2.

```cpp
for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
        std::cout << "(" << x << ", " << y << ") ";
    }
    std::cout << std::endl;
}
```

## Challenges

### Challenge 1: 3D Cube
Add a third loop for `z` (0-1). Print coordinates (x, y, z).

### Challenge 2: Breaking Nested Loops
Try to `break` out of the inner loop. Does it stop the outer loop?
How would you stop the outer loop from inside the inner loop?
*Hint: Use a boolean flag or `return`.*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 2: Breaking out of nested loops
    bool stop = false;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (i == 2 && j == 2) {
                stop = true;
                break; // Breaks inner loop only
            }
            std::cout << i << "," << j << " ";
        }
        if (stop) break; // Breaks outer loop
        std::cout << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented 2D nested loops
✅ Formatted output with `setw`
✅ Implemented 3D nested loops (Challenge 1)
✅ Controlled nested loop exit (Challenge 2)

## Key Learnings
- Inner loop runs fully for each outer loop iteration
- `break` only affects the current loop level
- Formatting output is crucial for grid data

## Next Steps
Proceed to **Lab 3.9: Goto** to see an alternative (but discouraged) way to jump.
