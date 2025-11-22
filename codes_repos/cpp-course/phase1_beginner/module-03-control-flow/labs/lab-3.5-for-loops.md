# Lab 3.5: For Loop Patterns

## Objective
Master the `for` loop by creating visual patterns.

## Instructions

### Step 1: Basic Line
Create `patterns.cpp`. Print 10 stars in a row.

```cpp
#include <iostream>

int main() {
    for (int i = 0; i < 10; ++i) {
        std::cout << "*";
    }
    std::cout << std::endl;
    return 0;
}
```

### Step 2: Square
Use nested loops to print a 5x5 square of stars.

```cpp
for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 5; ++col) {
        std::cout << "* ";
    }
    std::cout << std::endl;
}
```

### Step 3: Triangle
Modify the inner loop to print a triangle (1 star, then 2, then 3...).

```cpp
for (int row = 0; row < 5; ++row) {
    for (int col = 0; col <= row; ++col) { // Note: col <= row
        std::cout << "* ";
    }
    std::cout << std::endl;
}
```

## Challenges

### Challenge 1: Inverted Triangle
Print the triangle upside down (5 stars, then 4, etc.).

### Challenge 2: Pyramid
Print a centered pyramid:
```
    *
   ***
  *****
 *******
```
*Hint: You need a loop for spaces and a loop for stars inside the row loop.*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    int rows = 5;
    
    for (int i = 1; i <= rows; ++i) {
        // Print spaces
        for (int j = 1; j <= rows - i; ++j) {
            std::cout << " ";
        }
        // Print stars
        for (int k = 1; k <= 2 * i - 1; ++k) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented basic `for` loop
✅ Implemented nested loops
✅ Created square and triangle patterns
✅ Solved pyramid logic (Challenge 2)

## Key Learnings
- Loop counters (`i`, `j`, `k`)
- Nested loop execution flow
- Pattern logic

## Next Steps
Proceed to **Lab 3.6: Range-based For Loop** for modern iteration.
