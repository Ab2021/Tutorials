# Lab 11.3: Standard Exceptions

## Objective
Use the C++ Standard Library exception hierarchy (`<stdexcept>`).

## Instructions

### Step 1: Out of Range
Create `std_except.cpp`.
Accessing a vector out of bounds using `at()` throws `std::out_of_range`.

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

int main() {
    std::vector<int> v = {1, 2, 3};
    try {
        v.at(10); // Throws
    } catch (const std::out_of_range& e) {
        std::cout << "Range Error: " << e.what() << std::endl;
    }
    return 0;
}
```

### Step 2: Runtime Error
Throw a generic runtime error.

```cpp
void process() {
    throw std::runtime_error("Disk full");
}
```

### Step 3: Catching std::exception
Since all standard exceptions inherit from `std::exception`, you can catch them all polymorphically.

```cpp
try {
    process();
} catch (const std::exception& e) {
    std::cout << "Standard Error: " << e.what() << std::endl;
}
```

## Challenges

### Challenge 1: Invalid Argument
Write a function `sqrt(double x)` that throws `std::invalid_argument` if `x < 0`.

### Challenge 2: Bad Alloc
Try to allocate a huge array `new int[1000000000000]` inside a try block.
Catch `std::bad_alloc`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <new>

double mySqrt(double x) {
    if (x < 0) throw std::invalid_argument("Negative input");
    return std::sqrt(x);
}

int main() {
    // Challenge 1
    try {
        mySqrt(-5);
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
    }
    
    // Challenge 2
    try {
        int* p = new int[1000000000000L];
    } catch (const std::bad_alloc& e) {
        std::cout << "Memory fail: " << e.what() << "\n";
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Caught `std::out_of_range`
✅ Threw `std::runtime_error`
✅ Caught via base `std::exception`
✅ Threw `std::invalid_argument` (Challenge 1)
✅ Caught `std::bad_alloc` (Challenge 2)

## Key Learnings
- Prefer standard exceptions over primitives
- `e.what()` returns the error message
- Catching `std::exception&` handles most library errors

## Next Steps
Proceed to **Lab 11.4: Custom Exceptions** to define your own.
