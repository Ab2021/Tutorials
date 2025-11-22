# Lab 2.1: Type Exploration and Limits

## Objective
Explore the size and range of fundamental C++ data types on your system.

## Instructions

### Step 1: Size of Types
Create `types_demo.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "Size of int: " << sizeof(int) << " bytes" << std::endl;
    // TODO: Print sizes of char, bool, float, double, long long
    
    return 0;
}
```

### Step 2: Numeric Limits
Include `<limits>` and print the min/max values:

```cpp
#include <iostream>
#include <limits>

int main() {
    std::cout << "Int Min: " << std::numeric_limits<int>::min() << std::endl;
    std::cout << "Int Max: " << std::numeric_limits<int>::max() << std::endl;
    
    // TODO: Print limits for other types
    
    return 0;
}
```

### Step 3: Overflow
Try to overflow an integer:
```cpp
int max = std::numeric_limits<int>::max();
std::cout << "Max: " << max << std::endl;
std::cout << "Max + 1: " << (max + 1) << std::endl; // What happens?
```

## Challenges

### Challenge 1: Unsigned Types
Explore `unsigned int`. What is its minimum value? What happens when you subtract 1 from it?

### Challenge 2: Floating Point Precision
Print `1.0 / 3.0` with high precision using `std::setprecision`.
```cpp
#include <iomanip>
std::cout << std::setprecision(20) << (1.0 / 3.0) << std::endl;
```
Compare `float` vs `double` precision.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <limits>
#include <iomanip>

int main() {
    std::cout << "Size of long long: " << sizeof(long long) << " bytes" << std::endl;
    
    std::cout << "Unsigned Int Max: " << std::numeric_limits<unsigned int>::max() << std::endl;
    
    unsigned int u_min = 0;
    std::cout << "0 - 1 (unsigned): " << (u_min - 1) << std::endl; // Underflow to max
    
    float f = 1.0f / 3.0f;
    double d = 1.0 / 3.0;
    
    std::cout << std::setprecision(20);
    std::cout << "Float:  " << f << std::endl;
    std::cout << "Double: " << d << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Printed sizes of all fundamental types
✅ Printed min/max values
✅ Observed integer overflow behavior
✅ Observed floating point precision differences

## Key Learnings
- `sizeof` operator
- `std::numeric_limits`
- Integer overflow/underflow behavior
- Precision difference between float and double

## Next Steps
Proceed to **Lab 2.2: Auto Type Deduction** to let the compiler do the work.
