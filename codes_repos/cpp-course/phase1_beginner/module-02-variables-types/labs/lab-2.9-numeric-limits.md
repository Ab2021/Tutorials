# Lab 2.9: Numeric Limits Exploration

## Objective
Explore special numeric values like Infinity, NaN (Not a Number), and Epsilon.

## Instructions

### Step 1: Infinity and NaN
Create `limits_demo.cpp`:

```cpp
#include <iostream>
#include <limits>
#include <cmath> // For std::isnan, std::isinf

int main() {
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();
    
    std::cout << "Infinity: " << inf << std::endl;
    std::cout << "NaN: " << nan << std::endl;
    
    std::cout << "Inf + 1: " << (inf + 1) << std::endl;
    std::cout << "Inf / Inf: " << (inf / inf) << std::endl; // NaN
    std::cout << "1.0 / 0.0: " << (1.0 / 0.0) << std::endl; // Inf
    
    return 0;
}
```

### Step 2: Checking for Special Values
Use `std::isnan` and `std::isinf` to check values.

```cpp
if (std::isinf(inf)) std::cout << "It is infinite" << std::endl;
if (std::isnan(nan)) std::cout << "It is NaN" << std::endl;
```

### Step 3: Machine Epsilon
Epsilon is the smallest difference between 1.0 and the next representable value.

```cpp
double eps = std::numeric_limits<double>::epsilon();
std::cout << "Epsilon: " << eps << std::endl;
```

## Challenges

### Challenge 1: Floating Point Comparison
Write a function `bool areEqual(double a, double b)` that returns true if the difference is less than epsilon.
```cpp
bool areEqual(double a, double b) {
    return std::abs(a - b) < std::numeric_limits<double>::epsilon();
}
```
Test it with `1.0 / 3.0 * 3.0` vs `1.0`.

### Challenge 2: NaN Propagation
What happens if you use NaN in a calculation?
`double result = 10.0 + nan;`
Check if `result == result`. (Hint: NaN is never equal to itself!)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <limits>
#include <cmath>

bool areEqual(double a, double b) {
    return std::abs(a - b) < std::numeric_limits<double>::epsilon();
}

int main() {
    double nan = std::numeric_limits<double>::quiet_NaN();
    
    // Challenge 2
    if (nan == nan) {
        std::cout << "NaN == NaN" << std::endl;
    } else {
        std::cout << "NaN != NaN" << std::endl;
    }
    
    double x = 1.0 / 3.0;
    double y = x * 3.0;
    
    if (areEqual(y, 1.0)) {
        std::cout << "Math works!" << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Generated Infinity and NaN
✅ Checked for special values
✅ Understood Epsilon
✅ Implemented safe float comparison

## Key Learnings
- Floating point math has special states
- Never compare floats with `==`
- NaN propagates through calculations
- NaN is not equal to anything, including itself

## Next Steps
Proceed to **Lab 2.10: Type-Safe Units Library** for the final challenge.
