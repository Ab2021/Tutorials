# Lab 4.2: Parameters and Return Values

## Objective
Understand how to pass data into functions and get results back.

## Instructions

### Step 1: Multiple Parameters
Create `params.cpp`. Write a function `calculate` that takes two integers and an operation code (char).

```cpp
#include <iostream>

int calculate(int a, int b, char op) {
    if (op == '+') return a + b;
    if (op == '-') return a - b;
    return 0; // Default
}

int main() {
    std::cout << "10 + 5 = " << calculate(10, 5, '+') << std::endl;
    return 0;
}
```

### Step 2: Boolean Return
Write a function `isEven` that returns `true` if a number is even.

```cpp
bool isEven(int n) {
    return (n % 2 == 0);
}
```

### Step 3: Early Return
Modify `calculate` to handle division. If dividing by zero, print error and return 0 immediately.

```cpp
if (op == '/') {
    if (b == 0) {
        std::cerr << "Error: Div by zero" << std::endl;
        return 0;
    }
    return a / b;
}
```

## Challenges

### Challenge 1: Returning Multiple Values (Struct)
C++ functions return only one value. To return more, use a struct.
Create `struct Stats { int sum; int product; };`.
Write `Stats compute(int a, int b)` that returns both.

### Challenge 2: Returning Multiple Values (Pair)
Use `std::pair` (from `<utility>`) to return two values without defining a struct.
```cpp
#include <utility>
std::pair<int, int> getMinMax(int a, int b);
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <utility>

struct Stats {
    int sum;
    int product;
};

Stats compute(int a, int b) {
    return {a + b, a * b};
}

std::pair<int, int> getMinMax(int a, int b) {
    if (a < b) return {a, b};
    return {b, a};
}

int main() {
    Stats s = compute(5, 10);
    std::cout << "Sum: " << s.sum << ", Product: " << s.product << std::endl;
    
    std::pair<int, int> p = getMinMax(20, 10);
    std::cout << "Min: " << p.first << ", Max: " << p.second << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Passed multiple parameters
✅ Returned values correctly
✅ Used early return pattern
✅ Returned multiple values using struct/pair (Challenge 1/2)

## Key Learnings
- Functions can take any number of parameters
- `return` exits the function immediately
- Structs and pairs are common ways to return multiple values

## Next Steps
Proceed to **Lab 4.3: Pass by Value vs Reference** to understand memory usage.
