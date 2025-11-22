# Lab 4.10: Lambda Expressions Basics

## Objective
Introduction to lambda expressions (anonymous functions) in C++.

## Instructions

### Step 1: Basic Lambda
Create `lambda.cpp`. Define and call a lambda immediately.

```cpp
#include <iostream>

int main() {
    auto hello = []() {
        std::cout << "Hello form Lambda!" << std::endl;
    };
    
    hello(); // Call it
    return 0;
}
```

### Step 2: Parameters and Returns
Lambdas can take arguments and return values.

```cpp
auto add = [](int a, int b) -> int {
    return a + b;
};

std::cout << "Sum: " << add(5, 3) << std::endl;
```

### Step 3: Captures
Lambdas can "capture" local variables.
`[=]` captures by value (copy).
`[&]` captures by reference (original).

```cpp
int x = 10;
auto printX = [x]() { // Capture x by value
    std::cout << x << std::endl;
};

x = 20;
printX(); // Prints 10 (copy was made)
```

## Challenges

### Challenge 1: Capture by Reference
Modify Step 3 to capture `x` by reference `[&x]`.
Change `x` to 20. Call lambda. It should print 20.

### Challenge 2: ForEach with Lambda
Use `std::for_each` (from `<algorithm>`) with a lambda to print a vector.

```cpp
#include <vector>
#include <algorithm>

std::vector<int> v = {1, 2, 3};
std::for_each(v.begin(), v.end(), [](int n) {
    // Print n
});
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Challenge 1
    int x = 10;
    auto printRef = [&x]() {
        std::cout << "Ref: " << x << std::endl;
    };
    x = 20;
    printRef(); // Prints 20
    
    // Challenge 2
    std::vector<int> v = {10, 20, 30};
    std::for_each(v.begin(), v.end(), [](int n) {
        std::cout << n << " ";
    });
    std::cout << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Defined and called a lambda
✅ Used parameters and return types
✅ Understood capture by value vs reference
✅ Used lambda with STL algorithm (Challenge 2)

## Key Learnings
- `[]` Capture clause
- `()` Parameter list
- `{}` Body
- Lambdas are objects (functors)

## Next Steps
Congratulations! You've completed Module 4.

Proceed to **Module 5: Pointers and References** to dive deep into memory management.
