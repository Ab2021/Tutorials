# Lab 14.1: Basic Lambda Syntax

## Objective
Learn the fundamental syntax of lambda expressions.

## Instructions

### Step 1: Simple Lambda
Create `lambda_basics.cpp`.

```cpp
#include <iostream>

int main() {
    // Lambda with no parameters
    auto greet = []() { std::cout << "Hello!\n"; };
    greet();
    
    // Lambda with parameters
    auto add = [](int a, int b) { return a + b; };
    std::cout << "Sum: " << add(5, 3) << "\n";
    
    // Lambda with explicit return type
    auto divide = [](double a, double b) -> double {
        if (b == 0) return 0.0;
        return a / b;
    };
    
    return 0;
}
```

### Step 2: Inline Lambda
Use lambda directly without storing it.

```cpp
int result = [](int x) { return x * x; }(5); // IIFE
std::cout << "Square: " << result << "\n";
```

### Step 3: Lambda as Parameter
Pass lambda to a function.

```cpp
void execute(int x, int (*func)(int)) {
    std::cout << func(x) << "\n";
}

execute(10, [](int n) { return n * 2; });
```

## Challenges

### Challenge 1: Multi-Statement Lambda
Write a lambda that takes an int and prints whether it's even or odd, then returns the number squared.

### Challenge 2: Trailing Return Type
When is `-> type` required? Try removing it from a complex lambda and see if it still compiles.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 1
    auto process = [](int n) {
        if (n % 2 == 0) std::cout << n << " is even\n";
        else std::cout << n << " is odd\n";
        return n * n;
    };
    
    int result = process(7);
    std::cout << "Squared: " << result << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created basic lambdas
✅ Used lambdas inline (IIFE)
✅ Passed lambdas to functions
✅ Wrote multi-statement lambda (Challenge 1)

## Key Learnings
- Lambdas are anonymous functions
- `auto` deduces the lambda type
- Return type is usually inferred

## Next Steps
Proceed to **Lab 14.2: Capture Mechanisms** to access outer variables.
