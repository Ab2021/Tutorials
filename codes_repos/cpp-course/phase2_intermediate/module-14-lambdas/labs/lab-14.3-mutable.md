# Lab 14.3: Mutable Lambdas

## Objective
Use the `mutable` keyword to modify captured-by-value variables.

## Instructions

### Step 1: The Problem
Create `mutable_lambda.cpp`.
Captured-by-value variables are const.

```cpp
#include <iostream>

int main() {
    int count = 0;
    
    auto increment = [count]() {
        // count++; // Error: cannot modify
        std::cout << count << "\n";
    };
    
    return 0;
}
```

### Step 2: Mutable Keyword
Allow modification of the captured copy.

```cpp
auto increment = [count]() mutable {
    count++; // OK: modifies the lambda's copy
    std::cout << "Lambda count: " << count << "\n";
};

increment(); // 1
increment(); // 2
std::cout << "Original count: " << count << "\n"; // Still 0
```

### Step 3: Stateful Lambdas
Lambdas can maintain state across calls.

```cpp
auto counter = [n = 0]() mutable {
    return ++n;
};

std::cout << counter() << "\n"; // 1
std::cout << counter() << "\n"; // 2
std::cout << counter() << "\n"; // 3
```

## Challenges

### Challenge 1: Fibonacci Generator
Create a lambda that returns the next Fibonacci number each time it's called.
Use two captured variables to track state.

### Challenge 2: Mutable vs Reference
Compare behavior:
- `[x]() mutable { x++; }` (modifies copy)
- `[&x]() { x++; }` (modifies original)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 1: Fibonacci
    auto fib = [a = 0, b = 1]() mutable {
        int result = a;
        int next = a + b;
        a = b;
        b = next;
        return result;
    };
    
    for (int i = 0; i < 10; ++i) {
        std::cout << fib() << " ";
    }
    std::cout << "\n";
    
    // Challenge 2: Comparison
    int x = 10;
    auto mut = [x]() mutable { x++; std::cout << "Mut: " << x << "\n"; };
    mut(); // 11
    mut(); // 12
    std::cout << "Original x: " << x << "\n"; // 10
    
    int y = 10;
    auto ref = [&y]() { y++; std::cout << "Ref: " << y << "\n"; };
    ref(); // 11
    std::cout << "Original y: " << y << "\n"; // 11
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `mutable` keyword
✅ Created stateful lambda
✅ Implemented Fibonacci generator (Challenge 1)
✅ Compared mutable vs reference (Challenge 2)

## Key Learnings
- `mutable` allows modifying captured-by-value variables
- Each lambda instance has its own state
- Useful for generators and accumulators

## Next Steps
Proceed to **Lab 14.4: Generic Lambdas** for template-like behavior.
