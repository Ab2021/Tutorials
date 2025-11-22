# Lab 14.9: Constexpr Lambdas (C++17)

## Objective
Use lambdas in constant expressions for compile-time computation.

## Instructions

### Step 1: Constexpr Lambda
Create `constexpr_lambda.cpp`.
Lambdas are implicitly `constexpr` if possible (C++17).

```cpp
#include <iostream>

int main() {
    constexpr auto square = [](int x) { return x * x; };
    
    constexpr int val = square(5); // Compile-time
    static_assert(val == 25);
    
    int arr[square(3)]; // Array size = 9
    
    return 0;
}
```

### Step 2: Compile-Time Factorial
```cpp
constexpr auto factorial = [](int n) {
    auto impl = [](int n, auto& self) -> int {
        return n <= 1 ? 1 : n * self(n - 1, self);
    };
    return impl(n, impl);
};

constexpr int fact5 = factorial(5);
static_assert(fact5 == 120);
```

### Step 3: Constexpr Capture
```cpp
constexpr int base = 10;
constexpr auto add_base = [base](int x) { return x + base; };
constexpr int result = add_base(5);
```

## Challenges

### Challenge 1: Compile-Time String Length
Write a constexpr lambda that computes string length.
```cpp
constexpr auto strlen_lambda = [](const char* s) {
    int len = 0;
    while (*s++) len++;
    return len;
};
```

### Challenge 2: Constexpr Array Initialization
Use a constexpr lambda to initialize an array at compile time.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <array>

int main() {
    // Challenge 1
    constexpr auto strlen_ct = [](const char* s) {
        int len = 0;
        while (*s++) len++;
        return len;
    };
    
    constexpr auto len = strlen_ct("Hello");
    static_assert(len == 5);
    
    // Challenge 2: Array init
    constexpr auto make_squares = []() {
        std::array<int, 10> arr{};
        for (int i = 0; i < 10; ++i) {
            arr[i] = i * i;
        }
        return arr;
    };
    
    constexpr auto squares = make_squares();
    static_assert(squares[5] == 25);
    
    for (auto s : squares) std::cout << s << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created constexpr lambda
✅ Used lambda in constant expressions
✅ Implemented compile-time factorial
✅ Initialized array at compile time (Challenge 2)

## Key Learnings
- Lambdas are implicitly constexpr if they meet requirements
- Enables powerful compile-time metaprogramming
- Captured variables must be constexpr

## Next Steps
Proceed to **Lab 14.10: Event System** to build a real application.
