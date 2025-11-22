# Lab 13.3: Constexpr Functions

## Objective
Use `constexpr` to perform computations at compile time.

## Instructions

### Step 1: Constexpr Variables
Create `constexpr_demo.cpp`.

```cpp
#include <iostream>

constexpr int SIZE = 100; // Compile-time constant
int arr[SIZE]; // OK: SIZE is known at compile time

int main() {
    constexpr int x = 5 * 5; // Computed at compile time
    std::cout << x << std::endl;
    return 0;
}
```

### Step 2: Constexpr Functions (C++11)
Functions that CAN be evaluated at compile time.

```cpp
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    constexpr int fact5 = factorial(5); // Compile-time
    int arr[factorial(4)]; // Array size must be compile-time constant
    
    int runtime = 6;
    int fact6 = factorial(runtime); // Runtime evaluation (allowed)
    return 0;
}
```

### Step 3: Constexpr Constructors (C++11/14)
```cpp
class Point {
    int x, y;
public:
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int getX() const { return x; }
};

constexpr Point p(10, 20);
constexpr int val = p.getX(); // Compile-time
```

## Challenges

### Challenge 1: Fibonacci
Write a `constexpr` Fibonacci function.
Create an array of size `fib(10)`.

### Challenge 2: String Length (C++17)
C++17 allows `constexpr` on more complex functions.
Write `constexpr size_t strlen_ct(const char* s)` that computes string length at compile time.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

constexpr int fib(int n) {
    return (n <= 1) ? n : fib(n - 1) + fib(n - 2);
}

constexpr size_t strlen_ct(const char* s) {
    return *s ? 1 + strlen_ct(s + 1) : 0;
}

int main() {
    // Challenge 1
    constexpr int fibVal = fib(10);
    int arr[fibVal]; // Size = 55
    std::cout << "Fib(10) = " << fibVal << ", Array size: " << sizeof(arr)/sizeof(int) << "\n";
    
    // Challenge 2
    constexpr auto len = strlen_ct("Hello");
    static_assert(len == 5, "Length should be 5");
    std::cout << "Length: " << len << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `constexpr` variables
✅ Wrote `constexpr` functions
✅ Used `constexpr` constructors
✅ Implemented compile-time Fibonacci (Challenge 1)
✅ Implemented compile-time strlen (Challenge 2)

## Key Learnings
- `constexpr` enables zero-runtime-cost abstractions
- Functions can be used at compile time OR runtime
- C++14/17/20 progressively relaxed `constexpr` restrictions

## Next Steps
Proceed to **Lab 13.4: Range-Based For** for cleaner loops.
