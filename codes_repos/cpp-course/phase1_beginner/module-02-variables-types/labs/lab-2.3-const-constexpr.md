# Lab 2.3: Const and Constexpr

## Objective
Understand the difference between run-time constants (`const`) and compile-time constants (`constexpr`).

## Instructions

### Step 1: Const Variables
Create `const_demo.cpp`:

```cpp
#include <iostream>

int main() {
    const int max_users = 100;
    // max_users = 101; // TODO: Uncomment and see error
    
    int n;
    std::cout << "Enter n: ";
    std::cin >> n;
    
    const int user_input = n; // Valid: known at runtime
    // user_input = 5; // Error
    
    return 0;
}
```

### Step 2: Constexpr Variables
```cpp
constexpr int limit = 50;
int arr[limit]; // Valid: limit is compile-time constant

// int x = 10;
// constexpr int y = x; // Error: x is not constant
```

### Step 3: Constexpr Functions
Define a function that can run at compile time:

```cpp
constexpr int square(int x) {
    return x * x;
}

int main() {
    constexpr int s = square(5); // Computed at compile time!
    int arr[s]; // Valid array size
    return 0;
}
```

## Challenges

### Challenge 1: Factorial
Write a `constexpr` function to calculate factorial. Use it to define the size of an array.

### Challenge 2: Const Pointers
Explore the difference:
```cpp
int x = 10;
const int* p1 = &x; // Pointer to const int (cannot change value)
int* const p2 = &x; // Const pointer to int (cannot change pointer)
const int* const p3 = &x; // Const pointer to const int
```
Try to modify `*p1`, `p1`, `*p2`, `p2` and see what fails.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

int main() {
    constexpr int fact5 = factorial(5); // 120
    std::cout << "5! = " << fact5 << std::endl;
    
    int arr[fact5]; // Valid
    
    int x = 10;
    int y = 20;
    
    const int* p1 = &x;
    // *p1 = 15; // Error
    p1 = &y; // OK
    
    int* const p2 = &x;
    *p2 = 15; // OK
    // p2 = &y; // Error
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood `const` vs `constexpr`
✅ Created a `constexpr` function
✅ Used `constexpr` result for array size
✅ Distinguished between `const T*` and `T* const`

## Key Learnings
- `const`: "I promise not to change this"
- `constexpr`: "This can be computed at compile time"
- `constexpr` implies `const`
- Compile-time computation improves performance

## Next Steps
Proceed to **Lab 2.4: Type Conversion Safety** to handle types correctly.
