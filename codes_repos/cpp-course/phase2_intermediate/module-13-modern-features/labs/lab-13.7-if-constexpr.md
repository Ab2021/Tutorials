# Lab 13.7: If Constexpr (C++17)

## Objective
Use `if constexpr` for compile-time branching in templates.

## Instructions

### Step 1: The Problem
Create `if_constexpr.cpp`.
Regular `if` evaluates at runtime. Both branches must compile.

```cpp
template <typename T>
void process(T val) {
    if (std::is_integral_v<T>) {
        std::cout << "Integer: " << val * 2 << "\n";
    } else {
        std::cout << "Other: " << val << "\n";
        // val * 2 would fail for strings, but this branch compiles anyway!
    }
}
```

### Step 2: If Constexpr
Only the taken branch is instantiated.

```cpp
#include <type_traits>

template <typename T>
void process2(T val) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << val * 2 << "\n";
    } else {
        std::cout << "Other: " << val.size() << "\n"; // OK if T has .size()
    }
}
```

### Step 3: Recursive Example
```cpp
template <typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first << " ";
    if constexpr (sizeof...(args) > 0) {
        print(args...); // Only compiled if there are more args
    }
}
```

## Challenges

### Challenge 1: Type Dispatch
Write a function `describe<T>()` that prints different messages based on type:
- "Integral" for int, long, etc.
- "Floating" for float, double
- "Other" for everything else

### Challenge 2: Optimization
Use `if constexpr` to choose between two algorithms based on a compile-time condition (e.g., array size).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <type_traits>
#include <string>

template <typename T>
void describe() {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integral type\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Floating point type\n";
    } else {
        std::cout << "Other type\n";
    }
}

template <typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first;
    if constexpr (sizeof...(args) > 0) {
        std::cout << ", ";
        print(args...);
    } else {
        std::cout << "\n";
    }
}

int main() {
    describe<int>();
    describe<double>();
    describe<std::string>();
    
    print(1, 2.5, "Hello", 'A');
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `if constexpr` in templates
✅ Avoided compilation errors in unused branches
✅ Implemented type dispatch (Challenge 1)
✅ Used for variadic template recursion

## Key Learnings
- `if constexpr` discards unused branches at compile time
- Essential for generic programming
- Replaces many SFINAE use cases

## Next Steps
Proceed to **Lab 13.8: Variant and Any** for type-safe unions.
