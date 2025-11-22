# Lab 9.7: Variadic Templates (Intro)

## Objective
Learn how to write templates that accept any number of arguments (C++11).

## Instructions

### Step 1: The Base Case
Create `variadic.cpp`. Recursive variadic templates need a base case to stop recursion.

```cpp
#include <iostream>

void print() {
    std::cout << std::endl; // End of line
}
```

### Step 2: The Recursive Case
`typename... Args` is a parameter pack.

```cpp
template <typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first << " ";
    print(args...); // Recursive call with remaining args
}
```

### Step 3: Usage
```cpp
int main() {
    print(1, 2.5, "Hello", 'A');
    return 0;
}
```
*Logic: Prints 1, calls print(2.5, ...). Prints 2.5, calls print("Hello", ...). Etc.*

## Challenges

### Challenge 1: Count Arguments
Write a function `countArgs(Args... args)` that returns `sizeof...(args)`.
This operator returns the number of elements in the pack.

### Challenge 2: Sum Function
Write a variadic `sum` function that adds all arguments together.
Base case: `sum(T v) { return v; }`.
Recursive: `return first + sum(rest...);`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

// Challenge 1
template <typename... Args>
int countArgs(Args... args) {
    return sizeof...(args);
}

// Challenge 2
template <typename T>
T sum(T v) { return v; }

template <typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}

int main() {
    std::cout << "Count: " << countArgs(1, 2, 3) << std::endl; // 3
    std::cout << "Sum: " << sum(1, 2, 3, 4, 5) << std::endl; // 15
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented variadic function template
✅ Implemented base case and recursive step
✅ Used `sizeof...` operator (Challenge 1)
✅ Implemented variadic sum (Challenge 2)

## Key Learnings
- `...` denotes a parameter pack
- Recursive expansion was the standard way in C++11
- `sizeof...` counts arguments at compile time

## Next Steps
Proceed to **Lab 9.8: Fold Expressions** for the modern C++17 way.
