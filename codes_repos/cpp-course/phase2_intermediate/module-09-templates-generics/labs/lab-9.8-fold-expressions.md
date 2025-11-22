# Lab 9.8: Fold Expressions (C++17)

## Objective
Simplify variadic templates using C++17 Fold Expressions.

## Instructions

### Step 1: Sum without Recursion
Create `fold.cpp`.

```cpp
#include <iostream>

template <typename... Args>
auto sum(Args... args) {
    return (args + ...); // Unary right fold
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << std::endl;
    return 0;
}
```
*Much cleaner than Lab 9.7!*

### Step 2: Print with Separator
Binary left fold.

```cpp
template <typename... Args>
void print(Args... args) {
    (std::cout << ... << args) << std::endl;
}
```

### Step 3: Complex Logic
Check if ALL arguments are true.

```cpp
template <typename... Args>
bool allTrue(Args... args) {
    return (args && ...);
}
```

## Challenges

### Challenge 1: Comma Operator
Use the comma operator to run a function on every argument.
`template <typename... Args> void process(Args... args) { (func(args), ...); }`

### Challenge 2: Average
Write a function `average(Args... args)` that calculates sum using fold expression and divides by `sizeof...(args)`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

template <typename... Args>
double average(Args... args) {
    return (args + ...) / (double)sizeof...(args);
}

void doWork(int x) { std::cout << "Working " << x << "\n"; }

template <typename... Args>
void process(Args... args) {
    (doWork(args), ...); // Comma fold
}

int main() {
    std::cout << "Avg: " << average(10, 20, 30) << std::endl;
    process(1, 2, 3);
    return 0;
}
```
</details>

## Success Criteria
✅ Used Unary Fold `(args + ...)`
✅ Used Binary Fold `(init + ... + args)`
✅ Implemented `allTrue` logic
✅ Implemented comma fold (Challenge 1)

## Key Learnings
- Fold expressions replace recursion for most variadic tasks
- Supported operators: `+`, `-`, `*`, `/`, `&&`, `||`, `,`, `<<`, `>>`
- Significantly reduces compile time and code complexity

## Next Steps
Proceed to **Lab 9.9: Concepts** for C++20 constraints.
