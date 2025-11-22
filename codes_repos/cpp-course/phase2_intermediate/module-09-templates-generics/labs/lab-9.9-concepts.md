# Lab 9.9: Concepts and Constraints (C++20)

## Objective
Learn how to constrain templates to accept only specific types, improving error messages.

## Instructions

### Step 1: The Problem
Create `concepts.cpp`.
`template <typename T> T add(T a, T b) { return a + b; }`
If you pass a struct without `operator+`, you get a massive, cryptic error message.

### Step 2: The Constraint
Use `requires` to restrict T to integral types.

```cpp
#include <iostream>
#include <concepts>

template <typename T>
requires std::integral<T>
T add(T a, T b) {
    return a + b;
}

int main() {
    std::cout << add(1, 2) << std::endl;
    // add(1.5, 2.5); // Error: double is not integral
    return 0;
}
```
*Observe the clean error message.*

### Step 3: Shorthand Syntax
```cpp
void print(std::integral auto x) {
    std::cout << x << std::endl;
}
```

## Challenges

### Challenge 1: Custom Concept
Define a concept `Addable` that checks if `a + b` is valid.
```cpp
template <typename T>
concept Addable = requires(T a, T b) {
    a + b;
};
```
Use it in a template.

### Challenge 2: Floating Point
Write a function that accepts ONLY floating point numbers (`std::floating_point`).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <concepts>

template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template <Addable T>
T sum(T a, T b) { return a + b; }

void onlyFloat(std::floating_point auto x) {
    std::cout << "Float: " << x << std::endl;
}

int main() {
    sum(1, 2);
    // onlyFloat(10); // Error
    onlyFloat(3.14);
    return 0;
}
```
</details>

## Success Criteria
✅ Used standard concepts (`std::integral`)
✅ Used `requires` clause
✅ Defined custom concept (Challenge 1)
✅ Used abbreviated template syntax (Challenge 2)

## Key Learnings
- Concepts act as "interfaces" for templates
- They produce readable error messages
- They allow overloading based on constraints
- C++20 feature (requires compiler support)

## Next Steps
Proceed to **Lab 9.10: Generic Matrix Class** to apply everything.
