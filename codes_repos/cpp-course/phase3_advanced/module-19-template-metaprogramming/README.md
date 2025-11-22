# Module 19: Template Metaprogramming

## Overview
Template metaprogramming is a powerful technique that allows you to perform computations and make decisions at compile time. This module covers advanced template techniques that enable you to write highly generic and efficient code.

## Learning Objectives
By the end of this module, you will be able to:
- Understand and use type traits
- Apply SFINAE for template selection
- Write compile-time algorithms
- Create expression templates
- Implement advanced template patterns
- Use C++20 concepts for constraints

## Key Concepts

### 1. Type Traits
Type traits provide compile-time information about types.
```cpp
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // Handle integers
    } else if constexpr (std::is_floating_point_v<T>) {
        // Handle floats
    }
}
```

### 2. SFINAE (Substitution Failure Is Not An Error)
Enable or disable template instantiations based on type properties.
```cpp
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
add(T a, T b) {
    return a + b;
}
```

### 3. Variadic Templates
Work with arbitrary numbers of template parameters.
```cpp
template<typename... Args>
auto sum(Args... args) {
    return (args + ...); // Fold expression
}
```

### 4. Compile-Time Computation
Perform calculations at compile time using `constexpr`.
```cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```

### 5. Expression Templates
Optimize complex expressions by delaying evaluation.
```cpp
template<typename E>
class VecExpression {
    // Lazy evaluation of vector operations
};
```

## Rust Comparison

### Type Traits
**C++:**
```cpp
if constexpr (std::is_integral_v<T>) { }
```

**Rust:**
```rust
where T: num::Integer { }
```

### Compile-Time Computation
**C++:**
```cpp
constexpr int value = factorial(5);
```

**Rust:**
```rust
const VALUE: i32 = factorial(5);
```

## Labs

1. **Lab 19.1**: Type Traits Basics
2. **Lab 19.2**: Custom Type Traits
3. **Lab 19.3**: SFINAE Techniques
4. **Lab 19.4**: Tag Dispatch
5. **Lab 19.5**: Compile-Time Algorithms
6. **Lab 19.6**: Variadic Template Patterns
7. **Lab 19.7**: Expression Templates
8. **Lab 19.8**: Template Recursion
9. **Lab 19.9**: C++20 Concepts
10. **Lab 19.10**: Metaprogramming Library (Capstone)

## Additional Resources
- "C++ Templates: The Complete Guide" by Vandevoorde & Josuttis
- "Modern C++ Design" by Andrei Alexandrescu
- cppreference.com - Template metaprogramming

## Next Module
After completing this module, proceed to **Module 20: Design Patterns**.
