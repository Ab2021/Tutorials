# Lab 19.1: Type Traits Basics

## Objective
Learn to use standard type traits for compile-time type information.

## Instructions

### Step 1: Basic Type Traits
Create `type_traits_basics.cpp`.

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
void analyzeType() {
    std::cout << "Type analysis:\n";
    std::cout << "  Is integral: " << std::is_integral_v<T> << "\n";
    std::cout << "  Is floating point: " << std::is_floating_point_v<T> << "\n";
    std::cout << "  Is pointer: " << std::is_pointer_v<T> << "\n";
    std::cout << "  Is const: " << std::is_const_v<T> << "\n";
    std::cout << "  Size: " << sizeof(T) << " bytes\n";
}

int main() {
    analyzeType<int>();
    analyzeType<const double>();
    analyzeType<int*>();
    
    return 0;
}
```

### Step 2: Conditional Compilation
```cpp
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integer: " << value << "\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing float: " << value << "\n";
    } else {
        std::cout << "Processing other type\n";
    }
}
```

### Step 3: Type Modifications
```cpp
template<typename T>
void demonstrateModifications() {
    using NoConst = std::remove_const_t<T>;
    using NoRef = std::remove_reference_t<T>;
    using Pointer = std::add_pointer_t<T>;
    
    std::cout << "Original is const: " << std::is_const_v<T> << "\n";
    std::cout << "Removed const is const: " << std::is_const_v<NoConst> << "\n";
}
```

## Challenges

### Challenge 1: Type Checker
Create a function that checks if a type is arithmetic (integral or floating point).

### Challenge 2: Safe Cast
Implement a safe casting function using type traits.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <type_traits>
#include <string>

// Challenge 1: Type checker
template<typename T>
constexpr bool is_arithmetic_type() {
    return std::is_integral_v<T> || std::is_floating_point_v<T>;
}

template<typename T>
void printIfArithmetic(T value) {
    if constexpr (is_arithmetic_type<T>()) {
        std::cout << "Arithmetic value: " << value << "\n";
    } else {
        std::cout << "Not an arithmetic type\n";
    }
}

// Challenge 2: Safe cast
template<typename To, typename From>
std::enable_if_t<std::is_convertible_v<From, To>, To>
safe_cast(From value) {
    return static_cast<To>(value);
}

// Comprehensive type analysis
template<typename T>
void comprehensiveAnalysis() {
    std::cout << "\n=== Type Analysis ===\n";
    std::cout << "Is fundamental: " << std::is_fundamental_v<T> << "\n";
    std::cout << "Is arithmetic: " << std::is_arithmetic_v<T> << "\n";
    std::cout << "Is signed: " << std::is_signed_v<T> << "\n";
    std::cout << "Is unsigned: " << std::is_unsigned_v<T> << "\n";
    std::cout << "Is class: " << std::is_class_v<T> << "\n";
    std::cout << "Is polymorphic: " << std::is_polymorphic_v<T> << "\n";
}

int main() {
    // Challenge 1
    printIfArithmetic(42);
    printIfArithmetic(3.14);
    printIfArithmetic(std::string("hello"));
    
    // Challenge 2
    int i = 42;
    double d = safe_cast<double>(i);
    std::cout << "Casted value: " << d << "\n";
    
    // Comprehensive analysis
    comprehensiveAnalysis<int>();
    comprehensiveAnalysis<std::string>();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used standard type traits
✅ Applied `if constexpr` for conditional compilation
✅ Modified types with trait transformations
✅ Created type checker (Challenge 1)
✅ Implemented safe cast (Challenge 2)

## Key Learnings
- Type traits provide compile-time type information
- `if constexpr` enables conditional compilation
- Type modifications create new types from existing ones
- `_v` suffix provides value, `_t` suffix provides type

## Next Steps
Proceed to **Lab 19.2: Custom Type Traits**.
