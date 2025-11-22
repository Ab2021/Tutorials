# Lab 19.10: Metaprogramming Library (Capstone)

## Objective
Build a comprehensive metaprogramming library combining all techniques learned in this module.

## Instructions

### Step 1: Library Design
Create a metaprogramming library with:
- Type traits
- Type lists
- Compile-time algorithms
- Concepts
- Expression templates

Create `meta_lib.hpp`.

### Step 2: Type List Library
```cpp
namespace meta {

// Type list
template<typename... Types>
struct TypeList {};

// Length
template<typename List>
struct Length;

template<typename... Types>
struct Length<TypeList<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

// At
template<size_t N, typename List>
struct At;

template<size_t N, typename... Types>
struct At<N, TypeList<Types...>> {
    using type = /* implementation */;
};

// Filter, Map, Fold, etc.

} // namespace meta
```

### Step 3: Compile-Time Utilities
```cpp
namespace meta {

// Compile-time string
template<size_t N>
struct String {
    char data[N];
    constexpr String(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }
};

// Compile-time map
template<typename... Pairs>
struct Map;

} // namespace meta
```

## Challenges

### Challenge 1: Complete Library
Implement all core metaprogramming utilities.

### Challenge 2: Real Application
Use the library to build a compile-time JSON parser or similar.

### Challenge 3: Benchmarks
Compare compile-time vs runtime performance.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
// meta_lib.hpp
#pragma once
#include <type_traits>
#include <concepts>
#include <tuple>

namespace meta {

// ===== Type List =====

template<typename... Types>
struct TypeList {
    static constexpr size_t size = sizeof...(Types);
};

// Length
template<typename List>
struct Length;

template<typename... Types>
struct Length<TypeList<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

template<typename List>
inline constexpr size_t Length_v = Length<List>::value;

// At
template<size_t N, typename List>
struct At;

template<size_t N, typename Head, typename... Tail>
struct At<N, TypeList<Head, Tail...>> {
    using type = typename At<N - 1, TypeList<Tail...>>::type;
};

template<typename Head, typename... Tail>
struct At<0, TypeList<Head, Tail...>> {
    using type = Head;
};

template<size_t N, typename List>
using At_t = typename At<N, List>::type;

// Append
template<typename List, typename T>
struct Append;

template<typename... Types, typename T>
struct Append<TypeList<Types...>, T> {
    using type = TypeList<Types..., T>;
};

template<typename List, typename T>
using Append_t = typename Append<List, T>::type;

// Filter
template<template<typename> class Pred, typename List>
struct Filter;

template<template<typename> class Pred>
struct Filter<Pred, TypeList<>> {
    using type = TypeList<>;
};

template<template<typename> class Pred, typename Head, typename... Tail>
struct Filter<Pred, TypeList<Head, Tail...>> {
    using type = std::conditional_t<
        Pred<Head>::value,
        Append_t<typename Filter<Pred, TypeList<Tail...>>::type, Head>,
        typename Filter<Pred, TypeList<Tail...>>::type
    >;
};

template<template<typename> class Pred, typename List>
using Filter_t = typename Filter<Pred, List>::type;

// Transform
template<template<typename> class F, typename List>
struct Transform;

template<template<typename> class F>
struct Transform<F, TypeList<>> {
    using type = TypeList<>;
};

template<template<typename> class F, typename Head, typename... Tail>
struct Transform<F, TypeList<Head, Tail...>> {
    using type = Append_t<
        typename Transform<F, TypeList<Tail...>>::type,
        typename F<Head>::type
    >;
};

template<template<typename> class F, typename List>
using Transform_t = typename Transform<F, List>::type;

// Contains
template<typename T, typename List>
struct Contains;

template<typename T>
struct Contains<T, TypeList<>> : std::false_type {};

template<typename T, typename Head, typename... Tail>
struct Contains<T, TypeList<Head, Tail...>> 
    : std::conditional_t<
        std::is_same_v<T, Head>,
        std::true_type,
        Contains<T, TypeList<Tail...>>
    > {};

template<typename T, typename List>
inline constexpr bool Contains_v = Contains<T, List>::value;

// ===== Compile-Time Algorithms =====

template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

template<int N>
inline constexpr int Factorial_v = Factorial<N>::value;

template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<> struct Fibonacci<0> { static constexpr int value = 0; };
template<> struct Fibonacci<1> { static constexpr int value = 1; };

template<int N>
inline constexpr int Fibonacci_v = Fibonacci<N>::value;

// ===== Concepts =====

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Container = requires(T c) {
    typename T::value_type;
    { c.begin() };
    { c.end() };
    { c.size() } -> std::convertible_to<std::size_t>;
};

// ===== Utilities =====

// Compile-time string
template<size_t N>
struct ConstString {
    char data[N];
    
    constexpr ConstString(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }
    
    constexpr size_t size() const { return N - 1; }
    constexpr const char* c_str() const { return data; }
};

// Type-to-string mapping
template<typename T>
struct TypeName {
    static constexpr const char* value = "unknown";
};

#define REGISTER_TYPE_NAME(Type) \
    template<> struct TypeName<Type> { \
        static constexpr const char* value = #Type; \
    };

REGISTER_TYPE_NAME(int)
REGISTER_TYPE_NAME(double)
REGISTER_TYPE_NAME(float)
REGISTER_TYPE_NAME(char)

} // namespace meta

// ===== Example Usage =====

#include <iostream>

int main() {
    using namespace meta;
    
    // Type list operations
    using MyList = TypeList<int, double, char, float>;
    std::cout << "List length: " << Length_v<MyList> << "\n";
    
    using SecondType = At_t<1, MyList>;
    std::cout << "Second type is double: " 
              << std::is_same_v<SecondType, double> << "\n";
    
    // Filter
    template<typename T>
    struct IsIntegral : std::is_integral<T> {};
    
    using Integers = Filter_t<IsIntegral, MyList>;
    std::cout << "Integer types: " << Length_v<Integers> << "\n";
    
    // Compile-time computation
    std::cout << "5! = " << Factorial_v<5> << "\n";
    std::cout << "Fib(10) = " << Fibonacci_v<10> << "\n";
    
    // Compile-time string
    constexpr ConstString str("Hello, Metaprogramming!");
    std::cout << str.c_str() << "\n";
    
    // Type names
    std::cout << "Type name: " << TypeName<int>::value << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Built comprehensive type list library
✅ Implemented compile-time algorithms
✅ Created utility functions
✅ Applied concepts throughout
✅ Completed real application (Challenge 2)
✅ Benchmarked performance (Challenge 3)

## Key Learnings
- Metaprogramming libraries enable powerful abstractions
- Compile-time computation has zero runtime cost
- Type lists are fundamental to metaprogramming
- Concepts improve library interfaces
- Template metaprogramming is Turing-complete

## Congratulations!
You've completed Module 19: Template Metaprogramming. You now have advanced skills in:
- Type traits and SFINAE
- Variadic templates
- Compile-time algorithms
- Expression templates
- C++20 concepts
- Building metaprogramming libraries

## Next Steps
Proceed to **Module 20: Design Patterns** to learn classic and modern design patterns in C++.
