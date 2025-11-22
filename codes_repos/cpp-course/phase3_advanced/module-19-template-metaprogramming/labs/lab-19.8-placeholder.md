# Lab 19.8: Template Recursion

## Objective
Master template recursion techniques for compile-time computation and type manipulation.

## Instructions

### Step 1: Basic Template Recursion
Create `template_recursion.cpp`.

```cpp
#include <iostream>

// Compile-time list length
template<typename... Types>
struct Length;

template<>
struct Length<> {
    static constexpr size_t value = 0;
};

template<typename Head, typename... Tail>
struct Length<Head, Tail...> {
    static constexpr size_t value = 1 + Length<Tail...>::value;
};
```

### Step 2: Type List Operations
```cpp
// Get Nth type
template<size_t N, typename... Types>
struct TypeAt;

template<typename Head, typename... Tail>
struct TypeAt<0, Head, Tail...> {
    using type = Head;
};

template<size_t N, typename Head, typename... Tail>
struct TypeAt<N, Head, Tail...> {
    using type = typename TypeAt<N - 1, Tail...>::type;
};
```

### Step 3: Type List Transformation
```cpp
// Add pointer to all types
template<typename... Types>
struct AddPointer;

template<>
struct AddPointer<> {
    using type = std::tuple<>;
};

template<typename Head, typename... Tail>
struct AddPointer<Head, Tail...> {
    using type = /* implementation */;
};
```

## Challenges

### Challenge 1: Type List Filtering
Filter types from a type list based on a predicate.

### Challenge 2: Type List Reversal
Reverse the order of types in a type list.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <type_traits>
#include <tuple>

// Type list utilities

// Length
template<typename... Types>
struct Length {
    static constexpr size_t value = sizeof...(Types);
};

// TypeAt
template<size_t N, typename... Types>
struct TypeAt;

template<typename Head, typename... Tail>
struct TypeAt<0, Head, Tail...> {
    using type = Head;
};

template<size_t N, typename Head, typename... Tail>
struct TypeAt<N, Head, Tail...> {
    using type = typename TypeAt<N - 1, Tail...>::type;
};

template<size_t N, typename... Types>
using TypeAt_t = typename TypeAt<N, Types...>::type;

// Append
template<typename List, typename T>
struct Append;

template<template<typename...> class List, typename... Types, typename T>
struct Append<List<Types...>, T> {
    using type = List<Types..., T>;
};

// Challenge 1: Filter
template<template<typename> class Pred, typename... Types>
struct Filter;

template<template<typename> class Pred>
struct Filter<Pred> {
    using type = std::tuple<>;
};

template<template<typename> class Pred, typename Head, typename... Tail>
struct Filter<Pred, Head, Tail...> {
    using type = std::conditional_t<
        Pred<Head>::value,
        typename Append<typename Filter<Pred, Tail...>::type, Head>::type,
        typename Filter<Pred, Tail...>::type
    >;
};

// Predicate: is pointer
template<typename T>
struct IsPointer : std::is_pointer<T> {};

// Challenge 2: Reverse
template<typename... Types>
struct Reverse;

template<>
struct Reverse<> {
    using type = std::tuple<>;
};

template<typename Head, typename... Tail>
struct Reverse<Head, Tail...> {
    using type = typename Append<
        typename Reverse<Tail...>::type,
        Head
    >::type;
};

// Additional operations

// Contains
template<typename T, typename... Types>
struct Contains;

template<typename T>
struct Contains<T> : std::false_type {};

template<typename T, typename Head, typename... Tail>
struct Contains<T, Head, Tail...> 
    : std::conditional_t<
        std::is_same_v<T, Head>,
        std::true_type,
        Contains<T, Tail...>
    > {};

// Transform
template<template<typename> class F, typename... Types>
struct Transform;

template<template<typename> class F>
struct Transform<F> {
    using type = std::tuple<>;
};

template<template<typename> class F, typename Head, typename... Tail>
struct Transform<F, Head, Tail...> {
    using type = typename Append<
        typename Transform<F, Tail...>::type,
        typename F<Head>::type
    >::type;
};

// Example transformation: add pointer
template<typename T>
struct AddPointer {
    using type = T*;
};

// Demonstration
template<typename... Types>
void printTypes() {
    std::cout << "Type list with " << sizeof...(Types) << " types\n";
}

int main() {
    // Length
    std::cout << "Length: " << Length<int, double, char>::value << "\n";
    
    // TypeAt
    using SecondType = TypeAt_t<1, int, double, char>;
    std::cout << "Second type is double: " 
              << std::is_same_v<SecondType, double> << "\n";
    
    // Filter pointers
    using Mixed = std::tuple<int, int*, double, char*, float>;
    using Pointers = typename Filter<IsPointer, int, int*, double, char*, float>::type;
    
    // Reverse
    using Original = std::tuple<int, double, char>;
    using Reversed = typename Reverse<int, double, char>::type;
    
    // Transform
    using WithPointers = typename Transform<AddPointer, int, double, char>::type;
    
    std::cout << "Metaprogramming complete!\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented template recursion
✅ Created type list operations
✅ Built type transformations
✅ Implemented type filtering (Challenge 1)
✅ Created type list reversal (Challenge 2)

## Key Learnings
- Template recursion processes types at compile time
- Base cases terminate recursion
- Type lists enable powerful metaprogramming
- Compile-time type manipulation has zero runtime cost

## Next Steps
Proceed to **Lab 19.9: C++20 Concepts**.
