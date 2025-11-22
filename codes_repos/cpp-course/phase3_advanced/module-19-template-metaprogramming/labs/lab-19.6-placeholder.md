# Lab 19.6: Variadic Template Patterns

## Objective
Master advanced patterns using variadic templates for flexible, generic code.

## Instructions

### Step 1: Parameter Pack Expansion
Create `variadic_patterns.cpp`.

```cpp
#include <iostream>

// Print all arguments
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << " "), ...); // Fold expression
    std::cout << "\n";
}

// Count arguments
template<typename... Args>
constexpr size_t count(Args...) {
    return sizeof...(Args);
}
```

### Step 2: Recursive Variadic Templates
```cpp
// Base case
void printRecursive() {}

// Recursive case
template<typename T, typename... Args>
void printRecursive(T first, Args... rest) {
    std::cout << first << " ";
    printRecursive(rest...);
}
```

### Step 3: Variadic Class Templates
```cpp
template<typename... Types>
class Tuple;

template<>
class Tuple<> {};

template<typename Head, typename... Tail>
class Tuple<Head, Tail...> : private Tuple<Tail...> {
    Head value;
public:
    Tuple(Head h, Tail... t) : Tuple<Tail...>(t...), value(h) {}
    Head& head() { return value; }
    Tuple<Tail...>& tail() { return *this; }
};
```

## Challenges

### Challenge 1: Type-Safe printf
Implement a type-safe printf using variadic templates.

### Challenge 2: Tuple Operations
Create `get<N>` and `apply` functions for the Tuple class.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <sstream>

// Challenge 1: Type-safe printf
template<typename... Args>
std::string format(const std::string& fmt, Args... args) {
    std::ostringstream oss;
    size_t pos = 0;
    
    auto printArg = [&](auto arg) {
        size_t found = fmt.find("{}", pos);
        if (found != std::string::npos) {
            oss << fmt.substr(pos, found - pos) << arg;
            pos = found + 2;
        }
    };
    
    (printArg(args), ...);
    oss << fmt.substr(pos);
    
    return oss.str();
}

// Challenge 2: Tuple operations

// get<N> implementation
template<size_t N, typename... Types>
struct TupleElement;

template<typename Head, typename... Tail>
struct TupleElement<0, Tuple<Head, Tail...>> {
    using type = Head;
    
    static Head& get(Tuple<Head, Tail...>& t) {
        return t.head();
    }
};

template<size_t N, typename Head, typename... Tail>
struct TupleElement<N, Tuple<Head, Tail...>> {
    using type = typename TupleElement<N - 1, Tuple<Tail...>>::type;
    
    static auto& get(Tuple<Head, Tail...>& t) {
        return TupleElement<N - 1, Tuple<Tail...>>::get(t.tail());
    }
};

template<size_t N, typename... Types>
auto& get(Tuple<Types...>& t) {
    return TupleElement<N, Tuple<Types...>>::get(t);
}

// apply function
template<typename F, typename... Args>
auto apply(F&& f, Tuple<Args...>& t) {
    return applyImpl(std::forward<F>(f), t, std::index_sequence_for<Args...>{});
}

template<typename F, typename Tuple, size_t... Is>
auto applyImpl(F&& f, Tuple& t, std::index_sequence<Is...>) {
    return f(get<Is>(t)...);
}

// Additional patterns

// All same type check
template<typename T, typename... Args>
struct AllSame : std::conjunction<std::is_same<T, Args>...> {};

// Sum of all arguments
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

// Max of all arguments
template<typename T>
constexpr T max(T a) { return a; }

template<typename T, typename... Args>
constexpr T max(T a, Args... args) {
    T rest = max(args...);
    return a > rest ? a : rest;
}

int main() {
    // Type-safe printf
    std::cout << format("Hello, {}! You are {} years old.", "Alice", 30) << "\n";
    
    // Tuple operations
    Tuple<int, double, std::string> t(42, 3.14, "hello");
    std::cout << "Element 0: " << get<0>(t) << "\n";
    std::cout << "Element 1: " << get<1>(t) << "\n";
    std::cout << "Element 2: " << get<2>(t) << "\n";
    
    // Apply
    auto printer = [](int a, double b, const std::string& c) {
        std::cout << a << ", " << b << ", " << c << "\n";
    };
    apply(printer, t);
    
    // Sum
    std::cout << "Sum: " << sum(1, 2, 3, 4, 5) << "\n";
    
    // Max
    std::cout << "Max: " << max(5, 2, 8, 1, 9) << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used parameter pack expansion
✅ Implemented recursive variadic templates
✅ Created variadic class templates
✅ Built type-safe printf (Challenge 1)
✅ Implemented tuple operations (Challenge 2)

## Key Learnings
- Fold expressions simplify parameter pack expansion
- Recursive templates process packs element-by-element
- Variadic templates enable flexible APIs
- `sizeof...` counts template parameters

## Next Steps
Proceed to **Lab 19.7: Expression Templates**.
