# Lab 19.9: C++20 Concepts

## Objective
Use C++20 concepts to constrain templates with clear, readable requirements.

## Instructions

### Step 1: Basic Concepts
Create `concepts.cpp`.

```cpp
#include <iostream>
#include <concepts>

// Basic concept
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

// Using the concept
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// Compound concept
template<typename T>
concept Incrementable = requires(T x) {
    { ++x } -> std::same_as<T&>;
    { x++ } -> std::same_as<T>;
};
```

### Step 2: Requires Clauses
```cpp
template<typename T>
requires std::integral<T>
T multiply(T a, T b) {
    return a * b;
}

// Or with concept directly
template<std::integral T>
T divide(T a, T b) {
    return a / b;
}
```

### Step 3: Requires Expressions
```cpp
template<typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print(const T& value) {
    std::cout << value << "\n";
}
```

### Step 4: Concept Composition
```cpp
template<typename T>
concept SignedIntegral = std::integral<T> && std::signed_integral<T>;

template<typename T>
concept Container = requires(T c) {
    typename T::value_type;
    { c.begin() } -> std::same_as<typename T::iterator>;
    { c.end() } -> std::same_as<typename T::iterator>;
    { c.size() } -> std::convertible_to<std::size_t>;
};
```

## Challenges

### Challenge 1: Custom Concepts
Create concepts for `Comparable`, `Hashable`, and `Serializable`.

### Challenge 2: Concept Overloading
Use concepts to overload functions based on type properties.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <concepts>
#include <vector>
#include <string>

// Challenge 1: Custom concepts

template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
};

template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept Serializable = requires(T t) {
    { t.serialize() } -> std::convertible_to<std::string>;
    { T::deserialize(std::string{}) } -> std::same_as<T>;
};

// Challenge 2: Concept overloading

// For integral types
template<std::integral T>
void process(T value) {
    std::cout << "Processing integer: " << value << "\n";
}

// For floating point types
template<std::floating_point T>
void process(T value) {
    std::cout << "Processing float: " << value << "\n";
}

// For containers
template<typename T>
concept Container = requires(T c) {
    typename T::value_type;
    { c.begin() };
    { c.end() };
    { c.size() } -> std::convertible_to<std::size_t>;
};

template<Container T>
void process(const T& container) {
    std::cout << "Processing container with " << container.size() << " elements\n";
}

// Advanced concepts

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<typename T>
concept Range = requires(T r) {
    { std::begin(r) };
    { std::end(r) };
};

template<typename T, typename U>
concept ConvertibleTo = std::convertible_to<T, U>;

// Subsumption example
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = Arithmetic<T> && std::floating_point<T>;

template<Arithmetic T>
void compute(T value) {
    std::cout << "Arithmetic computation\n";
}

template<FloatingPoint T>  // More constrained, preferred
void compute(T value) {
    std::cout << "Floating point computation\n";
}

// Real-world example: Generic algorithm
template<Range R, typename Pred>
requires std::predicate<Pred, typename R::value_type>
auto filter(const R& range, Pred pred) {
    std::vector<typename R::value_type> result;
    for (const auto& item : range) {
        if (pred(item)) {
            result.push_back(item);
        }
    }
    return result;
}

int main() {
    // Basic usage
    std::cout << add(5, 3) << "\n";
    std::cout << add(2.5, 1.5) << "\n";
    
    // Concept overloading
    process(42);
    process(3.14);
    process(std::vector<int>{1, 2, 3});
    
    // Subsumption
    compute(42);
    compute(3.14);
    
    // Generic algorithm
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
    auto evens = filter(numbers, [](int x) { return x % 2 == 0; });
    
    std::cout << "Even numbers: ";
    for (int x : evens) std::cout << x << " ";
    std::cout << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created basic concepts
✅ Used requires clauses
✅ Wrote requires expressions
✅ Composed concepts
✅ Implemented custom concepts (Challenge 1)
✅ Used concept overloading (Challenge 2)

## Key Learnings
- Concepts provide clear template constraints
- More readable than SFINAE
- Better error messages
- Concept subsumption enables overloading
- Requires expressions check type properties

## Next Steps
Proceed to **Lab 19.10: Metaprogramming Library (Capstone)**.
