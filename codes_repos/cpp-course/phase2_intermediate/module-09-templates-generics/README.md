# Module 9: Templates and Generics

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the concept of Generic Programming
- Write Function Templates to avoid code duplication
- Create Class Templates for generic data structures
- Master Template Specialization for specific types
- Understand Non-Type Template Parameters
- Introduction to Variadic Templates (C++11)
- Understand Concepts and Constraints (C++20)

---

## 📖 Theoretical Concepts

### 9.1 Function Templates

Write code that works with any type.

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
// Usage: add<int>(1, 2) or add(1, 2) (deduction)
```

### 9.2 Class Templates

Generic classes (like `std::vector`).

```cpp
template <typename T>
class Box {
    T value;
public:
    Box(T v) : value(v) {}
    T get() { return value; }
};
```

### 9.3 Template Specialization

Custom logic for specific types.

```cpp
template <>
class Box<char> {
    // Specialized implementation for char
};
```

### 9.4 Non-Type Template Parameters

Passing values (integers, pointers) instead of types.

```cpp
template <typename T, int Size>
class Array {
    T data[Size];
};
// Usage: Array<int, 10> arr;
```

### 9.5 Concepts (C++20)

Constraining templates to specific interfaces.

```cpp
template <typename T>
requires std::integral<T>
T add(T a, T b) { return a + b; }
```

---

## 🦀 Rust vs C++ Comparison

### Generics vs Templates
**C++ Templates:**
- "Duck Typing" at compile time.
- Code is generated for each instantiation.
- Extremely powerful (Turing complete metaprogramming).
- Errors can be verbose (though Concepts help).

**Rust Generics:**
- Bound by Traits.
- Checked before instantiation.
- Cleaner error messages.
- Less flexible than C++ templates (no non-type parameters in the same way, though `const generics` exist now).

### Syntax
**C++:**
```cpp
template <typename T>
T max(T a, T b) { return (a > b) ? a : b; }
```

**Rust:**
```rust
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
```

---

## 🔑 Key Takeaways

1. Templates enable code reuse without runtime overhead.
2. The compiler generates a new version of the function/class for each type used.
3. Use `typename` or `class` interchangeably in template declarations.
4. Template definitions usually go in header files (to be visible for instantiation).
5. C++20 Concepts significantly improve template error messages and safety.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 9.1:** Function Templates
2. **Lab 9.2:** Class Templates
3. **Lab 9.3:** Multiple Template Parameters
4. **Lab 9.4:** Template Specialization
5. **Lab 9.5:** Non-Type Template Parameters
6. **Lab 9.6:** Default Template Arguments
7. **Lab 9.7:** Variadic Templates (Intro)
8. **Lab 9.8:** Fold Expressions (C++17)
9. **Lab 9.9:** Concepts and Constraints (C++20)
10. **Lab 9.10:** Building a Generic Matrix Class

After completing the labs, move on to **Module 10: Standard Template Library (STL)**.
