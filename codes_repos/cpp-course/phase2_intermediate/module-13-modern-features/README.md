# Module 13: Modern C++ Features (C++11/14/17/20)

## 🎯 Learning Objectives

By the end of this module, you will:
- Master `auto` type deduction and when to use it
- Use `nullptr` instead of `NULL`
- Understand `constexpr` for compile-time computation
- Use Range-based for loops effectively
- Master Uniform Initialization (`{}`)
- Use Structured Bindings (C++17)
- Understand `if constexpr` (C++17)
- Use `std::variant` and `std::any` (C++17)
- Understand Designated Initializers (C++20)
- Use the Spaceship Operator `<=>` (C++20)

---

## 📖 Theoretical Concepts

### 13.1 Auto (C++11)

Let the compiler deduce types.

```cpp
auto x = 5; // int
auto y = 3.14; // double
auto z = std::vector<int>{1, 2, 3}; // vector<int>
```

### 13.2 Nullptr (C++11)

Type-safe null pointer.

```cpp
int* p = nullptr; // Not NULL or 0
```

### 13.3 Constexpr (C++11/14/17)

Compile-time constants and functions.

```cpp
constexpr int square(int x) { return x * x; }
constexpr int val = square(5); // Computed at compile time
```

### 13.4 Range-Based For (C++11)

```cpp
for (auto& item : container) { ... }
```

### 13.5 Structured Bindings (C++17)

Unpack tuples/pairs/structs.

```cpp
auto [x, y] = std::make_pair(1, 2);
```

### 13.6 If Constexpr (C++17)

Compile-time conditional compilation.

```cpp
if constexpr (std::is_integral_v<T>) { ... }
```

---

## 🦀 Rust vs C++ Comparison

### Type Inference
**C++:** `auto` (must be initialized).
**Rust:** `let` (always inferred unless annotated).

### Null Safety
**C++:** `nullptr` is still a raw pointer (can dangle).
**Rust:** `Option<T>` forces explicit handling of "no value".

### Compile-Time Execution
**C++:** `constexpr` functions.
**Rust:** `const fn` (more restrictive but guaranteed compile-time).

### Pattern Matching
**C++:** `std::variant` + `std::visit`.
**Rust:** Native `match` expression (more ergonomic).

---

## 🔑 Key Takeaways

1. Use `auto` for complex types, but not for clarity-critical code.
2. Always use `nullptr`, never `NULL` or `0`.
3. `constexpr` enables zero-runtime-cost abstractions.
4. Range-based for is cleaner and safer than index loops.
5. Structured bindings make tuple/pair code readable.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 13.1:** Auto Type Deduction
2. **Lab 13.2:** Nullptr and Type Safety
3. **Lab 13.3:** Constexpr Functions
4. **Lab 13.4:** Range-Based For Loops
5. **Lab 13.5:** Uniform Initialization
6. **Lab 13.6:** Structured Bindings (C++17)
7. **Lab 13.7:** If Constexpr (C++17)
8. **Lab 13.8:** Variant and Any (C++17)
9. **Lab 13.9:** Designated Initializers (C++20)
10. **Lab 13.10:** Spaceship Operator (C++20)

After completing the labs, move on to **Module 14: Lambda Expressions (Deep Dive)**.
