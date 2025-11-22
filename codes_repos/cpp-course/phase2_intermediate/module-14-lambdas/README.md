# Module 14: Lambda Expressions (Deep Dive)

## 🎯 Learning Objectives

By the end of this module, you will:
- Master lambda syntax and capture mechanisms
- Understand capture by value vs reference
- Use `mutable` lambdas
- Create generic lambdas (C++14)
- Use init-capture (C++14)
- Understand lambda types and `std::function`
- Use lambdas with STL algorithms
- Master IIFE (Immediately Invoked Function Expression)
- Use `constexpr` lambdas (C++17)
- Understand template lambdas (C++20)

---

## 📖 Theoretical Concepts

### 14.1 Lambda Syntax

```cpp
[capture](parameters) -> return_type { body }
```

Example:
```cpp
auto add = [](int a, int b) { return a + b; };
```

### 14.2 Capture Modes

- `[]`: Capture nothing
- `[=]`: Capture all by value
- `[&]`: Capture all by reference
- `[x]`: Capture `x` by value
- `[&x]`: Capture `x` by reference
- `[=, &x]`: Capture all by value except `x` by reference

### 14.3 Generic Lambdas (C++14)

```cpp
auto print = [](auto x) { std::cout << x; };
```

### 14.4 Init-Capture (C++14)

```cpp
auto ptr = std::make_unique<int>(5);
auto lambda = [p = std::move(ptr)]() { return *p; };
```

---

## 🦀 Rust vs C++ Comparison

### Closures
**C++:** Lambdas with explicit capture.
**Rust:** Closures with automatic capture inference (`Fn`, `FnMut`, `FnOnce` traits).

### Syntax
**C++:** `[capture](params) { body }`
**Rust:** `|params| { body }`

### Move Semantics
**C++:** Must use init-capture or `std::move` in capture.
**Rust:** `move` keyword moves all captured variables.

---

## 🔑 Key Takeaways

1. Lambdas are anonymous function objects (functors).
2. Capture by reference `[&]` can lead to dangling references.
3. `mutable` allows modifying captured-by-value variables.
4. Generic lambdas are templates in disguise.
5. Prefer lambdas over `std::bind` (deprecated).

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 14.1:** Basic Lambda Syntax
2. **Lab 14.2:** Capture Mechanisms
3. **Lab 14.3:** Mutable Lambdas
4. **Lab 14.4:** Generic Lambdas (C++14)
5. **Lab 14.5:** Init-Capture (C++14)
6. **Lab 14.6:** Lambda Types and std::function
7. **Lab 14.7:** Lambdas with STL Algorithms
8. **Lab 14.8:** IIFE Pattern
9. **Lab 14.9:** Constexpr Lambdas (C++17)
10. **Lab 14.10:** Event System (Capstone)

After completing the labs, move on to **Module 15: Smart Pointers (Deep Dive)**.
