# Module 11: Error Handling and Exceptions

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the C++ Exception mechanism (`try`, `catch`, `throw`).
- Use the Standard Exception Hierarchy (`std::exception`).
- Create Custom Exception classes.
- Understand Exception Safety Guarantees (Basic, Strong, Nothrow).
- Use `noexcept` specifier correctly.
- Learn about modern error handling alternatives (`std::optional`, `std::expected`).
- Use Assertions for debugging (`assert`, `static_assert`).

---

## 📖 Theoretical Concepts

### 11.1 The Try-Catch Block

```cpp
try {
    // Risky code
    throw std::runtime_error("Something went wrong");
} catch (const std::exception& e) {
    std::cerr << "Caught: " << e.what() << '\n';
}
```

### 11.2 Standard Exceptions

Inherit from `std::exception`.
- `std::logic_error`: Bugs in logic (e.g., `out_of_range`).
- `std::runtime_error`: Unpredictable errors (e.g., file not found).

### 11.3 Exception Safety

1.  **Basic Guarantee:** Invariants preserved, no leaks.
2.  **Strong Guarantee:** Operation either succeeds or has no effect (Transactional).
3.  **No-Throw Guarantee:** Function never throws (`noexcept`).

### 11.4 Modern Alternatives

**`std::optional<T>` (C++17):** Value might be present or not.
**`std::expected<T, E>` (C++23):** Returns value OR error (like Rust `Result`).

---

## 🦀 Rust vs C++ Comparison

### Mechanism
**C++:** Uses Exceptions (`throw`/`catch`). Unwinding the stack is expensive. Exceptions are for *exceptional* circumstances.
**Rust:** Uses Return Values (`Result<T, E>`). No exceptions (except `panic!` for unrecoverable errors).

### Safety
**C++:** Functions can throw anything (unless marked `noexcept`). Hard to know what throws.
**Rust:** Error types are part of the function signature (`-> Result<...>`). Forces handling.

### Philosophy
**C++:** "Zero Overhead" (mostly). You pay only if you throw.
**Rust:** Explicit error handling is preferred over hidden control flow.

---

## 🔑 Key Takeaways

1.  **Throw by value, catch by reference.** (`catch (const std::exception& e)`).
2.  Destructors should **never** throw.
3.  Use `std::runtime_error` for most cases; derive from it for custom errors.
4.  Use `noexcept` for move constructors and destructors.
5.  Consider `std::optional` for non-error "missing value" cases.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1.  **Lab 11.1:** Basic Try-Catch
2.  **Lab 11.2:** Multiple Catch Blocks
3.  **Lab 11.3:** Standard Exceptions
4.  **Lab 11.4:** Custom Exceptions
5.  **Lab 11.5:** Stack Unwinding and RAII
6.  **Lab 11.6:** Exception Safety Levels
7.  **Lab 11.7:** The `noexcept` Specifier
8.  **Lab 11.8:** Assertions (`assert` vs `static_assert`)
9.  **Lab 11.9:** `std::optional` (C++17)
10. **Lab 11.10:** Robust File Reader (Capstone)

After completing the labs, move on to **Module 12: File I/O and Streams**.
