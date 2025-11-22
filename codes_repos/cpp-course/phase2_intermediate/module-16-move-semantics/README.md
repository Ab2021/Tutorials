# Module 16: Move Semantics (Deep Dive)

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand rvalue references (`T&&`)
- Master move constructors and move assignment operators
- Understand `std::move` and `std::forward`
- Implement the Rule of Five
- Understand perfect forwarding
- Use move semantics for performance optimization
- Understand value categories (lvalue, rvalue, xvalue, prvalue, glvalue)
- Avoid common move semantics pitfalls
- Implement move-only types
- Understand return value optimization (RVO/NRVO)

---

## 📖 Theoretical Concepts

### 16.1 Rvalue References

```cpp
int&& rref = 5; // Binds to temporary
std::string&& s = std::string("Hello"); // Binds to temporary
```

### 16.2 Move Constructor

```cpp
class Buffer {
    int* data;
public:
    // Move constructor
    Buffer(Buffer&& other) noexcept 
        : data(other.data) {
        other.data = nullptr; // Leave in valid state
    }
};
```

### 16.3 std::move

Casts to rvalue reference (doesn't actually move anything).

```cpp
std::vector<int> v1 = {1, 2, 3};
std::vector<int> v2 = std::move(v1); // Move, v1 is now empty
```

### 16.4 Perfect Forwarding

```cpp
template <typename T>
void wrapper(T&& arg) {
    func(std::forward<T>(arg)); // Preserves value category
}
```

---

## 🦀 Rust vs C++ Comparison

### Move Semantics
**C++:** Explicit via `std::move`, leaves source in valid-but-unspecified state.
**Rust:** Implicit move by default, source becomes inaccessible (compile error if used).

### Copy vs Move
**C++:** Copy by default (expensive), must explicitly move.
**Rust:** Move by default (cheap), must explicitly clone.

### Safety
**C++:** Can use moved-from object (undefined behavior if not in valid state).
**Rust:** Borrow checker prevents use after move at compile time.

---

## 🔑 Key Takeaways

1. Move semantics enable efficient transfer of resources.
2. `std::move` is just a cast, doesn't move anything.
3. Move constructors should be `noexcept`.
4. Moved-from objects must be in a valid state.
5. RVO/NRVO often eliminates the need for explicit moves.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 16.1:** Rvalue References Basics
2. **Lab 16.2:** Move Constructor
3. **Lab 16.3:** Move Assignment Operator
4. **Lab 16.4:** Rule of Five
5. **Lab 16.5:** std::move Deep Dive
6. **Lab 16.6:** Perfect Forwarding
7. **Lab 16.7:** Move-Only Types
8. **Lab 16.8:** RVO and NRVO
9. **Lab 16.9:** Performance Optimization
10. **Lab 16.10:** String Builder (Capstone)

After completing the labs, move on to **Module 17: Concurrency and Multithreading**.
