# Module 15: Smart Pointers (Deep Dive)

## 🎯 Learning Objectives

By the end of this module, you will:
- Master `std::unique_ptr` for exclusive ownership
- Understand `std::shared_ptr` and reference counting
- Use `std::weak_ptr` to break circular references
- Understand custom deleters
- Master `std::make_unique` and `std::make_shared`
- Avoid common smart pointer pitfalls
- Understand ownership semantics
- Use smart pointers with polymorphism
- Implement RAII patterns with smart pointers
- Understand performance implications

---

## 📖 Theoretical Concepts

### 15.1 Unique Pointer

Exclusive ownership, non-copyable, movable.

```cpp
std::unique_ptr<int> p = std::make_unique<int>(42);
// p2 = p; // Error: cannot copy
auto p2 = std::move(p); // OK: transfer ownership
```

### 15.2 Shared Pointer

Shared ownership with reference counting.

```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(42);
std::shared_ptr<int> p2 = p1; // Both own the object
// Object deleted when last shared_ptr is destroyed
```

### 15.3 Weak Pointer

Non-owning observer, breaks circular references.

```cpp
std::weak_ptr<int> wp = sp;
if (auto sp2 = wp.lock()) { // Check if still alive
    // Use sp2
}
```

### 15.4 Custom Deleters

```cpp
auto deleter = [](FILE* f) { fclose(f); };
std::unique_ptr<FILE, decltype(deleter)> file(fopen("file.txt", "r"), deleter);
```

---

## 🦀 Rust vs C++ Comparison

### Ownership
**C++:** `unique_ptr` = exclusive ownership, `shared_ptr` = shared ownership.
**Rust:** `Box<T>` = exclusive, `Rc<T>` = shared (single-threaded), `Arc<T>` = shared (thread-safe).

### Reference Counting
**C++:** `shared_ptr` uses atomic reference counting (thread-safe but overhead).
**Rust:** `Rc` is non-atomic (faster, single-threaded), `Arc` is atomic.

### Weak References
**C++:** `weak_ptr` for breaking cycles.
**Rust:** `Weak<T>` for both `Rc` and `Arc`.

### Safety
**C++:** Can still have dangling pointers if misused (e.g., storing raw pointer from `get()`).
**Rust:** Borrow checker prevents dangling references at compile time.

---

## 🔑 Key Takeaways

1. Prefer `unique_ptr` by default (zero overhead when not moved).
2. Use `shared_ptr` only when shared ownership is truly needed.
3. Always use `make_unique`/`make_shared` (exception-safe, efficient).
4. Never mix raw pointers and smart pointers for the same object.
5. Use `weak_ptr` to break circular references.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 15.1:** Unique Pointer Basics
2. **Lab 15.2:** Unique Pointer with Arrays
3. **Lab 15.3:** Shared Pointer Basics
4. **Lab 15.4:** Weak Pointer and Circular References
5. **Lab 15.5:** Custom Deleters
6. **Lab 15.6:** Make Functions
7. **Lab 15.7:** Smart Pointers and Polymorphism
8. **Lab 15.8:** Performance Comparison
9. **Lab 15.9:** Common Pitfalls
10. **Lab 15.10:** Resource Manager (Capstone)

After completing the labs, move on to **Module 16: Move Semantics (Deep Dive)**.
