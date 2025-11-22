# Module 6: Memory Management Basics

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the difference between Stack and Heap memory
- Manage dynamic memory manually (`new`/`delete`)
- Identify and prevent memory leaks
- Understand the RAII idiom (Resource Acquisition Is Initialization)
- Implement deep copying to prevent double-free errors
- Write destructors for proper cleanup
- Build a custom class that manages its own memory

---

## 📖 Theoretical Concepts

### 6.1 Stack vs Heap

**Stack:**
- Fast allocation (moving a pointer)
- Automatic cleanup (scope exit)
- Limited size (stack overflow risk)
- Contiguous memory

**Heap:**
- Slower allocation (finding free block)
- Manual cleanup required (or smart pointers)
- Large size (limited by RAM)
- Fragmented memory

### 6.2 Dynamic Allocation

```cpp
int* p = new int;       // Allocate single int
int* arr = new int[10]; // Allocate array
delete p;               // Free single
delete[] arr;           // Free array
```

### 6.3 The Rule of Three

If a class manages a resource (like a raw pointer), it needs:
1. **Destructor:** To free the resource.
2. **Copy Constructor:** To perform a deep copy.
3. **Copy Assignment Operator:** To perform a deep copy on assignment.

### 6.4 RAII (Resource Acquisition Is Initialization)

The most important idiom in C++.
- Acquire resource in constructor.
- Release resource in destructor.
- Guaranteed cleanup even if exceptions are thrown.

```cpp
class Wrapper {
    int* ptr;
public:
    Wrapper() { ptr = new int(10); }
    ~Wrapper() { delete ptr; } // Automatic cleanup
};
```

---

## 🦀 Rust vs C++ Comparison

### Memory Model
**C++:**
- Manual memory management (historically).
- Modern C++ uses RAII and smart pointers (`std::unique_ptr`, `std::shared_ptr`).
- Copy constructors are implicit (shallow copy by default).

**Rust:**
- Ownership model built-in.
- "Move" by default (no implicit deep copies).
- `Drop` trait is equivalent to C++ Destructor.
- No `new`/`delete` keywords; uses `Box::new()`.

### Deep Copy
**C++:**
```cpp
// Implicit copy constructor called
MyClass a;
MyClass b = a; // Copies data (shallow or deep depending on implementation)
```

**Rust:**
```rust
let a = MyStruct::new();
let b = a.clone(); // Explicit clone required for deep copy
// let b = a; // Moves ownership (a becomes invalid)
```

---

## 🔑 Key Takeaways

1. Prefer Stack allocation whenever possible.
2. Every `new` must have a matching `delete`.
3. Every `new[]` must have a matching `delete[]`.
4. Use RAII classes to manage resources automatically.
5. If you write a destructor, you probably need a copy constructor and assignment operator (Rule of Three).

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 6.1:** Stack vs Heap Memory
2. **Lab 6.2:** New and Delete (Deep Dive)
3. **Lab 6.3:** Array Allocation and Deallocation
4. **Lab 6.4:** Memory Leaks and Detection
5. **Lab 6.5:** RAII Introduction
6. **Lab 6.6:** Simple Smart Pointer
7. **Lab 6.7:** Copy Constructors and Deep Copies
8. **Lab 6.8:** Destructors and Cleanup
9. **Lab 6.9:** Move Semantics Basics
10. **Lab 6.10:** Custom String Class

After completing the labs, move on to **Module 7: Classes and Objects**.
