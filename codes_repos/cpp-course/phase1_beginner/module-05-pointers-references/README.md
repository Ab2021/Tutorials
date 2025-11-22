# Module 5: Pointers and References

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand memory addresses and how to access them
- Master pointer syntax: `*` (dereference) and `&` (address-of)
- Differentiate between pointers and references
- Perform pointer arithmetic safely
- Understand the relationship between arrays and pointers
- Use `nullptr` for safety
- Manage dynamic memory with `new` and `delete` (introductory)

---

## 📖 Theoretical Concepts

### 5.1 What is a Pointer?

A pointer is a variable that stores the **memory address** of another variable.

```cpp
int x = 10;
int* ptr = &x; // ptr stores the address of x
```

- `&` (Address-of operator): Gets the address.
- `*` (Dereference operator): Accesses the value at the address.

### 5.2 References vs Pointers

| Feature | Pointer (`int*`) | Reference (`int&`) |
|---------|------------------|--------------------|
| Nullability | Can be `nullptr` | Cannot be null |
| Reassignable | Can point to different vars | Bound at initialization |
| Initialization | Optional (but dangerous) | Required |
| Syntax | Explicit dereference `*ptr` | Implicit access `ref` |

**Best Practice:** Use references by default. Use pointers when you need nullability or re-assignability.

### 5.3 Pointer Arithmetic

Pointers can be incremented to move to the next memory location based on type size.

```cpp
int arr[] = {10, 20, 30};
int* p = arr;
p++; // Moves forward by sizeof(int) bytes (usually 4)
// *p is now 20
```

### 5.4 Const Correctness

- `const int* p`: Pointer to **constant integer**. (Cannot change value).
- `int* const p`: **Constant pointer** to integer. (Cannot change address).
- `const int* const p`: Constant pointer to constant integer.

### 5.5 Dynamic Memory (Intro)

Allocating memory on the heap.

```cpp
int* p = new int(42); // Allocate
delete p;             // Deallocate
p = nullptr;          // Safety
```

**Warning:** Forgetting `delete` causes **memory leaks**.

---

## 🦀 Rust vs C++ Comparison

### Pointers
**C++:**
- Raw pointers (`*`) are common but unsafe.
- Manual memory management (`new`/`delete`).
- Null pointers (`nullptr`) can cause crashes.

**Rust:**
- References (`&`) are safe, non-null, and checked by borrow checker.
- Raw pointers (`*const`, `*mut`) exist but require `unsafe` block.
- No `nullptr` (use `Option<Box<T>>`).
- No manual `delete` (RAII handles it).

### Safety
**C++:**
```cpp
int* p = nullptr;
*p = 10; // Undefined Behavior (Crash)
```

**Rust:**
```rust
// Won't compile without unsafe
let p: *mut i32 = std::ptr::null_mut();
unsafe { *p = 10; } // Explicit opt-in to danger
```

---

## 🔑 Key Takeaways

1. Initialize pointers to `nullptr` if they don't point to valid memory.
2. Prefer references (`&`) over pointers for function parameters.
3. Pointer arithmetic is powerful but dangerous (buffer overflows).
4. Always pair `new` with `delete`.
5. Read pointer declarations right-to-left (e.g., `int* const` is a constant pointer to int).

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 5.1:** Pointer Basics (address-of, dereference)
2. **Lab 5.2:** Nullptr and Safety
3. **Lab 5.3:** Pointer Arithmetic
4. **Lab 5.4:** Arrays and Pointers
5. **Lab 5.5:** Const Pointers vs Pointers to Const
6. **Lab 5.6:** References (Review and Deep Dive)
7. **Lab 5.7:** Pointers to Pointers
8. **Lab 5.8:** Dynamic Memory (new/delete basics)
9. **Lab 5.9:** Void Pointers (and why to avoid)
10. **Lab 5.10:** Function Pointers (Callbacks)

After completing the labs, move on to **Module 6: Memory Management Basics**.
