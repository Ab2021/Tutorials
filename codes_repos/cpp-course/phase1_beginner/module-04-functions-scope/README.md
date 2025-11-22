# Module 4: Functions and Scope

## 🎯 Learning Objectives

By the end of this module, you will:
- Define and call functions
- Understand parameter passing (value, reference, const reference)
- Master function overloading and default arguments
- Understand variable scope (local, global, static)
- Use `inline` functions for performance
- Introduction to recursion and lambda expressions

---

## 📖 Theoretical Concepts

### 4.1 Function Basics

**Declaration (Prototype):** Tells the compiler the function exists.
```cpp
int add(int a, int b);
```

**Definition:** The actual code.
```cpp
int add(int a, int b) {
    return a + b;
}
```

### 4.2 Parameter Passing

**Pass by Value:** Copy is made. Changes don't affect original.
```cpp
void func(int x); 
```

**Pass by Reference:** No copy. Changes affect original.
```cpp
void func(int& x);
```

**Pass by Const Reference:** No copy. Cannot change original. Efficient for large objects.
```cpp
void func(const std::string& str);
```

### 4.3 Function Overloading

Same name, different parameter list.
```cpp
void print(int x);
void print(double x);
void print(std::string x);
```

### 4.4 Default Arguments

Must be at the end of the parameter list.
```cpp
void log(std::string msg, int level = 1);
```

### 4.5 Scope and Lifetime

- **Local Scope:** Inside `{}`. Destroyed when block ends.
- **Global Scope:** Outside all functions. Lives for program duration.
- **Static Local:** Inside function, but lives for program duration. Retains value between calls.

### 4.6 Inline Functions

Suggestion to compiler to replace call with code body.
```cpp
inline int square(int x) { return x * x; }
```

### 4.7 Lambdas (Introduction)

Anonymous functions.
```cpp
auto add = [](int a, int b) { return a + b; };
```

---

## 🦀 Rust vs C++ Comparison

### Function Syntax
**C++:**
```cpp
int add(int a, int b) {
    return a + b;
}
```

**Rust:**
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b // Implicit return
}
```

### Overloading
**C++:** Supports function overloading.
**Rust:** Does NOT support overloading. Use different names or traits.

### Default Arguments
**C++:** Supported.
**Rust:** Not supported. Use `Option` or builder pattern.

### Return Values
**C++:** `void` for no return.
**Rust:** `()` (unit type) for no return.

---

## 🔑 Key Takeaways

1. Pass large objects by `const reference` to avoid copying.
2. Use function overloading to create intuitive APIs.
3. Avoid global variables; use `static` if state persistence is needed.
4. Default arguments can simplify function calls but must be trailing.
5. Lambdas are powerful for short, local functions (callbacks).

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 4.1:** Basic Function Definition
2. **Lab 4.2:** Parameters and Return Values
3. **Lab 4.3:** Pass by Value vs Reference
4. **Lab 4.4:** Const References
5. **Lab 4.5:** Function Overloading
6. **Lab 4.6:** Default Arguments
7. **Lab 4.7:** Inline Functions
8. **Lab 4.8:** Static Variables
9. **Lab 4.9:** Recursion (Factorial/Fibonacci)
10. **Lab 4.10:** Lambda Expressions Basics

After completing the labs, move on to **Module 5: Pointers and References**.
