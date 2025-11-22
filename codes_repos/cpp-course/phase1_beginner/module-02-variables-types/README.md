# Module 2: Variables and Types

## 🎯 Learning Objectives

By the end of this module, you will:
- Master C++ fundamental data types
- Understand type inference with `auto`
- Use `const` and `constexpr` for immutability
- Perform safe type conversions
- Work with strongly-typed enums
- Understand type sizes and limits

---

## 📖 Theoretical Concepts

### 2.1 Fundamental Types

C++ has several built-in types:

| Type | Description | Size (Typical) | Range |
|------|-------------|----------------|-------|
| `int` | Integer | 4 bytes | -2B to +2B |
| `double` | Floating point | 8 bytes | +/- 1.7e308 |
| `char` | Character | 1 byte | -128 to 127 |
| `bool` | Boolean | 1 byte | true / false |
| `void` | Empty type | - | - |
| `long long` | Large integer | 8 bytes | -9e18 to +9e18 |

**Initialization:**
```cpp
int a = 10;         // Copy initialization
int b(10);          // Direct initialization
int c{10};          // Brace initialization (Recommended)
```

### 2.2 Type Inference (auto)

Introduced in C++11, `auto` lets the compiler deduce the type:

```cpp
auto x = 42;        // int
auto y = 3.14;      // double
auto z = "Hello";   // const char*
```

**Best Practice:** Use `auto` when the type is obvious (e.g., `auto x = 5;`) or hard to type (iterators).

### 2.3 Constants

**const:** Value cannot be changed after initialization.
```cpp
const int MAX_USERS = 100;
// MAX_USERS = 101; // Error
```

**constexpr:** Value must be known at compile-time.
```cpp
constexpr int ARRAY_SIZE = 10 * 2; // Calculated at compile time
int arr[ARRAY_SIZE]; // Valid
```

### 2.4 Type Conversions

**Implicit:** Automatic conversion (can lose data).
```cpp
int x = 3.14; // x becomes 3 (warning)
```

**Explicit (C-style):** Avoid this.
```cpp
int x = (int)3.14;
```

**Explicit (C++ style):**
```cpp
int x = static_cast<int>(3.14); // Safe, clear intent
```

### 2.5 Enums

**Old Enums (Unscoped):**
```cpp
enum Color { RED, GREEN, BLUE };
Color c = RED; // Pollutes global namespace
```

**Enum Class (Scoped, Strongly Typed):**
```cpp
enum class Color { Red, Green, Blue };
Color c = Color::Red; // Must use scope
// int x = Color::Red; // Error: No implicit conversion to int
```

---

## 🦀 Rust vs C++ Comparison

### Variable Declaration

| Feature | C++ | Rust |
|---------|-----|------|
| **Syntax** | `type name = value;` | `let name: type = value;` |
| **Inference** | `auto name = value;` | `let name = value;` |
| **Mutability** | Mutable by default | Immutable by default |
| **Make Mutable** | Just declare | `let mut` |
| **Make Const** | `const` | Default behavior |

**C++:**
```cpp
int x = 5;      // Mutable
const int y = 5; // Immutable
```

**Rust:**
```rust
let mut x = 5;  // Mutable
let y = 5;      // Immutable
```

### Type System

**C++:**
- Weakly typed in some areas (implicit conversions)
- `int` size varies by platform (usually 32-bit)
- `char` varies (signed/unsigned)

**Rust:**
- Strongly typed (no implicit conversions)
- Explicit sizes: `i32`, `i64`, `u8`, `f64`
- `char` is always 4 bytes (Unicode scalar)

---

## 🔑 Key Takeaways

1. Prefer brace initialization `int x{5};` to prevent narrowing conversions.
2. Use `auto` to let the compiler deduce types, but keep code readable.
3. Use `const` for variables that shouldn't change.
4. Use `constexpr` for values known at compile-time.
5. Prefer `static_cast` over C-style casts.
6. Always use `enum class` instead of plain `enum`.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 2.1:** Type exploration and limits
2. **Lab 2.2:** Auto type deduction exercises
3. **Lab 2.3:** Const and constexpr practice
4. **Lab 2.4:** Type conversion safety
5. **Lab 2.5:** Enum class vs traditional enums
6. **Lab 2.6:** Temperature converter with strong types
7. **Lab 2.7:** Custom literals
8. **Lab 2.8:** Type size and alignment
9. **Lab 2.9:** Numeric limits exploration
10. **Lab 2.10:** Building a type-safe units library

After completing the labs, move on to **Module 3: Control Flow**.
