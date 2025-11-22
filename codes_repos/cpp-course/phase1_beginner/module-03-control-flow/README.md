# Module 3: Control Flow

## 🎯 Learning Objectives

By the end of this module, you will:
- Master conditional logic with `if`, `else if`, `else`
- Use `switch` statements effectively (including C++17 init-statements)
- Implement loops: `while`, `do-while`, `for`
- Use modern range-based `for` loops
- Control loop execution with `break`, `continue`, and `return`
- Understand scope within control structures

---

## 📖 Theoretical Concepts

### 3.1 Conditional Statements

**If-Else:**
```cpp
if (condition) {
    // code
} else if (other_condition) {
    // code
} else {
    // code
}
```

**If with Initializer (C++17):**
Limits the scope of the variable.
```cpp
if (auto val = getValue(); val > 10) {
    // val is valid here
}
// val is NOT valid here
```

### 3.2 Switch Statements

Used for checking a value against multiple constants.

```cpp
switch (expression) {
    case 1:
        // code
        break;
    case 2:
    case 3: // Fallthrough
        // code
        break;
    default:
        // code
}
```

**Best Practice:** Always use `break` unless fallthrough is intentional (comment it!).

### 3.3 Loops

**While Loop:**
Checks condition *before* execution.
```cpp
while (condition) {
    // code
}
```

**Do-While Loop:**
Checks condition *after* execution (runs at least once).
```cpp
do {
    // code
} while (condition);
```

**For Loop:**
Standard C-style loop.
```cpp
for (int i = 0; i < 10; ++i) {
    // code
}
```

**Range-Based For Loop (C++11):**
Iterates over containers or arrays.
```cpp
int arr[] = {1, 2, 3};
for (int x : arr) {
    // x is a copy
}
for (const auto& x : arr) {
    // x is a const reference (efficient)
}
```

### 3.4 Jump Statements

- `break`: Exits the nearest loop or switch.
- `continue`: Skips the rest of the current iteration.
- `return`: Exits the function.
- `goto`: Jumps to a label (Avoid unless necessary!).

---

## 🦀 Rust vs C++ Comparison

### If Statements
**C++:** Statement (doesn't return a value).
```cpp
int x;
if (cond) x = 1; else x = 2;
```

**Rust:** Expression (returns a value).
```rust
let x = if cond { 1 } else { 2 };
```

### Switch vs Match
**C++:** `switch` only works on integers and enums. Fallthrough by default.
```cpp
switch (x) {
    case 1: doSomething(); break;
}
```

**Rust:** `match` works on patterns, ranges, structs. No fallthrough. Exhaustive checking.
```rust
match x {
    1 => do_something(),
    2..=5 => do_other(),
    _ => default_action(),
}
```

### Loops
**C++:** `for`, `while`, `do-while`.
**Rust:** `for`, `while`, `loop` (infinite). No `do-while`.

---

## 🔑 Key Takeaways

1. Use `if` with initializers (C++17) to keep scope tight.
2. Prefer range-based `for` loops for iterating over collections.
3. Use `const auto&` in range loops to avoid copying.
4. Be careful with `switch` fallthrough; use `[[fallthrough]]` attribute (C++17) if intentional.
5. Avoid `goto` leads to "spaghetti code".

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 3.1:** If/Else logic puzzles
2. **Lab 3.2:** Switch statement menu system
3. **Lab 3.3:** While loop number guessing game
4. **Lab 3.4:** Do-while loop input validation
5. **Lab 3.5:** For loop patterns (pyramids)
6. **Lab 3.6:** Range-based for loop with arrays
7. **Lab 3.7:** Break and Continue usage
8. **Lab 3.8:** Nested loops (multiplication table)
9. **Lab 3.9:** Goto (and why to avoid it)
10. **Lab 3.10:** Building a text-based adventure game

After completing the labs, move on to **Module 4: Functions and Scope**.
