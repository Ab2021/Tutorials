# Lab 13.2: Nullptr and Type Safety

## Objective
Understand why `nullptr` is superior to `NULL` and `0`.

## Instructions

### Step 1: The Problem with NULL
Create `nullptr_demo.cpp`.
`NULL` is just a macro for `0` (or `(void*)0`).

```cpp
#include <iostream>

void func(int x) { std::cout << "int: " << x << "\n"; }
void func(int* p) { std::cout << "pointer\n"; }

int main() {
    func(0); // Calls func(int), not func(int*)!
    // func(NULL); // Ambiguous! Might call int version
    func(nullptr); // Unambiguous: calls func(int*)
    return 0;
}
```

### Step 2: Type Safety
`nullptr` has type `std::nullptr_t`.

```cpp
int* p1 = nullptr; // OK
int* p2 = 0; // OK but old style
// int x = nullptr; // Error: can't convert nullptr_t to int
```

### Step 3: Template Compatibility
```cpp
template <typename T>
void process(T* ptr) {
    if (ptr == nullptr) std::cout << "Null pointer\n";
}
```

## Challenges

### Challenge 1: Overload Resolution
Create two overloads: `void test(long)` and `void test(void*)`.
Call with `0`, `NULL`, and `nullptr`. Observe which is called.

### Challenge 2: Smart Pointers
Verify that `unique_ptr` and `shared_ptr` can be compared with `nullptr`.
```cpp
std::unique_ptr<int> p;
if (p == nullptr) { ... }
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>

void test(long x) { std::cout << "long: " << x << "\n"; }
void test(void* p) { std::cout << "pointer\n"; }

int main() {
    // Challenge 1
    test(0); // Calls long (0 is int, promotes to long)
    // test(NULL); // Ambiguous or calls long (depends on platform)
    test(nullptr); // Calls pointer
    
    // Challenge 2
    std::unique_ptr<int> p;
    if (p == nullptr) std::cout << "Smart pointer is null\n";
    
    p = std::make_unique<int>(42);
    if (p != nullptr) std::cout << "Smart pointer has value\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood `NULL` ambiguity
✅ Used `nullptr` for type safety
✅ Demonstrated overload resolution (Challenge 1)
✅ Used `nullptr` with smart pointers (Challenge 2)

## Key Learnings
- Always use `nullptr`, never `NULL` or `0` for pointers
- `nullptr` is a keyword, not a macro
- Works seamlessly with templates and overloading

## Next Steps
Proceed to **Lab 13.3: Constexpr** for compile-time computation.
