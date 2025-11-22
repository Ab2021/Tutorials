# Lab 5.2: Nullptr and Safety

## Objective
Learn how to safely handle pointers using `nullptr` and checks.

## Instructions

### Step 1: Uninitialized Pointer (Danger)
Create `safety.cpp`.
```cpp
#include <iostream>

int main() {
    int* ptr; // Uninitialized (contains garbage address)
    // *ptr = 10; // CRASH! Undefined Behavior.
    
    // Don't actually run the above line unless you want to see a crash.
    return 0;
}
```

### Step 2: Nullptr
Initialize with `nullptr`.

```cpp
int* ptr = nullptr;
```

### Step 3: Safety Check
Always check before using.

```cpp
if (ptr != nullptr) {
    std::cout << *ptr << std::endl;
} else {
    std::cout << "Pointer is null." << std::endl;
}
```

### Step 4: NULL vs nullptr
`NULL` is a macro (usually 0). `nullptr` is a type-safe keyword (C++11). Always use `nullptr`.

## Challenges

### Challenge 1: Safe Delete (Preview)
Write a function `void safePrint(int* p)` that only prints if `p` is valid.
Call it with `nullptr` and a valid address.

### Challenge 2: The "0" Confusion
Try passing `NULL` and `0` to a function overloaded for `int` and `int*`.
```cpp
void func(int x) { std::cout << "Int"; }
void func(int* p) { std::cout << "Ptr"; }
// func(NULL); // Ambiguous?
// func(nullptr); // Calls Ptr version
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void safePrint(int* p) {
    if (p) { // Implicit check for nullptr
        std::cout << "Value: " << *p << std::endl;
    } else {
        std::cout << "Null pointer passed." << std::endl;
    }
}

void func(int x) { std::cout << "Called Int version" << std::endl; }
void func(int* p) { std::cout << "Called Ptr version" << std::endl; }

int main() {
    int x = 10;
    safePrint(&x);
    safePrint(nullptr);
    
    // func(NULL); // Might be ambiguous or call Int version depending on compiler
    func(nullptr); // Definitely calls Ptr version
    
    return 0;
}
```
</details>

## Success Criteria
✅ Initialized pointer to `nullptr`
✅ Implemented null check
✅ Understood difference between `NULL` and `nullptr`
✅ Avoided dereferencing invalid memory

## Key Learnings
- Uninitialized pointers are dangerous
- `nullptr` is the safe "empty" state
- Always check for null before dereferencing

## Next Steps
Proceed to **Lab 5.3: Pointer Arithmetic** to move through memory.
