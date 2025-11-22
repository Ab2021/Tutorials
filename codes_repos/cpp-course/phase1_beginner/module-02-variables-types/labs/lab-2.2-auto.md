# Lab 2.2: Auto Type Deduction

## Objective
Practice using the `auto` keyword for type inference and understand when to use it.

## Instructions

### Step 1: Basic Auto
Create `auto_demo.cpp`:

```cpp
#include <iostream>
#include <typeinfo> // For typeid

int main() {
    auto a = 42;
    auto b = 3.14;
    auto c = "Hello";
    auto d = true;
    
    // Note: typeid(x).name() output is compiler-dependent (e.g., 'i' for int)
    std::cout << "Type of a: " << typeid(a).name() << std::endl;
    std::cout << "Type of b: " << typeid(b).name() << std::endl;
    
    return 0;
}
```

### Step 2: Auto with Modifiers
Experiment with `const` and references:

```cpp
int x = 10;
const auto y = x; // const int
auto& z = x;      // int& (reference)
z = 20;           // Changes x
```

### Step 3: Trailing Return Type
C++ allows `auto` in function return types:

```cpp
auto add(int a, int b) -> int {
    return a + b;
}
// Or simply (C++14+):
auto subtract(int a, int b) {
    return a - b;
}
```

## Challenges

### Challenge 1: Auto and Pointers
Declare a pointer to an int. Use `auto` to declare another variable that holds that pointer. Verify the type.

### Challenge 2: Auto and Initialization
What happens if you try:
```cpp
auto x; // Error?
auto y = {1, 2, 3}; // What type is this?
```
*Hint: It might be `std::initializer_list<int>`.*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <typeinfo>
#include <initializer_list>

int main() {
    int val = 5;
    int* ptr = &val;
    
    auto ptr_auto = ptr; // int*
    
    // auto x; // Error: Declaration of variable 'x' with deduced type 'auto' requires an initializer
    
    auto list = {1, 2, 3}; // std::initializer_list<int>
    
    std::cout << "List size: " << list.size() << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `auto` for basic types
✅ Used `auto` with `const` and references
✅ Understood that `auto` requires initialization
✅ Identified type deduction for initializer lists

## Key Learnings
- `auto` deduces type from initializer
- `auto` drops top-level `const` and references unless specified
- `auto` is useful for complex types (like iterators)

## Next Steps
Proceed to **Lab 2.3: Const and Constexpr** to master immutability.
