# Lab 11.2: Multiple Catch Blocks

## Objective
Handle different types of exceptions separately.

## Instructions

### Step 1: Multiple Throw Types
Create `multi_catch.cpp`.
Write a function that throws different types based on input.

```cpp
#include <iostream>
#include <string>

void test(int choice) {
    if (choice == 1) throw 101; // int
    if (choice == 2) throw 3.14; // double
    if (choice == 3) throw std::string("Error"); // string
}
```

### Step 2: Multiple Catch Blocks
Order matters! Specific types first, generic types last.

```cpp
int main() {
    try {
        test(2);
    } catch (int e) {
        std::cout << "Int error: " << e << "\n";
    } catch (double e) {
        std::cout << "Double error: " << e << "\n";
    } catch (std::string& e) {
        std::cout << "String error: " << e << "\n";
    }
    return 0;
}
```

### Step 3: Catching by Reference
Always catch objects by reference (const ref usually) to avoid slicing and copying.
`catch (const std::string& e)`

## Challenges

### Challenge 1: Inheritance Catching
If you have `Base` and `Derived` exception classes.
If you `catch (Base& b)` first, it will catch `Derived` too!
Demonstrate this by defining two classes and trying to catch them.

### Challenge 2: Re-throwing
Inside a catch block, use `throw;` (no arguments) to re-throw the current exception to a higher-level handler.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Base {};
class Derived : public Base {};

void risky() {
    try {
        throw Derived();
    } catch (Base& b) {
        std::cout << "Caught Base (or Derived)\n";
        throw; // Re-throw
    }
}

int main() {
    try {
        risky();
    } catch (Derived& d) {
        std::cout << "Caught Derived in main\n";
    } catch (...) {
        std::cout << "Caught something else\n";
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented multiple catch blocks
✅ Caught different primitive types
✅ Understood catch ordering (Challenge 1)
✅ Used `throw;` to re-throw (Challenge 2)

## Key Learnings
- Catch blocks are checked sequentially
- Always catch specific types before general types (Derived before Base)
- Catch by reference to prevent slicing

## Next Steps
Proceed to **Lab 11.3: Standard Exceptions** to use the built-in hierarchy.
