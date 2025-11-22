# Lab 11.7: The noexcept Specifier

## Objective
Mark functions as non-throwing to allow compiler optimizations.

## Instructions

### Step 1: Syntax
Create `noexcept_demo.cpp`.

```cpp
#include <iostream>
#include <vector>

void safe() noexcept {
    std::cout << "Safe\n";
}

void risky() {
    throw 1;
}
```

### Step 2: Checking noexcept
Use the `noexcept` operator to check at compile time.

```cpp
int main() {
    std::cout << "Is safe() noexcept? " << noexcept(safe()) << std::endl;
    std::cout << "Is risky() noexcept? " << noexcept(risky()) << std::endl;
    return 0;
}
```

### Step 3: Violating noexcept
What happens if a `noexcept` function throws?
`std::terminate()` is called immediately. The stack is NOT unwound.

```cpp
void crash() noexcept {
    throw 1; // Crash!
}
```

## Challenges

### Challenge 1: Move Constructor
Create a class `BigData`.
Implement a Move Constructor marked `noexcept`.
`std::vector` will only use your move constructor if it is `noexcept`. Otherwise, it copies (for safety).

### Challenge 2: Conditional noexcept
`void func() noexcept(sizeof(int) == 4);`
Make a function `noexcept` only if a condition is true.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class BigData {
public:
    BigData() {}
    // Move Ctor - Marked noexcept
    BigData(BigData&&) noexcept { std::cout << "Move\n"; }
    // Copy Ctor
    BigData(const BigData&) { std::cout << "Copy\n"; }
};

int main() {
    std::vector<BigData> v;
    v.push_back(BigData());
    
    std::cout << "-- Resize --\n";
    // If Move is noexcept, vector uses it during resize.
    // If not, it uses Copy to ensure strong guarantee.
    v.resize(10); 
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `noexcept` specifier
✅ Used `noexcept` operator
✅ Understood `std::terminate` behavior
✅ Optimized vector resizing with `noexcept` move (Challenge 1)

## Key Learnings
- Mark Move Constructors and Destructors as `noexcept`
- If you lie to the compiler (`noexcept` but throw), the program crashes hard
- `std::vector` optimization relies on `noexcept`

## Next Steps
Proceed to **Lab 11.8: Assertions** for debugging.
