# Lab 11.8: Assertions (assert vs static_assert)

## Objective
Use assertions to catch programmer errors during development.

## Instructions

### Step 1: Runtime Assertions
Create `assertions.cpp`.
Include `<cassert>`.

```cpp
#include <iostream>
#include <cassert>

void process(int* ptr) {
    assert(ptr != nullptr && "Pointer must not be null");
    std::cout << *ptr << std::endl;
}

int main() {
    int x = 10;
    process(&x);
    // process(nullptr); // Aborts execution
    return 0;
}
```
*Note: `assert` is removed in Release builds (`-DNDEBUG`).*

### Step 2: Compile-Time Assertions
`static_assert` checks conditions during compilation.

```cpp
template <typename T>
void checkType() {
    static_assert(sizeof(T) >= 4, "Type too small!");
}

// checkType<char>(); // Compile Error!
```

## Challenges

### Challenge 1: Custom Message
Trigger a `static_assert` with a custom message.
Try `static_assert(false, "This code should not run");` inside a template specialization.

### Challenge 2: Release Build
Compile with `-DNDEBUG` (e.g., `g++ assertions.cpp -DNDEBUG`).
Verify that the runtime `assert(nullptr)` is ignored and likely crashes (segfault) instead of aborting cleanly.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cassert>
#include <type_traits>

template <typename T>
void mustBeInteger() {
    static_assert(std::is_integral<T>::value, "T must be an integer");
}

int main() {
    // Runtime
    int age = -5;
    // assert(age >= 0); // Uncomment to test
    
    // Compile Time
    mustBeInteger<int>();
    // mustBeInteger<float>(); // Compile Error
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `assert` for runtime checks
✅ Used `static_assert` for compile-time checks
✅ Understood NDEBUG behavior (Challenge 2)

## Key Learnings
- Use `assert` for logic errors (impossible states)
- Use Exceptions for runtime errors (file not found, bad input)
- Use `static_assert` to validate types and constants at compile time

## Next Steps
Proceed to **Lab 11.9: std::optional** for modern error handling.
