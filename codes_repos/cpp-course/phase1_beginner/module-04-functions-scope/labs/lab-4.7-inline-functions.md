# Lab 4.7: Inline Functions

## Objective
Understand how `inline` functions work and when to use them for performance.

## Instructions

### Step 1: Define Inline Function
Create `inline_demo.cpp`.

```cpp
#include <iostream>

inline int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 10);
    // Compiler likely replaces this with: int result = 5 + 10;
    std::cout << result << std::endl;
    return 0;
}
```

### Step 2: Header Files
Inline functions usually go in header files, not `.cpp` files, because the compiler needs to see the body to inline it.

Create `math_utils.h`:
```cpp
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

inline int square(int x) {
    return x * x;
}

#endif
```

### Step 3: Use it
Include and use it in `main`.

## Challenges

### Challenge 1: Complex Inline?
Write a large, complex function (loops, I/O) and mark it `inline`.
Does the compiler *have* to inline it?
*Answer: No, `inline` is just a suggestion. Modern compilers decide based on cost/benefit.*

### Challenge 2: Linker Errors
Define a non-inline function in a header file and include it in two `.cpp` files. Link them.
*Error: Multiple definition.*
Add `inline` keyword.
*Success! `inline` also suppresses the multiple definition error.*

## Solution

<details>
<summary>Click to reveal solution</summary>

**math_utils.h**
```cpp
#pragma once

inline int cube(int x) {
    return x * x * x;
}
```

**main.cpp**
```cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << cube(3) << std::endl;
    return 0;
}
```
</details>

## Success Criteria
✅ Defined an inline function
✅ Placed inline function in header
✅ Understood that `inline` is a hint
✅ Understood `inline` prevents One Definition Rule (ODR) violations

## Key Learnings
- Use `inline` for small, frequently called functions (getters/setters)
- Put inline functions in headers
- `inline` affects linking (ODR) as much as optimization

## Next Steps
Proceed to **Lab 4.8: Static Variables** to maintain state.
