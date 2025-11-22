# Lab 1.7: Working with Header Files

## Objective
Understand the role of header files, include guards, and the difference between declaration and definition.

## Instructions

### Step 1: The Problem (Duplicate Definitions)
Create a file `bad_header.h`:
```cpp
// NO INCLUDE GUARDS!
int global_val = 10; // Definition in header (BAD!)
```

Create `main.cpp`:
```cpp
#include "bad_header.h"
#include "bad_header.h" // Accidental double include

int main() {
    return 0;
}
```

Try to compile: `g++ main.cpp -o app`.
*You should see a "redefinition" error.*

### Step 2: Fix with Include Guards
Modify `bad_header.h` to use standard include guards:

```cpp
#ifndef BAD_HEADER_H
#define BAD_HEADER_H

extern int global_val; // Declaration only

#endif
```

And create `bad_header.cpp` for the definition:
```cpp
#include "bad_header.h"

int global_val = 10; // Definition
```

Compile: `g++ main.cpp bad_header.cpp -o app`.

### Step 3: Modern Fix (#pragma once)
Create `modern_header.h`:
```cpp
#pragma once

void say_hello();
```

Create `modern_header.cpp`:
```cpp
#include <iostream>
#include "modern_header.h"

void say_hello() {
    std::cout << "Hello from pragma once!" << std::endl;
}
```

Modify `main.cpp` to use it:
```cpp
#include "modern_header.h"

int main() {
    say_hello();
    return 0;
}
```

Compile and run.

## Challenges

### Challenge 1: Circular Dependencies
Create two headers `A.h` and `B.h`.
- `A.h` includes `B.h`.
- `B.h` includes `A.h`.
Try to compile. What happens?
*Hint: This is why forward declarations are important.*

### Challenge 2: Forward Declaration
Fix the circular dependency in Challenge 1 using forward declarations instead of including the headers.
```cpp
class B; // Forward declaration

class A {
    B* b_ptr;
};
```

## Solution

<details>
<summary>Click to reveal solution</summary>

**Circular Dependency Fix (A.h)**
```cpp
#pragma once

class B; // Forward declaration

class A {
public:
    B* b; // Pointer allows forward declaration
};
```

**B.h**
```cpp
#pragma once

class A; // Forward declaration

class B {
public:
    A* a;
};
```
</details>

## Success Criteria
✅ Understood "redefinition" errors
✅ Implemented standard include guards (`#ifndef`)
✅ Implemented `#pragma once`
✅ Solved circular dependency with forward declaration

## Key Learnings
- Never define variables in headers (unless `inline` or `constexpr`)
- Always use include guards
- `#pragma once` is a modern, widely supported alternative
- Forward declarations can speed up compilation and break cycles

## Next Steps
Proceed to **Lab 1.8: Namespace Basics** to learn how to avoid name collisions.
