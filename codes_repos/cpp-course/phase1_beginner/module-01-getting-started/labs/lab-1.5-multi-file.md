# Lab 1.5: Multi-file Programs

## Objective
Learn how to split a C++ program into multiple source files and header files for better organization.

## Instructions

### Step 1: Create Header File
Create a file named `math_utils.h`. This file will contain the **declarations** of our functions.

```cpp
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// Function declaration
int add(int a, int b);
int subtract(int a, int b);

#endif
```

*Note: The `#ifndef`, `#define`, `#endif` lines are "Include Guards". They prevent the file from being included multiple times.*

### Step 2: Create Implementation File
Create a file named `math_utils.cpp`. This file will contain the **definitions** (actual code) of our functions.

```cpp
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
```

### Step 3: Create Main File
Create a file named `main.cpp` that uses the functions.

```cpp
#include <iostream>
#include "math_utils.h" // Quote includes for local files

int main() {
    int x = 10;
    int y = 5;
    
    std::cout << "Add: " << add(x, y) << std::endl;
    std::cout << "Subtract: " << subtract(x, y) << std::endl;
    
    return 0;
}
```

### Step 4: Compile and Run
You need to compile **both** source files together.

```bash
g++ main.cpp math_utils.cpp -o my_app
./my_app
```

## Challenges

### Challenge 1: Add Multiplication
1. Add `int multiply(int a, int b);` to `math_utils.h`.
2. Implement the function in `math_utils.cpp`.
3. Call it in `main.cpp`.
4. Recompile and run.

### Challenge 2: Object Files
Try compiling in two steps:
1. Compile `math_utils.cpp` to an object file: `g++ -c math_utils.cpp -o math_utils.o`
2. Compile `main.cpp` to an object file: `g++ -c main.cpp -o main.o`
3. Link them together: `g++ main.o math_utils.o -o my_app`

This is what build systems like CMake do automatically!

## Solution

<details>
<summary>Click to reveal solution</summary>

**math_utils.h**
```cpp
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b); // Challenge 1

#endif
```

**math_utils.cpp**
```cpp
#include "math_utils.h"

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; } // Challenge 1
```

**main.cpp**
```cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << "Mult: " << multiply(10, 5) << std::endl;
    return 0;
}
```
</details>

## Success Criteria
✅ Created `.h` and `.cpp` files correctly
✅ Used include guards
✅ Successfully compiled multiple files into one executable
✅ Added new function across all files

## Key Learnings
- Separation of declaration (header) and definition (source)
- Purpose of include guards
- Compiling multiple source files
- Linking object files

## Next Steps
Proceed to **Lab 1.6: Basic CMake Project** to automate this build process.
