# Lab 1.4: Understanding Compilation Flags

## Objective
Learn how to use compiler flags to control warnings, optimization, and C++ standards.

## Instructions

### Step 1: Create a "Buggy" Source File
Create a file named `flags_demo.cpp` with some intentional bad practices (but valid syntax):

```cpp
#include <iostream>

int main() {
    int x; // Uninitialized variable
    
    if (x == 5) { 
        std::cout << "x is 5" << std::endl;
    }
    
    long unsigned int y = -1; // Implicit conversion warning
    
    return 0;
}
```

### Step 2: Compile with Defaults
Compile normally:
```bash
g++ flags_demo.cpp -o demo
```
*Note: You might not see many warnings by default.*

### Step 3: Enable Warnings
Compile with "Wall" (Warn All) and "Wextra":
```bash
g++ -Wall -Wextra flags_demo.cpp -o demo
```
*Observe the output. The compiler should now warn you about `x` being uninitialized and the comparison.*

### Step 4: Specify C++ Standard
Compile forcing a specific standard (e.g., C++20):
```bash
g++ -std=c++20 -Wall -Wextra flags_demo.cpp -o demo
```

### Step 5: Treat Warnings as Errors
Force the compilation to fail if there are warnings:
```bash
g++ -Werror -Wall -Wextra flags_demo.cpp -o demo
```

### Step 6: Your Task
Fix the code in `flags_demo.cpp` so it compiles cleanly even with `-Werror`.

## Challenges

### Challenge 1: Optimization Flags
Create a loop that does a lot of work (e.g., counts to 1 billion).
Compile with `-O0` (no optimization) and time the execution.
Compile with `-O3` (max optimization) and time the execution.
*On Linux/macOS use the `time` command: `time ./demo`*

### Challenge 2: Preprocessor Output
Run the compiler with the `-E` flag to see the preprocessed output (what the code looks like after `#include`s are processed).
```bash
g++ -E flags_demo.cpp > preprocessed.txt
```
Open `preprocessed.txt` and look at the massive amount of code added from `<iostream>`.

## Solution

<details>
<summary>Click to reveal fixed code</summary>

```cpp
#include <iostream>

int main() {
    int x = 0; // Fixed: Initialized variable
    
    if (x == 5) { 
        std::cout << "x is 5" << std::endl;
    }
    
    // Fixed: Use appropriate type or cast, or just ignore if intentional
    // For this demo, let's just print it to use the variable
    long unsigned int y = static_cast<long unsigned int>(-1); 
    std::cout << "y: " << y << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understand the purpose of `-Wall` and `-Wextra`
✅ Successfully compiled with `-std=c++20`
✅ Fixed code to pass `-Werror`
✅ Observed difference between `-O0` and `-O3`

## Key Learnings
- Compiler warnings are friends, not enemies
- How to enforce code quality with flags
- Impact of optimization levels
- Specifying language standards

## Next Steps
Proceed to **Lab 1.5: Multi-file Programs** to learn how to structure larger applications.
