# Lab 6.5: RAII Introduction

## Objective
Understand **Resource Acquisition Is Initialization (RAII)**, the most important pattern in C++.

## Instructions

### Step 1: The Wrapper Class
Create `raii.cpp`. Create a class that manages an int pointer.

```cpp
#include <iostream>

class IntWrapper {
    int* ptr;
public:
    IntWrapper(int value) {
        std::cout << "Acquiring resource\n";
        ptr = new int(value);
    }
    
    ~IntWrapper() {
        std::cout << "Releasing resource\n";
        delete ptr;
    }
    
    int getValue() { return *ptr; }
};
```

### Step 2: Automatic Cleanup
Use it in a scope.

```cpp
int main() {
    {
        IntWrapper w(42);
        std::cout << "Value: " << w.getValue() << std::endl;
    } // Destructor called here automatically!
    
    std::cout << "End of main" << std::endl;
    return 0;
}
```

### Step 3: Exception Safety
RAII works even with exceptions.

```cpp
try {
    IntWrapper w(10);
    throw std::runtime_error("Error!");
} catch (...) {
    std::cout << "Caught exception" << std::endl;
}
// Destructor still called!
```

## Challenges

### Challenge 1: File Handle RAII
Create a `FileHandler` class that opens a file in constructor (`fopen`) and closes it in destructor (`fclose`).
*Note: Use C-style FILE* for this exercise to demonstrate wrapping raw resources.*

### Challenge 2: Copying Problem
What happens if you copy `IntWrapper`?
`IntWrapper w2 = w1;`
Both pointers point to the same memory. Both destructors will try to delete it. Double Free!
(We will fix this in Lab 6.7).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstdio>

class FileHandler {
    FILE* file;
public:
    FileHandler(const char* filename) {
        file = std::fopen(filename, "w");
        std::cout << "File opened\n";
    }
    ~FileHandler() {
        if (file) {
            std::fclose(file);
            std::cout << "File closed\n";
        }
    }
    void write(const char* text) {
        if (file) std::fprintf(file, "%s", text);
    }
};

int main() {
    {
        FileHandler fh("test.txt");
        fh.write("Hello RAII");
    } // File closed automatically
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented RAII class
✅ Verified destructor execution at scope exit
✅ Verified exception safety
✅ Wrapped a file handle (Challenge 1)

## Key Learnings
- Resources (memory, files, locks) should be wrapped in objects
- Constructor acquires, Destructor releases
- Stack unwinding guarantees destructor execution

## Next Steps
Proceed to **Lab 6.6: Simple Smart Pointer** to build a reusable RAII wrapper.
