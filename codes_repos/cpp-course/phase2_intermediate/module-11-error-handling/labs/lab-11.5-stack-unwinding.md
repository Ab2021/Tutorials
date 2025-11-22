# Lab 11.5: Stack Unwinding and RAII

## Objective
Observe how exceptions trigger destructor calls (Stack Unwinding) and why RAII is crucial for exception safety.

## Instructions

### Step 1: The Resource
Create `unwinding.cpp`.
Create a noisy class.

```cpp
#include <iostream>

class Resource {
    std::string name;
public:
    Resource(std::string n) : name(n) { std::cout << "Acquire " << name << "\n"; }
    ~Resource() { std::cout << "Release " << name << "\n"; }
};
```

### Step 2: The Throw
Throw an exception while resources are alive.

```cpp
void risky() {
    Resource r1("A");
    Resource r2("B");
    throw std::runtime_error("Boom");
    Resource r3("C"); // Never created
}

int main() {
    try {
        risky();
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
    return 0;
}
```
*Output: Acquire A -> Acquire B -> Release B -> Release A -> Caught.*
*Note: Destructors run in reverse order.*

### Step 3: The Leak (Without RAII)
Show what happens with raw pointers.

```cpp
void leak() {
    int* p = new int(5);
    throw 1;
    delete p; // Never reached! Memory Leak!
}
```

## Challenges

### Challenge 1: Smart Pointer Safety
Fix the leak in Step 3 using `std::unique_ptr<int>`.
Verify that memory is freed (you can't easily see it, but you know RAII works).

### Challenge 2: Constructor Exception
Throw an exception INSIDE a constructor.
Verify that the destructor for that object is NOT called (because it was never fully constructed).
But destructors for *already constructed members* ARE called.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

class Member {
public:
    ~Member() { std::cout << "~Member\n"; }
};

class Broken {
    Member m;
public:
    Broken() {
        throw std::runtime_error("Fail in Ctor");
    }
    ~Broken() { std::cout << "~Broken\n"; } // Won't run
};

int main() {
    // Challenge 1
    try {
        std::unique_ptr<int> p = std::make_unique<int>(5);
        throw 1;
    } catch(...) { std::cout << "Smart pointer cleaned up\n"; }
    
    // Challenge 2
    try {
        Broken b;
    } catch(...) {
        std::cout << "Caught ctor error\n";
    }
    // Output: ~Member -> Caught
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed stack unwinding
✅ Verified destructor execution order
✅ Identified memory leak with raw pointers
✅ Fixed leak with RAII (Challenge 1)
✅ Understood constructor exception behavior (Challenge 2)

## Key Learnings
- Exceptions destroy local objects automatically
- Raw pointers + Exceptions = Leaks
- RAII (Smart Pointers) is the ONLY way to write Exception-Safe code

## Next Steps
Proceed to **Lab 11.6: Exception Safety Levels** to write robust code.
