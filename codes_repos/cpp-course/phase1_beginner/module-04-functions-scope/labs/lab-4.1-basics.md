# Lab 4.1: Basic Function Definition

## Objective
Learn how to define, declare, and call simple functions.

## Instructions

### Step 1: Define and Call
Create `functions.cpp`. Define a function `sayHello` before `main`.

```cpp
#include <iostream>

void sayHello() {
    std::cout << "Hello, World!" << std::endl;
}

int main() {
    sayHello();
    return 0;
}
```

### Step 2: Function with Arguments
Add a function `greet` that takes a name.

```cpp
void greet(std::string name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}
```

### Step 3: Function Prototypes
Move the function definitions *after* `main`. Try to compile. It should fail.
Fix it by adding prototypes *before* `main`.

```cpp
// Prototype
void sayHello();
void greet(std::string name);

int main() {
    sayHello();
    greet("Alice");
    return 0;
}

// Definition
void sayHello() {
    // ...
}
```

## Challenges

### Challenge 1: Math Function
Create a function `int square(int n)` that returns the square of the number. Call it in main and print the result.

### Challenge 2: Separation
Put the prototypes in a header file `my_funcs.h` and definitions in `my_funcs.cpp`. Call them from `main.cpp`. (Review of Module 1 concepts).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

// Prototypes
void sayHello();
void greet(std::string name);
int square(int n);

int main() {
    sayHello();
    greet("Bob");
    std::cout << "5 squared is " << square(5) << std::endl;
    return 0;
}

// Definitions
void sayHello() {
    std::cout << "Hello, World!" << std::endl;
}

void greet(std::string name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

int square(int n) {
    return n * n;
}
```
</details>

## Success Criteria
✅ Defined and called a void function
✅ Defined and called a function with arguments
✅ Used function prototypes
✅ Implemented a function returning a value

## Key Learnings
- Functions must be declared before use
- Prototypes allow definitions to be placed anywhere (or in other files)
- Return types must match the function signature

## Next Steps
Proceed to **Lab 4.2: Parameters and Return Values** to handle data flow.
