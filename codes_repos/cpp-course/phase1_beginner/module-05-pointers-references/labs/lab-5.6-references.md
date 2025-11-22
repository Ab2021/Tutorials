# Lab 5.6: References (Deep Dive)

## Objective
Understand references as safer, non-null aliases to variables.

## Instructions

### Step 1: Basic Reference
Create `references.cpp`.

```cpp
#include <iostream>

int main() {
    int original = 100;
    int& ref = original; // Must initialize!
    
    std::cout << "Original: " << original << std::endl;
    std::cout << "Ref: " << ref << std::endl;
    
    ref = 200;
    std::cout << "Original after ref change: " << original << std::endl;
    
    return 0;
}
```

### Step 2: Reassignment?
Try to make `ref` point to another variable.
```cpp
int other = 500;
ref = other; // Does this rebind ref? Or assign value?
```
*Answer: It assigns the value 500 to `original`. References cannot be reseated.*

### Step 3: Address of Reference
Print `&original` and `&ref`. They should be identical.

## Challenges

### Challenge 1: Dangling Reference
Write a function that returns a reference to a local variable.
```cpp
int& badFunc() {
    int x = 10;
    return x; // Warning!
}
```
Call it and try to print the result. This is Undefined Behavior.

### Challenge 2: Reference to Pointer
Create a reference to a pointer.
```cpp
int x = 10;
int* ptr = &x;
int*& refPtr = ptr;
```
Change `ptr` via `refPtr`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int& badFunc() {
    int x = 10;
    return x; // Compiler warning: reference to local variable returned
}

int main() {
    int x = 10;
    int& ref = x;
    
    std::cout << "Addr x: " << &x << std::endl;
    std::cout << "Addr ref: " << &ref << std::endl;
    
    // Challenge 2
    int* ptr = &x;
    int*& refPtr = ptr;
    
    int y = 20;
    refPtr = &y; // Changes ptr to point to y
    
    std::cout << "Ptr points to: " << *ptr << std::endl; // 20
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and used references
✅ Verified references share memory address
✅ Understood references cannot be reseated
✅ Identified dangling reference risk (Challenge 1)

## Key Learnings
- References are aliases, not separate objects
- They must be initialized
- They cannot be null
- Returning references to locals is a common bug

## Next Steps
Proceed to **Lab 5.7: Pointers to Pointers** for multi-level indirection.
