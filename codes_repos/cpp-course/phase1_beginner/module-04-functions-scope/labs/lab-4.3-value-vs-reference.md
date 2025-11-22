# Lab 4.3: Pass by Value vs Reference

## Objective
Understand the crucial difference between passing a copy (Value) and passing the original object (Reference).

## Instructions

### Step 1: Pass by Value (The Fail)
Create `swap_demo.cpp`. Try to swap two numbers using pass-by-value.

```cpp
#include <iostream>

void swapVal(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    std::cout << "Inside swapVal: " << a << " " << b << std::endl;
}

int main() {
    int x = 10, y = 20;
    swapVal(x, y);
    std::cout << "Main after swapVal: " << x << " " << y << std::endl;
    // x and y did NOT change!
    return 0;
}
```

### Step 2: Pass by Reference (The Fix)
Change the parameters to references (`&`).

```cpp
void swapRef(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
// In main: swapRef(x, y);
```
*Now x and y should be swapped.*

### Step 3: Visualizing Addresses
Print the memory addresses to prove they are the same variable.

```cpp
void printAddr(int& n) {
    std::cout << "Func Addr: " << &n << std::endl;
}

int main() {
    int x = 10;
    std::cout << "Main Addr: " << &x << std::endl;
    printAddr(x); // Should match
    return 0;
}
```

## Challenges

### Challenge 1: Increment Function
Write a function `void increment(int& n)` that adds 1 to the passed variable.
Try calling it with a literal: `increment(5);`. What happens? Why?

### Challenge 2: Reference to Pointer
Write a function that changes where a pointer points to.
`void changePointer(int*& ptr)`
*Hint: This is a reference to a pointer!*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void increment(int& n) {
    n++;
}

void changePointer(int*& ptr) {
    static int safe = 100;
    ptr = &safe;
}

int main() {
    int x = 10;
    increment(x);
    std::cout << x << std::endl; // 11
    
    // increment(5); // Error: cannot bind non-const lvalue reference to rvalue
    
    int val = 50;
    int* p = &val;
    std::cout << "Ptr points to: " << *p << std::endl;
    changePointer(p);
    std::cout << "Ptr now points to: " << *p << std::endl; // 100
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed that pass-by-value does not modify original
✅ Successfully swapped variables using pass-by-reference
✅ Verified memory addresses match
✅ Understood why references cannot bind to literals (Challenge 1)

## Key Learnings
- `&` in parameter list means "Reference"
- References act as aliases to the original variable
- Use references when you need to modify the argument

## Next Steps
Proceed to **Lab 4.4: Const References** for efficiency and safety.
