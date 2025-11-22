# Lab 5.10: Function Pointers (Callbacks)

## Objective
Learn how to store addresses of functions and call them dynamically (Callbacks).

## Instructions

### Step 1: Declare Function Pointer
Create `callbacks.cpp`.

```cpp
#include <iostream>

void hello() {
    std::cout << "Hello!" << std::endl;
}

void goodbye() {
    std::cout << "Goodbye!" << std::endl;
}

int main() {
    // Pointer to function taking void, returning void
    void (*funcPtr)() = nullptr;
    
    funcPtr = &hello;
    funcPtr(); // Calls hello
    
    funcPtr = &goodbye;
    funcPtr(); // Calls goodbye
    
    return 0;
}
```

### Step 2: Passing as Argument
Write a function `repeat` that takes a function pointer and a count.

```cpp
void repeat(void (*action)(), int times) {
    for (int i = 0; i < times; ++i) {
        action();
    }
}
// repeat(hello, 3);
```

### Step 3: Typedef / Using
Function pointer syntax is ugly. Make it pretty.

```cpp
using Action = void (*)();
void repeat(Action action, int times);
```

## Challenges

### Challenge 1: Calculator Callback
Create `int operate(int a, int b, int (*op)(int, int))`.
Pass `add` and `subtract` functions to it.

### Challenge 2: Array Sorter
Use `qsort` (from `<cstdlib>`) to sort an array of ints.
You need to provide a comparison function: `int compare(const void* a, const void* b)`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstdlib>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }

int operate(int a, int b, int (*op)(int, int)) {
    return op(a, b);
}

// For qsort
int compare(const void* a, const void* b) {
    int int_a = *static_cast<const int*>(a);
    int int_b = *static_cast<const int*>(b);
    return int_a - int_b;
}

int main() {
    // Challenge 1
    std::cout << "Add: " << operate(10, 5, add) << std::endl;
    std::cout << "Sub: " << operate(10, 5, subtract) << std::endl;
    
    // Challenge 2
    int arr[] = {5, 2, 9, 1, 5, 6};
    std::qsort(arr, 6, sizeof(int), compare);
    
    std::cout << "Sorted: ";
    for(int x : arr) std::cout << x << " ";
    std::cout << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Declared and used function pointer
✅ Passed function as argument (callback)
✅ Used `using` alias for readability
✅ Used standard library callback `qsort` (Challenge 2)

## Key Learnings
- Functions have addresses too
- Callbacks allow flexible behavior injection
- Syntax is tricky; use `using` or `typedef`
- Modern C++ prefers `std::function` (covered later)

## Next Steps
Congratulations! You've completed Module 5. You've mastered the most difficult part of C++ for beginners: Pointers.

Proceed to **Module 6: Memory Management Basics** to solidify these concepts.
