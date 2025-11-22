# Lab 5.5: Const Pointers vs Pointers to Const

## Objective
Master the "const" rules for pointers. Read right-to-left!

## Instructions

### Step 1: Pointer to Const
Create `const_ptrs.cpp`.
"Pointer to a constant integer". You can change where it points, but not the value.

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;
    
    const int* p1 = &x;
    // *p1 = 15; // Error: Read-only
    p1 = &y; // OK: Can point to something else
    
    return 0;
}
```

### Step 2: Const Pointer
"Constant pointer to an integer". You can change the value, but not where it points.

```cpp
    int* const p2 = &x;
    *p2 = 15; // OK: Can change value
    // p2 = &y; // Error: Cannot re-point
```

### Step 3: Const Pointer to Const
"Constant pointer to a constant integer". Locked down.

```cpp
    const int* const p3 = &x;
    // *p3 = 20; // Error
    // p3 = &y; // Error
```

## Challenges

### Challenge 1: String Literals
`const char* str = "Hello";`
Try to do `str[0] = 'h';`. Crash or Error?
Why is `char* str = "Hello";` (without const) deprecated/dangerous?

### Challenge 2: Function Parameters
Write a function that takes a buffer and its size. It should read the buffer but not modify it.
`void process(const int* buffer, int size)`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void process(const int* buffer, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << buffer[i] << " ";
        // buffer[i] = 0; // Error
    }
    std::cout << std::endl;
}

int main() {
    int arr[] = {1, 2, 3};
    process(arr, 3);
    
    const char* str = "Hello";
    // str[0] = 'h'; // Error (or crash if cast away const)
    
    return 0;
}
```
</details>

## Success Criteria
✅ Distinguished `const T*` vs `T* const`
✅ Implemented function with `const T*` parameter
✅ Understood string literal constness

## Key Learnings
- Read pointer declarations right-to-left
- Use `const T*` for read-only access to arrays/buffers
- String literals are effectively `const char*`

## Next Steps
Proceed to **Lab 5.6: References** for a safer alternative.
