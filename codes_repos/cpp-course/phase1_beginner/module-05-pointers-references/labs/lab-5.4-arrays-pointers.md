# Lab 5.4: Arrays and Pointers

## Objective
Understand the close relationship (and differences) between arrays and pointers.

## Instructions

### Step 1: Decay
Create `decay.cpp`. Arrays "decay" into pointers when passed to functions.

```cpp
#include <iostream>

void printSize(int* arr) {
    std::cout << "Size in func: " << sizeof(arr) << std::endl; // Size of pointer (8)
}

int main() {
    int arr[10];
    std::cout << "Size in main: " << sizeof(arr) << std::endl; // 40 (10 * 4)
    printSize(arr);
    return 0;
}
```

### Step 2: Array Syntax with Pointers
You can use `[]` on pointers!

```cpp
int* p = arr;
p[0] = 5; // Same as *p = 5
p[1] = 10; // Same as *(p + 1) = 10
```

### Step 3: Pointer Syntax with Arrays
You can use `*` on arrays!

```cpp
*arr = 100; // Same as arr[0] = 100
*(arr + 1) = 200;
```

## Challenges

### Challenge 1: Prevent Decay
Pass the array by reference to preserve size info.
`void printSizeRef(int (&arr)[10])`
Check `sizeof` inside this function.

### Challenge 2: Range-based Loop on Pointer?
Try to use a range-based for loop on a pointer.
```cpp
int* p = arr;
// for (int x : p) ... // Error!
```
Why? Pointers don't know their size. Arrays do (in the scope they are defined).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void printSizeRef(int (&arr)[10]) {
    std::cout << "Size in ref func: " << sizeof(arr) << std::endl; // 40
}

int main() {
    int arr[10];
    printSizeRef(arr);
    
    int* p = arr;
    // for(int x : p) {} // Compilation error: begin/end not defined for int*
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed array decay
✅ Used `[]` on pointers and `*` on arrays
✅ Passed array by reference (Challenge 1)
✅ Understood why range loops fail on pointers

## Key Learnings
- Arrays are not pointers, but convert to them easily
- `sizeof` behaves differently on arrays vs pointers
- Passing arrays to functions usually loses size information

## Next Steps
Proceed to **Lab 5.5: Const Pointers** to master const correctness.
