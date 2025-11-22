# Lab 5.3: Pointer Arithmetic

## Objective
Understand how pointers can be used to navigate arrays.

## Instructions

### Step 1: Array and Pointer
Create `arithmetic.cpp`. Point to the start of an array.

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int* ptr = arr; // Points to arr[0]
    
    std::cout << "First: " << *ptr << std::endl;
    
    return 0;
}
```

### Step 2: Increment
Move the pointer forward.

```cpp
ptr++; // Moves 4 bytes (sizeof int) forward
std::cout << "Second: " << *ptr << std::endl; // 20
```

### Step 3: Addition
Access relative positions.

```cpp
std::cout << "Third: " << *(ptr + 1) << std::endl; // 30
```

### Step 4: Loop
Iterate using pointers.

```cpp
ptr = arr; // Reset
for (int i = 0; i < 5; ++i) {
    std::cout << *ptr << " ";
    ptr++;
}
```

## Challenges

### Challenge 1: Reverse Iteration
Reset pointer to the **end** of the array (`&arr[4]`).
Loop backwards using `ptr--` until you print all elements.

### Challenge 2: Distance
Subtract two pointers.
```cpp
int* start = &arr[0];
int* end = &arr[4];
std::ptrdiff_t dist = end - start;
```
Print the distance. What unit is it in? (Bytes or Elements?)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    
    // Challenge 1
    int* ptr = &arr[4];
    for (int i = 0; i < 5; ++i) {
        std::cout << *ptr << " ";
        ptr--;
    }
    std::cout << std::endl;
    
    // Challenge 2
    int* start = &arr[0];
    int* end = &arr[4];
    std::cout << "Distance: " << (end - start) << " elements" << std::endl; // 4
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `ptr++` to traverse array
✅ Used `*(ptr + n)` to access elements
✅ Iterated backwards (Challenge 1)
✅ Calculated pointer distance (Challenge 2)

## Key Learnings
- Pointer arithmetic respects type size
- `ptr + 1` adds `sizeof(T)` bytes
- Subtracting pointers gives number of elements

## Next Steps
Proceed to **Lab 5.4: Arrays and Pointers** to see how they relate.
