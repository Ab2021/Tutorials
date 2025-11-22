# Lab 6.1: Stack vs Heap Memory

## Objective
Visualize the difference between Stack (automatic) and Heap (dynamic) memory.

## Instructions

### Step 1: Address Comparison
Create `stack_heap.cpp`.

```cpp
#include <iostream>

int main() {
    int stackVar = 10;
    int* heapVar = new int(20);
    
    std::cout << "Stack Address: " << &stackVar << std::endl;
    std::cout << "Heap Address:  " << heapVar << std::endl;
    
    delete heapVar;
    return 0;
}
```
*Note: Stack addresses are usually high, Heap addresses are usually low (or vice versa depending on OS).*

### Step 2: Stack Lifetime
Demonstrate that stack variables die at end of scope.

```cpp
int* dangling = nullptr;
{
    int temp = 50;
    dangling = &temp;
} 
// *dangling is now unsafe!
```

### Step 3: Heap Lifetime
Demonstrate that heap variables survive scope.

```cpp
int* p = nullptr;
{
    p = new int(50);
}
std::cout << "Heap value still exists: " << *p << std::endl;
delete p;
```

## Challenges

### Challenge 1: Stack Overflow
Write a recursive function without a base case (or a very deep one) to crash the program with a Stack Overflow.
`void crash() { crash(); }`

### Challenge 2: Heap Exhaustion
Try to allocate a massive amount of memory (e.g., `new int[1000000000]`) in a loop until it fails (throws `std::bad_alloc`).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void stackOverflow() {
    int arr[1000]; // Consume stack space
    stackOverflow();
}

int main() {
    // Challenge 1: Uncomment to crash
    // stackOverflow();
    
    // Challenge 2
    try {
        while (true) {
            int* huge = new int[100000000]; // 400 MB
            std::cout << "Allocated 400 MB" << std::endl;
        }
    } catch (const std::bad_alloc& e) {
        std::cout << "Memory exhausted: " << e.what() << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed address differences
✅ Understood scope lifetime differences
✅ Caused Stack Overflow (Challenge 1)
✅ Handled Heap Exhaustion (Challenge 2)

## Key Learnings
- Stack is automatic and fast but limited
- Heap is manual and large but slower
- Stack variables die at `}`, Heap variables die at `delete`

## Next Steps
Proceed to **Lab 6.2: New and Delete** to practice manual management.
