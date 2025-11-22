# Lab 6.4: Memory Leaks and Detection

## Objective
Understand what memory leaks are and how to simulate/detect them.

## Instructions

### Step 1: Create a Leak
Create `leak.cpp`.

```cpp
#include <iostream>

void createLeak() {
    int* p = new int(5);
    // Oops, forgot delete!
}

int main() {
    for (int i = 0; i < 1000; ++i) {
        createLeak();
    }
    std::cout << "Leaked 4KB of memory..." << std::endl;
    return 0;
}
```

### Step 2: Tracking Allocations (Manual)
Since we don't have external tools installed, let's write a simple tracker.
Overload global `new` and `delete`.

```cpp
int allocations = 0;

void* operator new(size_t size) {
    allocations++;
    std::cout << "Allocating " << size << " bytes\n";
    return malloc(size);
}

void operator delete(void* p) noexcept {
    allocations--;
    std::cout << "Freeing memory\n";
    free(p);
}
```

### Step 3: Verify Leak
Run the code. If `allocations` is not 0 at the end, you have a leak.

## Challenges

### Challenge 1: Fix the Leak
Modify `createLeak` to properly delete the memory. Verify `allocations` returns to 0.

### Challenge 2: Leak in Exception
Throw an exception before delete.
```cpp
void risky() {
    int* p = new int(5);
    throw std::runtime_error("Oops");
    delete p; // Unreachable! Leak!
}
```
Catch the exception in main. Check allocation count.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstdlib>

int allocations = 0;

void* operator new(size_t size) {
    allocations++;
    return malloc(size);
}

void operator delete(void* p) noexcept {
    allocations--;
    free(p);
}

void fixedLeak() {
    int* p = new int(5);
    delete p;
}

int main() {
    fixedLeak();
    
    if (allocations == 0) {
        std::cout << "No leaks detected!" << std::endl;
    } else {
        std::cout << "Leaks detected: " << allocations << std::endl;
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Created a memory leak
✅ Implemented simple allocation tracker
✅ Detected the leak
✅ Fixed the leak

## Key Learnings
- Leaks occur when `delete` is missed
- Exceptions can cause leaks (skipped code)
- Tools (Valgrind, ASan) are usually used, but manual tracking works for learning

## Next Steps
Proceed to **Lab 6.5: RAII Introduction** to fix leaks forever.
