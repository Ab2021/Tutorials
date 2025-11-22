# Lab 5.8: Dynamic Memory (new/delete)

## Objective
Learn how to allocate memory on the heap and manage its lifetime.

## Instructions

### Step 1: Single Allocation
Create `heap_demo.cpp`.

```cpp
#include <iostream>

int main() {
    int* p = new int(42); // Allocate int with value 42
    std::cout << "Value: " << *p << std::endl;
    
    delete p; // Free memory
    p = nullptr; // Good practice
    
    return 0;
}
```

### Step 2: Array Allocation
Allocate an array of 10 integers.

```cpp
int* arr = new int[10];
for (int i = 0; i < 10; ++i) arr[i] = i;

delete[] arr; // MUST use delete[] for arrays!
arr = nullptr;
```

### Step 3: Memory Leak
What happens if you don't delete?
```cpp
void leak() {
    int* p = new int(100);
    // No delete! Memory is lost until program ends.
}
```

## Challenges

### Challenge 1: Interactive Array
Ask user for array size. Allocate that many integers. Fill them. Print them. Delete them.
(This is something you can't do with standard stack arrays like `int arr[size]` in standard C++!)

### Challenge 2: Double Free
Try to delete the same pointer twice.
```cpp
int* p = new int(5);
delete p;
delete p; // Crash?
```
Fix it by setting `p = nullptr` after first delete. `delete nullptr` is safe (no-op).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    int size;
    std::cout << "Enter array size: ";
    std::cin >> size;
    
    int* arr = new int[size]; // Dynamic allocation
    
    for (int i = 0; i < size; ++i) {
        arr[i] = i * 10;
    }
    
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] arr; // Cleanup
    arr = nullptr;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `new` and `delete`
✅ Used `new[]` and `delete[]`
✅ Understood memory leaks
✅ Handled dynamic array size (Challenge 1)

## Key Learnings
- Heap memory persists until deleted
- Always pair `new` with `delete`
- Use `delete[]` for arrays
- Set pointers to `nullptr` after deletion

## Next Steps
Proceed to **Lab 5.9: Void Pointers** to see generic pointers.
