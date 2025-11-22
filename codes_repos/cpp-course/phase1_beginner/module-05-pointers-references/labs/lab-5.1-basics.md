# Lab 5.1: Pointer Basics

## Objective
Understand how to declare pointers, assign addresses, and access values.

## Instructions

### Step 1: Address-of Operator (&)
Create `pointers.cpp`.

```cpp
#include <iostream>

int main() {
    int num = 42;
    std::cout << "Value: " << num << std::endl;
    std::cout << "Address: " << &num << std::endl;
    
    return 0;
}
```

### Step 2: Declaring a Pointer
Declare a pointer variable and store the address of `num`.

```cpp
int* ptr = &num;
std::cout << "Pointer holds: " << ptr << std::endl;
```

### Step 3: Dereferencing (*)
Access the value through the pointer.

```cpp
std::cout << "Value via pointer: " << *ptr << std::endl;
```

### Step 4: Modification
Change the value using the pointer.

```cpp
*ptr = 100;
std::cout << "New value of num: " << num << std::endl;
```

## Challenges

### Challenge 1: Pointer Size
Print `sizeof(ptr)` and `sizeof(num)`.
On a 64-bit system, a pointer is usually 8 bytes, regardless of what it points to (char*, int*, double*). Verify this.

### Challenge 2: Double Pointer
Create a pointer to a pointer.
```cpp
int** ptr2 = &ptr;
```
Print the value of `num` using `ptr2` (requires double dereference).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    int num = 42;
    int* ptr = &num;
    
    std::cout << "Address: " << ptr << std::endl;
    std::cout << "Value: " << *ptr << std::endl;
    
    *ptr = 100;
    std::cout << "Num is now: " << num << std::endl;
    
    // Challenge 1
    std::cout << "Size of int*: " << sizeof(ptr) << std::endl;
    std::cout << "Size of char*: " << sizeof(char*) << std::endl;
    
    // Challenge 2
    int** ptr2 = &ptr;
    std::cout << "Value via ptr2: " << **ptr2 << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `&` to get address
✅ Declared `int*` correctly
✅ Used `*` to read and write value
✅ Accessed value via double pointer (Challenge 2)

## Key Learnings
- Pointers store addresses
- `*` has two meanings: declaration (`int*`) and dereference (`*ptr`)
- Changing `*ptr` changes the original variable

## Next Steps
Proceed to **Lab 5.2: Nullptr and Safety** to avoid crashes.
