# Lab 5.9: Void Pointers

## Objective
Understand `void*` as a generic pointer type and its dangers.

## Instructions

### Step 1: Generic Pointer
Create `void_ptr.cpp`. `void*` can hold any address.

```cpp
#include <iostream>

int main() {
    int n = 10;
    double d = 3.14;
    
    void* ptr;
    
    ptr = &n; // Point to int
    std::cout << "Address of n: " << ptr << std::endl;
    
    ptr = &d; // Point to double
    std::cout << "Address of d: " << ptr << std::endl;
    
    return 0;
}
```

### Step 2: Dereferencing?
Try to print `*ptr`.
`// std::cout << *ptr; // Error: 'void*' is not a pointer-to-object type`
You must cast it back to the correct type.

### Step 3: Casting
```cpp
ptr = &n;
int* intPtr = static_cast<int*>(ptr);
std::cout << "Value: " << *intPtr << std::endl;
```

## Challenges

### Challenge 1: Generic Printer
Write a function `void printBytes(void* data, int size)` that prints the memory byte-by-byte (as hex).
Cast `data` to `unsigned char*` to iterate.

### Challenge 2: Type Tagging
Create a struct `Any` that holds a `void*` and an enum `Type { INT, DOUBLE }`.
Write a print function that checks the type and casts accordingly.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <iomanip>

void printBytes(void* data, int size) {
    unsigned char* bytePtr = static_cast<unsigned char*>(data);
    for (int i = 0; i < size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)bytePtr[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

enum Type { INT, DOUBLE };
struct Any {
    void* ptr;
    Type type;
};

void printAny(Any a) {
    if (a.type == INT) 
        std::cout << "Int: " << *static_cast<int*>(a.ptr) << std::endl;
    else if (a.type == DOUBLE)
        std::cout << "Double: " << *static_cast<double*>(a.ptr) << std::endl;
}

int main() {
    int x = 0x12345678;
    printBytes(&x, sizeof(x)); // Little-endian: 78 56 34 12
    
    Any a = {&x, INT};
    printAny(a);
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `void*` to store different addresses
✅ Cast `void*` back to specific type
✅ Implemented byte inspector (Challenge 1)
✅ Implemented tagged union pattern (Challenge 2)

## Key Learnings
- `void*` is a raw address with no type info
- Must cast to use
- Dangerous because type safety is lost
- Used in low-level C APIs (like `malloc`, `memcpy`)

## Next Steps
Proceed to **Lab 5.10: Function Pointers** to treat code as data.
