# Lab 6.6: Simple Smart Pointer

## Objective
Build a simplified version of `std::unique_ptr` to understand how smart pointers work under the hood.

## Instructions

### Step 1: The Class Structure
Create `smart_ptr.cpp`. Define a template class.

```cpp
#include <iostream>

template <typename T>
class SmartPtr {
    T* ptr;
public:
    explicit SmartPtr(T* p = nullptr) : ptr(p) {
        std::cout << "SmartPtr created\n";
    }
    
    ~SmartPtr() {
        delete ptr;
        std::cout << "SmartPtr destroyed\n";
    }
};
```

### Step 2: Operator Overloading
Make it behave like a pointer.

```cpp
    T& operator*() { return *ptr; }
    T* operator->() { return ptr; }
```

### Step 3: Usage
```cpp
struct Test {
    void sayHello() { std::cout << "Hello\n"; }
};

int main() {
    SmartPtr<Test> sp(new Test());
    sp->sayHello();
    return 0;
}
```

## Challenges

### Challenge 1: Prevent Copying
Smart pointers that own memory shouldn't be copied (double free risk).
Delete the copy constructor and assignment operator.
```cpp
SmartPtr(const SmartPtr&) = delete;
SmartPtr& operator=(const SmartPtr&) = delete;
```
Try to copy `sp` in main to verify the error.

### Challenge 2: Release
Add a `T* release()` method that returns the raw pointer and sets the internal pointer to `nullptr` (transferring ownership).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

template <typename T>
class SmartPtr {
    T* ptr;
public:
    explicit SmartPtr(T* p = nullptr) : ptr(p) {}
    ~SmartPtr() { delete ptr; }
    
    // Challenge 1: No Copy
    SmartPtr(const SmartPtr&) = delete;
    SmartPtr& operator=(const SmartPtr&) = delete;
    
    T& operator*() { return *ptr; }
    T* operator->() { return ptr; }
    
    // Challenge 2: Release
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
};

struct Test { void greet() { std::cout << "Hi\n"; } };

int main() {
    SmartPtr<Test> sp(new Test());
    sp->greet();
    
    // SmartPtr<Test> sp2 = sp; // Error!
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented template smart pointer class
✅ Overloaded `*` and `->` operators
✅ Prevented copying (Challenge 1)
✅ Implemented `release()` (Challenge 2)

## Key Learnings
- Smart pointers are wrappers around raw pointers
- Operator overloading makes them feel like pointers
- Unique ownership requires disabling copying

## Next Steps
Proceed to **Lab 6.7: Copy Constructors** to handle copying correctly.
