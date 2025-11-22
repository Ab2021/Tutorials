# Lab 8.7: Virtual Destructors

## Objective
Understand why base class destructors must be `virtual` when using polymorphism.

## Instructions

### Step 1: The Leak
Create `virt_destructor.cpp`.

```cpp
#include <iostream>

class Base {
public:
    Base() { std::cout << "Base Ctor\n"; }
    ~Base() { std::cout << "Base Dtor\n"; } // Non-virtual!
};

class Derived : public Base {
    int* data;
public:
    Derived() { 
        data = new int[100];
        std::cout << "Derived Ctor\n"; 
    }
    ~Derived() { 
        delete[] data;
        std::cout << "Derived Dtor\n"; 
    }
};

int main() {
    Base* b = new Derived();
    delete b; // Problem!
    return 0;
}
```
*Output: Base Ctor -> Derived Ctor -> Base Dtor. (Derived Dtor is MISSING! Memory Leak!)*

### Step 2: The Fix
Add `virtual` to `~Base()`.

```cpp
virtual ~Base() { std::cout << "Base Dtor\n"; }
```
*Output: ... Derived Dtor -> Base Dtor. (Correct)*

## Challenges

### Challenge 1: Pure Virtual Destructor
Can a destructor be pure virtual?
`virtual ~Base() = 0;`
Yes, but you MUST provide a definition (body) for it, because derived destructors call it.
`Base::~Base() {}`

### Challenge 2: Final Class (C++11)
Prevent inheritance using `final`.
`class Derived final : public Base {};`
Try to inherit from `Derived`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() { std::cout << "Base Dtor\n"; }
};

class Derived final : public Base {
public:
    ~Derived() { std::cout << "Derived Dtor\n"; }
};

// class More : public Derived {}; // Error: Derived is final

int main() {
    Base* b = new Derived();
    delete b;
    return 0;
}
```
</details>

## Success Criteria
✅ Reproduced memory leak with non-virtual destructor
✅ Fixed leak with virtual destructor
✅ Understood `final` keyword (Challenge 2)

## Key Learnings
- If a class is meant to be a base class, give it a virtual destructor
- Deleting a derived object via a base pointer requires a virtual destructor
- `final` prevents further inheritance

## Next Steps
Proceed to **Lab 8.8: Multiple Inheritance** to combine classes.
