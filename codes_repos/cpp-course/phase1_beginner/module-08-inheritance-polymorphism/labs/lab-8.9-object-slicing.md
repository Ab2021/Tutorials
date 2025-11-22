# Lab 8.9: Object Slicing

## Objective
Understand what happens when you pass a derived object by value to a base parameter.

## Instructions

### Step 1: The Setup
Create `slicing.cpp`.

```cpp
#include <iostream>
#include <string>

class Base {
public:
    int x = 10;
    virtual void show() { std::cout << "Base: " << x << "\n"; }
};

class Derived : public Base {
public:
    int y = 20;
    void show() override { std::cout << "Derived: " << x << ", " << y << "\n"; }
};
```

### Step 2: Slicing by Value
Write a function taking Base by value.

```cpp
void printByValue(Base b) {
    b.show();
}

int main() {
    Derived d;
    printByValue(d); // Calls Base::show()!
    return 0;
}
```
*Why? `b` is a COPY of the Base part of `d`. The `y` part is "sliced" off. The vtable points to Base.*

### Step 3: The Fix (Pass by Reference)
Change to reference.

```cpp
void printByRef(Base& b) {
    b.show();
}
// printByRef(d); // Calls Derived::show()
```

## Challenges

### Challenge 1: Slicing in Vectors
`std::vector<Base> list;`
`list.push_back(Derived());`
Iterate and call `show()`. It will be sliced.
Fix: Use `std::vector<Base*>`.

### Challenge 2: Clone Pattern
Implement a `virtual Base* clone() const` method to allow safe copying of polymorphic objects.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class Base {
public:
    virtual void show() { std::cout << "Base\n"; }
    virtual Base* clone() const { return new Base(*this); }
    virtual ~Base() {}
};

class Derived : public Base {
public:
    void show() override { std::cout << "Derived\n"; }
    Base* clone() const override { return new Derived(*this); }
};

int main() {
    std::vector<Base*> list;
    list.push_back(new Derived());
    
    for(auto b : list) b->show();
    
    // Cleanup
    for(auto b : list) delete b;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Demonstrated object slicing
✅ Fixed slicing using references
✅ Fixed slicing in containers using pointers (Challenge 1)
✅ Implemented Clone pattern (Challenge 2)

## Key Learnings
- Never pass polymorphic objects by value
- Slicing loses data and polymorphic behavior
- Use pointers or references for polymorphism

## Next Steps
Proceed to **Lab 8.10: Shape Hierarchy System** to build a complete system.
