# Lab 8.3: Constructor and Destructor Order

## Objective
Understand the sequence of construction and destruction in inheritance hierarchies.

## Instructions

### Step 1: The Chain
Create `order.cpp`.

```cpp
#include <iostream>

class Parent {
public:
    Parent() { std::cout << "Parent Created\n"; }
    ~Parent() { std::cout << "Parent Destroyed\n"; }
};

class Child : public Parent {
public:
    Child() { std::cout << "Child Created\n"; }
    ~Child() { std::cout << "Child Destroyed\n"; }
};

int main() {
    {
        Child c;
    }
    return 0;
}
```
*Output: Parent Created -> Child Created -> Child Destroyed -> Parent Destroyed.*

### Step 2: Passing Arguments to Base
If Parent has no default constructor, Child MUST call it explicitly.

```cpp
class Parent {
    int id;
public:
    Parent(int i) : id(i) { std::cout << "Parent " << id << "\n"; }
};

class Child : public Parent {
public:
    Child(int i) : Parent(i) { // Call Base Ctor
        std::cout << "Child\n"; 
    }
};
```

## Challenges

### Challenge 1: Member Initialization
Add a member object to Child.
Verify the order: Base Ctor -> Member Ctor -> Child Ctor.

### Challenge 2: Deep Hierarchy
Create `GrandChild`.
Verify order: Parent -> Child -> GrandChild.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Member {
public:
    Member() { std::cout << "Member Created\n"; }
    ~Member() { std::cout << "Member Destroyed\n"; }
};

class Base {
public:
    Base() { std::cout << "Base Created\n"; }
    ~Base() { std::cout << "Base Destroyed\n"; }
};

class Derived : public Base {
    Member m;
public:
    Derived() { std::cout << "Derived Created\n"; }
    ~Derived() { std::cout << "Derived Destroyed\n"; }
};

int main() {
    Derived d;
    return 0;
}
```
</details>

## Success Criteria
✅ Observed construction order (Base -> Derived)
✅ Observed destruction order (Derived -> Base)
✅ Passed arguments to base constructor
✅ Verified member initialization order (Challenge 1)

## Key Learnings
- Always initialize base classes in the initializer list
- Construction builds up (foundation first)
- Destruction tears down (roof first)

## Next Steps
Proceed to **Lab 8.4: Method Overriding** to change behavior.
