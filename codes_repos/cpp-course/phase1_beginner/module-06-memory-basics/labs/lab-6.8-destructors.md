# Lab 6.8: Destructors and Cleanup

## Objective
Understand destructor timing and inheritance issues.

## Instructions

### Step 1: Destruction Order
Create `destructors.cpp`.

```cpp
#include <iostream>

class Component {
    std::string name;
public:
    Component(std::string n) : name(n) { std::cout << name << " created\n"; }
    ~Component() { std::cout << name << " destroyed\n"; }
};

int main() {
    Component c1("Main");
    {
        Component c2("Inner");
    } // c2 dies
    std::cout << "Exiting main\n";
    return 0;
} // c1 dies
```
*Observe: Reverse order of construction.*

### Step 2: Virtual Destructor (Preview)
When using inheritance, base class destructors MUST be virtual.

```cpp
class Base {
public:
    virtual ~Base() { std::cout << "Base destroyed\n"; }
};

class Derived : public Base {
public:
    ~Derived() { std::cout << "Derived destroyed\n"; }
};

int main() {
    Base* b = new Derived();
    delete b; // Should call Derived then Base
    return 0;
}
```

### Step 3: The Leak
Remove `virtual` from `~Base()`. Run it.
*Only `~Base()` is called! `~Derived()` is skipped, causing leaks.*

## Challenges

### Challenge 1: Member Destruction
Add a `Component` member to `Derived`.
Verify that the member's destructor is called automatically when `Derived` is destroyed.

### Challenge 2: Array Destruction
Create an array of objects.
`Component* arr = new Component[3];`
`delete[] arr;`
Verify that the destructor is called 3 times.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

class Component {
    std::string name;
public:
    Component(std::string n = "Anon") : name(n) {}
    ~Component() { std::cout << "Component " << name << " destroyed\n"; }
};

class Base {
public:
    virtual ~Base() { std::cout << "Base destroyed\n"; }
};

class Derived : public Base {
    Component c;
public:
    Derived() : c("Member") {}
    ~Derived() { std::cout << "Derived destroyed\n"; }
};

int main() {
    Base* b = new Derived();
    delete b; 
    // Output: Derived destroyed -> Component Member destroyed -> Base destroyed
    
    std::cout << "---\n";
    
    Component* arr = new Component[2];
    delete[] arr;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Observed LIFO destruction order
✅ Implemented virtual destructor
✅ Demonstrated leak without virtual destructor
✅ Verified array destruction (Challenge 2)

## Key Learnings
- Destructors run in reverse order of construction
- Always make base class destructors `virtual`
- Members are destroyed automatically

## Next Steps
Proceed to **Lab 6.9: Move Semantics** for performance.
