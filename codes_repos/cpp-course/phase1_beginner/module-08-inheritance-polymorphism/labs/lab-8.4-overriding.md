# Lab 8.4: Method Overriding

## Objective
Learn how to redefine base class methods in derived classes.

## Instructions

### Step 1: Redefinition
Create `overriding.cpp`.

```cpp
#include <iostream>

class Animal {
public:
    void speak() { std::cout << "Animal sound\n"; }
};

class Dog : public Animal {
public:
    void speak() { std::cout << "Woof!\n"; }
};

int main() {
    Dog d;
    d.speak(); // Woof!
    return 0;
}
```

### Step 2: The Problem (Slicing/Hiding)
Assign Dog to Animal variable.

```cpp
Animal a = d; // Object Slicing (copies only Animal part)
a.speak(); // Animal sound
```

### Step 3: Calling Base Method
Call the parent's method from the child.

```cpp
class Dog : public Animal {
public:
    void speak() {
        Animal::speak(); // Call base
        std::cout << "Woof!\n";
    }
};
```

## Challenges

### Challenge 1: Pointer Behavior (Non-Virtual)
`Animal* ptr = &d;`
`ptr->speak();`
What does it print? "Animal sound". Why? Because `speak` is not virtual (Static Binding).

### Challenge 2: Hiding Overloads
If Base has `speak(int)` and Derived has `speak()`, `speak(int)` is hidden!
Use `using Animal::speak;` in Derived to unhide it.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Base {
public:
    void func() { std::cout << "Base func\n"; }
    void func(int i) { std::cout << "Base func(int)\n"; }
};

class Derived : public Base {
public:
    // using Base::func; // Uncomment to fix Challenge 2
    void func() { std::cout << "Derived func\n"; }
};

int main() {
    Derived d;
    d.func();
    // d.func(10); // Error: Hidden by Derived::func()
    
    d.Base::func(10); // Explicit call works
    
    return 0;
}
```
</details>

## Success Criteria
✅ Overrode a base method
✅ Called base method using scope resolution `::`
✅ Observed static binding with pointers (Challenge 1)
✅ Identified name hiding issue (Challenge 2)

## Key Learnings
- Redefining a method hides all base overloads
- Without `virtual`, pointers call based on pointer type, not object type

## Next Steps
Proceed to **Lab 8.5: Polymorphism** to fix the pointer issue.
