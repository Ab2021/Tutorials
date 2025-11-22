# Lab 7.1: Basic Class Definition

## Objective
Learn how to define a class and create objects.

## Instructions

### Step 1: Define Class
Create `class_basics.cpp`.

```cpp
#include <iostream>
#include <string>

class Person {
public: // Access specifier
    std::string name;
    int age;
    
    void introduce() {
        std::cout << "Hi, I'm " << name << " and I'm " << age << " years old." << std::endl;
    }
};

int main() {
    Person p1;
    p1.name = "Alice";
    p1.age = 30;
    p1.introduce();
    
    return 0;
}
```

### Step 2: Multiple Objects
Create another object `p2`.

```cpp
Person p2;
p2.name = "Bob";
p2.age = 25;
p2.introduce();
```

### Step 3: Struct vs Class
Change `class` to `struct`. Remove `public:`.
Does it still work? Yes, because struct members are public by default.
Change back to `class` and remove `public:`. Does it work? No (private by default).

## Challenges

### Challenge 1: Member Function
Add a function `void birthday()` that increments age and prints "Happy Birthday!".

### Challenge 2: Pointer to Object
Create a pointer to `p1`.
`Person* ptr = &p1;`
Call `introduce()` using the pointer arrow operator `->`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;
    
    void introduce() {
        std::cout << "Hi, I'm " << name << " (" << age << ")" << std::endl;
    }
    
    // Challenge 1
    void birthday() {
        age++;
        std::cout << "Happy Birthday " << name << "!" << std::endl;
    }
};

int main() {
    Person p1;
    p1.name = "Alice";
    p1.age = 30;
    
    // Challenge 2
    Person* ptr = &p1;
    ptr->introduce();
    ptr->birthday();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Defined a class
✅ Instantiated objects
✅ Accessed members using `.`
✅ Accessed members using `->` (Challenge 2)

## Key Learnings
- Classes bundle data and behavior
- `public:` makes members accessible
- `.` operator for objects, `->` for pointers

## Next Steps
Proceed to **Lab 7.2: Access Specifiers** to hide data.
