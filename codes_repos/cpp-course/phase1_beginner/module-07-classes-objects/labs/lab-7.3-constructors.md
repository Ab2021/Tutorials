# Lab 7.3: Constructors and Initializer Lists

## Objective
Learn how to initialize objects using constructors and the member initializer list.

## Instructions

### Step 1: Default Constructor
Create `constructors.cpp`.

```cpp
#include <iostream>
#include <string>

class Student {
    std::string name;
    int id;
public:
    // Default Constructor
    Student() {
        name = "Unknown";
        id = 0;
        std::cout << "Default Ctor\n";
    }
    
    void print() { std::cout << name << " (" << id << ")\n"; }
};

int main() {
    Student s1; // Calls default ctor
    s1.print();
    return 0;
}
```

### Step 2: Parameterized Constructor
Add a constructor that takes arguments.

```cpp
    Student(std::string n, int i) {
        name = n;
        id = i;
        std::cout << "Param Ctor\n";
    }
```
Usage: `Student s2("Alice", 123);`

### Step 3: Member Initializer List (Best Practice)
Rewrite the parameterized constructor.

```cpp
    Student(std::string n, int i) : name(n), id(i) {
        std::cout << "Init List Ctor\n";
    }
```
*Why? It initializes variables directly rather than assigning them later. More efficient.*

## Challenges

### Challenge 1: Delegating Constructors (C++11)
Make the default constructor call the parameterized constructor.
```cpp
Student() : Student("Unknown", 0) { }
```

### Challenge 2: Explicit Constructor
Mark the constructor `explicit`.
`explicit Student(int id);`
Try `Student s = 500;`. It should fail.
(Implicit conversion prevention).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

class Student {
    std::string name;
    int id;
public:
    // Delegating Constructor
    Student() : Student("Unknown", 0) {}
    
    // Initializer List
    Student(std::string n, int i) : name(n), id(i) {}
    
    void print() { std::cout << name << ": " << id << std::endl; }
};

int main() {
    Student s1;
    Student s2("Bob", 999);
    
    s1.print();
    s2.print();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented default and parameterized constructors
✅ Used member initializer list
✅ Used delegating constructor (Challenge 1)
✅ Understood `explicit` keyword (Challenge 2)

## Key Learnings
- Constructors run when object is created
- Initializer lists are preferred over assignment in body
- `explicit` prevents accidental type conversions

## Next Steps
Proceed to **Lab 7.4: Encapsulation** to formalize getters and setters.
