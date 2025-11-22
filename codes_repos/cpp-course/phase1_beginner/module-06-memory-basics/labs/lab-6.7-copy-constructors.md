# Lab 6.7: Copy Constructors and Deep Copies

## Objective
Learn how to implement Deep Copy to safely copy objects that manage memory.

## Instructions

### Step 1: The Problem (Shallow Copy)
Create `deep_copy.cpp`.

```cpp
#include <iostream>

class Box {
    int* data;
public:
    Box(int val) { data = new int(val); }
    ~Box() { delete data; }
    int getValue() { return *data; }
};

int main() {
    Box b1(10);
    Box b2 = b1; // Default copy ctor (Shallow Copy)
    // b2.data points to same memory as b1.data
    // Destructor of b2 deletes memory.
    // Destructor of b1 deletes SAME memory -> Crash!
    return 0;
}
```

### Step 2: Copy Constructor (Deep Copy)
Implement a custom copy constructor.

```cpp
    Box(const Box& other) {
        std::cout << "Copying...\n";
        data = new int(*other.data); // Allocate NEW memory and copy value
    }
```

### Step 3: Copy Assignment Operator
Handle `b2 = b1;` (assignment, not initialization).

```cpp
    Box& operator=(const Box& other) {
        if (this == &other) return *this; // Self-assignment check
        
        delete data; // Clean up old memory
        data = new int(*other.data); // Allocate new
        return *this;
    }
```

## Challenges

### Challenge 1: Verify Addresses
Add a method `void printAddr()` to print the address of `data`.
Verify that `b1` and `b2` have *different* data addresses after a deep copy.

### Challenge 2: Rule of Three
You now have Destructor, Copy Ctor, and Copy Assignment. This is the "Rule of Three".
Try removing one and see what happens (e.g., remove copy assignment and try `b1 = b2`).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Box {
    int* data;
public:
    Box(int val) { data = new int(val); }
    
    // Copy Constructor
    Box(const Box& other) {
        data = new int(*other.data);
    }
    
    // Copy Assignment
    Box& operator=(const Box& other) {
        if (this != &other) {
            delete data;
            data = new int(*other.data);
        }
        return *this;
    }
    
    ~Box() { delete data; }
    
    void printAddr() { std::cout << "Data Addr: " << data << std::endl; }
};

int main() {
    Box b1(10);
    Box b2 = b1; // Copy Ctor
    
    b1.printAddr();
    b2.printAddr(); // Should be different
    
    Box b3(20);
    b3 = b1; // Copy Assignment
    
    return 0;
}
```
</details>

## Success Criteria
✅ Reproduced crash with shallow copy
✅ Implemented Deep Copy Constructor
✅ Implemented Copy Assignment Operator
✅ Handled self-assignment

## Key Learnings
- Default copy is shallow (copies pointer value, not data)
- Deep copy allocates new memory and copies data
- Rule of Three: If you need one, you need all three.

## Next Steps
Proceed to **Lab 6.8: Destructors** to refine cleanup.
