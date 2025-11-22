# Lab 6.2: New and Delete (Deep Dive)

## Objective
Practice allocating and deallocating single objects.

## Instructions

### Step 1: Allocation
Create `new_delete.cpp`.

```cpp
#include <iostream>

struct User {
    std::string name;
    int id;
};

int main() {
    User* u = new User; // Default constructor
    u->name = "Alice";
    u->id = 101;
    
    std::cout << u->name << std::endl;
    
    delete u; // Cleanup
    return 0;
}
```

### Step 2: Initialization
Use direct initialization.

```cpp
int* p1 = new int(10); // Value 10
int* p2 = new int;     // Uninitialized (garbage)
int* p3 = new int();   // Value 0 (Value initialization)
```

### Step 3: Placement New (Advanced Preview)
Just know it exists: You can construct an object in pre-allocated memory.
`new (address) Type();` (Don't use this yet).

## Challenges

### Challenge 1: Double Free
Allocate an int, delete it, then delete it again.
Observe the crash or error message.

### Challenge 2: Use After Free
Allocate, delete, then try to print the value.
`delete p; std::cout << *p;`
This is Undefined Behavior (might print garbage, might crash).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    int* p = new int(5);
    std::cout << *p << std::endl;
    delete p;
    
    // Challenge 1: Double Free
    // delete p; // Crash!
    
    // Challenge 2: Use After Free
    // std::cout << *p << std::endl; // Undefined Behavior
    
    p = nullptr; // Safety
    delete p; // Safe
    
    return 0;
}
```
</details>

## Success Criteria
✅ Allocated struct on heap
✅ Used different initialization syntax
✅ Triggered Double Free error (Challenge 1)
✅ Understood Use After Free risk (Challenge 2)

## Key Learnings
- `new` returns a pointer
- `delete` frees the memory
- Accessing memory after delete is dangerous
- Setting to `nullptr` prevents double free crashes

## Next Steps
Proceed to **Lab 6.3: Array Allocation** for managing lists.
