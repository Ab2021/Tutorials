# Lab 6.9: Move Semantics Basics (Intro)

## Objective
Understand the concept of "Moving" resources instead of copying them (C++11).

## Instructions

### Step 1: The Cost of Copying
Imagine a class holding a huge array. Copying it is slow.
Moving it means just stealing the pointer.

### Step 2: Move Constructor
Create `move_demo.cpp`.

```cpp
#include <iostream>
#include <utility> // for std::move

class HugeData {
    int* data;
public:
    HugeData() { 
        data = new int[1000]; 
        std::cout << "Allocated\n";
    }
    ~HugeData() { 
        delete[] data; 
        std::cout << "Deleted\n";
    }
    
    // Copy Ctor (Slow)
    HugeData(const HugeData& other) {
        std::cout << "Copying...\n";
        data = new int[1000];
        // copy values...
    }
    
    // Move Ctor (Fast!)
    HugeData(HugeData&& other) noexcept {
        std::cout << "Moving...\n";
        data = other.data; // Steal pointer
        other.data = nullptr; // Nullify source
    }
};
```

### Step 3: Triggering Move
Use `std::move` to cast an lvalue to an rvalue.

```cpp
int main() {
    HugeData a;
    HugeData b = std::move(a); // Calls Move Ctor
    // a is now empty/null
    return 0;
}
```

## Challenges

### Challenge 1: Move Assignment
Implement the Move Assignment Operator `operator=(HugeData&& other)`.
Remember to check for self-assignment and delete own data first.

### Challenge 2: Vector Push Back
Create a `std::vector<HugeData>`.
`v.push_back(HugeData());`
Does it copy or move? (It should move because the temporary is an rvalue).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <utility>

class HugeData {
    int* data;
public:
    HugeData() : data(new int[100]) {}
    ~HugeData() { delete[] data; }
    
    HugeData(const HugeData& other) : data(new int[100]) {
        std::cout << "Copy\n";
    }
    
    HugeData(HugeData&& other) noexcept : data(other.data) {
        other.data = nullptr;
        std::cout << "Move\n";
    }
    
    HugeData& operator=(HugeData&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};

int main() {
    std::vector<HugeData> v;
    v.push_back(HugeData()); // Should print "Move"
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented Move Constructor
✅ Used `std::move`
✅ Understood pointer stealing mechanism
✅ Implemented Move Assignment (Challenge 1)

## Key Learnings
- Move semantics optimize performance by avoiding deep copies
- `&&` denotes an rvalue reference
- Source object must be left in a valid (but unspecified) state (usually null)

## Next Steps
Proceed to **Lab 6.10: Custom String Class** to put it all together.
