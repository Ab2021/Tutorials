# Lab 16.4: Rule of Five

## Objective
Understand and implement the Rule of Five for complete resource management.

## Instructions

### Step 1: The Rule
If you define any of these five, define all five:
1. Destructor
2. Copy constructor
3. Copy assignment operator
4. Move constructor
5. Move assignment operator

Create `rule_of_five.cpp`.

```cpp
#include <iostream>
#include <utility>

class Resource {
    int* data;
public:
    // Constructor
    Resource(int value = 0) : data(new int(value)) {
        std::cout << "Constructed\n";
    }
    
    // 1. Destructor
    ~Resource() {
        delete data;
        std::cout << "Destroyed\n";
    }
    
    // 2. Copy constructor
    Resource(const Resource& other) 
        : data(new int(*other.data)) {
        std::cout << "Copy constructed\n";
    }
    
    // 3. Copy assignment
    Resource& operator=(const Resource& other) {
        std::cout << "Copy assigned\n";
        if (this != &other) {
            *data = *other.data;
        }
        return *this;
    }
    
    // 4. Move constructor
    Resource(Resource&& other) noexcept 
        : data(other.data) {
        other.data = nullptr;
        std::cout << "Move constructed\n";
    }
    
    // 5. Move assignment
    Resource& operator=(Resource&& other) noexcept {
        std::cout << "Move assigned\n";
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

### Step 2: Rule of Zero
If you don't manage resources, use default implementations.

```cpp
class Simple {
    std::string name;
    std::vector<int> data;
    // Compiler generates all five correctly!
};
```

## Challenges

### Challenge 1: Deleted Functions
Implement a move-only type by deleting copy operations.

### Challenge 2: Defaulted Functions
Use `= default` for some operations and custom for others.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

// Challenge 1: Move-only type
class MoveOnly {
    int* data;
public:
    MoveOnly(int v) : data(new int(v)) {}
    ~MoveOnly() { delete data; }
    
    // Delete copy operations
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    
    // Default move operations
    MoveOnly(MoveOnly&&) noexcept = default;
    MoveOnly& operator=(MoveOnly&&) noexcept = default;
};

int main() {
    MoveOnly m1(42);
    // MoveOnly m2 = m1; // Error: deleted
    MoveOnly m3 = std::move(m1); // OK
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented all five special member functions
✅ Understood Rule of Zero
✅ Created move-only type (Challenge 1)
✅ Used `= default` and `= delete` (Challenge 2)

## Key Learnings
- Rule of Five ensures complete resource management
- Rule of Zero: prefer using standard containers
- Delete copy operations for move-only types
- Use `= default` when compiler-generated is correct

## Next Steps
Proceed to **Lab 16.5: std::move Deep Dive**.
