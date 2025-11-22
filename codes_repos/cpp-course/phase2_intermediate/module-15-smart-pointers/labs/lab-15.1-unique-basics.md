# Lab 15.1: Unique Pointer Basics

## Objective
Master `std::unique_ptr` for exclusive ownership.

## Instructions

### Step 1: Basic Usage
Create `unique_ptr_basics.cpp`.

```cpp
#include <iostream>
#include <memory>

int main() {
    // Create unique_ptr
    std::unique_ptr<int> p1 = std::make_unique<int>(42);
    
    std::cout << "Value: " << *p1 << "\n";
    std::cout << "Address: " << p1.get() << "\n";
    
    // Automatic cleanup when p1 goes out of scope
    return 0;
}
```

### Step 2: Transfer Ownership
```cpp
std::unique_ptr<int> p2 = std::move(p1);
// p1 is now nullptr
if (!p1) std::cout << "p1 is null\n";
std::cout << "p2: " << *p2 << "\n";
```

### Step 3: Reset and Release
```cpp
p2.reset(); // Deletes object, sets to nullptr
p2.reset(new int(100)); // Deletes old, takes new

int* raw = p2.release(); // Gives up ownership, returns raw pointer
delete raw; // Manual cleanup required
```

## Challenges

### Challenge 1: Class with unique_ptr
Create a class that owns a resource via `unique_ptr`.
```cpp
class Widget {
    std::unique_ptr<int> data;
public:
    Widget() : data(std::make_unique<int>(0)) {}
    void set(int val) { *data = val; }
    int get() const { return *data; }
};
```

### Challenge 2: Return unique_ptr
Write a factory function that returns `unique_ptr`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>

class Widget {
    std::unique_ptr<int> data;
public:
    Widget() : data(std::make_unique<int>(0)) {}
    void set(int val) { *data = val; }
    int get() const { return *data; }
};

std::unique_ptr<Widget> createWidget(int val) {
    auto w = std::make_unique<Widget>();
    w->set(val);
    return w; // Move semantics
}

int main() {
    auto w = createWidget(42);
    std::cout << "Widget value: " << w->get() << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and used `unique_ptr`
✅ Transferred ownership with `std::move`
✅ Used `reset()` and `release()`
✅ Returned `unique_ptr` from function (Challenge 2)

## Key Learnings
- `unique_ptr` has zero overhead when not moved
- Cannot be copied, only moved
- Automatically deletes managed object

## Next Steps
Proceed to **Lab 15.2: Unique Pointer with Arrays**.
