# Lab 15.9: Common Pitfalls

## Objective
Learn to avoid common smart pointer mistakes.

## Instructions

### Step 1: Double Delete
Create `pitfalls.cpp`.

```cpp
#include <iostream>
#include <memory>

int main() {
    int* raw = new int(42);
    
    std::unique_ptr<int> p1(raw);
    // std::unique_ptr<int> p2(raw); // DANGER: Double delete!
    
    return 0;
}
```

### Step 2: Storing get() Result
```cpp
auto p = std::make_unique<int>(42);
int* raw = p.get(); // OK for temporary use

p.reset(); // p deletes the object
// *raw; // DANGER: Dangling pointer!
```

### Step 3: Shared Pointer from This
```cpp
class Widget {
public:
    std::shared_ptr<Widget> getShared() {
        // return std::shared_ptr<Widget>(this); // WRONG!
        // Creates new control block, double delete
    }
};

// Correct way: inherit from enable_shared_from_this
class Widget2 : public std::enable_shared_from_this<Widget2> {
public:
    std::shared_ptr<Widget2> getShared() {
        return shared_from_this();
    }
};
```

### Step 4: Circular References
Already covered in Lab 15.4, but worth repeating: use `weak_ptr`!

## Challenges

### Challenge 1: Find the Bug
```cpp
void process(std::shared_ptr<int> p) { /* ... */ }

int main() {
    process(std::shared_ptr<int>(new int(42)));
    process(std::shared_ptr<int>(new int(42)));
}
```
What's wrong? (Hint: Exception safety)

### Challenge 2: Fixing enable_shared_from_this
Create a class that safely returns `shared_ptr` to itself.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>

// Challenge 2: Correct usage
class Task : public std::enable_shared_from_this<Task> {
    std::string name;
public:
    Task(std::string n) : name(n) {}
    
    void schedule() {
        // Can safely get shared_ptr to this
        auto self = shared_from_this();
        std::cout << "Scheduled: " << name << "\n";
        // self can be stored in a container, etc.
    }
};

int main() {
    // Must be created as shared_ptr
    auto task = std::make_shared<Task>("MyTask");
    task->schedule();
    
    // WRONG: Cannot call shared_from_this on stack object
    // Task t2("StackTask");
    // t2.schedule(); // Throws bad_weak_ptr
    
    return 0;
}
```

**Challenge 1 Answer:**
The code is fine, but could be more efficient:
```cpp
// Better:
process(std::make_shared<int>(42));
```
</details>

## Success Criteria
✅ Identified double delete risk
✅ Understood `get()` dangers
✅ Used `enable_shared_from_this` correctly
✅ Identified exception safety issue (Challenge 1)

## Key Learnings
- Never create multiple smart pointers from same raw pointer
- Don't store result of `get()` long-term
- Use `enable_shared_from_this` for `shared_from_this()`
- Beware of circular references

## Next Steps
Proceed to **Lab 15.10: Resource Manager** for the capstone project.
