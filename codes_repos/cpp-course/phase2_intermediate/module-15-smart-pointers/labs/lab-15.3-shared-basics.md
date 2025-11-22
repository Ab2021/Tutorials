# Lab 15.3: Shared Pointer Basics

## Objective
Understand shared ownership with `std::shared_ptr`.

## Instructions

### Step 1: Basic Usage
Create `shared_ptr_basics.cpp`.

```cpp
#include <iostream>
#include <memory>

int main() {
    std::shared_ptr<int> p1 = std::make_shared<int>(42);
    std::cout << "Value: " << *p1 << "\n";
    std::cout << "Use count: " << p1.use_count() << "\n";
    
    {
        std::shared_ptr<int> p2 = p1; // Share ownership
        std::cout << "Use count: " << p1.use_count() << "\n"; // 2
    } // p2 destroyed
    
    std::cout << "Use count: " << p1.use_count() << "\n"; // 1
    
    return 0;
} // p1 destroyed, object deleted
```

### Step 2: Copying is Allowed
```cpp
std::shared_ptr<int> p3 = p1; // Copy
std::shared_ptr<int> p4(p1); // Copy
```

### Step 3: Reset
```cpp
p1.reset(); // Decrements count, may delete object
p1.reset(new int(100)); // Assign new object
```

## Challenges

### Challenge 1: Shared Resource
Create a class `Logger` that is shared among multiple objects.
```cpp
class Logger {
public:
    void log(std::string msg) { std::cout << msg << "\n"; }
};

class Service {
    std::shared_ptr<Logger> logger;
public:
    Service(std::shared_ptr<Logger> l) : logger(l) {}
    void doWork() { logger->log("Working..."); }
};
```

### Challenge 2: Aliasing Constructor
`shared_ptr` can share ownership but point to different objects (aliasing).
```cpp
struct Data { int x, y; };
auto p = std::make_shared<Data>();
std::shared_ptr<int> px(p, &p->x); // Shares ownership of Data, points to x
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <string>

class Logger {
public:
    ~Logger() { std::cout << "Logger destroyed\n"; }
    void log(const std::string& msg) { std::cout << "[LOG] " << msg << "\n"; }
};

class Service {
    std::shared_ptr<Logger> logger;
    std::string name;
public:
    Service(std::shared_ptr<Logger> l, std::string n) 
        : logger(l), name(n) {}
    
    void doWork() {
        logger->log(name + " is working");
    }
};

int main() {
    auto logger = std::make_shared<Logger>();
    
    Service s1(logger, "Service1");
    Service s2(logger, "Service2");
    
    s1.doWork();
    s2.doWork();
    
    std::cout << "Logger use count: " << logger.use_count() << "\n"; // 3
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created and shared `shared_ptr`
✅ Observed reference counting
✅ Shared resource among objects (Challenge 1)

## Key Learnings
- `shared_ptr` uses reference counting
- Thread-safe reference counting (atomic operations)
- Object deleted when last `shared_ptr` is destroyed

## Next Steps
Proceed to **Lab 15.4: Weak Pointer** to break cycles.
