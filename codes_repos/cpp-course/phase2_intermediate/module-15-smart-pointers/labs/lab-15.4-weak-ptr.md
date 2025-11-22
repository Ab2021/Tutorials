# Lab 15.4: Weak Pointer and Circular References

## Objective
Use `std::weak_ptr` to break circular references.

## Instructions

### Step 1: The Problem
Create `weak_ptr.cpp`.
Circular references cause memory leaks.

```cpp
#include <iostream>
#include <memory>

struct Node {
    std::shared_ptr<Node> next;
    ~Node() { std::cout << "Node destroyed\n"; }
};

int main() {
    auto n1 = std::make_shared<Node>();
    auto n2 = std::make_shared<Node>();
    
    n1->next = n2;
    n2->next = n1; // Circular reference!
    
    return 0;
} // Memory leak! Nodes never destroyed
```

### Step 2: The Solution
Use `weak_ptr` to break the cycle.

```cpp
struct Node2 {
    std::shared_ptr<Node2> next;
    std::weak_ptr<Node2> prev; // Weak reference
    ~Node2() { std::cout << "Node2 destroyed\n"; }
};

int main() {
    auto n1 = std::make_shared<Node2>();
    auto n2 = std::make_shared<Node2>();
    
    n1->next = n2;
    n2->prev = n1; // Weak, doesn't increase count
    
    return 0;
} // Both destroyed correctly
```

### Step 3: Using weak_ptr
```cpp
std::weak_ptr<int> wp;
{
    auto sp = std::make_shared<int>(42);
    wp = sp;
    
    if (auto sp2 = wp.lock()) { // Try to get shared_ptr
        std::cout << *sp2 << "\n";
    }
} // sp destroyed

if (wp.expired()) {
    std::cout << "Object is gone\n";
}
```

## Challenges

### Challenge 1: Observer Pattern
Implement an observer pattern where observers hold weak references to the subject.

### Challenge 2: Cache
Implement a simple cache that holds weak references to objects, allowing them to be deleted when no longer used.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Subject;

class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(int value) = 0;
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers;
    int value = 0;
    
public:
    void attach(std::shared_ptr<Observer> obs) {
        observers.push_back(obs);
    }
    
    void setValue(int v) {
        value = v;
        notify();
    }
    
    void notify() {
        // Clean up expired observers
        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [](const std::weak_ptr<Observer>& wp) { return wp.expired(); }),
            observers.end()
        );
        
        for (auto& wp : observers) {
            if (auto sp = wp.lock()) {
                sp->update(value);
            }
        }
    }
};

class ConcreteObserver : public Observer {
    std::string name;
public:
    ConcreteObserver(std::string n) : name(n) {}
    ~ConcreteObserver() { std::cout << name << " destroyed\n"; }
    
    void update(int value) override {
        std::cout << name << " received: " << value << "\n";
    }
};

int main() {
    Subject subject;
    
    {
        auto obs1 = std::make_shared<ConcreteObserver>("Observer1");
        auto obs2 = std::make_shared<ConcreteObserver>("Observer2");
        
        subject.attach(obs1);
        subject.attach(obs2);
        
        subject.setValue(10);
    } // Observers destroyed
    
    subject.setValue(20); // No observers left
    
    return 0;
}
```
</details>

## Success Criteria
✅ Identified circular reference problem
✅ Used `weak_ptr` to break cycle
✅ Used `lock()` and `expired()`
✅ Implemented observer pattern (Challenge 1)

## Key Learnings
- Circular `shared_ptr` references cause leaks
- `weak_ptr` doesn't increase reference count
- Always check `lock()` before using

## Next Steps
Proceed to **Lab 15.5: Custom Deleters**.
