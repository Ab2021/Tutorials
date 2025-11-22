# Lab 20.7: Observer Pattern

## Objective
Implement the Observer pattern for event-driven programming.

## Instructions

### Step 1: Basic Observer
Create `observer_pattern.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

// Observer interface
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(float temperature) = 0;
};

// Subject
class WeatherStation {
    std::vector<Observer*> observers;
    float temperature;
    
public:
    void attach(Observer* obs) {
        observers.push_back(obs);
    }
    
    void detach(Observer* obs) {
        observers.erase(
            std::remove(observers.begin(), observers.end(), obs),
            observers.end()
        );
    }
    
    void setTemperature(float temp) {
        temperature = temp;
        notify();
    }
    
    void notify() {
        for (auto* obs : observers) {
            obs->update(temperature);
        }
    }
};

// Concrete observers
class PhoneDisplay : public Observer {
public:
    void update(float temperature) override {
        std::cout << "Phone: Temperature is " << temperature << "°C\n";
    }
};

class WindowDisplay : public Observer {
public:
    void update(float temperature) override {
        std::cout << "Window: Temperature is " << temperature << "°C\n";
    }
};
```

### Step 2: Modern Observer with Smart Pointers
```cpp
class ModernObserver {
public:
    virtual ~ModernObserver() = default;
    virtual void onEvent(const std::string& event) = 0;
};

class ModernSubject {
    std::vector<std::weak_ptr<ModernObserver>> observers;
    
public:
    void subscribe(std::shared_ptr<ModernObserver> obs) {
        observers.push_back(obs);
    }
    
    void notify(const std::string& event) {
        // Remove expired observers
        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [](const auto& wp) { return wp.expired(); }),
            observers.end()
        );
        
        // Notify active observers
        for (auto& wp : observers) {
            if (auto sp = wp.lock()) {
                sp->onEvent(event);
            }
        }
    }
};
```

## Challenges

### Challenge 1: Push vs Pull
Implement both push and pull observer models.

### Challenge 2: Thread-Safe Observer
Make the observer pattern thread-safe.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>

// Pull model
class PullSubject {
    std::vector<class PullObserver*> observers;
    int state;
    
public:
    void attach(PullObserver* obs) { observers.push_back(obs); }
    void setState(int s) { state = s; notify(); }
    int getState() const { return state; }
    void notify();
};

class PullObserver {
public:
    virtual ~PullObserver() = default;
    virtual void update(PullSubject* subject) = 0;
};

void PullSubject::notify() {
    for (auto* obs : observers) {
        obs->update(this);
    }
}

class ConcretePullObserver : public PullObserver {
public:
    void update(PullSubject* subject) override {
        std::cout << "Pull: State is " << subject->getState() << "\n";
    }
};

// Challenge 2: Thread-safe observer
class ThreadSafeSubject {
    mutable std::mutex mutex;
    std::vector<std::weak_ptr<ModernObserver>> observers;
    
public:
    void subscribe(std::shared_ptr<ModernObserver> obs) {
        std::lock_guard<std::mutex> lock(mutex);
        observers.push_back(obs);
    }
    
    void notify(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex);
        
        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [](const auto& wp) { return wp.expired(); }),
            observers.end()
        );
        
        for (auto& wp : observers) {
            if (auto sp = wp.lock()) {
                sp->onEvent(event);
            }
        }
    }
};

int main() {
    // Pull model
    PullSubject subject;
    ConcretePullObserver observer;
    subject.attach(&observer);
    subject.setState(42);
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented basic observer
✅ Created modern observer with smart pointers
✅ Implemented push/pull models (Challenge 1)
✅ Made thread-safe observer (Challenge 2)

## Key Learnings
- Observer enables one-to-many dependencies
- Push model sends data, pull model queries
- Weak pointers prevent memory leaks
- Thread safety requires synchronization

## Next Steps
Proceed to **Lab 20.10: Pattern Combination (Capstone)**.
