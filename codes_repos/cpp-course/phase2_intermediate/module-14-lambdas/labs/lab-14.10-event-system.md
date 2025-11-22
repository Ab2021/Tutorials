# Lab 14.10: Event System (Capstone)

## Objective
Build a simple event/observer system using lambdas and `std::function`.

## Instructions

### Step 1: Event Class
Create `event_system.cpp`.
Store a list of callbacks.

```cpp
#include <iostream>
#include <vector>
#include <functional>
#include <string>

class Event {
    std::vector<std::function<void()>> listeners;
public:
    void subscribe(std::function<void()> callback) {
        listeners.push_back(std::move(callback));
    }
    
    void trigger() {
        for (auto& listener : listeners) {
            listener();
        }
    }
};
```

### Step 2: Typed Events
Support events with parameters.

```cpp
template <typename... Args>
class TypedEvent {
    std::vector<std::function<void(Args...)>> listeners;
public:
    void subscribe(std::function<void(Args...)> callback) {
        listeners.push_back(std::move(callback));
    }
    
    void trigger(Args... args) {
        for (auto& listener : listeners) {
            listener(args...);
        }
    }
};
```

### Step 3: Usage
```cpp
int main() {
    TypedEvent<std::string, int> onMessage;
    
    onMessage.subscribe([](std::string msg, int code) {
        std::cout << "Listener 1: " << msg << " (" << code << ")\n";
    });
    
    onMessage.subscribe([](std::string msg, int code) {
        std::cout << "Listener 2: Got " << msg << "\n";
    });
    
    onMessage.trigger("Hello", 200);
    
    return 0;
}
```

## Challenges

### Challenge 1: Unsubscribe
Add an `unsubscribe` method that removes a specific callback.
Return a subscription ID when subscribing.

### Challenge 2: Once
Add a `subscribeOnce` method that automatically unsubscribes after first trigger.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <memory>

template <typename... Args>
class Event {
    struct Listener {
        int id;
        std::function<void(Args...)> callback;
        bool once;
    };
    
    std::vector<Listener> listeners;
    int nextId = 0;
    
public:
    int subscribe(std::function<void(Args...)> callback, bool once = false) {
        int id = nextId++;
        listeners.push_back({id, std::move(callback), once});
        return id;
    }
    
    int subscribeOnce(std::function<void(Args...)> callback) {
        return subscribe(std::move(callback), true);
    }
    
    void unsubscribe(int id) {
        listeners.erase(
            std::remove_if(listeners.begin(), listeners.end(),
                [id](const Listener& l) { return l.id == id; }),
            listeners.end()
        );
    }
    
    void trigger(Args... args) {
        auto it = listeners.begin();
        while (it != listeners.end()) {
            it->callback(args...);
            if (it->once) {
                it = listeners.erase(it);
            } else {
                ++it;
            }
        }
    }
};

int main() {
    Event<std::string> onEvent;
    
    int id1 = onEvent.subscribe([](std::string msg) {
        std::cout << "Persistent: " << msg << "\n";
    });
    
    onEvent.subscribeOnce([](std::string msg) {
        std::cout << "Once: " << msg << "\n";
    });
    
    std::cout << "First trigger:\n";
    onEvent.trigger("Hello");
    
    std::cout << "\nSecond trigger:\n";
    onEvent.trigger("World");
    
    onEvent.unsubscribe(id1);
    
    std::cout << "\nThird trigger (after unsubscribe):\n";
    onEvent.trigger("Goodbye");
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented event subscription system
✅ Created typed events with parameters
✅ Implemented unsubscribe (Challenge 1)
✅ Implemented subscribeOnce (Challenge 2)

## Key Learnings
- Lambdas are perfect for callbacks
- `std::function` allows storing heterogeneous callables
- Event systems are common in GUI and game programming

## Next Steps
Congratulations! You've completed Module 14.

Proceed to **Module 15: Smart Pointers (Deep Dive)** to master memory management.
