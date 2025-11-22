# Lab 7.9: Composition

## Objective
Learn how to build complex classes by combining smaller classes ("Has-a" relationship).

## Instructions

### Step 1: Component Classes
Create `composition.cpp`.

```cpp
#include <iostream>
#include <string>

class Engine {
    int hp;
public:
    Engine(int horsepower) : hp(horsepower) {
        std::cout << "Engine created (" << hp << " HP)\n";
    }
    void start() { std::cout << "Engine started\n"; }
};

class Wheel {
public:
    Wheel() { std::cout << "Wheel created\n"; }
    void roll() { std::cout << "Wheel rolling\n"; }
};
```

### Step 2: Composite Class
A Car *has an* Engine and 4 Wheels.

```cpp
class Car {
    Engine engine;
    Wheel wheels[4];
public:
    // Constructor Initialization List is CRITICAL here
    Car(int hp) : engine(hp) {
        std::cout << "Car created\n";
    }
    
    void drive() {
        engine.start();
        for(auto& w : wheels) w.roll();
        std::cout << "Car moving\n";
    }
};
```

### Step 3: Usage
```cpp
int main() {
    Car myCar(300);
    myCar.drive();
    return 0;
}
```

## Challenges

### Challenge 1: Initialization Order
Add print statements to destructors.
Observe the order of destruction. (Car -> Wheels -> Engine).

### Challenge 2: Pointer Composition
Change `Engine engine` to `Engine* engine`.
Allocate it in constructor (`new`) and delete in destructor.
This is "Aggregation" (or Composition with dynamic lifetime).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Engine {
public:
    Engine() { std::cout << "Engine created\n"; }
    ~Engine() { std::cout << "Engine destroyed\n"; }
};

class Car {
    Engine* engine; // Dynamic composition
public:
    Car() { 
        engine = new Engine(); 
        std::cout << "Car created\n";
    }
    ~Car() { 
        delete engine; 
        std::cout << "Car destroyed\n";
    }
};

int main() {
    Car c;
    return 0;
}
```
</details>

## Success Criteria
✅ Created classes as members of other classes
✅ Used initializer list for member constructors
✅ Understood construction/destruction order
✅ Implemented dynamic composition (Challenge 2)

## Key Learnings
- Composition > Inheritance (usually)
- "Has-a" relationship
- Initializer lists are required if members don't have default constructors

## Next Steps
Proceed to **Lab 7.10: Bank Account System** to apply everything.
