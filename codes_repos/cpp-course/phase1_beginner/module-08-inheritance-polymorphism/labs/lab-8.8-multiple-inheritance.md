# Lab 8.8: Multiple Inheritance and the Diamond Problem

## Objective
Inherit from multiple classes and solve the ambiguity of the Diamond Problem.

## Instructions

### Step 1: Multiple Inheritance
Create `multiple.cpp`.

```cpp
#include <iostream>

class Camera {
public:
    void takePhoto() { std::cout << "Snap!\n"; }
};

class Phone {
public:
    void call() { std::cout << "Ring ring!\n"; }
};

class SmartPhone : public Camera, public Phone {
};

int main() {
    SmartPhone sp;
    sp.takePhoto();
    sp.call();
    return 0;
}
```

### Step 2: The Diamond Problem
Create a common ancestor.

```cpp
class Device {
public:
    int id;
    Device() { std::cout << "Device created\n"; }
};

class Camera : public Device {};
class Phone : public Device {};
class SmartPhone : public Camera, public Phone {};
```

### Step 3: Ambiguity
`SmartPhone` now has TWO `Device` parts (one via Camera, one via Phone).
`sp.id = 10;` // Error: Ambiguous. Which `id`?

### Step 4: Virtual Inheritance (The Fix)
Inherit `virtually` from Device.

```cpp
class Camera : virtual public Device {};
class Phone : virtual public Device {};
```
Now `SmartPhone` shares a single `Device` instance.

## Challenges

### Challenge 1: Constructor Call
In virtual inheritance, the most derived class (`SmartPhone`) is responsible for initializing the virtual base (`Device`).
Add a constructor to `Device(int id)` and try to initialize it from `SmartPhone`.

### Challenge 2: Interface Segregation
Use multiple inheritance to implement multiple interfaces (Abstract Classes).
`class Worker : public IEngineer, public IManager`.
This is the safest use of multiple inheritance.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Device {
public:
    Device(int i) { std::cout << "Device " << i << "\n"; }
};

class Camera : virtual public Device {
public:
    Camera() : Device(0) {} // Ignored when part of SmartPhone
};

class Phone : virtual public Device {
public:
    Phone() : Device(0) {} // Ignored when part of SmartPhone
};

class SmartPhone : public Camera, public Phone {
public:
    // Must initialize virtual base directly
    SmartPhone() : Device(42), Camera(), Phone() {}
};

int main() {
    SmartPhone sp;
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented multiple inheritance
✅ Reproduced Diamond Problem ambiguity
✅ Solved using `virtual` inheritance
✅ Initialized virtual base class (Challenge 1)

## Key Learnings
- Multiple inheritance is powerful but complex
- Diamond problem creates duplicate base instances
- Virtual inheritance ensures a single shared base instance
- Prefer Composition or Interface Inheritance over complex MI

## Next Steps
Proceed to **Lab 8.9: Object Slicing** to avoid data loss.
