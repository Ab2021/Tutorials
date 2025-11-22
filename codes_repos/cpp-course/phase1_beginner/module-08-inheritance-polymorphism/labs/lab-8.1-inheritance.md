# Lab 8.1: Basic Inheritance

## Objective
Understand how to create a derived class that inherits functionality from a base class.

## Instructions

### Step 1: Base Class
Create `inheritance.cpp`.

```cpp
#include <iostream>
#include <string>

class Vehicle {
public:
    std::string brand = "Ford";
    void honk() {
        std::cout << "Tuut, tuut!" << std::endl;
    }
};
```

### Step 2: Derived Class
Create a class `Car` that inherits from `Vehicle`.

```cpp
class Car : public Vehicle {
public:
    std::string model = "Mustang";
};
```

### Step 3: Usage
Create a `Car` object. Access members from both classes.

```cpp
int main() {
    Car myCar;
    myCar.honk(); // From Vehicle
    std::cout << myCar.brand << " " << myCar.model << std::endl;
    return 0;
}
```

## Challenges

### Challenge 1: Another Derived Class
Create a `Motorcycle` class inheriting from `Vehicle`.
Add a specific method `wheelie()`.

### Challenge 2: Inheritance Chain
Create `SportsCar` inheriting from `Car`.
Verify it has access to `brand` (from Vehicle) and `model` (from Car).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

class Vehicle {
public:
    std::string brand = "Generic";
    void honk() { std::cout << "Honk!\n"; }
};

class Car : public Vehicle {
public:
    std::string model = "Sedan";
};

class Motorcycle : public Vehicle {
public:
    void wheelie() { std::cout << "Doing a wheelie!\n"; }
};

class SportsCar : public Car {
public:
    void turbo() { std::cout << "Turbo boost!\n"; }
};

int main() {
    Motorcycle m;
    m.honk();
    m.wheelie();
    
    SportsCar s;
    s.honk(); // Vehicle
    std::cout << s.brand << " " << s.model << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created Base and Derived classes
✅ Accessed base members from derived object
✅ Created multiple derived classes (Challenge 1)
✅ Created multi-level inheritance (Challenge 2)

## Key Learnings
- `class Derived : public Base` syntax
- Derived classes inherit all public members
- "Is-A" relationship (Car Is-A Vehicle)

## Next Steps
Proceed to **Lab 8.2: Access Specifiers** to control inheritance visibility.
