# Lab 7.6: Static Members

## Objective
Understand how to share data and functions across all objects of a class.

## Instructions

### Step 1: Static Data Member
Create `static_members.cpp`.

```cpp
#include <iostream>

class Robot {
public:
    static int count; // Declaration
    int id;
    
    Robot() {
        id = ++count;
        std::cout << "Robot " << id << " created.\n";
    }
    
    ~Robot() {
        count--;
        std::cout << "Robot " << id << " destroyed.\n";
    }
};

// Definition (Must be outside class)
int Robot::count = 0;

int main() {
    std::cout << "Count: " << Robot::count << std::endl;
    
    Robot r1;
    Robot r2;
    
    std::cout << "Count: " << Robot::count << std::endl;
    
    {
        Robot r3;
    } // r3 destroyed
    
    std::cout << "Count: " << Robot::count << std::endl;
    
    return 0;
}
```

### Step 2: Static Member Function
Add a static function to access the static variable.

```cpp
static int getCount() {
    return count;
}
// Call: Robot::getCount();
```
*Note: Static functions cannot access non-static members (like `id`).*

## Challenges

### Challenge 1: Singleton Pattern (Preview)
Create a class `Config` that has a private constructor and a static method `getInstance()` that returns a reference to a static local instance.
This ensures only one instance exists.

### Challenge 2: Math Utility Class
Create a class `Math` with only static methods (`add`, `sub`, `mul`) and no data.
Prevent instantiation by making the constructor private.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Config {
private:
    Config() { std::cout << "Config initialized\n"; }
public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }
    void show() { std::cout << "Showing config\n"; }
};

class Math {
private:
    Math() = delete; // Prevent instantiation
public:
    static int add(int a, int b) { return a + b; }
};

int main() {
    Config& c1 = Config::getInstance();
    Config& c2 = Config::getInstance(); // Same instance
    c1.show();
    
    std::cout << "Sum: " << Math::add(5, 10) << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Declared and defined static member variable
✅ Implemented static member function
✅ Tracked object count across instances
✅ Implemented Singleton (Challenge 1)

## Key Learnings
- Static members belong to the class, not objects
- Static variables need definition outside the class
- Static functions have no `this` pointer

## Next Steps
Proceed to **Lab 7.7: Friend Functions** to break encapsulation (carefully).
