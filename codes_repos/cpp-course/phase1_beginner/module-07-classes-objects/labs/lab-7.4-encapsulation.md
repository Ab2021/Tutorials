# Lab 7.4: Encapsulation (Getters/Setters)

## Objective
Practice the pattern of hiding data and providing controlled access.

## Instructions

### Step 1: The Class
Create `encapsulation.cpp`.

```cpp
#include <iostream>

class Thermostat {
private:
    int temperature;
public:
    Thermostat(int startTemp) : temperature(startTemp) {}
    
    int getTemperature() { return temperature; }
    
    void setTemperature(int t) {
        if (t >= 0 && t <= 100) {
            temperature = t;
        } else {
            std::cout << "Invalid temperature!\n";
        }
    }
};
```

### Step 2: Usage
```cpp
int main() {
    Thermostat t(20);
    t.setTemperature(25);
    std::cout << "Current: " << t.getTemperature() << std::endl;
    
    t.setTemperature(200); // Should fail
    return 0;
}
```

### Step 3: Read-Only Property
Add a private member `int id` (set in constructor).
Provide `getId()` but NO `setId()`.
Now `id` is immutable from the outside.

## Challenges

### Challenge 1: Write-Only Property?
Create a member `secretCode`. Provide `setSecret()` but no getter.
Add a method `bool checkSecret(int guess)` to verify it.

### Challenge 2: Derived Property
Add `double getTemperatureFahrenheit()`.
It doesn't store F, it calculates it from C: `(C * 9/5) + 32`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Thermostat {
    int temperature;
    int secretCode = 1234;
public:
    Thermostat(int t) : temperature(t) {}
    
    int getTemperature() { return temperature; }
    
    void setTemperature(int t) {
        if (t >= 0 && t <= 100) temperature = t;
    }
    
    // Challenge 2
    double getFahrenheit() {
        return (temperature * 9.0 / 5.0) + 32;
    }
    
    // Challenge 1
    void setSecret(int code) { secretCode = code; }
    bool checkSecret(int guess) { return guess == secretCode; }
};

int main() {
    Thermostat t(25);
    std::cout << "Fahrenheit: " << t.getFahrenheit() << std::endl;
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented Getters and Setters
✅ Added validation logic
✅ Created read-only property
✅ Created derived property (Challenge 2)

## Key Learnings
- Getters/Setters decouple internal representation from external interface
- Validation logic belongs in setters
- Derived properties don't need storage

## Next Steps
Proceed to **Lab 7.5: Const Member Functions** to ensure safety.
