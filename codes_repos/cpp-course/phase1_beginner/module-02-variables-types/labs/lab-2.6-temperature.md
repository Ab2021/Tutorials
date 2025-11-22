# Lab 2.6: Temperature Converter with Strong Types

## Objective
Build a temperature converter that uses `enum class` to prevent mixing up units.

## Instructions

### Step 1: Define Scales
Create `converter.cpp`:

```cpp
#include <iostream>

enum class Scale { Celsius, Fahrenheit, Kelvin };

// Helper to print scale name
const char* scaleName(Scale s) {
    switch (s) {
        case Scale::Celsius:    return "Celsius";
        case Scale::Fahrenheit: return "Fahrenheit";
        case Scale::Kelvin:     return "Kelvin";
        default:                return "Unknown";
    }
}
```

### Step 2: Conversion Function
Write a function that takes a value, a source scale, and a target scale.

```cpp
double convert(double value, Scale from, Scale to) {
    if (from == to) return value;
    
    // Convert everything to Celsius first
    double celsius = value;
    switch (from) {
        case Scale::Fahrenheit: celsius = (value - 32.0) * 5.0 / 9.0; break;
        case Scale::Kelvin:     celsius = value - 273.15; break;
        case Scale::Celsius:    break;
    }
    
    // Convert Celsius to target
    switch (to) {
        case Scale::Fahrenheit: return (celsius * 9.0 / 5.0) + 32.0;
        case Scale::Kelvin:     return celsius + 273.15;
        case Scale::Celsius:    return celsius;
    }
    return celsius; // Should not reach here
}
```

### Step 3: Main Loop
Create a simple UI in `main` to ask for input.

```cpp
int main() {
    std::cout << "Enter temp in Celsius: ";
    double c;
    std::cin >> c;
    
    double f = convert(c, Scale::Celsius, Scale::Fahrenheit);
    double k = convert(c, Scale::Celsius, Scale::Kelvin);
    
    std::cout << c << " C = " << f << " F" << std::endl;
    std::cout << c << " C = " << k << " K" << std::endl;
    
    return 0;
}
```

## Challenges

### Challenge 1: Strong Type Wrapper
Instead of passing `double` and `Scale` separately, create a struct:
```cpp
struct Temperature {
    double value;
    Scale scale;
};
```
Update the convert function to take a `Temperature` object.

### Challenge 2: Operator Overloading (Preview)
Try to implement `==` for `Temperature` structs. Two temperatures are equal if they represent the same physical heat (e.g., 0 C == 32 F).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cmath> // For std::abs

enum class Scale { Celsius, Fahrenheit, Kelvin };

struct Temperature {
    double value;
    Scale scale;
};

double toCelsius(Temperature t) {
    switch (t.scale) {
        case Scale::Fahrenheit: return (t.value - 32.0) * 5.0 / 9.0;
        case Scale::Kelvin:     return t.value - 273.15;
        case Scale::Celsius:    return t.value;
    }
    return 0.0;
}

bool operator==(const Temperature& a, const Temperature& b) {
    return std::abs(toCelsius(a) - toCelsius(b)) < 0.001;
}

int main() {
    Temperature t1{0.0, Scale::Celsius};
    Temperature t2{32.0, Scale::Fahrenheit};
    
    if (t1 == t2) {
        std::cout << "Freezing point matches!" << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `enum class` for scales
✅ Implemented conversion logic
✅ Created a struct to group value and scale
✅ Implemented equality check (Challenge 2)

## Key Learnings
- Grouping related data in structs
- Using enums for state/options
- Separating logic (conversion) from I/O

## Next Steps
Proceed to **Lab 2.7: Custom Literals** to write cleaner code like `10.0_kg`.
