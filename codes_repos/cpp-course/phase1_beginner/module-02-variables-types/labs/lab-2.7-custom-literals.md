# Lab 2.7: Custom Literals

## Objective
Learn how to define user-defined literals to create more readable code for units.

## Instructions

### Step 1: Define Literals
Create `literals.cpp`. We want to support `10.0_kg` and `10.0_lb`.

```cpp
#include <iostream>

// Literals must start with underscore (standard ones don't)
long double operator"" _kg(long double x) {
    return x; // Base unit
}

long double operator"" _lb(long double x) {
    return x * 0.453592; // Convert to kg
}

// For integers
long double operator"" _kg(unsigned long long x) {
    return static_cast<long double>(x);
}
```

### Step 2: Use Literals
```cpp
int main() {
    long double weight1 = 10.0_kg;
    long double weight2 = 22.0_lb; // Approx 10 kg
    
    std::cout << "Weight 1: " << weight1 << " kg" << std::endl;
    std::cout << "Weight 2: " << weight2 << " kg" << std::endl;
    
    if (weight1 > weight2) {
        std::cout << "Weight 1 is heavier" << std::endl;
    } else {
        std::cout << "Weight 2 is heavier or equal" << std::endl;
    }
    
    return 0;
}
```

### Step 3: Distance Literals
Implement literals for `_km`, `_m`, `_cm`. Use meters as the base unit.

## Challenges

### Challenge 1: Time Literals
Implement `_h`, `_min`, `_s` for time, converting everything to seconds.
Calculate how many seconds are in `1.5_h + 30.0_min`.

### Challenge 2: String Literals
Implement a literal that takes a string and reverses it (just print it for now).
```cpp
void operator"" _print_rev(const char* str, size_t len) {
    for (int i = len - 1; i >= 0; --i) std::cout << str[i];
    std::cout << std::endl;
}

"Hello"_print_rev; // Prints olleH
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

// Distance (base: meters)
long double operator"" _km(long double x) { return x * 1000.0; }
long double operator"" _m(long double x)  { return x; }
long double operator"" _cm(long double x) { return x / 100.0; }

// Time (base: seconds)
long double operator"" _h(long double x)   { return x * 3600.0; }
long double operator"" _min(long double x) { return x * 60.0; }
long double operator"" _s(long double x)   { return x; }

int main() {
    auto dist = 1.5_km + 500.0_m;
    std::cout << "Distance: " << dist << " m" << std::endl;
    
    auto time = 1.5_h + 30.0_min;
    std::cout << "Time: " << time << " s" << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Defined custom literals for mass
✅ Defined custom literals for distance
✅ Used literals in expressions
✅ Implemented string literal (Challenge 2)

## Key Learnings
- `operator"" _suffix` syntax
- Using base units for consistent calculations
- Improving code readability

## Next Steps
Proceed to **Lab 2.8: Type Size and Alignment** to understand memory layout.
