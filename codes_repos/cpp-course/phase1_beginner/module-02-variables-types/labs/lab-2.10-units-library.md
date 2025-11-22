# Lab 2.10: Building a Type-Safe Units Library

## Objective
Combine structs, operator overloading, and strong types to build a simple physics library that prevents unit errors (like adding meters to seconds).

## Instructions

### Step 1: Define Units
Create `units.cpp`. Define structs for `Meters` and `Seconds`.

```cpp
#include <iostream>

struct Meters {
    double value;
};

struct Seconds {
    double value;
};

struct Speed {
    double value;
};
```

### Step 2: Operator Overloading
Allow adding Meters to Meters, but NOT Meters to Seconds.

```cpp
Meters operator+(Meters a, Meters b) {
    return Meters{a.value + b.value};
}

Seconds operator+(Seconds a, Seconds b) {
    return Seconds{a.value + b.value};
}

// Note: No operator+(Meters, Seconds) defined!
```

### Step 3: Division
Allow dividing Meters by Seconds to get Speed.

```cpp
Speed operator/(Meters m, Seconds s) {
    return Speed{m.value / s.value};
}
```

### Step 4: Main Logic
```cpp
int main() {
    Meters dist1{100.0};
    Meters dist2{50.0};
    
    Seconds time{10.0};
    
    Meters totalDist = dist1 + dist2;
    // Meters err = dist1 + time; // Compile error!
    
    Speed s = totalDist / time;
    
    std::cout << "Speed: " << s.value << " m/s" << std::endl;
    
    return 0;
}
```

## Challenges

### Challenge 1: Output Stream
Overload `operator<<` for all types so you can do `std::cout << dist1`.
```cpp
std::ostream& operator<<(std::ostream& os, Meters m) {
    os << m.value << " m";
    return os;
}
```

### Challenge 2: Literals Integration
Combine with Lab 2.7 to allow:
`Meters m = 100.0_m;`
`Seconds s = 10.0_s;`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

struct Meters { double value; };
struct Seconds { double value; };
struct Speed { double value; };

Meters operator+(Meters a, Meters b) { return {a.value + b.value}; }
Seconds operator+(Seconds a, Seconds b) { return {a.value + b.value}; }
Speed operator/(Meters m, Seconds s) { return {m.value / s.value}; }

std::ostream& operator<<(std::ostream& os, Meters m) { return os << m.value << " m"; }
std::ostream& operator<<(std::ostream& os, Seconds s) { return os << s.value << " s"; }
std::ostream& operator<<(std::ostream& os, Speed s) { return os << s.value << " m/s"; }

Meters operator"" _m(long double x) { return {static_cast<double>(x)}; }
Seconds operator"" _s(long double x) { return {static_cast<double>(x)}; }

int main() {
    auto d = 100.0_m;
    auto t = 9.58_s; // Usain Bolt
    
    auto speed = d / t;
    
    std::cout << "Distance: " << d << std::endl;
    std::cout << "Time: " << t << std::endl;
    std::cout << "Speed: " << speed << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created distinct types for units
✅ Implemented type-safe addition
✅ Implemented division resulting in new type
✅ Prevented invalid operations at compile time

## Key Learnings
- Using the type system to enforce physical laws
- Operator overloading for custom types
- Compile-time safety vs runtime checks

## Next Steps
Congratulations! You've completed Module 2. You now have a deep understanding of C++ types.

Proceed to **Module 3: Control Flow** to add logic to your programs.
