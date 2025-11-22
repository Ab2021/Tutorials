# Lab 2.5: Enum Class vs Traditional Enums

## Objective
Understand the benefits of scoped enums (`enum class`) over traditional enums.

## Instructions

### Step 1: Traditional Enum Problems
Create `enum_demo.cpp`:

```cpp
#include <iostream>

enum Color { RED, GREEN, BLUE };
enum TrafficLight { RED, YELLOW, GREEN }; // Error: RED and GREEN redefinition!

int main() {
    Color c = RED;
    int x = c; // Implicit conversion to int (0)
    
    if (c == 0) { // Compares to int
        std::cout << "Red is 0" << std::endl;
    }
    
    return 0;
}
```
*Note: This code might fail to compile due to redefinition.*

### Step 2: Enum Class (Scoped)
Fix the issues using `enum class`:

```cpp
#include <iostream>

enum class Color { Red, Green, Blue };
enum class TrafficLight { Red, Yellow, Green }; // OK: Scoped

int main() {
    Color c = Color::Red;
    // int x = c; // Error: No implicit conversion
    
    // Explicit cast needed if you want the int value
    int x = static_cast<int>(c);
    
    if (c == Color::Red) {
        std::cout << "It is Red" << std::endl;
    }
    
    return 0;
}
```

### Step 3: Switch with Enum Class
```cpp
switch (c) {
    case Color::Red:   std::cout << "Red\n"; break;
    case Color::Green: std::cout << "Green\n"; break;
    case Color::Blue:  std::cout << "Blue\n"; break;
}
```

## Challenges

### Challenge 1: Underlying Type
Specify the underlying type of an enum class to save space (e.g., `char`).
```cpp
enum class SmallEnum : char { A, B, C };
```
Check `sizeof(SmallEnum)`.

### Challenge 2: Enum Methods?
C++ enums don't have methods (unlike Java/Rust). Create a helper function `toString(Color c)` that returns the string name of the color.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

enum class Color : char { Red, Green, Blue };

std::string toString(Color c) {
    switch (c) {
        case Color::Red:   return "Red";
        case Color::Green: return "Green";
        case Color::Blue:  return "Blue";
        default:           return "Unknown";
    }
}

int main() {
    std::cout << "Size: " << sizeof(Color) << " bytes" << std::endl; // 1 byte
    
    Color c = Color::Green;
    std::cout << "Color: " << toString(c) << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Fixed name collision using `enum class`
✅ Understood lack of implicit conversion
✅ Specified underlying type
✅ Created helper function for string representation

## Key Learnings
- `enum class` is scoped (avoids name pollution)
- `enum class` is strongly typed (no implicit int conversion)
- Can specify underlying type (int, char, etc.)

## Next Steps
Proceed to **Lab 2.6: Temperature Converter** to apply these concepts.
