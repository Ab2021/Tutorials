# Lab 2.4: Type Conversion Safety

## Objective
Learn safe ways to convert between types and avoid data loss.

## Instructions

### Step 1: Implicit Conversion (The Bad)
Create `cast_demo.cpp`:

```cpp
#include <iostream>

int main() {
    double pi = 3.14159;
    int x = pi; // Implicit conversion (warning?)
    
    std::cout << "Pi: " << pi << std::endl;
    std::cout << "X: " << x << std::endl; // Data loss!
    
    return 0;
}
```

### Step 2: Static Cast (The Good)
Use `static_cast` to make your intent clear:

```cpp
int y = static_cast<int>(pi);
```

### Step 3: Narrowing Conversion
Try brace initialization to prevent narrowing:

```cpp
// int z{pi}; // Error: Narrowing conversion not allowed
int z{static_cast<int>(pi)}; // OK with explicit cast
```

## Challenges

### Challenge 1: Char to Int
Convert a `char` to an `int` to see its ASCII value.
```cpp
char c = 'A';
// Print integer value
```

### Challenge 2: Signed to Unsigned
Be careful converting negative numbers to unsigned types!
```cpp
int neg = -1;
unsigned int u = static_cast<unsigned int>(neg);
std::cout << u << std::endl; // What is this huge number?
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    char c = 'A';
    int ascii = static_cast<int>(c);
    std::cout << "ASCII of A: " << ascii << std::endl; // 65
    
    int neg = -1;
    unsigned int u = static_cast<unsigned int>(neg);
    std::cout << "-1 as unsigned: " << u << std::endl; // 4294967295 (max uint)
    
    return 0;
}
```
</details>

## Success Criteria
✅ Recognized implicit conversion risks
✅ Used `static_cast` for explicit conversion
✅ Used brace initialization to prevent narrowing
✅ Observed signed/unsigned conversion behavior

## Key Learnings
- Avoid C-style casts `(int)x`
- Use `static_cast<type>(x)`
- Brace initialization `{}` prevents accidental data loss
- Converting negative integers to unsigned causes wrap-around

## Next Steps
Proceed to **Lab 2.5: Enum Class** to work with strongly typed enumerations.
