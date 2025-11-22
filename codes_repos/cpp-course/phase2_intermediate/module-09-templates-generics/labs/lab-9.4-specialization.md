# Lab 9.4: Template Specialization

## Objective
Provide custom implementations for specific types.

## Instructions

### Step 1: Generic Class
Create `specialization.cpp`.

```cpp
#include <iostream>

template <typename T>
class Printer {
public:
    void print(T val) {
        std::cout << "Generic: " << val << std::endl;
    }
};
```

### Step 2: Full Specialization
Specialize for `char`.

```cpp
template <>
class Printer<char> {
public:
    void print(char val) {
        std::cout << "Char Specialization: '" << val << "'" << std::endl;
    }
};
```

### Step 3: Usage
```cpp
int main() {
    Printer<int> p1;
    p1.print(10); // Generic
    
    Printer<char> p2;
    p2.print('A'); // Specialized
    
    return 0;
}
```

## Challenges

### Challenge 1: Function Specialization
Specialize a function `bool isEqual(T a, T b)` for `double`.
(Floating point equality should use an epsilon, not `==`).

### Challenge 2: Partial Specialization
Create `class Box<T>`.
Partially specialize for pointers `Box<T*>`.
Print "Pointer to..." instead of just the value.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cmath>

// Challenge 1
template <typename T>
bool isEqual(T a, T b) { return a == b; }

template <>
bool isEqual<double>(double a, double b) {
    return std::abs(a - b) < 0.0001;
}

// Challenge 2
template <typename T>
class Box {
public:
    void show() { std::cout << "Generic Box\n"; }
};

template <typename T>
class Box<T*> {
public:
    void show() { std::cout << "Pointer Box\n"; }
};

int main() {
    std::cout << isEqual(10, 10) << "\n";
    std::cout << isEqual(1.0000001, 1.0) << "\n";
    
    Box<int> b1; b1.show();
    Box<int*> b2; b2.show();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented full class specialization
✅ Implemented function specialization (Challenge 1)
✅ Implemented partial class specialization (Challenge 2)

## Key Learnings
- Specialization allows optimizing or fixing behavior for specific types
- `template <>` denotes full specialization
- Partial specialization works for classes but NOT functions

## Next Steps
Proceed to **Lab 9.5: Non-Type Template Parameters** to pass values.
