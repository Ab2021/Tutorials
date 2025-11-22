# Lab 9.1: Function Templates

## Objective
Learn how to write a single function that works with multiple data types.

## Instructions

### Step 1: The Problem (Duplication)
Create `func_templates.cpp`.
Write two `swap` functions: one for `int` and one for `double`.

```cpp
#include <iostream>

void swapInt(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

void swapDouble(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}
```

### Step 2: The Solution (Template)
Replace them with a single template function.

```cpp
template <typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}
```

### Step 3: Usage
Call it with different types.

```cpp
int main() {
    int x = 1, y = 2;
    mySwap(x, y); // Deduce T = int
    std::cout << x << " " << y << std::endl;
    
    double d1 = 1.5, d2 = 2.5;
    mySwap(d1, d2); // Deduce T = double
    std::cout << d1 << " " << d2 << std::endl;
    
    return 0;
}
```

## Challenges

### Challenge 1: Explicit Instantiation
Call `mySwap` by explicitly specifying the type: `mySwap<int>(x, y)`.
Try `mySwap<double>(x, y)` with integers. Does it compile? (It might warn about references).

### Challenge 2: Max Function
Write a template function `T myMax(T a, T b)` that returns the larger value.
Test it with `int`, `double`, and `std::string`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

template <typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
T myMax(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    int a = 10, b = 20;
    mySwap<int>(a, b);
    
    std::cout << "Max(10, 20): " << myMax(10, 20) << std::endl;
    std::cout << "Max(A, B): " << myMax<std::string>("Alice", "Bob") << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented function template
✅ Used template type deduction
✅ Used explicit template instantiation
✅ Implemented generic Max function (Challenge 2)

## Key Learnings
- `template <typename T>` introduces a generic type `T`
- Compiler generates code for each used type
- Reduces code duplication significantly

## Next Steps
Proceed to **Lab 9.2: Class Templates** to create generic containers.
