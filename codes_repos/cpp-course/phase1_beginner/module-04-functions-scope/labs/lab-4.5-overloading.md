# Lab 4.5: Function Overloading

## Objective
Learn how to define multiple functions with the same name but different parameters.

## Instructions

### Step 1: Basic Overloading
Create `overload.cpp`. Define `print` for int and string.

```cpp
#include <iostream>
#include <string>

void print(int i) {
    std::cout << "Integer: " << i << std::endl;
}

void print(std::string s) {
    std::cout << "String: " << s << std::endl;
}

int main() {
    print(42);
    print("Hello");
    return 0;
}
```

### Step 2: Number of Arguments
Overload based on argument count.

```cpp
void print(int a, int b) {
    std::cout << "Pair: " << a << ", " << b << std::endl;
}
```

### Step 3: Ambiguity
Be careful!

```cpp
void func(int x);
void func(double x);

// func(5); // Calls int version
// func(5.5); // Calls double version
// func('a'); // Calls int version (char promotes to int)
```

## Challenges

### Challenge 1: Vector Overload
Overload `print` to take a `std::vector<int>`.
*Hint: You need `#include <vector>`.*

### Challenge 2: Ambiguous Call
Create a situation where the compiler throws an "ambiguous call" error.
Example:
```cpp
void test(int x, double y = 0);
void test(int x);
// test(5); // Error!
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <string>

void print(int i) { std::cout << "Int: " << i << std::endl; }
void print(std::string s) { std::cout << "Str: " << s << std::endl; }
void print(const std::vector<int>& v) {
    std::cout << "Vector: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    print(10);
    print("Test");
    std::vector<int> vec = {1, 2, 3};
    print(vec);
    return 0;
}
```
</details>

## Success Criteria
✅ Created overloaded functions
✅ Overloaded based on type and count
✅ Resolved calls correctly
✅ Identified ambiguous scenarios (Challenge 2)

## Key Learnings
- Overloading makes APIs intuitive (e.g., `print` works for everything)
- Compiler selects the "best match"
- Implicit conversions can cause ambiguity

## Next Steps
Proceed to **Lab 4.6: Default Arguments** to simplify function calls.
