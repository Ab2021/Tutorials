# Lab 14.2: Capture Mechanisms

## Objective
Master the different ways to capture variables from the enclosing scope.

## Instructions

### Step 1: Capture by Value
Create `captures.cpp`.

```cpp
#include <iostream>

int main() {
    int x = 10;
    
    auto lambda = [x]() { 
        std::cout << "Captured x: " << x << "\n";
        // x = 20; // Error: captured by value is const
    };
    
    lambda();
    x = 100; // Doesn't affect lambda's copy
    lambda(); // Still prints 10
    
    return 0;
}
```

### Step 2: Capture by Reference
```cpp
int y = 5;
auto ref_lambda = [&y]() {
    y += 10; // Modifies the original
    std::cout << "y: " << y << "\n";
};
ref_lambda(); // y is now 15
```

### Step 3: Capture All
```cpp
int a = 1, b = 2;
auto all_val = [=]() { return a + b; }; // Capture all by value
auto all_ref = [&]() { a++; b++; }; // Capture all by reference
```

### Step 4: Mixed Capture
```cpp
int m = 1, n = 2;
auto mixed = [=, &n]() { 
    // m is by value, n is by reference
    n = m + 10;
};
```

## Challenges

### Challenge 1: Dangling Reference
Create a lambda that captures by reference, then return it from a function. Call it outside and observe undefined behavior.

### Challenge 2: This Capture
In a class method, capture `this` to access members.
```cpp
class Counter {
    int count = 0;
public:
    auto getIncrementer() {
        return [this]() { return ++count; };
    }
};
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <functional>

// Challenge 1: Dangling reference (DON'T DO THIS)
std::function<void()> makeDangerous() {
    int local = 42;
    return [&local]() { std::cout << local << "\n"; }; // DANGER!
}

// Challenge 2: This capture
class Counter {
    int count = 0;
public:
    auto getIncrementer() {
        return [this]() { return ++count; };
    }
};

int main() {
    Counter c;
    auto inc = c.getIncrementer();
    std::cout << inc() << "\n"; // 1
    std::cout << inc() << "\n"; // 2
    
    return 0;
}
```
</details>

## Success Criteria
✅ Captured by value `[x]`
✅ Captured by reference `[&x]`
✅ Used `[=]` and `[&]`
✅ Used mixed capture
✅ Understood dangling reference risk (Challenge 1)

## Key Learnings
- `[=]` copies all used variables
- `[&]` references all used variables
- Captured-by-value variables are const by default
- Beware of lifetime issues with `[&]`

## Next Steps
Proceed to **Lab 14.3: Mutable Lambdas** to modify captured values.
