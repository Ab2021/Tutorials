# Lab 16.5: std::move Deep Dive

## Objective
Understand what `std::move` actually does and when to use it.

## Instructions

### Step 1: What std::move Does
`std::move` doesn't move anything! It's just a cast to rvalue reference.

Create `std_move.cpp`.

```cpp
#include <iostream>
#include <utility>

void process(int& x) {
    std::cout << "Lvalue reference\n";
}

void process(int&& x) {
    std::cout << "Rvalue reference\n";
}

int main() {
    int x = 42;
    process(x);              // Lvalue reference
    process(std::move(x));   // Rvalue reference (cast)
    
    // x is still valid! std::move didn't move anything
    std::cout << "x = " << x << "\n"; // Still 42
    
    return 0;
}
```

### Step 2: When to Use std::move
```cpp
#include <vector>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = std::move(s1); // Transfer ownership
    // s1 is now in valid but unspecified state
    
    std::vector<std::string> vec;
    vec.push_back(s2);              // Copy
    vec.push_back(std::move(s2));   // Move
    
    return 0;
}
```

### Step 3: Common Mistakes
```cpp
// DON'T: Move from const
const std::string s = "Hello";
auto s2 = std::move(s); // Copies! Can't move from const

// DON'T: Use after move (undefined behavior)
std::string s1 = "Hello";
auto s2 = std::move(s1);
std::cout << s1; // Dangerous! s1 is unspecified

// DO: Reassign after move
s1 = "New value"; // OK, s1 is usable again
```

## Challenges

### Challenge 1: Move in Return
Understand when to use `std::move` in return statements.

```cpp
std::string makeString() {
    std::string s = "Hello";
    return s;           // Good: copy elision
    // return std::move(s); // Bad: prevents copy elision!
}
```

### Challenge 2: Conditional Move
Implement a function that conditionally moves or copies.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <vector>

// Challenge 2: Conditional move
template<typename T>
void addToVector(std::vector<T>& vec, T&& item, bool shouldMove) {
    if (shouldMove) {
        vec.push_back(std::move(item)); // Move
    } else {
        vec.push_back(item); // Copy
    }
}

class Widget {
    std::string name;
public:
    Widget(std::string n) : name(std::move(n)) {
        std::cout << "Constructed: " << name << "\n";
    }
    
    Widget(const Widget& other) : name(other.name) {
        std::cout << "Copied: " << name << "\n";
    }
    
    Widget(Widget&& other) noexcept : name(std::move(other.name)) {
        std::cout << "Moved: " << name << "\n";
    }
};

int main() {
    // Demonstrate std::move is just a cast
    Widget w1("Original");
    Widget w2(std::move(w1)); // Move constructor called
    
    // w1 is still a valid object, just in unspecified state
    w1 = Widget("Reassigned"); // OK to reassign
    
    // Return value optimization
    auto makeWidget = []() {
        Widget w("Local");
        return w; // Don't use std::move here!
    };
    
    Widget w3 = makeWidget();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood `std::move` is just a cast
✅ Used `std::move` appropriately
✅ Avoided common mistakes
✅ Understood return value optimization (Challenge 1)

## Rust Comparison
```rust
// Rust moves by default
let s1 = String::from("Hello");
let s2 = s1; // s1 is moved, no longer accessible

// Explicit copy in Rust
let s3 = s2.clone();
```

## Key Learnings
- `std::move` is a cast to rvalue reference
- Moved-from objects are in valid but unspecified state
- Don't use `std::move` in return statements (prevents RVO)
- Can't move from `const` objects

## Next Steps
Proceed to **Lab 16.6: Perfect Forwarding**.
