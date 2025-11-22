# Lab 13.1: Auto Type Deduction

## Objective
Learn when and how to use `auto` for type deduction.

## Instructions

### Step 1: Basic Auto
Create `auto_demo.cpp`.

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    auto x = 5; // int
    auto y = 3.14; // double
    auto s = "Hello"; // const char*
    auto str = std::string("World"); // std::string
    
    std::cout << x << " " << y << " " << s << " " << str << std::endl;
    return 0;
}
```

### Step 2: Auto with Containers
Avoid verbose iterator types.

```cpp
std::vector<int> v = {1, 2, 3};
// Old way
std::vector<int>::iterator it = v.begin();

// Modern way
auto it2 = v.begin();
```

### Step 3: Auto with Functions
```cpp
auto add(int a, int b) -> int { // Trailing return type (C++11)
    return a + b;
}

// C++14: Return type deduction
auto multiply(int a, int b) {
    return a * b;
}
```

## Challenges

### Challenge 1: AAA (Almost Always Auto)
Rewrite this code using `auto` everywhere possible:
```cpp
std::map<std::string, int> ages;
ages["Alice"] = 30;
for (std::pair<const std::string, int>& p : ages) {
    std::cout << p.first << ": " << p.second << "\n";
}
```

### Challenge 2: When NOT to Use Auto
Identify cases where `auto` hurts readability:
- `auto x = 0;` (Is it int? long? unsigned?)
- `auto flag = getFlag();` (What type is returned?)

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    // Challenge 1: AAA
    auto ages = std::map<std::string, int>{};
    ages["Alice"] = 30;
    ages["Bob"] = 25;
    
    for (auto& [name, age] : ages) { // Structured binding + auto
        std::cout << name << ": " << age << "\n";
    }
    
    // Challenge 2: Be explicit when clarity matters
    int count = 0; // Better than auto count = 0
    bool isValid = true; // Better than auto isValid = true
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `auto` for basic types
✅ Used `auto` for iterators
✅ Used `auto` for function return types
✅ Identified when NOT to use `auto` (Challenge 2)

## Key Learnings
- `auto` reduces boilerplate
- Essential for complex template types
- Don't sacrifice clarity for brevity

## Next Steps
Proceed to **Lab 13.2: Nullptr** for type-safe null pointers.
