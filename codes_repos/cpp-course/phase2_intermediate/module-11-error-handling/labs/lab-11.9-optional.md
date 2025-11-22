# Lab 11.9: std::optional (C++17)

## Objective
Use `std::optional` to represent values that might be missing, avoiding exceptions for common cases.

## Instructions

### Step 1: The Problem
Create `optional.cpp`.
A function `findUser(id)` usually returns a pointer or throws if not found.
`std::optional` is safer.

```cpp
#include <iostream>
#include <optional>
#include <string>

std::optional<std::string> findUser(int id) {
    if (id == 1) return "Alice";
    return std::nullopt; // Empty
}
```

### Step 2: Checking Value
```cpp
int main() {
    auto user = findUser(1);
    if (user.has_value()) { // or if(user)
        std::cout << "Found: " << *user << "\n";
    }
    
    auto missing = findUser(99);
    std::cout << "Missing: " << missing.value_or("Guest") << "\n";
    
    return 0;
}
```

### Step 3: Accessing
- `*opt`: Unsafe access (UB if empty).
- `opt.value()`: Safe access (throws `std::bad_optional_access` if empty).

## Challenges

### Challenge 1: Optional Int
Write a function `std::optional<int> parse(string s)`.
If `s` is a valid number, return it. Else return `nullopt`.
Use `std::stoi` and try-catch inside.

### Challenge 2: Monadic Operations (C++23 Preview)
(Just read about it): `opt.and_then(...)` allows chaining operations.
For now, implement manual chaining:
If user found, get their email. If email found, print it.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <optional>
#include <string>

std::optional<int> parseInt(const std::string& s) {
    try {
        return std::stoi(s);
    } catch (...) {
        return std::nullopt;
    }
}

int main() {
    auto val = parseInt("123");
    if (val) std::cout << *val << "\n";
    
    auto bad = parseInt("abc");
    if (!bad) std::cout << "Failed to parse\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::optional` return type
✅ Checked for value using `has_value()` or `if(opt)`
✅ Used `value_or()` for defaults
✅ Implemented safe parsing (Challenge 1)

## Key Learnings
- `std::optional` expresses "Value or Nothing" explicitly
- Better than returning pointers (no ownership confusion)
- Better than exceptions for expected failures (e.g., "User not found")

## Next Steps
Proceed to **Lab 11.10: Robust File Reader** to apply everything.
