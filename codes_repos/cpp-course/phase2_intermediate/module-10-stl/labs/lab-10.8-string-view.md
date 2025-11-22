# Lab 10.8: String View (C++17)

## Objective
Avoid expensive string copies using `std::string_view`.

## Instructions

### Step 1: The Problem
Create `string_view.cpp`.
Pass a substring to a function.

```cpp
#include <iostream>
#include <string>

void printString(const std::string& s) {
    std::cout << s << "\n";
}

int main() {
    std::string text = "Hello World";
    // Creates a temporary string copy for "Hello"
    printString(text.substr(0, 5)); 
    return 0;
}
```

### Step 2: The Solution
Use `std::string_view`. It's a pointer + length. No allocation.

```cpp
#include <string_view>

void printView(std::string_view sv) {
    std::cout << sv << "\n";
}

int main() {
    std::string text = "Hello World";
    // No copy! Just points to the buffer.
    printView(std::string_view(text).substr(0, 5));
    return 0;
}
```

### Step 3: Literals
```cpp
using namespace std::literals;
auto sv = "Hello"sv; // std::string_view
```

## Challenges

### Challenge 1: Safety Check
`string_view` does NOT own the data.
Create a function that returns a `string_view` to a local string.
Call it in main. (Undefined Behavior / Crash).
*Lesson: Never return a view to a local variable.*

### Challenge 2: Parsing
Write a function `trim(string_view)` that returns a new view without leading/trailing spaces.
Use `sv.remove_prefix` and `sv.remove_suffix`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string_view>

std::string_view trim(std::string_view sv) {
    while (!sv.empty() && isspace(sv.front())) sv.remove_prefix(1);
    while (!sv.empty() && isspace(sv.back())) sv.remove_suffix(1);
    return sv;
}

int main() {
    std::string_view sv = "  Hello  ";
    std::cout << "'" << trim(sv) << "'\n";
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::string_view` as function parameter
✅ Avoided string copies
✅ Understood lifetime risks (Challenge 1)
✅ Implemented zero-copy trim (Challenge 2)

## Key Learnings
- Use `string_view` for read-only string arguments
- It is lightweight (pointer + size)
- BEWARE of lifetime issues (dangling views)

## Next Steps
Proceed to **Lab 10.9: Ranges** for the future of STL.
