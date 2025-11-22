# Lab 4.4: Const References

## Objective
Learn how to pass large objects efficiently without allowing modification.

## Instructions

### Step 1: The Problem with Value
Create `const_ref.cpp`. Pass a string by value.

```cpp
#include <iostream>
#include <string>

void printVal(std::string s) {
    std::cout << s << std::endl;
}
// This creates a COPY of the string. Slow for long strings!
```

### Step 2: The Problem with Reference
Pass by reference avoids copy, but allows modification.

```cpp
void printRef(std::string& s) {
    s += " (modified)"; // Oops! We didn't want this.
    std::cout << s << std::endl;
}
```

### Step 3: Const Reference (The Best of Both)
Use `const std::string&`.

```cpp
void printConstRef(const std::string& s) {
    // s += " test"; // Error: s is const
    std::cout << s << std::endl;
}
```

### Step 4: Binding to Literals
Const references CAN bind to literals (temporaries).

```cpp
printConstRef("Hello"); // Works!
// printRef("Hello"); // Error: non-const ref cannot bind to temporary
```

## Challenges

### Challenge 1: Large Struct
Create a large struct (e.g., array of 1000 ints).
Write a function that takes it by value and one by const ref.
(Optional: Measure time difference if you loop 1M times).

### Challenge 2: Output Parameters
Sometimes we use non-const refs for outputs and const refs for inputs.
Write `void split(const std::string& input, std::string& part1, std::string& part2)` that splits a string in half.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <vector>

struct BigData {
    int data[1000];
};

void process(const BigData& d) {
    // Efficient read access
    int x = d.data[0];
}

void split(const std::string& input, std::string& part1, std::string& part2) {
    int mid = input.length() / 2;
    part1 = input.substr(0, mid);
    part2 = input.substr(mid);
}

int main() {
    std::string full = "HelloWorld";
    std::string p1, p2;
    
    split(full, p1, p2);
    std::cout << p1 << " | " << p2 << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented pass-by-const-reference
✅ Prevented accidental modification
✅ Successfully passed string literals to const ref
✅ Implemented input/output parameter pattern (Challenge 2)

## Key Learnings
- `const T&` is the standard way to pass objects in C++
- It avoids copying (efficiency)
- It prevents modification (safety)
- It accepts temporaries (literals)

## Next Steps
Proceed to **Lab 4.5: Function Overloading** to create flexible APIs.
