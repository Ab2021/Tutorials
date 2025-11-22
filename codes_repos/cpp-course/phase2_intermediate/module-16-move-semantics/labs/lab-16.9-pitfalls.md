# Lab 16.9: Common Pitfalls

## Objective
Learn to avoid common mistakes when using move semantics.

## Instructions

### Step 1: Using After Move
Create `pitfalls.cpp`.

```cpp
#include <iostream>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = std::move(s1);
    
    // DANGER: s1 is in unspecified state
    // std::cout << s1; // Undefined behavior!
    
    // OK: Reassign or call methods that don't depend on state
    s1 = "New value"; // OK
    s1.clear();       // OK
    
    return 0;
}
```

### Step 2: Moving from const
```cpp
const std::string s = "Hello";
auto s2 = std::move(s); // Copies! Can't move from const
```

### Step 3: Unnecessary std::move
```cpp
std::string makeString() {
    std::string s = "Hello";
    return std::move(s); // BAD! Prevents RVO
}

// Good version
std::string makeStringGood() {
    std::string s = "Hello";
    return s; // Let compiler optimize
}
```

### Step 4: Forgetting noexcept
```cpp
class Widget {
public:
    // Without noexcept, std::vector uses copy instead of move!
    Widget(Widget&& other) { /* ... */ }
    
    // Better:
    Widget(Widget&& other) noexcept { /* ... */ }
};
```

### Step 5: Self-Move Assignment
```cpp
class String {
    char* data;
public:
    String& operator=(String&& other) noexcept {
        // Must check for self-assignment!
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

## Challenges

### Challenge 1: Debug Moved-From State
Create a class that detects use-after-move.

### Challenge 2: Fix Broken Code
Find and fix move-related bugs in provided code.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <stdexcept>

// Challenge 1: Debug class
class DebugString {
    std::string data;
    bool moved_from = false;
    
public:
    DebugString(const char* s) : data(s) {}
    
    DebugString(DebugString&& other) noexcept 
        : data(std::move(other.data)) {
        other.moved_from = true;
    }
    
    DebugString& operator=(DebugString&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            moved_from = false;
            other.moved_from = true;
        }
        return *this;
    }
    
    const std::string& get() const {
        if (moved_from) {
            throw std::runtime_error("Accessing moved-from object!");
        }
        return data;
    }
};

// Challenge 2: Broken code examples
class BrokenWidget {
    int* data;
    
public:
    // BROKEN: Missing noexcept
    BrokenWidget(BrokenWidget&& other) {
        data = other.data;
        other.data = nullptr;
    }
    
    // FIXED:
    BrokenWidget(BrokenWidget&& other) noexcept {
        data = other.data;
        other.data = nullptr;
    }
};

std::string brokenReturn() {
    std::string s = "Hello";
    return std::move(s); // BROKEN: Prevents RVO
}

std::string fixedReturn() {
    std::string s = "Hello";
    return s; // FIXED: Allow RVO
}

void demonstratePitfalls() {
    // Pitfall 1: Use after move
    std::string s1 = "Hello";
    std::string s2 = std::move(s1);
    // Don't use s1 here!
    
    // Pitfall 2: Moving from const
    const std::string s3 = "World";
    std::string s4 = std::move(s3); // Copies!
    
    // Pitfall 3: Unnecessary move in return
    auto s5 = brokenReturn(); // Works but suboptimal
    auto s6 = fixedReturn();  // Better
    
    // Pitfall 4: Self-move
    std::string s7 = "Test";
    s7 = std::move(s7); // Must be handled correctly!
}

int main() {
    try {
        DebugString s1("Hello");
        DebugString s2 = std::move(s1);
        
        std::cout << s2.get() << "\n"; // OK
        // std::cout << s1.get() << "\n"; // Throws!
        
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
    
    demonstratePitfalls();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Avoided use-after-move bugs
✅ Didn't move from `const`
✅ Avoided `std::move` in returns
✅ Marked move operations `noexcept`
✅ Handled self-move assignment

## Common Pitfalls Checklist
- [ ] Using moved-from objects
- [ ] Moving from `const` objects
- [ ] Using `std::move` in return statements
- [ ] Forgetting `noexcept` on move operations
- [ ] Not checking self-assignment in move assignment
- [ ] Assuming moved-from state is empty
- [ ] Moving when copy is needed

## Key Learnings
- Moved-from objects are in valid but unspecified state
- Can't move from `const` objects
- Never use `std::move` in return statements
- Always mark move operations `noexcept`
- Check for self-assignment in move assignment

## Next Steps
Proceed to **Lab 16.10: Resource Manager (Capstone)**.
