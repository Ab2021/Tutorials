# Lab 16.3: Move Assignment Operator

## Objective
Implement move assignment for efficient resource transfer during assignment.

## Instructions

### Step 1: Basic Move Assignment
Create `move_assignment.cpp`.

```cpp
#include <iostream>
#include <cstring>

class String {
    char* data;
public:
    String(const char* s) {
        data = new char[strlen(s) + 1];
        strcpy(data, s);
    }
    
    ~String() { delete[] data; }
    
    // Move assignment operator
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data; // Free existing resource
            data = other.data; // Transfer ownership
            other.data = nullptr; // Leave in valid state
        }
        return *this;
    }
};
```

### Step 2: Self-Assignment Check
```cpp
String& operator=(String&& other) noexcept {
    if (this != &other) { // Critical!
        // ...
    }
    return *this;
}
```

### Step 3: Copy-and-Swap Idiom
```cpp
String& operator=(String other) { // Pass by value
    swap(*this, other); // Swap with temporary
    return *this; // other destroyed, freeing old data
}
```

## Challenges

### Challenge 1: Implement Both
Implement both copy assignment and move assignment. Observe which is called.

### Challenge 2: Benchmark
Compare performance of copy vs move assignment with large data.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstring>
#include <chrono>

class String {
    char* data;
    size_t size;
    
public:
    String(const char* s = "") {
        size = strlen(s);
        data = new char[size + 1];
        strcpy(data, s);
        std::cout << "Constructed\n";
    }
    
    ~String() { delete[] data; }
    
    // Copy assignment
    String& operator=(const String& other) {
        std::cout << "Copy assigned\n";
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new char[size + 1];
            strcpy(data, other.data);
        }
        return *this;
    }
    
    // Move assignment
    String& operator=(String&& other) noexcept {
        std::cout << "Move assigned\n";
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    const char* c_str() const { return data; }
};

int main() {
    String s1("Hello");
    String s2("World");
    
    s1 = s2; // Copy assignment
    s1 = String("Temp"); // Move assignment
    s1 = std::move(s2); // Explicit move assignment
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented move assignment operator
✅ Handled self-assignment
✅ Marked `noexcept`
✅ Compared copy vs move performance (Challenge 2)

## Key Learnings
- Move assignment transfers ownership during assignment
- Must free existing resource before taking new one
- Self-assignment check is critical
- `noexcept` enables optimizations

## Next Steps
Proceed to **Lab 16.4: Rule of Five**.
