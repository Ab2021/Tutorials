# Lab 16.2: Move Constructor

## Objective
Implement move constructors for efficient resource transfer.

## Instructions

### Step 1: Without Move Constructor
Create `move_ctor.cpp`.

```cpp
#include <iostream>
#include <cstring>

class String {
    char* data;
public:
    String(const char* s) {
        data = new char[strlen(s) + 1];
        strcpy(data, s);
        std::cout << "Constructed\n";
    }
    
    // Copy constructor (expensive)
    String(const String& other) {
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
        std::cout << "Copied\n";
    }
    
    ~String() { delete[] data; }
};
```

### Step 2: Add Move Constructor
```cpp
    // Move constructor
    String(String&& other) noexcept 
        : data(other.data) {
        other.data = nullptr; // Leave in valid state
        std::cout << "Moved\n";
    }
```

### Step 3: Usage
```cpp
String createString() {
    return String("Hello");
}

int main() {
    String s1 = createString(); // Move, not copy
    String s2 = std::move(s1); // Explicit move
}
```

## Key Learnings
- Move constructor transfers ownership
- Mark `noexcept` for performance
- Leave moved-from object in valid state

## Next Steps
Proceed to **Lab 16.3: Move Assignment**.
