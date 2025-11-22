# Lab 11.4: Custom Exceptions

## Objective
Create custom exception classes by inheriting from `std::exception`.

## Instructions

### Step 1: Define the Class
Create `custom_except.cpp`.
Inherit from `std::exception` and override `what()`.

```cpp
#include <iostream>
#include <exception>

class NetworkError : public std::exception {
public:
    const char* what() const noexcept override {
        return "Network Connection Failed";
    }
};
```

### Step 2: Throw and Catch
```cpp
void connect() {
    throw NetworkError();
}

int main() {
    try {
        connect();
    } catch (const NetworkError& e) {
        std::cout << "Custom: " << e.what() << std::endl;
    }
    return 0;
}
```

### Step 3: Inheriting from Runtime Error
Often easier to inherit from `std::runtime_error` because it handles the string storage for you.

```cpp
#include <stdexcept>
#include <string>

class DatabaseError : public std::runtime_error {
public:
    DatabaseError(const std::string& msg) 
        : std::runtime_error("DB Error: " + msg) {}
};
```

## Challenges

### Challenge 1: Error Codes
Add an integer `errorCode` member to `NetworkError`.
Initialize it in the constructor.
Access it in the catch block.

### Challenge 2: Nested Exceptions
Throw a `DatabaseError` inside a function.
Catch it in main using `std::exception&`.
Verify `what()` prints the correct message.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <exception>
#include <string>

class NetworkError : public std::exception {
    int code;
    std::string msg;
public:
    NetworkError(int c) : code(c), msg("Network Error " + std::to_string(c)) {}
    
    const char* what() const noexcept override {
        return msg.c_str();
    }
    
    int getCode() const { return code; }
};

int main() {
    try {
        throw NetworkError(404);
    } catch (const NetworkError& e) {
        std::cout << e.what() << " (Code: " << e.getCode() << ")\n";
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Created custom exception class
✅ Overrode `what()`
✅ Inherited from `std::runtime_error`
✅ Added custom data fields (Challenge 1)

## Key Learnings
- Inherit from `std::exception` (or `runtime_error`) for compatibility
- `what()` must be `const noexcept`
- Custom exceptions can carry rich error data (codes, context)

## Next Steps
Proceed to **Lab 11.5: Stack Unwinding** to understand RAII interaction.
