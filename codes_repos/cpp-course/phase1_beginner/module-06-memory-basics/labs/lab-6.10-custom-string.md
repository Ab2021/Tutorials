# Lab 6.10: Custom String Class (Capstone)

## Objective
Build a simplified `MyString` class that manages its own memory, implementing the Rule of Five.

## Instructions

### Step 1: Basic Class
Create `mystring.cpp`.

```cpp
#include <iostream>
#include <cstring> // strlen, strcpy

class MyString {
    char* buffer;
public:
    MyString(const char* str) {
        if (str) {
            buffer = new char[std::strlen(str) + 1];
            std::strcpy(buffer, str);
        } else {
            buffer = nullptr;
        }
    }
    
    ~MyString() {
        delete[] buffer;
    }
    
    void print() const {
        if (buffer) std::cout << buffer << std::endl;
    }
};
```

### Step 2: Copy Semantics (Rule of 3)
Implement Copy Constructor and Copy Assignment.

```cpp
    MyString(const MyString& other) {
        if (other.buffer) {
            buffer = new char[std::strlen(other.buffer) + 1];
            std::strcpy(buffer, other.buffer);
        } else {
            buffer = nullptr;
        }
    }
    
    MyString& operator=(const MyString& other) {
        if (this == &other) return *this;
        delete[] buffer;
        if (other.buffer) {
            buffer = new char[std::strlen(other.buffer) + 1];
            std::strcpy(buffer, other.buffer);
        } else {
            buffer = nullptr;
        }
        return *this;
    }
```

### Step 3: Move Semantics (Rule of 5)
Implement Move Constructor and Move Assignment.

```cpp
    MyString(MyString&& other) noexcept : buffer(other.buffer) {
        other.buffer = nullptr;
    }
    
    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] buffer;
            buffer = other.buffer;
            other.buffer = nullptr;
        }
        return *this;
    }
```

## Challenges

### Challenge 1: Concatenation
Implement `operator+` to join two strings.
`MyString operator+(const MyString& other)`
Allocate new buffer size = `len(this) + len(other) + 1`.

### Challenge 2: Comparison
Implement `bool operator==(const MyString& other)`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <cstring>
#include <utility>

class MyString {
    char* buffer;
public:
    MyString(const char* str = nullptr) {
        if (str) {
            buffer = new char[std::strlen(str) + 1];
            std::strcpy(buffer, str);
        } else buffer = nullptr;
    }
    
    ~MyString() { delete[] buffer; }
    
    // Copy
    MyString(const MyString& other) {
        if (other.buffer) {
            buffer = new char[std::strlen(other.buffer) + 1];
            std::strcpy(buffer, other.buffer);
        } else buffer = nullptr;
    }
    
    // Move
    MyString(MyString&& other) noexcept : buffer(other.buffer) {
        other.buffer = nullptr;
    }
    
    void print() const { if(buffer) std::cout << buffer << std::endl; }
    
    // Challenge 1: Concat
    MyString operator+(const MyString& other) {
        int len1 = buffer ? std::strlen(buffer) : 0;
        int len2 = other.buffer ? std::strlen(other.buffer) : 0;
        char* newBuf = new char[len1 + len2 + 1];
        if(buffer) std::strcpy(newBuf, buffer);
        if(other.buffer) std::strcpy(newBuf + len1, other.buffer);
        else if (!buffer) newBuf[0] = '\0';
        
        MyString res(newBuf);
        delete[] newBuf; // res made a copy, so delete temp
        return res;
    }
};

int main() {
    MyString s1("Hello");
    MyString s2(" World");
    MyString s3 = s1 + s2;
    s3.print();
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented Rule of Five
✅ Managed raw `char*` buffer
✅ Implemented deep copy
✅ Implemented move semantics
✅ Implemented concatenation (Challenge 1)

## Key Learnings
- Managing raw memory is hard work! (Use `std::string` in real code)
- Rule of Five ensures safety and performance
- Move semantics significantly reduce allocation overhead

## Next Steps
Congratulations! You've completed Module 6.

Proceed to **Module 7: Classes and Objects** to dive deeper into OOP.
