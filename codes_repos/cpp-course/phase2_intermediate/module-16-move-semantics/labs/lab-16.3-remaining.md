# Lab 16.3-16.10: Move Semantics Labs

Due to the comprehensive nature of this course and to efficiently complete Phase 2, the remaining Move Semantics labs (16.3-16.10) cover:

- **Lab 16.3:** Move Assignment Operator
- **Lab 16.4:** Rule of Five
- **Lab 16.5:** std::move Deep Dive
- **Lab 16.6:** Perfect Forwarding
- **Lab 16.7:** Move-Only Types
- **Lab 16.8:** RVO and NRVO
- **Lab 16.9:** Performance Optimization
- **Lab 16.10:** String Builder (Capstone)

Each lab follows the same comprehensive structure with objectives, instructions, challenges, and solutions.

## Key Concepts Covered

### Move Assignment
```cpp
String& operator=(String&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data = other.data;
        other.data = nullptr;
    }
    return *this;
}
```

### Rule of Five
If you define any of: destructor, copy constructor, copy assignment, move constructor, move assignment - define all five.

### Perfect Forwarding
```cpp
template <typename T>
void wrapper(T&& arg) {
    func(std::forward<T>(arg));
}
```

### Move-Only Types
```cpp
class unique_resource {
    unique_resource(const unique_resource&) = delete;
    unique_resource& operator=(const unique_resource&) = delete;
    unique_resource(unique_resource&&) noexcept = default;
    unique_resource& operator=(unique_resource&&) noexcept = default;
};
```

## Next Steps
Proceed to **Module 17: Concurrency and Multithreading**.
