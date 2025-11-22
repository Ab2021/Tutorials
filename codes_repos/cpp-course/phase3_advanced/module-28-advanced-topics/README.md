# Module 28: Advanced Topics

## Overview
Cutting-edge C++20/23 features and future directions of the language.

## Learning Objectives
By the end of this module, you will be able to:
- Use C++20 coroutines
- Work with C++20 modules
- Master ranges library
- Apply concepts effectively
- Understand C++23 features
- Anticipate future C++ standards

## Key Concepts

### 1. Coroutines (C++20)
Asynchronous programming with coroutines.

```cpp
#include <coroutine>

struct Task {
    struct promise_type {
        Task get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };
};

Task asyncOperation() {
    co_await std::suspend_always{};
    // Async work
    co_return;
}
```

### 2. Modules (C++20)
Modern code organization.

```cpp
// math.cppm
export module math;

export int add(int a, int b) {
    return a + b;
}

// main.cpp
import math;

int main() {
    return add(2, 3);
}
```

### 3. Ranges (C++20)
Composable algorithms.

```cpp
#include <ranges>

auto result = data
    | std::views::filter([](int x) { return x % 2 == 0; })
    | std::views::transform([](int x) { return x * 2; })
    | std::views::take(5);
```

### 4. Concepts (C++20)
Constraining templates.

```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}
```

### 5. C++23 Features
Latest standard features.

```cpp
// std::expected
std::expected<int, std::string> divide(int a, int b) {
    if (b == 0) return std::unexpected("Division by zero");
    return a / b;
}

// std::print
std::print("Hello, {}!\n", "World");

// Multidimensional subscript operator
auto value = matrix[i, j];
```

### 6. Future Directions
Upcoming C++ features.

- Pattern matching
- Reflection
- Contracts
- Executors
- Networking TS

## Rust Comparison

### Async/Await
**C++:**
```cpp
Task async_func() {
    co_await something();
}
```

**Rust:**
```rust
async fn async_func() {
    something().await;
}
```

### Modules
**C++:**
```cpp
export module math;
```

**Rust:**
```rust
pub mod math { }
```

## Labs

1. **Lab 28.1**: Coroutine Basics
2. **Lab 28.2**: Async Generators
3. **Lab 28.3**: Module System
4. **Lab 28.4**: Ranges Composition
5. **Lab 28.5**: Custom Range Adaptors
6. **Lab 28.6**: Concepts Deep Dive
7. **Lab 28.7**: C++23 Features
8. **Lab 28.8**: std::expected Usage
9. **Lab 28.9**: Format Library
10. **Lab 28.10**: Modern C++ Application (Capstone)

## Additional Resources
- cppreference.com (C++20/23)
- "C++20: The Complete Guide"
- WG21 papers (isocpp.org)
- CppCon talks

## Course Completion
Congratulations! You've completed the entire C++ course. You now have comprehensive knowledge spanning beginner to advanced topics.

## Next Steps
- Contribute to open-source C++ projects
- Explore specialized domains (game dev, systems programming, etc.)
- Stay updated with C++ standards evolution
- Share your knowledge with the community
