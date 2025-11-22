# Lab 13.8: Variant and Any (C++17)

## Objective
Use `std::variant` for type-safe unions and `std::any` for type-erased storage.

## Instructions

### Step 1: Variant
Create `variant_any.cpp`.
`std::variant` can hold one of several types.

```cpp
#include <iostream>
#include <variant>
#include <string>

int main() {
    std::variant<int, double, std::string> v;
    
    v = 42;
    std::cout << std::get<int>(v) << "\n";
    
    v = 3.14;
    std::cout << std::get<double>(v) << "\n";
    
    v = "Hello";
    std::cout << std::get<std::string>(v) << "\n";
    
    return 0;
}
```

### Step 2: Visiting
Use `std::visit` to handle all cases.

```cpp
std::visit([](auto&& arg) {
    std::cout << "Value: " << arg << "\n";
}, v);
```

### Step 3: Any
`std::any` can hold ANY type (type-erased).

```cpp
#include <any>

std::any a = 10;
a = 3.14;
a = std::string("Hello");

if (a.type() == typeid(std::string)) {
    std::cout << std::any_cast<std::string>(a) << "\n";
}
```

## Challenges

### Challenge 1: Error Handling
`std::variant` can represent Result<T, Error>.
```cpp
std::variant<int, std::string> divide(int a, int b) {
    if (b == 0) return std::string("Division by zero");
    return a / b;
}
```
Use `std::holds_alternative` to check which type is active.

### Challenge 2: Visitor Pattern
Create a visitor that handles int, double, and string differently.
```cpp
struct Printer {
    void operator()(int i) { std::cout << "Int: " << i << "\n"; }
    void operator()(double d) { std::cout << "Double: " << d << "\n"; }
    void operator()(const std::string& s) { std::cout << "String: " << s << "\n"; }
};
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <variant>
#include <string>

// Challenge 1
std::variant<int, std::string> divide(int a, int b) {
    if (b == 0) return std::string("Error: Division by zero");
    return a / b;
}

// Challenge 2
struct Printer {
    void operator()(int i) const { std::cout << "Int: " << i << "\n"; }
    void operator()(double d) const { std::cout << "Double: " << d << "\n"; }
    void operator()(const std::string& s) const { std::cout << "String: " << s << "\n"; }
};

int main() {
    auto result = divide(10, 2);
    if (std::holds_alternative<int>(result)) {
        std::cout << "Result: " << std::get<int>(result) << "\n";
    } else {
        std::cout << std::get<std::string>(result) << "\n";
    }
    
    std::variant<int, double, std::string> v = 42;
    std::visit(Printer{}, v);
    
    v = 3.14;
    std::visit(Printer{}, v);
    
    v = "Hello";
    std::visit(Printer{}, v);
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::variant` to hold multiple types
✅ Used `std::visit` for type-safe dispatch
✅ Used `std::any` for type erasure
✅ Implemented error handling with variant (Challenge 1)
✅ Created custom visitor (Challenge 2)

## Key Learnings
- `variant` is a type-safe union (like Rust `enum`)
- `visit` provides exhaustive pattern matching
- `any` is for truly dynamic types (use sparingly)

## Next Steps
Proceed to **Lab 13.9: Designated Initializers** for clearer struct initialization.
