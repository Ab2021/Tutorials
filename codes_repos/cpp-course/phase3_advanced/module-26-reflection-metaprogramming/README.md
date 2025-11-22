# Module 26: Reflection and Metaprogramming

## Overview
Advanced metaprogramming techniques including type introspection, compile-time reflection, and code generation.

## Learning Objectives
By the end of this module, you will be able to:
- Perform type introspection
- Use compile-time reflection
- Generate code at compile time
- Implement domain-specific languages (DSLs)
- Apply macro metaprogramming
- Create reflection libraries

## Key Concepts

### 1. Type Introspection
Examining type properties at compile time.

```cpp
template<typename T>
void introspect() {
    std::cout << "Type: " << typeid(T).name() << "\n";
    std::cout << "Size: " << sizeof(T) << "\n";
    std::cout << "Alignment: " << alignof(T) << "\n";
    std::cout << "Is class: " << std::is_class_v<T> << "\n";
}
```

### 2. Compile-Time Reflection
Reflecting on structure members.

```cpp
#define REFLECT(Type, ...) \
    template<> struct Reflector<Type> { \
        static constexpr auto fields() { \
            return std::make_tuple(__VA_ARGS__); \
        } \
    };

struct Person {
    std::string name;
    int age;
};

REFLECT(Person, 
    FIELD(name),
    FIELD(age)
)
```

### 3. Code Generation
Generating code at compile time.

```cpp
template<size_t N>
struct GenerateArray {
    static constexpr auto value = []() {
        std::array<int, N> arr{};
        for (size_t i = 0; i < N; ++i) {
            arr[i] = i * i;
        }
        return arr;
    }();
};
```

### 4. DSL Implementation
Creating domain-specific languages.

```cpp
auto query = from(users)
    .where([](const User& u) { return u.age > 18; })
    .select([](const User& u) { return u.name; })
    .orderBy([](const User& u) { return u.age; });
```

### 5. Macro Metaprogramming
Advanced preprocessor techniques.

```cpp
#define ENUM_WITH_STRING(EnumName, ...) \
    enum class EnumName { __VA_ARGS__ }; \
    inline const char* toString(EnumName e) { \
        switch(e) { \
            ENUM_CASES(__VA_ARGS__) \
        } \
    }
```

## Rust Comparison

### Reflection
**C++:**
```cpp
// Manual reflection with macros
```

**Rust:**
```rust
use serde::Serialize;
#[derive(Serialize)]
struct Person { }
```

### Macros
**C++:**
```cpp
#define MACRO(x) ...
```

**Rust:**
```rust
macro_rules! my_macro {
    ($x:expr) => { ... }
}
```

## Labs

1. **Lab 26.1**: Type Introspection
2. **Lab 26.2**: Member Detection
3. **Lab 26.3**: Compile-Time Reflection
4. **Lab 26.4**: Automatic Serialization
5. **Lab 26.5**: Code Generation
6. **Lab 26.6**: DSL Design
7. **Lab 26.7**: Expression Templates
8. **Lab 26.8**: Macro Metaprogramming
9. **Lab 26.9**: Reflection Library
10. **Lab 26.10**: ORM Framework (Capstone)

## Additional Resources
- Boost.Hana documentation
- "C++ Template Metaprogramming"
- Magic Get library

## Next Module
After completing this module, proceed to **Module 27: Compiler and Optimization**.
