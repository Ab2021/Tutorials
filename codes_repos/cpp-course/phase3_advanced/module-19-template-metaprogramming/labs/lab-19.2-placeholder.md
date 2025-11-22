# Lab 19.2: Custom Type Traits

## Objective
Create custom type traits to detect specific properties of types.

## Instructions

### Step 1: Basic Custom Trait
Create `custom_traits.cpp`.

```cpp
#include <iostream>
#include <type_traits>

// Custom trait to check if type has a size() method
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>> 
    : std::true_type {};

template<typename T>
inline constexpr bool has_size_v = has_size<T>::value;
```

### Step 2: Detecting Member Functions
```cpp
// Detect if type has begin() and end()
template<typename T, typename = void>
struct is_iterable : std::false_type {};

template<typename T>
struct is_iterable<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

template<typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;
```

### Step 3: Using Custom Traits
```cpp
template<typename T>
void printSize(const T& container) {
    if constexpr (has_size_v<T>) {
        std::cout << "Size: " << container.size() << "\n";
    } else {
        std::cout << "No size() method\n";
    }
}
```

## Challenges

### Challenge 1: Serializable Trait
Create a trait to detect if a type has `serialize()` and `deserialize()` methods.

### Challenge 2: Comparable Trait
Detect if a type supports comparison operators.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

// Challenge 1: Serializable trait
template<typename T, typename = void>
struct is_serializable : std::false_type {};

template<typename T>
struct is_serializable<T, std::void_t<
    decltype(std::declval<T>().serialize()),
    decltype(std::declval<T>().deserialize(std::string()))
>> : std::true_type {};

template<typename T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;

// Challenge 2: Comparable trait
template<typename T, typename = void>
struct is_comparable : std::false_type {};

template<typename T>
struct is_comparable<T, std::void_t<
    decltype(std::declval<T>() < std::declval<T>()),
    decltype(std::declval<T>() == std::declval<T>())
>> : std::true_type {};

template<typename T>
inline constexpr bool is_comparable_v = is_comparable<T>::value;

// Example serializable class
class Data {
public:
    std::string serialize() const { return "data"; }
    void deserialize(const std::string&) {}
};

int main() {
    std::cout << "vector has size: " << has_size_v<std::vector<int>> << "\n";
    std::cout << "int has size: " << has_size_v<int> << "\n";
    
    std::cout << "vector is iterable: " << is_iterable_v<std::vector<int>> << "\n";
    std::cout << "int is iterable: " << is_iterable_v<int> << "\n";
    
    std::cout << "Data is serializable: " << is_serializable_v<Data> << "\n";
    std::cout << "int is serializable: " << is_serializable_v<int> << "\n";
    
    std::cout << "int is comparable: " << is_comparable_v<int> << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created custom type traits
✅ Detected member functions
✅ Used `std::void_t` for SFINAE
✅ Implemented serializable trait (Challenge 1)
✅ Implemented comparable trait (Challenge 2)

## Key Learnings
- Custom traits use template specialization
- `std::void_t` simplifies SFINAE
- `std::declval` allows checking without instantiation
- Traits enable compile-time introspection

## Next Steps
Proceed to **Lab 19.3: SFINAE Techniques**.
