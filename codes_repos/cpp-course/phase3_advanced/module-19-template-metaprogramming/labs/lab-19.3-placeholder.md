# Lab 19.3: SFINAE Techniques

## Objective
Master SFINAE (Substitution Failure Is Not An Error) for template selection.

## Instructions

### Step 1: Basic SFINAE with enable_if
Create `sfinae.cpp`.

```cpp
#include <iostream>
#include <type_traits>

// Only enabled for integral types
template<typename T>
std::enable_if_t<std::is_integral_v<T>, void>
print(T value) {
    std::cout << "Integer: " << value << "\n";
}

// Only enabled for floating point types
template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, void>
print(T value) {
    std::cout << "Float: " << value << "\n";
}
```

### Step 2: Return Type SFINAE
```cpp
template<typename T>
auto getValue(T& container) 
    -> decltype(container.front()) {
    return container.front();
}
```

### Step 3: Expression SFINAE
```cpp
template<typename T>
auto add(T a, T b) -> decltype(a + b) {
    return a + b;
}
```

## Challenges

### Challenge 1: Container Printer
Create overloaded functions that print containers differently based on their properties.

### Challenge 2: Smart Add
Implement `add()` that works for arithmetic types and strings.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <type_traits>

// Challenge 1: Container printer with SFINAE

// For containers with random access
template<typename T>
std::enable_if_t<
    std::is_same_v<
        typename std::iterator_traits<typename T::iterator>::iterator_category,
        std::random_access_iterator_tag
    >, void>
printContainer(const T& container) {
    std::cout << "Random access container: ";
    for (size_t i = 0; i < container.size(); ++i) {
        std::cout << container[i] << " ";
    }
    std::cout << "\n";
}

// For other containers
template<typename T>
std::enable_if_t<
    !std::is_same_v<
        typename std::iterator_traits<typename T::iterator>::iterator_category,
        std::random_access_iterator_tag
    >, void>
printContainer(const T& container) {
    std::cout << "Sequential container: ";
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << "\n";
}

// Challenge 2: Smart add

// For arithmetic types
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
smartAdd(T a, T b) {
    return a + b;
}

// For strings
template<typename T>
std::enable_if_t<
    std::is_same_v<T, std::string> ||
    std::is_same_v<T, const char*>, 
    std::string>
smartAdd(T a, T b) {
    return std::string(a) + std::string(b);
}

int main() {
    // Challenge 1
    std::vector<int> vec = {1, 2, 3};
    std::list<int> lst = {4, 5, 6};
    
    printContainer(vec);
    printContainer(lst);
    
    // Challenge 2
    std::cout << "Int add: " << smartAdd(5, 3) << "\n";
    std::cout << "Double add: " << smartAdd(2.5, 1.5) << "\n";
    std::cout << "String add: " << smartAdd(std::string("Hello "), std::string("World")) << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `std::enable_if` for function selection
✅ Applied return type SFINAE
✅ Implemented expression SFINAE
✅ Created container printer (Challenge 1)
✅ Implemented smart add (Challenge 2)

## Key Learnings
- SFINAE removes invalid template instantiations
- `enable_if` controls function availability
- Return type SFINAE uses trailing return types
- Multiple overloads can coexist with SFINAE

## Next Steps
Proceed to **Lab 19.4: Tag Dispatch**.
