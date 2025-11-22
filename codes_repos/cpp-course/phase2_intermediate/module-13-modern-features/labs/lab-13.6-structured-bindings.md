# Lab 13.6: Structured Bindings (C++17)

## Objective
Unpack tuples, pairs, and structs into named variables.

## Instructions

### Step 1: Pair Unpacking
Create `structured_bindings.cpp`.

```cpp
#include <iostream>
#include <utility>
#include <map>

int main() {
    std::pair<int, std::string> p{1, "Alice"};
    
    // Old way
    int id = p.first;
    std::string name = p.second;
    
    // Structured binding
    auto [id2, name2] = p;
    std::cout << id2 << ": " << name2 << "\n";
    
    return 0;
}
```

### Step 2: Map Iteration
Makes map iteration beautiful.

```cpp
std::map<std::string, int> ages{{"Alice", 30}, {"Bob", 25}};

for (const auto& [name, age] : ages) {
    std::cout << name << " is " << age << " years old\n";
}
```

### Step 3: Struct Unpacking
```cpp
struct Point { int x, y; };
Point p{10, 20};
auto [x, y] = p;
```

## Challenges

### Challenge 1: Tuple Unpacking
Use `std::tuple` to return multiple values from a function.
```cpp
std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}
auto [i, d, s] = getData();
```

### Challenge 2: Array Unpacking
Structured bindings work on arrays too!
```cpp
int arr[] = {1, 2, 3};
auto [a, b, c] = arr;
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <tuple>
#include <map>

std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}

int main() {
    // Challenge 1: Tuple
    auto [i, d, s] = getData();
    std::cout << i << ", " << d << ", " << s << "\n";
    
    // Challenge 2: Array
    int arr[] = {10, 20, 30};
    auto [x, y, z] = arr;
    std::cout << x << " " << y << " " << z << "\n";
    
    // Map iteration
    std::map<std::string, int> scores{{"Alice", 95}, {"Bob", 87}};
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Unpacked pairs
✅ Used structured bindings in map iteration
✅ Unpacked structs
✅ Unpacked tuples (Challenge 1)
✅ Unpacked arrays (Challenge 2)

## Key Learnings
- Structured bindings make code more readable
- Works with pairs, tuples, structs, and arrays
- Essential for modern C++ map iteration

## Next Steps
Proceed to **Lab 13.7: If Constexpr** for compile-time conditionals.
