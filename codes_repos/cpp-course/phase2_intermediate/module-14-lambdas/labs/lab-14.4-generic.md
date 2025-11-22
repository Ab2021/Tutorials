# Lab 14.4: Generic Lambdas (C++14)

## Objective
Use `auto` parameters to create template-like lambdas.

## Instructions

### Step 1: Auto Parameters
Create `generic_lambda.cpp`.

```cpp
#include <iostream>
#include <string>

int main() {
    auto print = [](auto x) {
        std::cout << x << "\n";
    };
    
    print(42);
    print(3.14);
    print("Hello");
    print(std::string("World"));
    
    return 0;
}
```

### Step 2: Multiple Auto Parameters
```cpp
auto add = [](auto a, auto b) {
    return a + b;
};

std::cout << add(1, 2) << "\n"; // 3
std::cout << add(1.5, 2.5) << "\n"; // 4.0
std::cout << add(std::string("Hello"), std::string(" World")) << "\n";
```

### Step 3: Perfect Forwarding
```cpp
auto forward_call = [](auto&& func, auto&&... args) {
    return std::forward<decltype(func)>(func)(
        std::forward<decltype(args)>(args)...
    );
};
```

## Challenges

### Challenge 1: Type Constraints
Use `if constexpr` inside a generic lambda to handle different types.
```cpp
auto process = [](auto x) {
    if constexpr (std::is_integral_v<decltype(x)>) {
        return x * 2;
    } else {
        return x;
    }
};
```

### Challenge 2: Generic Comparator
Create a generic lambda for `std::sort` that works with any comparable type.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>

int main() {
    // Challenge 1: Type constraints
    auto process = [](auto x) {
        if constexpr (std::is_integral_v<decltype(x)>) {
            std::cout << "Int: " << x * 2 << "\n";
        } else if constexpr (std::is_floating_point_v<decltype(x)>) {
            std::cout << "Float: " << x / 2.0 << "\n";
        } else {
            std::cout << "Other: " << x << "\n";
        }
    };
    
    process(10);
    process(3.14);
    process("Hello");
    
    // Challenge 2: Generic comparator
    auto desc = [](const auto& a, const auto& b) {
        return a > b;
    };
    
    std::vector<int> nums = {3, 1, 4, 1, 5};
    std::sort(nums.begin(), nums.end(), desc);
    
    for (auto n : nums) std::cout << n << " ";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `auto` parameters
✅ Created multi-parameter generic lambda
✅ Used type traits with generic lambdas (Challenge 1)
✅ Created generic comparator (Challenge 2)

## Key Learnings
- Generic lambdas are templates
- Each call with different types generates a new instantiation
- Combine with `if constexpr` for type-specific behavior

## Next Steps
Proceed to **Lab 14.5: Init-Capture** to move objects into lambdas.
