# Lab 14.5: Init-Capture (C++14)

## Objective
Initialize captured variables with expressions, enabling move semantics.

## Instructions

### Step 1: Basic Init-Capture
Create `init_capture.cpp`.

```cpp
#include <iostream>

int main() {
    auto lambda = [x = 42]() {
        std::cout << "x: " << x << "\n";
    };
    
    lambda();
    // x doesn't exist in outer scope
    
    return 0;
}
```

### Step 2: Move Capture
Move unique_ptr into lambda.

```cpp
#include <memory>

auto ptr = std::make_unique<int>(100);
auto lambda = [p = std::move(ptr)]() {
    std::cout << "Value: " << *p << "\n";
};

// ptr is now nullptr
lambda();
```

### Step 3: Computed Capture
```cpp
int a = 5, b = 10;
auto lambda = [sum = a + b]() {
    std::cout << "Sum: " << sum << "\n";
};
```

## Challenges

### Challenge 1: Capture by Move
Create a `std::vector` and move it into a lambda. Verify the original vector is empty after.

### Challenge 2: Factory Pattern
Write a function that returns a lambda with a captured unique_ptr.
```cpp
auto makeProcessor() {
    auto data = std::make_unique<std::vector<int>>(100, 42);
    return [d = std::move(data)]() { return d->size(); };
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>
#include <memory>

auto makeProcessor() {
    auto data = std::make_unique<std::vector<int>>(100, 42);
    return [d = std::move(data)]() { 
        std::cout << "Size: " << d->size() << "\n";
        return d->at(0);
    };
}

int main() {
    // Challenge 1
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto lambda = [vec = std::move(v)]() {
        std::cout << "Lambda vec size: " << vec.size() << "\n";
    };
    
    std::cout << "Original vec size: " << v.size() << "\n"; // 0
    lambda();
    
    // Challenge 2
    auto proc = makeProcessor();
    int val = proc();
    std::cout << "Value: " << val << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used init-capture syntax
✅ Moved unique_ptr into lambda
✅ Captured computed values
✅ Implemented factory pattern (Challenge 2)

## Key Learnings
- Init-capture allows creating new variables in lambda scope
- Essential for move-only types (unique_ptr, thread)
- Syntax: `[name = expression]`

## Next Steps
Proceed to **Lab 14.6: Lambda Types** to understand storage.
