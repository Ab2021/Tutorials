# Lab 14.8: IIFE (Immediately Invoked Function Expression)

## Objective
Use lambdas for complex initialization and scoped logic.

## Instructions

### Step 1: Const Initialization
Create `iife.cpp`.
Initialize a const variable with complex logic.

```cpp
#include <iostream>

int main() {
    const int value = []() {
        int temp = 0;
        for (int i = 1; i <= 10; ++i) {
            temp += i;
        }
        return temp;
    }(); // Immediately invoked
    
    std::cout << "Value: " << value << "\n";
    
    return 0;
}
```

### Step 2: Conditional Initialization
```cpp
const std::string mode = [](bool debug) {
    if (debug) return "DEBUG";
    else return "RELEASE";
}(true);
```

### Step 3: Scoped Cleanup
```cpp
{
    auto cleanup = []() {
        std::cout << "Cleaning up...\n";
    };
    
    // Do work...
    
    cleanup(); // Explicit cleanup
}
```

## Challenges

### Challenge 1: Table Initialization
Initialize a lookup table using IIFE.
```cpp
const auto primes = []() {
    std::vector<int> p;
    // Compute first 10 primes
    return p;
}();
```

### Challenge 2: RAII Alternative
Use IIFE to ensure cleanup even with early returns.
```cpp
void process() {
    auto guard = [&]() {
        // Cleanup code
    };
    
    if (error) { guard(); return; }
    // More code
    guard();
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

int main() {
    // Challenge 1: Prime table
    const auto primes = []() {
        std::vector<int> p = {2};
        for (int n = 3; p.size() < 10; n += 2) {
            bool isPrime = true;
            for (int div : p) {
                if (n % div == 0) {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime) p.push_back(n);
        }
        return p;
    }();
    
    std::cout << "First 10 primes: ";
    for (int p : primes) std::cout << p << " ";
    std::cout << "\n";
    
    // IIFE for config
    const auto config = [](const char* env) {
        if (std::string(env) == "prod") return "Production Mode";
        return "Development Mode";
    }("dev");
    
    std::cout << config << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used IIFE for const initialization
✅ Used IIFE for conditional logic
✅ Initialized complex data structures (Challenge 1)

## Key Learnings
- IIFE allows complex const initialization
- Useful for one-time setup logic
- Keeps initialization code close to declaration

## Next Steps
Proceed to **Lab 14.9: Constexpr Lambdas** for compile-time execution.
