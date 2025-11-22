# Lab 14.6: Lambda Types and std::function

## Objective
Understand lambda types and when to use `std::function`.

## Instructions

### Step 1: Lambda Type
Create `lambda_types.cpp`.
Each lambda has a unique, compiler-generated type.

```cpp
#include <iostream>
#include <typeinfo>

int main() {
    auto l1 = []() {};
    auto l2 = []() {};
    
    // l1 = l2; // Error: Different types!
    
    std::cout << typeid(l1).name() << "\n";
    std::cout << typeid(l2).name() << "\n";
    
    return 0;
}
```

### Step 2: std::function
Type-erased wrapper for any callable.

```cpp
#include <functional>

std::function<int(int, int)> operation;

operation = [](int a, int b) { return a + b; };
std::cout << operation(5, 3) << "\n";

operation = [](int a, int b) { return a * b; };
std::cout << operation(5, 3) << "\n";
```

### Step 3: Function Pointers
Capture-less lambdas can convert to function pointers.

```cpp
int (*func_ptr)(int) = [](int x) { return x * 2; };
std::cout << func_ptr(10) << "\n";
```

## Challenges

### Challenge 1: Callback Storage
Create a class that stores a callback using `std::function`.
```cpp
class Button {
    std::function<void()> onClick;
public:
    void setOnClick(std::function<void()> callback) {
        onClick = callback;
    }
    void click() { if (onClick) onClick(); }
};
```

### Challenge 2: Performance
`std::function` has overhead (heap allocation, virtual call).
Use templates to avoid it:
```cpp
template <typename Func>
void execute(Func f) { f(); }
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <functional>

class Button {
    std::function<void()> onClick;
public:
    void setOnClick(std::function<void()> callback) {
        onClick = std::move(callback);
    }
    void click() { 
        if (onClick) onClick(); 
    }
};

template <typename Func>
void executeTemplate(Func f) {
    f(); // No overhead
}

void executeFunction(std::function<void()> f) {
    f(); // Overhead
}

int main() {
    Button btn;
    int count = 0;
    
    btn.setOnClick([&count]() {
        count++;
        std::cout << "Clicked " << count << " times\n";
    });
    
    btn.click();
    btn.click();
    btn.click();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood unique lambda types
✅ Used `std::function` for type erasure
✅ Converted lambda to function pointer
✅ Implemented callback system (Challenge 1)

## Key Learnings
- Each lambda has a unique type
- `std::function` allows storing different callables
- Use templates when possible to avoid `std::function` overhead
- Capture-less lambdas can be function pointers

## Next Steps
Proceed to **Lab 14.7: Lambdas with STL** for practical use.
