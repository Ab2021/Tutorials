# Lab 16.6: Perfect Forwarding

## Objective
Master perfect forwarding with `std::forward` for writing generic wrapper functions.

## Instructions

### Step 1: The Problem
Without perfect forwarding, we lose value category information.

Create `perfect_forwarding.cpp`.

```cpp
#include <iostream>

void process(int& x) { std::cout << "Lvalue\n"; }
void process(int&& x) { std::cout << "Rvalue\n"; }

template<typename T>
void wrapper(T&& arg) {
    process(arg); // Always calls lvalue version!
}

int main() {
    int x = 42;
    wrapper(x);    // Lvalue
    wrapper(42);   // Rvalue, but calls lvalue version!
    
    return 0;
}
```

### Step 2: The Solution - std::forward
```cpp
template<typename T>
void wrapper(T&& arg) {
    process(std::forward<T>(arg)); // Preserves value category
}

int main() {
    int x = 42;
    wrapper(x);    // Calls lvalue version
    wrapper(42);   // Calls rvalue version (correct!)
    
    return 0;
}
```

### Step 3: Universal References
`T&&` in template context is a universal (forwarding) reference.

```cpp
template<typename T>
void func(T&& arg);  // Universal reference

void func(int&& arg); // Rvalue reference (not template)
```

### Step 4: Reference Collapsing Rules
```cpp
// T& & → T&
// T& && → T&
// T&& & → T&
// T&& && → T&&
```

## Challenges

### Challenge 1: Factory Function
Create a generic factory that perfectly forwards arguments.

```cpp
template<typename T, typename... Args>
T* create(Args&&... args) {
    return new T(std::forward<Args>(args)...);
}
```

### Challenge 2: Wrapper Class
Implement a wrapper that stores and forwards a callable.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <string>

// Challenge 1: Factory function
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_forwarding(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Challenge 2: Wrapper class
template<typename F>
class FunctionWrapper {
    F func;
public:
    template<typename Func>
    FunctionWrapper(Func&& f) : func(std::forward<Func>(f)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) {
        return func(std::forward<Args>(args)...);
    }
};

class Widget {
    std::string name;
    int value;
public:
    Widget(const std::string& n, int v) 
        : name(n), value(v) {
        std::cout << "Lvalue string\n";
    }
    
    Widget(std::string&& n, int v) 
        : name(std::move(n)), value(v) {
        std::cout << "Rvalue string\n";
    }
    
    void display() const {
        std::cout << name << ": " << value << "\n";
    }
};

int main() {
    // Challenge 1: Factory
    std::string name = "Widget1";
    auto w1 = make_unique_forwarding<Widget>(name, 42);      // Lvalue
    auto w2 = make_unique_forwarding<Widget>("Widget2", 99); // Rvalue
    
    w1->display();
    w2->display();
    
    // Challenge 2: Wrapper
    auto lambda = [](int x, int y) { return x + y; };
    FunctionWrapper wrapper(lambda);
    std::cout << "Result: " << wrapper(10, 20) << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood the forwarding problem
✅ Used `std::forward` correctly
✅ Implemented generic factory (Challenge 1)
✅ Created function wrapper (Challenge 2)

## Rust Comparison
```rust
// Rust doesn't need perfect forwarding
// Ownership is explicit
fn wrapper<F, T>(f: F, arg: T) 
where F: FnOnce(T) {
    f(arg) // Ownership transferred
}
```

## Key Learnings
- `T&&` in template context is universal reference
- `std::forward` preserves value category
- Essential for writing generic wrapper functions
- Reference collapsing enables perfect forwarding

## Next Steps
Proceed to **Lab 16.7: Return Value Optimization**.
