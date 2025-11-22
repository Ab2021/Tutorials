# Lab 16.1: Rvalue References Basics

## Objective
Understand rvalue references and value categories.

## Instructions

### Step 1: Lvalue vs Rvalue
Create `rvalue_basics.cpp`.

```cpp
#include <iostream>

int main() {
    int x = 5; // x is lvalue
    int& lref = x; // Lvalue reference
    
    // int& lref2 = 5; // Error: cannot bind lvalue ref to rvalue
    int&& rref = 5; // Rvalue reference binds to temporary
    
    std::cout << rref << "\n";
    rref = 10; // Can modify through rvalue reference
    
    return 0;
}
```

### Step 2: Function Overloading
```cpp
void process(int& x) { std::cout << "Lvalue\n"; }
void process(int&& x) { std::cout << "Rvalue\n"; }

int main() {
    int a = 5;
    process(a); // Calls lvalue version
    process(10); // Calls rvalue version
    process(std::move(a)); // Calls rvalue version
}
```

### Step 3: Named Rvalue References are Lvalues
```cpp
void func(int&& x) {
    // x is an rvalue reference, but x itself is an lvalue!
    process(x); // Calls lvalue version
    process(std::move(x)); // Calls rvalue version
}
```

## Challenges

### Challenge 1: Const Rvalue Reference
What happens with `const int&&`? When is it useful?

### Challenge 2: Reference Collapsing
Understand reference collapsing rules:
- `T& &` → `T&`
- `T& &&` → `T&`
- `T&& &` → `T&`
- `T&& &&` → `T&&`

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

void process(int& x) { std::cout << "Lvalue\n"; }
void process(int&& x) { std::cout << "Rvalue\n"; }
void process(const int& x) { std::cout << "Const Lvalue\n"; }
void process(const int&& x) { std::cout << "Const Rvalue\n"; }

int main() {
    int a = 5;
    const int b = 10;
    
    process(a); // Lvalue
    process(5); // Rvalue
    process(b); // Const Lvalue
    process(std::move(b)); // Const Rvalue
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood lvalue vs rvalue
✅ Created rvalue references
✅ Overloaded functions for lvalue/rvalue
✅ Understood named rvalue references are lvalues

## Key Learnings
- Rvalue references bind to temporaries
- Enable move semantics and perfect forwarding
- Named rvalue references are lvalues

## Next Steps
Proceed to **Lab 16.2: Move Constructor**.
