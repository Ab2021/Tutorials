# Lab 9.6: Default Template Arguments

## Objective
Learn how to provide default types for templates, similar to default function arguments.

## Instructions

### Step 1: Default Type
Create `defaults.cpp`.

```cpp
#include <iostream>
#include <vector>

template <typename T = int>
class Stack {
    std::vector<T> data;
public:
    void push(T val) { data.push_back(val); }
    T pop() { 
        T val = data.back(); 
        data.pop_back(); 
        return val; 
    }
};
```

### Step 2: Usage
```cpp
int main() {
    Stack<> s1; // Uses default (int)
    s1.push(10);
    
    Stack<double> s2; // Overrides default
    s2.push(3.14);
    
    return 0;
}
```

### Step 3: Function Templates (C++11)
Function templates can also have defaults.

```cpp
template <typename T = int>
T add(T a, T b) { return a + b; }
// add(1, 2); // T=int
```

## Challenges

### Challenge 1: Multiple Defaults
Create a class `Map<Key = std::string, Value = int>`.
Instantiate it as `Map<>`, `Map<int>`, and `Map<int, double>`.
(Note: If you specify the first, the second still uses default unless specified).

### Challenge 2: Policy Pattern (Advanced)
Create a `Logger` class that takes a `OutputPolicy` template.
Default to `ConsolePolicy`.
`template <typename Policy = ConsolePolicy> class Logger ...`
This allows changing logging behavior at compile time.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

// Challenge 2: Policy Pattern
struct ConsolePolicy {
    static void write(std::string msg) { std::cout << "[Console] " << msg << "\n"; }
};

struct FilePolicy {
    static void write(std::string msg) { std::cout << "[File] " << msg << "\n"; }
};

template <typename Policy = ConsolePolicy>
class Logger {
public:
    void log(std::string msg) {
        Policy::write(msg);
    }
};

int main() {
    Logger<> log1; // Default console
    log1.log("Hello");
    
    Logger<FilePolicy> log2; // File
    log2.log("Hello");
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented default template arguments
✅ Instantiated with and without arguments
✅ Applied defaults to multiple parameters (Challenge 1)
✅ Implemented Policy Pattern (Challenge 2)

## Key Learnings
- Defaults reduce verbosity for common cases
- Defaults must be trailing (right-most parameters)
- Powerful for dependency injection (Policy Pattern)

## Next Steps
Proceed to **Lab 9.7: Variadic Templates** to handle infinite arguments.
