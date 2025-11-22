# Lab 13.5: Uniform Initialization (Brace Initialization)

## Objective
Use `{}` for consistent and safe initialization.

## Instructions

### Step 1: The Old Ways
Create `uniform_init.cpp`.
C++ had many initialization syntaxes.

```cpp
int x = 5; // Copy initialization
int y(10); // Direct initialization
int z{15}; // Uniform initialization (C++11)
```

### Step 2: Benefits of Braces
Prevents narrowing conversions.

```cpp
int a = 3.14; // OK but truncates (warning)
// int b{3.14}; // Error: narrowing conversion
int c{3}; // OK
```

### Step 3: Initializer Lists
Works with containers.

```cpp
std::vector<int> v{1, 2, 3, 4, 5};
std::map<std::string, int> m{{"Alice", 30}, {"Bob", 25}};
```

### Step 4: Structs
```cpp
struct Point { int x, y; };
Point p{10, 20}; // Aggregate initialization
```

## Challenges

### Challenge 1: Most Vexing Parse
This declares a function, not a variable!
```cpp
Widget w(); // Function declaration!
Widget w2{}; // Object initialization
```
Demonstrate this issue and fix it with `{}`.

### Challenge 2: Initializer List Constructor
If a class has an `initializer_list` constructor, `{}` prefers it.
```cpp
std::vector<int> v1(10, 5); // 10 elements, value 5
std::vector<int> v2{10, 5}; // 2 elements: 10 and 5
```
Verify this behavior.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class Widget {
public:
    Widget() { std::cout << "Default ctor\n"; }
};

int main() {
    // Challenge 1: Most Vexing Parse
    // Widget w(); // This is a function declaration!
    Widget w1{}; // This is an object
    
    // Challenge 2: Initializer list preference
    std::vector<int> v1(10, 5); // 10 elements of value 5
    std::vector<int> v2{10, 5}; // 2 elements: [10, 5]
    
    std::cout << "v1 size: " << v1.size() << ", v2 size: " << v2.size() << "\n";
    
    // Narrowing prevention
    double pi = 3.14;
    // int x{pi}; // Error: narrowing
    int y = pi; // OK but truncates (warning)
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `{}` for initialization
✅ Prevented narrowing conversions
✅ Initialized containers with `{}`
✅ Avoided Most Vexing Parse (Challenge 1)
✅ Understood initializer_list preference (Challenge 2)

## Key Learnings
- `{}` is the most consistent initialization syntax
- Prevents accidental narrowing
- Be aware of `initializer_list` constructor preference

## Next Steps
Proceed to **Lab 13.6: Structured Bindings** to unpack data elegantly.
