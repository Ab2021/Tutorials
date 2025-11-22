# Lab 7.5: Const Member Functions

## Objective
Learn how to mark functions as "safe" (read-only) using `const`.

## Instructions

### Step 1: Const Correctness
Create `const_funcs.cpp`.

```cpp
#include <iostream>

class Circle {
    double radius;
public:
    Circle(double r) : radius(r) {}
    
    // Const function: Cannot modify members
    double getRadius() const {
        return radius;
    }
    
    void setRadius(double r) {
        radius = r;
    }
};
```

### Step 2: Const Objects
Create a constant object.

```cpp
int main() {
    const Circle c(5.0);
    std::cout << c.getRadius() << std::endl; // OK: getRadius is const
    // c.setRadius(10.0); // Error: setRadius is not const
    return 0;
}
```

### Step 3: Mutable (Advanced)
Sometimes you want to modify a member even in a const function (e.g., a cache or mutex).
Mark the member `mutable`.

```cpp
mutable int accessCount = 0;
// In getRadius(): accessCount++;
```

## Challenges

### Challenge 1: Fix the Error
Remove `const` from `getRadius()`. Try to compile with `const Circle c`.
It will fail. Why? Because the compiler assumes non-const functions *might* modify the object.

### Challenge 2: Overloading on Const
You can have two versions of a function:
`int& getData()`
`const int& getData() const`
Implement this in a class holding an array.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Container {
    int data[10];
public:
    Container() { for(int i=0; i<10; ++i) data[i] = i; }
    
    // Read-write access
    int& get(int index) { return data[index]; }
    
    // Read-only access
    const int& get(int index) const { return data[index]; }
};

int main() {
    Container c;
    c.get(0) = 100; // Calls non-const
    
    const Container cc;
    // cc.get(0) = 100; // Error: Calls const version, returns const ref
    std::cout << cc.get(0) << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Marked getters as `const`
✅ Called methods on `const` objects
✅ Understood why non-const methods fail on const objects
✅ Overloaded based on constness (Challenge 2)

## Key Learnings
- Mark methods `const` if they don't change state
- Allows them to be used on `const` objects (and references)
- Essential for writing correct C++ code

## Next Steps
Proceed to **Lab 7.6: Static Members** for shared data.
