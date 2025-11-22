# Lab 7.8: Operator Overloading (Intro)

## Objective
Give meaning to standard operators (`+`, `-`, `==`) for your custom classes.

## Instructions

### Step 1: Vector2 Class
Create `operators.cpp`.

```cpp
#include <iostream>

class Vector2 {
public:
    float x, y;
    Vector2(float x, float y) : x(x), y(y) {}
    
    void print() const { std::cout << "(" << x << ", " << y << ")\n"; }
};
```

### Step 2: Overload +
Add a member function.

```cpp
    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }
```

### Step 3: Usage
```cpp
int main() {
    Vector2 v1(1, 2);
    Vector2 v2(3, 4);
    Vector2 v3 = v1 + v2;
    v3.print(); // (4, 6)
    return 0;
}
```

## Challenges

### Challenge 1: Overload ==
Implement `bool operator==(const Vector2& other) const`.
Check if x and y are equal.

### Challenge 2: Overload << (Stream Insertion)
Implement `std::ostream& operator<<(std::ostream& os, const Vector2& v)` as a friend function (outside class, or inside with `friend`).
Allows `std::cout << v1 << std::endl;`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Vector2 {
public:
    float x, y;
    Vector2(float x, float y) : x(x), y(y) {}
    
    // Member overload
    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }
    
    bool operator==(const Vector2& other) const {
        return x == other.x && y == other.y;
    }
    
    // Friend overload for cout
    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        os << "(" << v.x << ", " << v.y << ")";
        return os;
    }
};

int main() {
    Vector2 v1(1, 2);
    Vector2 v2(1, 2);
    
    if (v1 == v2) std::cout << "Equal!\n";
    std::cout << v1 + v2 << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Overloaded arithmetic operator (`+`)
✅ Overloaded comparison operator (`==`)
✅ Overloaded stream insertion (`<<`)
✅ Understood member vs non-member overloading

## Key Learnings
- Operators are just functions with special names
- `operator<<` must be a non-member (friend)
- Keep operator semantics intuitive (don't make `+` do subtraction)

## Next Steps
Proceed to **Lab 7.9: Composition** to build complex objects.
