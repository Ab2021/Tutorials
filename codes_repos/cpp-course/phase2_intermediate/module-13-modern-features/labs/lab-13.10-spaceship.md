# Lab 13.10: Spaceship Operator `<=>` (C++20)

## Objective
Use the three-way comparison operator to generate all comparison operators automatically.

## Instructions

### Step 1: The Old Way
Create `spaceship.cpp`.
Implementing all comparison operators is tedious.

```cpp
struct Point {
    int x, y;
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Point& other) const { return !(*this == other); }
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    bool operator<=(const Point& other) const { return !(other < *this); }
    bool operator>(const Point& other) const { return other < *this; }
    bool operator>=(const Point& other) const { return !(*this < other); }
};
```

### Step 2: The Spaceship Operator
One operator generates all six!

```cpp
#include <compare>

struct Point2 {
    int x, y;
    
    auto operator<=>(const Point2&) const = default;
};
```

### Step 3: Custom Ordering
```cpp
struct Person {
    std::string name;
    int age;
    
    auto operator<=>(const Person& other) const {
        // Compare by age first, then name
        if (auto cmp = age <=> other.age; cmp != 0) return cmp;
        return name <=> other.name;
    }
};
```

## Challenges

### Challenge 1: Return Type
The spaceship operator returns `std::strong_ordering`, `std::weak_ordering`, or `std::partial_ordering`.
Understand the difference:
- `strong_ordering`: Total order (int, string)
- `weak_ordering`: Equivalent values exist (case-insensitive strings)
- `partial_ordering`: Not all values are comparable (floats with NaN)

### Challenge 2: Equality
`<=>` does NOT generate `==` automatically. You must add:
```cpp
bool operator==(const Point2&) const = default;
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <compare>
#include <string>

struct Point {
    int x, y;
    
    auto operator<=>(const Point&) const = default;
    bool operator==(const Point&) const = default;
};

struct Person {
    std::string name;
    int age;
    
    auto operator<=>(const Person& other) const {
        if (auto cmp = age <=> other.age; cmp != 0) return cmp;
        return name <=> other.name;
    }
    
    bool operator==(const Person&) const = default;
};

int main() {
    Point p1{1, 2}, p2{1, 3};
    std::cout << "p1 < p2: " << (p1 < p2) << "\n";
    std::cout << "p1 == p2: " << (p1 == p2) << "\n";
    
    Person alice{"Alice", 30}, bob{"Bob", 25};
    std::cout << "alice > bob: " << (alice > bob) << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `operator<=>` with `= default`
✅ Implemented custom ordering
✅ Understood ordering categories (Challenge 1)
✅ Added `operator==` (Challenge 2)

## Key Learnings
- Spaceship operator drastically reduces boilerplate
- `= default` generates lexicographic comparison
- Must still define `operator==` separately

## Next Steps
Congratulations! You've completed Module 13.

Proceed to **Module 14: Lambda Expressions (Deep Dive)** to master functional programming in C++.
