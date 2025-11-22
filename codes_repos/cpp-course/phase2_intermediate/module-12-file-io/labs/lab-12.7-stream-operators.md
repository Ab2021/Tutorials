# Lab 12.7: Custom Stream Operators

## Objective
Overload `<<` and `>>` to make your classes work with `cout`, `cin`, and files.

## Instructions

### Step 1: Output Operator (<<)
Create `stream_ops.cpp`.
Must be a free function (usually `friend`).

```cpp
#include <iostream>

class Point {
    int x, y;
public:
    Point(int x=0, int y=0) : x(x), y(y) {}
    
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os; // Return stream for chaining
    }
};

int main() {
    Point p(10, 20);
    std::cout << "Point: " << p << std::endl;
    return 0;
}
```

### Step 2: Input Operator (>>)
Extract data.

```cpp
    friend std::istream& operator>>(std::istream& is, Point& p) {
        char c; // consume comma/parentheses
        // Expected format: (x, y) or x y
        is >> p.x >> p.y;
        return is;
    }
```

### Step 3: File Compatibility
Since `ofstream` inherits from `ostream`, your operator works for files automatically!

```cpp
std::ofstream f("point.txt");
f << p;
```

## Challenges

### Challenge 1: Robust Input
Modify `operator>>` to handle the format `(10, 20)`.
It should consume `(`, read int, consume `,`, read int, consume `)`.
Check `is.fail()` if format doesn't match.

### Challenge 2: Vector Output
Overload `<<` for `std::vector<Point>`.
Print `[(1, 2), (3, 4)]`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class Point {
    int x, y;
public:
    Point(int x=0, int y=0) : x(x), y(y) {}
    
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }
    
    friend std::istream& operator>>(std::istream& is, Point& p) {
        char c;
        is >> c; // (
        if (c != '(') { is.setstate(std::ios::failbit); return is; }
        is >> p.x >> c; // ,
        is >> p.y >> c; // )
        return is;
    }
};

// Challenge 2
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i < v.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

int main() {
    std::vector<Point> v = {{1, 2}, {3, 4}};
    std::cout << v << "\n";
    return 0;
}
```
</details>

## Success Criteria
✅ Overloaded `operator<<`
✅ Overloaded `operator>>`
✅ Verified file compatibility
✅ Implemented robust parsing (Challenge 1)

## Key Learnings
- Stream operators allow custom types to behave like built-ins
- Return `ostream&` to allow chaining (`cout << p << endl`)
- Input operators should validate format and set failbit on error

## Next Steps
Proceed to **Lab 12.8: Error Handling** to check stream states.
