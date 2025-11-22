# Lab 7.7: Friend Functions and Classes

## Objective
Learn how to grant access to private members to external functions or classes.

## Instructions

### Step 1: Friend Function
Create `friends.cpp`.

```cpp
#include <iostream>

class Box {
private:
    int width;
public:
    Box(int w) : width(w) {}
    
    // Grant access to printWidth
    friend void printWidth(Box b);
};

void printWidth(Box b) {
    // Can access private 'width' because it's a friend
    std::cout << "Width: " << b.width << std::endl;
}

int main() {
    Box b(10);
    printWidth(b);
    return 0;
}
```

### Step 2: Friend Class
Create a class `Storage` that needs access to `Box`.

```cpp
class Storage {
public:
    void store(Box& b, int w) {
        b.width = w; // Needs friendship
    }
};

// In Box: friend class Storage;
```

## Challenges

### Challenge 1: Two-way Friendship?
If A is a friend of B, is B a friend of A?
*Answer: No. Friendship is not mutual.*
Verify this by trying to access Storage private members from Box.

### Challenge 2: Operator Overloading (Preview)
`friend` is commonly used for overloading `<<` (cout).
`friend std::ostream& operator<<(std::ostream& os, const Box& b);`
Implement this to print "Box(width)".

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Box {
    int width;
public:
    Box(int w) : width(w) {}
    friend class Storage;
    friend std::ostream& operator<<(std::ostream& os, const Box& b);
};

class Storage {
public:
    void setWidth(Box& b, int w) {
        b.width = w;
    }
};

std::ostream& operator<<(std::ostream& os, const Box& b) {
    os << "Box(" << b.width << ")";
    return os;
}

int main() {
    Box b(10);
    Storage s;
    s.setWidth(b, 20);
    std::cout << b << std::endl;
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented friend function
✅ Implemented friend class
✅ Accessed private members from friend
✅ Overloaded `<<` using friend (Challenge 2)

## Key Learnings
- `friend` breaks encapsulation
- Use sparingly (e.g., for operator overloading or tightly coupled classes)
- Friendship is not inherited, mutual, or transitive

## Next Steps
Proceed to **Lab 7.8: Operator Overloading** to make types intuitive.
