# Lab 8.6: Abstract Classes and Pure Virtual Functions

## Objective
Create interfaces using Abstract Base Classes (ABCs).

## Instructions

### Step 1: Pure Virtual Function
Create `abstract.cpp`.

```cpp
#include <iostream>

class Shape {
public:
    virtual void draw() = 0; // Pure Virtual
    virtual double area() = 0;
};
```

### Step 2: Concrete Classes
Implement `Circle` and `Rectangle`.

```cpp
class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    void draw() override { std::cout << "Drawing Circle\n"; }
    double area() override { return 3.14 * radius * radius; }
};

class Rectangle : public Shape {
    double w, h;
public:
    Rectangle(double width, double height) : w(width), h(height) {}
    void draw() override { std::cout << "Drawing Rectangle\n"; }
    double area() override { return w * h; }
};
```

### Step 3: Usage
```cpp
int main() {
    // Shape s; // Error: Abstract class
    Shape* s1 = new Circle(5);
    Shape* s2 = new Rectangle(4, 5);
    
    s1->draw();
    std::cout << "Area: " << s1->area() << std::endl;
    
    delete s1;
    delete s2;
    return 0;
}
```

## Challenges

### Challenge 1: Partial Implementation
Create a class `Polygon` inheriting from `Shape` that implements `draw` but NOT `area`.
Try to instantiate `Polygon`. (It should still be abstract).

### Challenge 2: Interface Class
Create a class `IDrawable` with only `draw()`.
Make `Shape` inherit from `IDrawable`.
This mimics "Interface" inheritance common in Java/C#.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class IDrawable {
public:
    virtual void draw() = 0;
    virtual ~IDrawable() {}
};

class Shape : public IDrawable {
public:
    virtual double area() = 0;
};

class Circle : public Shape {
    double r;
public:
    Circle(double radius) : r(radius) {}
    void draw() override { std::cout << "O\n"; }
    double area() override { return 3.14 * r * r; }
};

int main() {
    IDrawable* d = new Circle(10);
    d->draw();
    delete d;
    return 0;
}
```
</details>

## Success Criteria
✅ Defined pure virtual functions
✅ Implemented concrete classes
✅ Verified abstract classes cannot be instantiated
✅ Created interface-style hierarchy (Challenge 2)

## Key Learnings
- `= 0` makes a function pure virtual
- A class with ANY pure virtual function is Abstract
- Derived classes must implement ALL pure virtuals to be concrete

## Next Steps
Proceed to **Lab 8.7: Virtual Destructors** to avoid leaks.
