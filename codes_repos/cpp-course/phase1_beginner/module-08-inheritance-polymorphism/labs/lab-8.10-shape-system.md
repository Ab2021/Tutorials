# Lab 8.10: Shape Hierarchy System (Capstone)

## Objective
Build a drawing system using Abstract Base Classes, Polymorphism, and Virtual Destructors.

## Instructions

### Step 1: Abstract Shape
Create `shapes.cpp`.
- `Shape` class with pure virtual `draw()` and `area()`.
- Virtual destructor.

```cpp
#include <iostream>
#include <vector>
#include <cmath>

class Shape {
public:
    virtual void draw() const = 0;
    virtual double area() const = 0;
    virtual ~Shape() {}
};
```

### Step 2: Concrete Shapes
Implement `Circle` (radius) and `Rectangle` (width, height).

```cpp
class Circle : public Shape {
    double r;
public:
    Circle(double radius) : r(radius) {}
    void draw() const override { std::cout << "Circle(r=" << r << ")\n"; }
    double area() const override { return 3.14159 * r * r; }
};

class Rectangle : public Shape {
    double w, h;
public:
    Rectangle(double width, double height) : w(width), h(height) {}
    void draw() const override { std::cout << "Rect(w=" << w << ", h=" << h << ")\n"; }
    double area() const override { return w * h; }
};
```

### Step 3: Canvas Class
Manages a list of shapes.

```cpp
class Canvas {
    std::vector<Shape*> shapes;
public:
    ~Canvas() {
        for (auto s : shapes) delete s;
    }
    
    void addShape(Shape* s) {
        shapes.push_back(s);
    }
    
    void render() const {
        for (const auto* s : shapes) s->draw();
    }
    
    double totalArea() const {
        double sum = 0;
        for (const auto* s : shapes) sum += s->area();
        return sum;
    }
};
```

## Challenges

### Challenge 1: Triangle Class
Add a `Triangle` class (base, height).

### Challenge 2: Move Semantics for Canvas
Implement `Canvas(Canvas&&)` to allow moving the canvas without copying all shapes.
(Remember to nullify the source vector or its pointers).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class Shape {
public:
    virtual void draw() const = 0;
    virtual double area() const = 0;
    virtual ~Shape() {}
};

class Circle : public Shape {
    double r;
public:
    Circle(double radius) : r(radius) {}
    void draw() const override { std::cout << "Circle " << r << "\n"; }
    double area() const override { return 3.14 * r * r; }
};

class Canvas {
    std::vector<Shape*> shapes;
public:
    ~Canvas() { clear(); }
    
    void add(Shape* s) { shapes.push_back(s); }
    
    void clear() {
        for(auto s : shapes) delete s;
        shapes.clear();
    }
    
    // Move Ctor
    Canvas(Canvas&& other) noexcept : shapes(std::move(other.shapes)) {
        other.shapes.clear(); // Vector move usually clears source, but good to be sure
    }
    
    void render() {
        for(auto s : shapes) {
            s->draw();
            std::cout << "Area: " << s->area() << "\n";
        }
    }
};

int main() {
    Canvas c;
    c.add(new Circle(5));
    c.render();
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented Abstract Base Class
✅ Implemented Polymorphism
✅ Managed memory with Virtual Destructor
✅ Implemented container for polymorphic objects
✅ Added new shape type (Challenge 1)

## Key Learnings
- Polymorphism allows treating different objects uniformly
- Virtual destructors are essential for cleanup
- Containers of pointers (`vector<Shape*>`) are the standard way to store polymorphic objects

## Next Steps
Congratulations! You've completed Module 8 and Phase 1 (Beginner).

You are now ready to move to **Phase 2: Intermediate C++**, starting with **Module 9: Templates and Generics**.
