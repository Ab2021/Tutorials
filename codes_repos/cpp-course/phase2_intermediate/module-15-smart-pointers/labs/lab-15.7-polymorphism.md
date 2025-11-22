# Lab 15.7: Smart Pointers and Polymorphism

## Objective
Use smart pointers with inheritance and virtual functions.

## Instructions

### Step 1: Polymorphic unique_ptr
Create `polymorphism.cpp`.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
};

class Circle : public Shape {
public:
    void draw() const override { std::cout << "Circle\n"; }
};

class Square : public Shape {
public:
    void draw() const override { std::cout << "Square\n"; }
};

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Square>());
    
    for (const auto& shape : shapes) {
        shape->draw();
    }
    
    return 0;
}
```

### Step 2: Factory Pattern
```cpp
std::unique_ptr<Shape> createShape(std::string type) {
    if (type == "circle") return std::make_unique<Circle>();
    if (type == "square") return std::make_unique<Square>();
    return nullptr;
}
```

### Step 3: Shared Ownership
```cpp
std::shared_ptr<Shape> s1 = std::make_shared<Circle>();
std::shared_ptr<Shape> s2 = s1; // Polymorphic shared ownership
```

## Challenges

### Challenge 1: Clone Pattern
Implement a `clone()` method that returns `unique_ptr<Shape>`.
```cpp
virtual std::unique_ptr<Shape> clone() const = 0;
```

### Challenge 2: Plugin System
Create a plugin manager that loads plugins (derived classes) and stores them in a container of smart pointers.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual std::unique_ptr<Shape> clone() const = 0;
};

class Circle : public Shape {
    int radius;
public:
    Circle(int r = 1) : radius(r) {}
    
    void draw() const override {
        std::cout << "Circle (r=" << radius << ")\n";
    }
    
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(*this);
    }
};

class Square : public Shape {
    int side;
public:
    Square(int s = 1) : side(s) {}
    
    void draw() const override {
        std::cout << "Square (s=" << side << ")\n";
    }
    
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Square>(*this);
    }
};

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5));
    shapes.push_back(std::make_unique<Square>(10));
    
    // Clone all shapes
    std::vector<std::unique_ptr<Shape>> clones;
    for (const auto& shape : shapes) {
        clones.push_back(shape->clone());
    }
    
    std::cout << "Clones:\n";
    for (const auto& shape : clones) {
        shape->draw();
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used smart pointers with polymorphism
✅ Implemented factory pattern
✅ Implemented clone pattern (Challenge 1)

## Key Learnings
- Smart pointers work seamlessly with polymorphism
- Virtual destructor is essential
- Factory pattern returns smart pointers

## Next Steps
Proceed to **Lab 15.8: Performance Comparison**.
