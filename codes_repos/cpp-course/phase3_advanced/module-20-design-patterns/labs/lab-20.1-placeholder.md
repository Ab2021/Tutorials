# Lab 20.1: Factory Pattern

## Objective
Implement the Factory design pattern for object creation abstraction.

## Instructions

### Step 1: Simple Factory
Create `factory_pattern.cpp`.

```cpp
#include <iostream>
#include <memory>
#include <string>

// Product interface
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual double area() const = 0;
};

// Concrete products
class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    void draw() const override {
        std::cout << "Drawing Circle\n";
    }
    double area() const override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    void draw() const override {
        std::cout << "Drawing Rectangle\n";
    }
    double area() const override {
        return width * height;
    }
};

// Factory
class ShapeFactory {
public:
    static std::unique_ptr<Shape> createShape(const std::string& type) {
        if (type == "circle") {
            return std::make_unique<Circle>(5.0);
        } else if (type == "rectangle") {
            return std::make_unique<Rectangle>(4.0, 6.0);
        }
        return nullptr;
    }
};
```

### Step 2: Factory Method Pattern
```cpp
// Creator interface
class ShapeCreator {
public:
    virtual ~ShapeCreator() = default;
    virtual std::unique_ptr<Shape> createShape() = 0;
    
    void renderShape() {
        auto shape = createShape();
        shape->draw();
        std::cout << "Area: " << shape->area() << "\n";
    }
};

// Concrete creators
class CircleCreator : public ShapeCreator {
public:
    std::unique_ptr<Shape> createShape() override {
        return std::make_unique<Circle>(5.0);
    }
};
```

## Challenges

### Challenge 1: Parameterized Factory
Add parameters to factory methods.

### Challenge 2: Registration System
Implement self-registering factories.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

// Product hierarchy
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual double area() const = 0;
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    void draw() const override { std::cout << "Circle(r=" << radius << ")\n"; }
    double area() const override { return 3.14159 * radius * radius; }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    void draw() const override { std::cout << "Rectangle(" << width << "x" << height << ")\n"; }
    double area() const override { return width * height; }
};

// Challenge 2: Self-registering factory
class ShapeRegistry {
    using Creator = std::function<std::unique_ptr<Shape>(double, double)>;
    std::unordered_map<std::string, Creator> creators;
    
    ShapeRegistry() = default;
    
public:
    static ShapeRegistry& instance() {
        static ShapeRegistry registry;
        return registry;
    }
    
    void registerShape(const std::string& name, Creator creator) {
        creators[name] = creator;
    }
    
    std::unique_ptr<Shape> create(const std::string& name, double param1, double param2 = 0) {
        auto it = creators.find(name);
        if (it != creators.end()) {
            return it->second(param1, param2);
        }
        return nullptr;
    }
};

// Auto-registration helper
template<typename T>
struct ShapeRegistrar {
    ShapeRegistrar(const std::string& name) {
        ShapeRegistry::instance().registerShape(name,
            [](double p1, double p2) -> std::unique_ptr<Shape> {
                if constexpr (std::is_constructible_v<T, double>) {
                    return std::make_unique<T>(p1);
                } else {
                    return std::make_unique<T>(p1, p2);
                }
            });
    }
};

// Register shapes
static ShapeRegistrar<Circle> circleReg("circle");
static ShapeRegistrar<Rectangle> rectangleReg("rectangle");

int main() {
    auto& registry = ShapeRegistry::instance();
    
    auto circle = registry.create("circle", 5.0);
    auto rect = registry.create("rectangle", 4.0, 6.0);
    
    circle->draw();
    std::cout << "Area: " << circle->area() << "\n";
    
    rect->draw();
    std::cout << "Area: " << rect->area() << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented simple factory
✅ Created factory method pattern
✅ Added parameterized factory (Challenge 1)
✅ Built registration system (Challenge 2)

## Key Learnings
- Factory pattern abstracts object creation
- Factory method delegates to subclasses
- Registration enables extensibility
- Smart pointers manage ownership

## Next Steps
Proceed to **Lab 20.2: Abstract Factory**.
