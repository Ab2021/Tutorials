# Module 20: Design Patterns

## Overview
Design patterns are reusable solutions to common software design problems. This module covers classic Gang of Four patterns and modern C++ implementations.

## Learning Objectives
By the end of this module, you will be able to:
- Implement creational design patterns
- Apply structural design patterns
- Use behavioral design patterns
- Adapt patterns for modern C++
- Choose appropriate patterns for problems
- Avoid pattern anti-patterns

## Key Concepts

### 1. Creational Patterns
Patterns for object creation mechanisms.

**Factory Pattern:**
```cpp
class ShapeFactory {
public:
    static std::unique_ptr<Shape> create(const std::string& type) {
        if (type == "circle") return std::make_unique<Circle>();
        if (type == "square") return std::make_unique<Square>();
        return nullptr;
    }
};
```

**Builder Pattern:**
```cpp
class CarBuilder {
    Car car;
public:
    CarBuilder& setEngine(Engine e) { car.engine = e; return *this; }
    CarBuilder& setWheels(int w) { car.wheels = w; return *this; }
    Car build() { return std::move(car); }
};
```

**Singleton Pattern:**
```cpp
class Singleton {
    static Singleton instance;
    Singleton() = default;
public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }
};
```

### 2. Structural Patterns
Patterns for composing classes and objects.

**Adapter Pattern:**
```cpp
class Adapter : public Target {
    Adaptee adaptee;
public:
    void request() override {
        adaptee.specificRequest();
    }
};
```

**Decorator Pattern:**
```cpp
class Decorator : public Component {
    std::unique_ptr<Component> component;
public:
    void operation() override {
        component->operation();
        // Add behavior
    }
};
```

### 3. Behavioral Patterns
Patterns for communication between objects.

**Observer Pattern:**
```cpp
class Subject {
    std::vector<Observer*> observers;
public:
    void attach(Observer* obs) { observers.push_back(obs); }
    void notify() {
        for (auto* obs : observers) obs->update();
    }
};
```

**Strategy Pattern:**
```cpp
class Context {
    std::unique_ptr<Strategy> strategy;
public:
    void setStrategy(std::unique_ptr<Strategy> s) {
        strategy = std::move(s);
    }
    void execute() { strategy->algorithm(); }
};
```

## Rust Comparison

### Factory Pattern
**C++:**
```cpp
auto shape = ShapeFactory::create("circle");
```

**Rust:**
```rust
let shape = ShapeFactory::create("circle");
// Rust uses enums for variants instead
```

### Observer Pattern
**C++:**
```cpp
subject.attach(&observer);
subject.notify();
```

**Rust:**
```rust
// Rust uses channels for observer pattern
let (tx, rx) = mpsc::channel();
```

## Labs

1. **Lab 20.1**: Factory Pattern
2. **Lab 20.2**: Abstract Factory
3. **Lab 20.3**: Builder Pattern
4. **Lab 20.4**: Singleton Pattern
5. **Lab 20.5**: Adapter Pattern
6. **Lab 20.6**: Decorator Pattern
7. **Lab 20.7**: Observer Pattern
8. **Lab 20.8**: Strategy Pattern
9. **Lab 20.9**: Command Pattern
10. **Lab 20.10**: Pattern Combination (Capstone)

## Additional Resources
- "Design Patterns" by Gang of Four
- "Modern C++ Design" by Andrei Alexandrescu
- refactoring.guru/design-patterns

## Next Module
After completing this module, proceed to **Module 21: Memory Management**.
