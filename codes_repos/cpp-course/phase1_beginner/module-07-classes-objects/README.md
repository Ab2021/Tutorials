# Module 7: Classes and Objects

## 🎯 Learning Objectives

By the end of this module, you will:
- Define classes and instantiate objects
- Master access specifiers (`public`, `private`, `protected`)
- Use constructors and member initializer lists
- Implement encapsulation with getters and setters
- Understand `const` correctness for member functions
- Use `static` members and `friend` functions
- Overload operators for custom types

---

## 📖 Theoretical Concepts

### 7.1 Classes vs Structs

In C++, `class` and `struct` are almost identical.
- **Struct:** Members are `public` by default.
- **Class:** Members are `private` by default.

```cpp
class Player {
private:
    int health;
public:
    void heal() { health = 100; }
};
```

### 7.2 Constructors

Initialize objects.
**Member Initializer List:** Preferred way to initialize.

```cpp
class Point {
    int x, y;
public:
    Point(int xVal, int yVal) : x(xVal), y(yVal) {} // Initializer list
};
```

### 7.3 Const Member Functions

Functions that do not modify the object state.

```cpp
int getX() const { return x; } // Safe to call on const objects
```

### 7.4 Static Members

Shared across all instances of the class.

```cpp
class Server {
    static int connectionCount; // Declaration
};
int Server::connectionCount = 0; // Definition
```

### 7.5 Operator Overloading

Give meaning to `+`, `-`, `==`, etc., for your types.

```cpp
Point operator+(const Point& other) {
    return Point(x + other.x, y + other.y);
}
```

---

## 🦀 Rust vs C++ Comparison

### Class Definition
**C++:**
```cpp
class Point {
    int x, y;
public:
    Point(int x, int y) : x(x), y(y) {}
    void move(int dx, int dy) { x += dx; y += dy; }
};
```

**Rust:**
Separates data (`struct`) from behavior (`impl`).
```rust
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self { Point { x, y } }
    fn move_by(&mut self, dx: i32, dy: i32) {
        self.x += dx;
        self.y += dy;
    }
}
```

### Inheritance
**C++:** Supports implementation inheritance (`class Dog : public Animal`).
**Rust:** No implementation inheritance. Uses Traits for shared behavior.

### Destructors
**C++:** `~ClassName()`.
**Rust:** `impl Drop for ClassName`.

---

## 🔑 Key Takeaways

1. Use `class` for invariants/encapsulation, `struct` for plain data.
2. Always use Member Initializer Lists for constructors.
3. Mark read-only methods as `const`.
4. Keep data `private` and provide accessors (`public`).
5. Operator overloading makes custom types intuitive but can be abused.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 7.1:** Basic Class Definition
2. **Lab 7.2:** Access Specifiers (public/private)
3. **Lab 7.3:** Constructors and Initializer Lists
4. **Lab 7.4:** Encapsulation (Getters/Setters)
5. **Lab 7.5:** Const Member Functions
6. **Lab 7.6:** Static Members
7. **Lab 7.7:** Friend Functions and Classes
8. **Lab 7.8:** Operator Overloading (Intro)
9. **Lab 7.9:** Composition (Objects in Objects)
10. **Lab 7.10:** Building a Bank Account System

After completing the labs, move on to **Module 8: Inheritance and Polymorphism**.
