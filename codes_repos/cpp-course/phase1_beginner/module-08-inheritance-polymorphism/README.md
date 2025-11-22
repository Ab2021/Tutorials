# Module 8: Inheritance and Polymorphism

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand how to derive classes from base classes
- Master access control in inheritance (`public`, `protected`, `private`)
- Implement Polymorphism using `virtual` functions
- Create Abstract Base Classes with Pure Virtual Functions
- Understand the importance of Virtual Destructors
- Avoid common pitfalls like Object Slicing and the Diamond Problem

---

## 📖 Theoretical Concepts

### 8.1 Inheritance

Creating a new class based on an existing one.

```cpp
class Animal {
public:
    void eat() { std::cout << "Eating...\n"; }
};

class Dog : public Animal { // Inherits from Animal
public:
    void bark() { std::cout << "Woof!\n"; }
};
```

### 8.2 Access Specifiers

- **Public Inheritance:** `public` -> `public`, `protected` -> `protected`.
- **Protected Inheritance:** `public` -> `protected`.
- **Private Inheritance:** `public` -> `private`.

### 8.3 Polymorphism

Treating derived objects as base objects. Requires `virtual` functions.

```cpp
class Base {
public:
    virtual void show() { std::cout << "Base\n"; }
};

class Derived : public Base {
public:
    void show() override { std::cout << "Derived\n"; }
};

Base* b = new Derived();
b->show(); // Prints "Derived" (Dynamic Dispatch)
```

### 8.4 Abstract Classes

Classes that cannot be instantiated. Contain at least one Pure Virtual Function.

```cpp
class Shape {
public:
    virtual void draw() = 0; // Pure Virtual
};
```

### 8.5 Virtual Destructor

**CRITICAL:** If a class has virtual functions, it MUST have a virtual destructor.

```cpp
virtual ~Base() {}
```

---

## 🦀 Rust vs C++ Comparison

### Inheritance
**C++:** Supports implementation inheritance (`class Dog : public Animal`).
**Rust:** Does NOT support implementation inheritance. It uses **Traits** for interface inheritance and **Composition** for code reuse.

### Polymorphism
**C++:** Runtime polymorphism via `virtual` functions and vtables.
**Rust:** Runtime polymorphism via **Trait Objects** (`Box<dyn Trait>`).

### Abstract Classes
**C++:** Abstract Base Classes (ABCs).
**Rust:** Traits with no default implementation.

---

## 🔑 Key Takeaways

1. Use `public` inheritance for "Is-A" relationships.
2. Mark overridden functions with `override` keyword.
3. Always make destructors `virtual` in base classes.
4. Prefer Composition ("Has-A") over Inheritance when possible.
5. Use Abstract Classes to define interfaces.

---

## ⏭️ Next Steps

Complete the labs in the `labs/` directory:

1. **Lab 8.1:** Basic Inheritance
2. **Lab 8.2:** Access Specifiers in Inheritance
3. **Lab 8.3:** Constructor/Destructor Order
4. **Lab 8.4:** Method Overriding
5. **Lab 8.5:** Virtual Functions and Polymorphism
6. **Lab 8.6:** Abstract Classes
7. **Lab 8.7:** Virtual Destructors
8. **Lab 8.8:** Multiple Inheritance
9. **Lab 8.9:** Object Slicing
10. **Lab 8.10:** Shape Hierarchy System

After completing the labs, move on to **Module 9: Templates and Generics**.
