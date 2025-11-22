# Lab 8.2: Access Specifiers in Inheritance

## Objective
Understand `protected` members and different types of inheritance.

## Instructions

### Step 1: Protected Members
Create `access_inheritance.cpp`.

```cpp
#include <iostream>

class Base {
private:
    int priv = 1;
protected:
    int prot = 2;
public:
    int pub = 3;
};

class Derived : public Base {
public:
    void show() {
        // std::cout << priv; // Error: Private not accessible
        std::cout << prot; // OK: Protected accessible in derived
        std::cout << pub;  // OK: Public accessible
    }
};
```

### Step 2: Access from Main
```cpp
int main() {
    Derived d;
    // d.prot; // Error: Protected not accessible from outside
    d.pub; // OK
    d.show();
    return 0;
}
```

### Step 3: Private Inheritance
Change `public Base` to `private Base`.
Now `pub` and `prot` become `private` inside `Derived`.
`d.pub` in main will fail.

## Challenges

### Challenge 1: Protected Inheritance
Change to `protected Base`.
`pub` and `prot` become `protected` in `Derived`.
Verify `d.pub` fails in main, but a class derived from `Derived` can access them.

### Challenge 2: Using Declaration
In Private Inheritance, restore access to specific members.
```cpp
class Derived : private Base {
public:
    using Base::pub; // Make pub public again
};
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class Base {
protected:
    int x = 10;
public:
    int y = 20;
};

class Derived : protected Base {
public:
    void print() {
        std::cout << x << " " << y << std::endl;
    }
};

class GrandChild : public Derived {
public:
    void access() {
        // x is still protected here because Derived used protected inheritance
        // If Derived used private inheritance, x would be inaccessible here
        // std::cout << x; 
    }
};

int main() {
    Derived d;
    // d.y; // Error (protected)
    d.print();
    return 0;
}
```
</details>

## Success Criteria
✅ Accessed `protected` members in derived class
✅ Verified `protected` is private to outsiders
✅ Experimented with Private Inheritance
✅ Restored access with `using` (Challenge 2)

## Key Learnings
- `protected` is for children only
- `public` inheritance is the standard "Is-A"
- `private` inheritance is "Implemented-In-Terms-Of" (Has-A logic)

## Next Steps
Proceed to **Lab 8.3: Constructor Order** to see how objects are built.
