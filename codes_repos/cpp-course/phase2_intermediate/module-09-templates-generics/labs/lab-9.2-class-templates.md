# Lab 9.2: Class Templates

## Objective
Create a class that can hold data of any type.

## Instructions

### Step 1: Generic Box
Create `class_templates.cpp`. Define a `Box` class.

```cpp
#include <iostream>

template <typename T>
class Box {
private:
    T content;
public:
    Box(T val) : content(val) {}
    
    T getContent() { return content; }
    void setContent(T val) { content = val; }
};
```

### Step 2: Instantiation
Unlike functions (pre-C++17), classes usually require explicit types.

```cpp
int main() {
    Box<int> intBox(123);
    Box<double> dblBox(3.14);
    
    std::cout << intBox.getContent() << std::endl;
    std::cout << dblBox.getContent() << std::endl;
    
    return 0;
}
```

### Step 3: Template Methods Definition
If defining methods outside the class, syntax is verbose.

```cpp
template <typename T>
void Box<T>::setContent(T val) {
    content = val;
}
```

## Challenges

### Challenge 1: Pair Class
Create a `Pair` class that holds two values of the SAME type `T`.
Methods: `getFirst()`, `getSecond()`, `swap()`.

### Challenge 2: CTAD (C++17)
Class Template Argument Deduction.
Try `Box b(5);` instead of `Box<int> b(5);`.
Does it compile? (Requires C++17 compiler flag).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

template <typename T>
class Pair {
    T first, second;
public:
    Pair(T a, T b) : first(a), second(b) {}
    
    T getFirst() { return first; }
    T getSecond() { return second; }
    
    void swap() {
        T temp = first;
        first = second;
        second = temp;
    }
};

int main() {
    Pair<int> p(1, 2);
    p.swap();
    std::cout << p.getFirst() << std::endl; // 2
    
    // C++17 CTAD
    // Pair p2(3.5, 4.5); // Deduced as Pair<double>
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented class template
✅ Instantiated with different types
✅ Defined template methods outside class
✅ Implemented generic Pair class (Challenge 1)

## Key Learnings
- Class templates allow generic data structures
- Syntax `Class<Type>` is required for instantiation
- Method definitions outside class need `template <typename T>` prefix

## Next Steps
Proceed to **Lab 9.3: Multiple Parameters** to mix types.
