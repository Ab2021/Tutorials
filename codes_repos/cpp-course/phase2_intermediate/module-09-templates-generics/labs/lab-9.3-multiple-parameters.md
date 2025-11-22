# Lab 9.3: Multiple Template Parameters

## Objective
Use multiple type parameters in templates.

## Instructions

### Step 1: Key-Value Pair
Create `multi_param.cpp`. Define a `Dictionary` entry.

```cpp
#include <iostream>
#include <string>

template <typename KeyType, typename ValueType>
class Entry {
    KeyType key;
    ValueType value;
public:
    Entry(KeyType k, ValueType v) : key(k), value(v) {}
    
    void print() {
        std::cout << key << ": " << value << std::endl;
    }
};
```

### Step 2: Usage
Mix different types.

```cpp
int main() {
    Entry<std::string, int> age("Alice", 30);
    Entry<int, std::string> id(101, "Bob");
    
    age.print();
    id.print();
    
    return 0;
}
```

### Step 3: Default Template Arguments
You can provide defaults!

```cpp
template <typename T = int>
class Container {
    T data;
};
// Container<> c; // T is int
```

## Challenges

### Challenge 1: Triple
Create a `Triple<T1, T2, T3>` class holding three values.

### Challenge 2: Function Return Type
Write a function `auto add(T1 a, T2 b)` that returns the sum.
What is the return type? `decltype(a + b)` or just `auto` (C++14).

```cpp
template <typename T1, typename T2>
auto add(T1 a, T2 b) {
    return a + b;
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

template <typename T1, typename T2, typename T3>
struct Triple {
    T1 first;
    T2 second;
    T3 third;
    
    void show() {
        std::cout << first << ", " << second << ", " << third << std::endl;
    }
};

template <typename T1, typename T2>
auto add(T1 a, T2 b) {
    return a + b;
}

int main() {
    Triple<int, double, char> t = {1, 3.14, 'A'};
    t.show();
    
    std::cout << add(5, 2.5) << std::endl; // 7.5
    return 0;
}
```
</details>

## Success Criteria
✅ Used multiple template parameters
✅ Instantiated with mixed types
✅ Used default template arguments
✅ Implemented mixed-type arithmetic (Challenge 2)

## Key Learnings
- Templates can take any number of type arguments
- `auto` return type deduction is powerful for mixed-type math
- Default arguments simplify usage

## Next Steps
Proceed to **Lab 9.4: Template Specialization** to handle edge cases.
