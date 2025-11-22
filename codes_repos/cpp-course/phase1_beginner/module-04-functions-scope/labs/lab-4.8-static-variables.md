# Lab 4.8: Static Variables

## Objective
Learn how to use `static` variables inside functions to preserve state between calls.

## Instructions

### Step 1: The Problem
Create `static_demo.cpp`. We want a function that counts how many times it has been called.

```cpp
#include <iostream>

void counter() {
    int count = 0; // Re-initialized every time!
    count++;
    std::cout << "Count: " << count << std::endl;
}

int main() {
    counter(); // 1
    counter(); // 1 (Oops)
    counter(); // 1 (Oops)
    return 0;
}
```

### Step 2: The Fix (Static)
Change `int count` to `static int count`.

```cpp
void counter() {
    static int count = 0; // Initialized only once
    count++;
    std::cout << "Count: " << count << std::endl;
}
```
*Run it again. It should print 1, 2, 3.*

### Step 3: Scope
Try to access `count` from `main`.
`// count = 10; // Error: count is local to the function`

## Challenges

### Challenge 1: Generator
Create a function `int generateID()` that returns a unique ID starting from 1000 each time it's called.

### Challenge 2: Thread Safety (Thought Experiment)
What happens if two threads call `counter()` at the exact same time?
*C++11 guarantees static local initialization is thread-safe, but modification (count++) is NOT.*

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int generateID() {
    static int id = 1000;
    return id++;
}

int main() {
    std::cout << "User 1 ID: " << generateID() << std::endl; // 1000
    std::cout << "User 2 ID: " << generateID() << std::endl; // 1001
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented static local variable
✅ Verified persistence of value
✅ Verified local scope (cannot access from main)
✅ Created a unique ID generator (Challenge 1)

## Key Learnings
- `static` locals live for the entire program lifetime
- They are initialized only the first time execution passes through
- Useful for counters, caches, or singletons

## Next Steps
Proceed to **Lab 4.9: Recursion** to make functions call themselves.
