# Lab 3.6: Range-based For Loop

## Objective
Learn the modern C++ way to iterate over collections using range-based for loops.

## Instructions

### Step 1: Basic Iteration
Create `range_loop.cpp`. Iterate over a simple array.

```cpp
#include <iostream>

int main() {
    int scores[] = {85, 90, 78, 92, 88};
    
    for (int score : scores) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### Step 2: Using Auto
Let the compiler deduce the type.

```cpp
for (auto score : scores) {
    // ...
}
```

### Step 3: Modifying Elements
To modify elements, you need a reference (`&`).

```cpp
for (auto& score : scores) {
    score += 5; // Bonus points!
}
```

### Step 4: Const Reference
To avoid copying large objects but prevent modification, use `const auto&`.

```cpp
for (const auto& score : scores) {
    std::cout << score << " ";
}
```

## Challenges

### Challenge 1: String Iteration
Iterate over a `std::string` (which is a collection of chars). Count how many vowels are in "Hello World".

### Challenge 2: Initializer List
Iterate directly over an initializer list without a variable.
```cpp
for (int x : {1, 2, 3, 4, 5}) {
    // ...
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

int main() {
    std::string text = "Hello World";
    int vowels = 0;
    
    for (char c : text) {
        char lower = std::tolower(c);
        if (lower == 'a' || lower == 'e' || lower == 'i' || 
            lower == 'o' || lower == 'u') {
            vowels++;
        }
    }
    
    std::cout << "Vowels: " << vowels << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used range-based for loop
✅ Understood `auto` vs `auto&` vs `const auto&`
✅ Iterated over different container types (array, string)
✅ Modified elements in place

## Key Learnings
- Range-based loops are cleaner and safer
- Use `&` to modify or avoid copies
- Use `const` to prevent accidental modification

## Next Steps
Proceed to **Lab 3.7: Break and Continue** to control loop flow.
