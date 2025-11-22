# Lab 3.9: Goto (and why to avoid it)

## Objective
Understand the `goto` statement, its valid use cases (rare), and why it's generally avoided.

## Instructions

### Step 1: The Spaghetti Code
Create `goto_demo.cpp`. Use `goto` to create a loop (don't do this in real code!).

```cpp
#include <iostream>

int main() {
    int i = 0;
    
start_loop:
    if (i >= 5) goto end_loop;
    std::cout << i << " ";
    i++;
    goto start_loop;

end_loop:
    std::cout << "\nDone" << std::endl;
    
    return 0;
}
```

### Step 2: A Valid Use Case?
Breaking out of deeply nested loops is one of the few accepted uses of `goto` in C++ (though flags or lambdas are often preferred).

```cpp
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            for (int z = 0; z < 10; ++z) {
                if (x + y + z == 15) {
                    std::cout << "Found it!" << std::endl;
                    goto found; // Jumps all the way out
                }
            }
        }
    }
    std::cout << "Not found." << std::endl;
    return 0;

found:
    std::cout << "Continuing..." << std::endl;
```

## Challenges

### Challenge 1: Rewrite without Goto
Rewrite Step 2 without using `goto`. Use a boolean flag or put the loops in a function and use `return`.

### Challenge 2: Infinite Goto
What happens if you forget the condition in Step 1?
`start: std::cout << "Hi"; goto start;`
Run it and use `Ctrl+C` to stop.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

// Challenge 1: Using function + return
void findTarget() {
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            for (int z = 0; z < 10; ++z) {
                if (x + y + z == 15) {
                    std::cout << "Found it!" << std::endl;
                    return; // Clean exit
                }
            }
        }
    }
}

int main() {
    findTarget();
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented a loop with `goto`
✅ Used `goto` to break nested loops
✅ Refactored `goto` code to use functions/return (Challenge 1)

## Key Learnings
- `goto` jumps to a label
- It can make code hard to follow ("spaghetti code")
- Valid use cases are extremely rare (mostly error handling in C, or deep breaks)
- Functions and `return` are usually better

## Next Steps
Proceed to **Lab 3.10: Text Adventure Game** to combine everything you've learned!
