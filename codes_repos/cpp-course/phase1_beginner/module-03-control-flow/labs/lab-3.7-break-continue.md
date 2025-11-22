# Lab 3.7: Break and Continue

## Objective
Learn how to alter the flow of a loop using `break` and `continue`.

## Instructions

### Step 1: Continue
Create `flow_control.cpp`. Print numbers 1 to 10, but skip 5.

```cpp
#include <iostream>

int main() {
    for (int i = 1; i <= 10; ++i) {
        if (i == 5) {
            continue; // Skip rest of this iteration
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

### Step 2: Break
Stop the loop when you reach 8.

```cpp
    for (int i = 1; i <= 10; ++i) {
        if (i == 8) {
            break; // Exit loop immediately
        }
        std::cout << i << " ";
    }
```

### Step 3: While Loop Break
Write an infinite loop `while(true)` that breaks when the user types 'q'.

```cpp
char input;
while (true) {
    std::cout << "Press 'q' to quit: ";
    std::cin >> input;
    if (input == 'q') break;
}
```

## Challenges

### Challenge 1: Prime Finder
Find the first prime number greater than 1000.
- Loop from 1000 upwards.
- Check if prime.
- If prime, print and `break`.

### Challenge 2: Input Filter
Ask user for 5 numbers.
- If number < 0, `continue` (don't count it).
- If number == 0, `break` (stop asking).
- Sum the positive numbers.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

bool isPrime(int n) {
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    // Challenge 1
    for (int i = 1001; ; ++i) {
        if (isPrime(i)) {
            std::cout << "First prime > 1000 is " << i << std::endl;
            break;
        }
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `continue` to skip iterations
✅ Used `break` to exit loops
✅ Implemented infinite loop with break condition
✅ Solved prime finder (Challenge 1)

## Key Learnings
- `break` exits the nearest enclosing loop
- `continue` jumps to the next iteration check
- Infinite loops with `break` are common patterns

## Next Steps
Proceed to **Lab 3.8: Nested Loops** to handle multi-dimensional logic.
