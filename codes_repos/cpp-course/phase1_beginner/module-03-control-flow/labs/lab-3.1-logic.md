# Lab 3.1: If/Else Logic Puzzles

## Objective
Master conditional logic using `if`, `else if`, and `else` statements.

## Instructions

### Step 1: Number Classifier
Create `logic.cpp`. Write a program that takes an integer input and prints:
- "Positive" if > 0
- "Negative" if < 0
- "Zero" if == 0

```cpp
#include <iostream>

int main() {
    int num;
    std::cout << "Enter a number: ";
    std::cin >> num;
    
    // TODO: Implement logic
    
    return 0;
}
```

### Step 2: Even or Odd
Extend the program to also check if the number is even or odd (only for non-zero numbers).

```cpp
if (num != 0) {
    if (num % 2 == 0) {
        std::cout << "Even" << std::endl;
    } else {
        std::cout << "Odd" << std::endl;
    }
}
```

### Step 3: Grade Calculator
Ask for a score (0-100) and print the grade:
- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- < 60: F

## Challenges

### Challenge 1: FizzBuzz
Write a loop from 1 to 20. For each number:
- If divisible by 3, print "Fizz"
- If divisible by 5, print "Buzz"
- If divisible by both, print "FizzBuzz"
- Otherwise, print the number

### Challenge 2: C++17 Init-Statement
Rewrite the even/odd check using C++17 `if` with initializer:
```cpp
if (int remainder = num % 2; remainder == 0) {
    // ...
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Challenge 1: FizzBuzz
    for (int i = 1; i <= 20; ++i) {
        if (i % 3 == 0 && i % 5 == 0) {
            std::cout << "FizzBuzz" << std::endl;
        } else if (i % 3 == 0) {
            std::cout << "Fizz" << std::endl;
        } else if (i % 5 == 0) {
            std::cout << "Buzz" << std::endl;
        } else {
            std::cout << i << std::endl;
        }
    }
    
    // Challenge 2: C++17 Init
    int num = 10;
    if (int r = num % 2; r == 0) {
        std::cout << num << " is Even" << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented basic if/else logic
✅ Handled nested conditions
✅ Implemented FizzBuzz correctly
✅ Used C++17 if-initializer

## Key Learnings
- Conditional branching
- Modulo operator for divisibility
- Scope of variables in C++17 if statements

## Next Steps
Proceed to **Lab 3.2: Switch Statement** to handle multiple choices.
