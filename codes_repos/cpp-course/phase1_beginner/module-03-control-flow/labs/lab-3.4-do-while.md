# Lab 3.4: Do-While Loop Input Validation

## Objective
Use a `do-while` loop to ensure code executes at least once, perfect for input validation.

## Instructions

### Step 1: Basic Validation
Create `validation.cpp`. Ask the user for a number between 1 and 10. Keep asking until they comply.

```cpp
#include <iostream>

int main() {
    int num;
    
    do {
        std::cout << "Enter a number (1-10): ";
        std::cin >> num;
    } while (num < 1 || num > 10);
    
    std::cout << "You entered: " << num << std::endl;
    
    return 0;
}
```

### Step 2: Error Message
Add an `if` inside the loop to print "Invalid input" if the number is out of range.

```cpp
    do {
        std::cout << "Enter a number (1-10): ";
        std::cin >> num;
        
        if (num < 1 || num > 10) {
            std::cout << "Invalid! Try again." << std::endl;
        }
    } while (num < 1 || num > 10);
```

## Challenges

### Challenge 1: Password Retry
Create a password checker.
- Define `const std::string password = "secret";`
- Ask user for password.
- Allow max 3 attempts.
- Use a `do-while` loop.

### Challenge 2: Input Stream Clearing
What happens if you enter "abc" instead of a number? The loop goes infinite!
Fix it by clearing the error state:
```cpp
if (std::cin.fail()) {
    std::cin.clear(); // Clear error flag
    std::cin.ignore(1000, '\n'); // Discard bad input
}
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>

int main() {
    const std::string correctPass = "secret";
    std::string input;
    int attempts = 0;
    
    do {
        std::cout << "Enter password: ";
        std::cin >> input;
        attempts++;
        
        if (input == correctPass) {
            std::cout << "Access Granted!" << std::endl;
            break;
        } else {
            std::cout << "Wrong password!" << std::endl;
        }
        
    } while (attempts < 3);
    
    if (input != correctPass) {
        std::cout << "Locked out!" << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented `do-while` loop
✅ Validated input range
✅ Handled invalid input types (Challenge 2)
✅ Implemented retry limit (Challenge 1)

## Key Learnings
- `do-while` guarantees at least one execution
- `std::cin` error handling
- Input buffer clearing

## Next Steps
Proceed to **Lab 3.5: For Loop Patterns** to practice iteration.
