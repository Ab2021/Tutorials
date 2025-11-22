# Lab 1.3: Building a Simple Calculator

## Objective
Create a simple calculator to practice basic I/O (Input/Output), variables, and arithmetic operations.

## Instructions

### Step 1: Create the Project
Create a file named `calculator.cpp`.

### Step 2: Starter Code

```cpp
#include <iostream>

int main() {
    std::cout << "=== Simple Calculator ===" << std::endl;
    
    // TODO: Declare variables for two numbers
    
    // TODO: Ask user for input
    
    // TODO: Perform arithmetic operations (+, -, *, /)
    
    // TODO: Print results
    
    return 0;
}
```

### Step 3: Your Task
1. Declare two `double` variables to store user input.
2. Use `std::cin` to get two numbers from the user.
3. Calculate sum, difference, product, and quotient.
4. Print the results with clear labels.

**Example Interaction:**
```
=== Simple Calculator ===
Enter first number: 10
Enter second number: 5

Results:
Addition: 15
Subtraction: 5
Multiplication: 50
Division: 2
```

### Hints
- Use `double` instead of `int` to handle decimal numbers.
- `std::cin >> variable;` reads input into a variable.
- Be careful with division by zero!

## Challenges

### Challenge 1: Division by Zero Check
Add an `if` statement to check if the second number is 0 before dividing. If it is 0, print "Cannot divide by zero" instead of the result.

### Challenge 2: Modulo Operator
Add a modulo (%) operation. Note: Modulo only works with integers. You'll need to cast your doubles to integers or declare new integer variables.
```cpp
int a = (int)num1;
int b = (int)num2;
int mod = a % b;
```

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    std::cout << "=== Simple Calculator ===" << std::endl;
    
    double num1, num2;
    
    // Input
    std::cout << "Enter first number: ";
    std::cin >> num1;
    
    std::cout << "Enter second number: ";
    std::cin >> num2;
    
    std::cout << "\nResults:" << std::endl;
    
    // Operations
    std::cout << "Addition: " << (num1 + num2) << std::endl;
    std::cout << "Subtraction: " << (num1 - num2) << std::endl;
    std::cout << "Multiplication: " << (num1 * num2) << std::endl;
    
    // Challenge 1: Division check
    if (num2 != 0) {
        std::cout << "Division: " << (num1 / num2) << std::endl;
    } else {
        std::cout << "Division: Cannot divide by zero" << std::endl;
    }
    
    // Challenge 2: Modulo
    int i1 = (int)num1;
    int i2 = (int)num2;
    
    if (i2 != 0) {
        std::cout << "Modulo (int): " << (i1 % i2) << std::endl;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Program accepts user input
✅ Performs basic arithmetic correctly
✅ Output is formatted and readable
✅ Handles division by zero (Challenge 1)

## Key Learnings
- Using `std::cin` for input
- Working with `double` vs `int`
- Basic arithmetic operators
- Simple conditional logic (`if`)

## Next Steps
Proceed to **Lab 1.4: Understanding Compilation Flags** to learn how to control the build process.
