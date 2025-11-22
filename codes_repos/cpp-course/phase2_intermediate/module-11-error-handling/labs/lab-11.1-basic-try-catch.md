# Lab 11.1: Basic Try-Catch

## Objective
Learn the fundamental syntax of exception handling in C++.

## Instructions

### Step 1: The Risky Function
Create `basic_exception.cpp`.
Write a function `divide(int a, int b)` that throws an integer `-1` if `b` is zero.

```cpp
#include <iostream>

double divide(int a, int b) {
    if (b == 0) {
        throw -1; // Throwing an int (Not recommended, but possible)
    }
    return static_cast<double>(a) / b;
}
```

### Step 2: Handling the Error
Wrap the call in a `try-catch` block.

```cpp
int main() {
    try {
        std::cout << divide(10, 2) << std::endl;
        std::cout << divide(10, 0) << std::endl; // Throws
        std::cout << "This line will not run" << std::endl;
    } catch (int e) {
        std::cout << "Caught error code: " << e << std::endl;
    }
    
    std::cout << "Program continues..." << std::endl;
    return 0;
}
```

### Step 3: Unhandled Exception
Remove the `try-catch` block and run it.
Observe the crash (program termination) when `divide(10, 0)` is called.

## Challenges

### Challenge 1: Throwing Strings
Modify `divide` to `throw "Division by zero"`.
Update `catch` to catch `const char* msg`.

### Challenge 2: Catch All
Add a `catch (...)` block. This catches *any* exception type.
It's useful as a last resort.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

double divide(int a, int b) {
    if (b == 0) throw "Division by zero!";
    return (double)a / b;
}

int main() {
    try {
        divide(10, 0);
    } catch (const char* msg) {
        std::cout << "Error: " << msg << std::endl;
    } catch (...) {
        std::cout << "Unknown error occurred" << std::endl;
    }
    return 0;
}
```
</details>

## Success Criteria
✅ Threw a primitive type (int)
✅ Caught the exception
✅ Observed program flow interruption
✅ Threw and caught a string (Challenge 1)
✅ Used catch-all `...` (Challenge 2)

## Key Learnings
- `throw` interrupts execution immediately
- `catch` handles the error and resumes execution *after* the catch block
- Unhandled exceptions terminate the program

## Next Steps
Proceed to **Lab 11.2: Multiple Catch Blocks** to handle specific errors.
