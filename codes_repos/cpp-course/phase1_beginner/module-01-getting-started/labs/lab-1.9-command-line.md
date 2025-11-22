# Lab 1.9: Command-line Arguments

## Objective
Learn how to accept and process arguments passed to your program from the command line.

## Instructions

### Step 1: The Main Signature
Create `args_demo.cpp`. Change the `main` function signature to accept arguments:

```cpp
#include <iostream>

int main(int argc, char* argv[]) {
    // argc: Argument Count (includes program name)
    // argv: Argument Vector (array of C-strings)
    
    std::cout << "Number of arguments: " << argc << std::endl;
    
    for (int i = 0; i < argc; ++i) {
        std::cout << "Arg " << i << ": " << argv[i] << std::endl;
    }
    
    return 0;
}
```

### Step 2: Run with Arguments
Compile: `g++ args_demo.cpp -o demo`
Run:
```bash
./demo hello world 123
```

Output should be:
```
Number of arguments: 4
Arg 0: ./demo
Arg 1: hello
Arg 2: world
Arg 3: 123
```

### Step 3: Simple Flag Parser
Modify the program to check for a specific flag, like `--help`.

```cpp
#include <iostream>
#include <string> // Needed for string comparison

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--help") {
            std::cout << "Usage: ./demo [name]" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Running program..." << std::endl;
    return 0;
}
```

## Challenges

### Challenge 1: Greeter
Make a program that takes a name as an argument and prints "Hello, [name]!".
If no name is provided, print "Hello, World!".

### Challenge 2: Sum of Numbers
Make a program that takes two numbers as arguments and prints their sum.
*Hint: Use `std::stoi()` to convert string to integer.*
```cpp
int num = std::stoi(argv[1]);
```

## Solution

<details>
<summary>Click to reveal solution</summary>

**Challenge 2: Sum**
```cpp
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./sum <num1> <num2>" << std::endl;
        return 1;
    }
    
    try {
        int a = std::stoi(argv[1]);
        int b = std::stoi(argv[2]);
        std::cout << "Sum: " << (a + b) << std::endl;
    } catch (...) {
        std::cerr << "Error: Invalid numbers provided." << std::endl;
        return 1;
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Understood `argc` and `argv`
✅ Iterated through arguments
✅ Parsed specific flags
✅ Converted string arguments to numbers

## Key Learnings
- `main` function parameters
- C-style strings in `argv`
- Converting strings to numbers (`std::stoi`)
- Basic input validation

## Next Steps
Proceed to **Lab 1.10: Build Configurations** to master Debug vs Release builds.
