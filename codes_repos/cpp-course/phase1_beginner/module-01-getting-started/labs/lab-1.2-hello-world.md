# Lab 1.2: Hello World Variations

## Objective
Write, compile, and run your first C++ program and explore different output streams.

## Instructions

### Step 1: Create the Source File

Create a new file named `hello.cpp` in your workspace.

### Step 2: Starter Code

Copy the following code into `hello.cpp`:

```cpp
#include <iostream>

int main() {
    // TODO: Print "Hello, C++!" to standard output
    
    // TODO: Print "This is an error message." to standard error
    
    return 0;
}
```

### Step 3: Your Task

1. Use `std::cout` to print "Hello, C++!" followed by a newline.
2. Use `std::cerr` to print "This is an error message." followed by a newline.
3. Compile the program using your compiler.
4. Run the executable.

**Compilation Command:**
```bash
g++ hello.cpp -o hello
# OR
clang++ hello.cpp -o hello
# OR (Windows MSVC)
cl /EHsc hello.cpp
```

**Run Command:**
```bash
./hello      # Linux/macOS
hello.exe    # Windows
```

### Hints
- `std::cout` is for standard output (normal text).
- `std::cerr` is for error messages (often unbuffered).
- `std::endl` inserts a newline and flushes the stream.
- `\n` can also be used for a newline.

### Example Output
```
Hello, C++!
This is an error message.
```

## Challenges

### Challenge 1: Multiple Lines
Modify the program to print a ASCII art shape (like a triangle or square) using multiple `std::cout` statements.

### Challenge 2: Return Values
Change `return 0;` to `return 1;`. Compile and run.
- On Linux/macOS, check the exit code with `echo $?`.
- On Windows, check with `echo %errorlevel%`.
- What does a non-zero return value usually mean?

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

int main() {
    // Standard output
    std::cout << "Hello, C++!" << std::endl;
    
    // Standard error
    std::cerr << "This is an error message." << std::endl;
    
    // Challenge 1: ASCII Art
    std::cout << "  *  " << std::endl;
    std::cout << " *** " << std::endl;
    std::cout << "*****" << std::endl;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Program compiles without errors
✅ "Hello, C++!" prints to stdout
✅ Error message prints to stderr
✅ Executable runs successfully

## Key Learnings
- Basic structure of a C++ program (`main` function)
- Difference between `std::cout` and `std::cerr`
- How to compile and run a simple program
- Return codes from `main`

## Next Steps
Proceed to **Lab 1.3: Building a Simple Calculator** to work with variables and input.
