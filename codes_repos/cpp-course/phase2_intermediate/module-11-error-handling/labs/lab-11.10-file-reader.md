# Lab 11.10: Robust File Reader (Capstone)

## Objective
Build a file reading utility that handles all possible errors gracefully using Exceptions, RAII, and Optional.

## Instructions

### Step 1: Custom Exception
Create `file_reader.cpp`.
Define `FileError` inheriting from `std::runtime_error`.

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <optional>

class FileError : public std::runtime_error {
public:
    FileError(const std::string& filename) 
        : std::runtime_error("Failed to open file: " + filename) {}
};
```

### Step 2: The Reader Function
Read lines from a file.
Use `std::ifstream`. Check `is_open()`.

```cpp
std::vector<std::string> readLines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileError(filename);
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}
```

### Step 3: Safe Wrapper
Create a wrapper that returns `std::optional` instead of throwing, for callers who don't want exceptions.

```cpp
std::optional<std::vector<std::string>> tryReadLines(const std::string& filename) {
    try {
        return readLines(filename);
    } catch (const FileError& e) {
        std::cerr << "Log: " << e.what() << "\n";
        return std::nullopt;
    }
}
```

## Challenges

### Challenge 1: Parse Numbers
Add a function `std::vector<int> readNumbers(filename)`.
Read strings, parse to int.
If parsing fails for a line, log a warning but continue (Partial Success).

### Challenge 2: RAII File Handle
Implement a `FileHandle` class that wraps `FILE*` (C-style) and throws in constructor if `fopen` fails, closes in destructor.
(Just to practice RAII + Exceptions).

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <optional>
#include <stdexcept>

// Challenge 2: RAII Wrapper
class FileHandle {
    FILE* file;
public:
    FileHandle(const char* name) {
        file = std::fopen(name, "r");
        if (!file) throw std::runtime_error("fopen failed");
    }
    ~FileHandle() {
        if (file) std::fclose(file);
    }
};

std::vector<std::string> readLines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open " + filename);
    
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(file, line)) lines.push_back(line);
    return lines;
}

int main() {
    // Test Exception Version
    try {
        auto lines = readLines("missing.txt");
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
    
    // Test Optional Version logic (simulated)
    auto result = std::optional<int>{};
    if (!result) std::cout << "Optional returned empty\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented custom file exception
✅ Used `ifstream` with error checking
✅ Created exception-free wrapper using `optional`
✅ Implemented RAII wrapper (Challenge 2)

## Key Learnings
- Exceptions are for error reporting
- RAII ensures cleanup during stack unwinding
- `optional` is great for non-critical failures
- Robust code handles errors at the appropriate level

## Next Steps
Congratulations! You've completed Module 11.

Proceed to **Module 12: File I/O and Streams** to master input/output.
