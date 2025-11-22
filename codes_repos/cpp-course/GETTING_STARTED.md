# 🚀 Getting Started with C++

## Welcome to C++!

This guide will help you get started with the C++ programming course and set up your development environment.

---

## Step 1: Install a C++ Compiler

Choose one of the following based on your operating system:

### Windows

#### Option 1: Visual Studio Community (Recommended for beginners)
1. Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/vs/community/)
2. During installation, select "Desktop development with C++"
3. This includes MSVC compiler, debugger, and IDE

#### Option 2: MinGW-w64 (GCC for Windows)
```powershell
# Using MSYS2 (Recommended)
winget install -e --id MSYS2.MSYS2

# Then open MSYS2 terminal and run:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-gdb
```

#### Option 3: Clang via LLVM
Download from [releases.llvm.org](https://releases.llvm.org/)

### Linux

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake gdb
```

#### Fedora/RHEL
```bash
sudo dnf install gcc-c++ cmake gdb
```

#### Arch Linux
```bash
sudo pacman -S base-devel cmake gdb
```

### macOS

#### Using Xcode Command Line Tools (Clang)
```bash
xcode-select --install
```

#### Using Homebrew (for CMake)
```bash
brew install cmake
```

### Verify Installation

```bash
# Check compiler
g++ --version       # For GCC
clang++ --version   # For Clang/Xcode
cl                  # For MSVC (in Developer Command Prompt)

# Check CMake
cmake --version

# Check debugger
gdb --version       # Linux/MinGW
lldb --version      # macOS/Clang
```

---

## Step 2: Choose Your Development Environment

### IDEs (Integrated Development Environments)

#### Visual Studio (Windows)
- **Pros:** Excellent debugger, IntelliSense, integrated tools
- **Cons:** Windows only, large installation
- **Best for:** Windows developers, beginners

#### CLion (JetBrains)
- **Pros:** Cross-platform, excellent refactoring, CMake integration
- **Cons:** Paid (free for students)
- **Best for:** Professional development

#### Code::Blocks
- **Pros:** Free, cross-platform, lightweight
- **Cons:** Less modern features
- **Best for:** Learning, simple projects

### Text Editors with C++ Support

#### Visual Studio Code (Recommended)
1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install these extensions:
   - **C/C++** (by Microsoft)
   - **CMake Tools**
   - **C++ TestMate** (for Google Test)
   - **clangd** (alternative to C/C++ extension)

**Configure VS Code:**
Create `.vscode/settings.json`:
```json
{
  "C_Cpp.default.cppStandard": "c++20",
  "C_Cpp.default.compilerPath": "/usr/bin/g++",
  "cmake.configureOnOpen": true
}
```

#### Vim/Neovim
- Install **YouCompleteMe** or **coc nvim** for autocomplete
- Use **vim-cmake** for CMake integration

#### Emacs
- Use **lsp-mode** with **ccls** or **clangd**

---

## Step 3: Install CMake (Build System)

CMake is essential for managing C++ projects.

### Installation

**Already installed if you followed Step 1 for Linux/macOS**

**Windows (if not using Visual Studio):**
```powershell
winget install -e --id Kitware.CMake
```

**Or download from:** [cmake.org/download](https://cmake.org/download/)

### Verify CMake
```bash
cmake --version
```

You should see version 3.15 or later (3.20+ recommended).

---

## Step 4: Your First C++ Program

### Method 1: Simple Compilation (No Build System)

Create `hello.cpp`:
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}
```

**Compile and run:**
```bash
# Linux/macOS/MinGW
g++ -std=c++20 hello.cpp -o hello
./hello

# Windows MSVC (in Developer Command Prompt)
cl /EHsc /std:c++20 hello.cpp
hello.exe
```

### Method 2: Using CMake (Recommended)

Create project structure:
```
my_first_project/
├── CMakeLists.txt
└── main.cpp
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyFirstProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(my_app main.cpp)
```

**main.cpp:**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello from CMake!" << std::endl;
    return 0;
}
```

**Build and run:**
```bash
mkdir build
cd build
cmake ..
cmake --build .

# Run
./my_app          # Linux/macOS
.\Debug\my_app    # Windows
```

---

## Step 5: Start the Course

1. ✅ Ensure compiler is installed
2. ✅ Set up your editor/IDE
3. ✅ Verify CMake works
4. ✅ Successfully compiled "Hello, World!"
5. 📚 **Start [Module 1: Getting Started](./phase1_beginner/module-01-getting-started/)**

---

## Essential Commands Reference

### Compilation

```bash
# Basic compilation
g++ -std=c++20 source.cpp -o output

# With debugging symbols
g++ -std=c++20 -g source.cpp -o output

# With optimizations
g++ -std=c++20 -O3 source.cpp -o output

# With warnings
g++ -std=c++20 -Wall -Wextra -Werror source.cpp -o output

# Multiple files
g++ -std=c++20 file1.cpp file2.cpp file3.cpp -o output
```

### CMake Workflow

```bash
# Configure
cmake -S . -B build

# Build
cmake --build build

# Build specific target
cmake --build build --target my_target

# Clean
cmake --build build --target clean

# Install
cmake --install build

# With different generator
cmake -S . -B build -G "Ninja"

# Release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

---

## C++ Language Standards

C++ has evolved significantly. This course focuses on **modern C++**:

| Standard | Year | Major Features |
|----------|------|----------------|
| C++98 | 1998 | Original standard, STL |
| C++03 | 2003 | Bug fixes |
| **C++11** | 2011 | auto, range-for, lambdas, smart pointers |
| **C++14** | 2014 | Generic lambdas, relaxed constexpr |
| **C++17** | 2017 | Structured bindings, std::optional, filesystem |
| **C++20** | 2020 | Concepts, ranges, coroutines, modules |
| **C++23** | 2023 | std::expected, ranges improvements |

**We recommend using C++20 for this course.**

---

## Common Beginner Mistakes

### 1. Forgetting to Include Headers
❌ **Wrong:**
```cpp
int main() {
    cout << "Hello!"; // Error: cout not declared
}
```

✅ **Correct:**
```cpp
#include <iostream>
using namespace std;  // or use std::cout

int main() {
    cout << "Hello!";
}
```

### 2. Missing return Statement
❌ **Wrong:**
```cpp
int add(int a, int b) {
    int sum = a + b;
    // Missing return!
}
```

✅ **Correct:**
```cpp
int add(int a, int b) {
    return a + b;
}
```

### 3. Not Initializing Variables
❌ **Wrong:**
```cpp
int x;  // Uninitialized!
x = x + 5;  // Undefined behavior
```

✅ **Correct:**
```cpp
int x = 0;
x = x + 5;  // Well-defined
```

### 4. Memory Leaks
❌ **Wrong:**
```cpp
int* ptr = new int(42);
// Forgot to delete!
```

✅ **Correct:**
```cpp
int* ptr = new int(42);
delete ptr;

// Or better, use smart pointers (learned in Phase 2):
auto ptr = std::make_unique<int>(42);
// Automatically cleaned up
```

---

## Troubleshooting

### "g++: command not found"
**Solution:** Compiler not installed or not in PATH
- Reinstall compiler
- Add to PATH environment variable

### "fatal error: iostream: No such file or directory"
**Solution:** Compiler installation incomplete
- Reinstall with standard library
- Check compiler documentation

### "CMake Error: CMake was unable to find a build program"
**Solution:** Build tools not installed
- Windows: Install Visual Studio or MinGW
- Linux: Install build-essential
- macOS: Install Xcode Command Line Tools

### Compilation is Very Slow
**Solution:** 
- Use forward declarations in headers
- Use precompiled headers
- Increase parallel build jobs: `cmake --build build -j 8`

### Linker Errors (undefined reference)
**Solution:**
- Missing implementation file in build
- Forgot to link required libraries
- Check CMakeLists.txt includes all source files

---

## Learning Tips

### 1. Embrace Compile-Time Errors
C++ compilers provide detailed error messages. **Read them carefully!**

Modern compilers (GCC 10+, Clang 10+) have excellent diagnostics.

### 2. Use Compiler Warnings
```bash
g++ -Wall -Wextra -Wpedantic -Werror source.cpp -o output
```

### 3. Use Sanitizers (Detect Bugs at Runtime)
```bash
# Address Sanitizer (memory errors)
g++ -fsanitize=address -g source.cpp -o output

# Undefined Behavior Sanitizer
g++ -fsanitize=undefined -g source.cpp -o output

# Thread Sanitizer (data races)
g++ -fsanitize=thread -g source.cpp -o output
```

### 4. Practice Daily
- Consistency beats intensity
- 30 minutes daily > 5 hours once a week

### 5. Read Quality Code
- Study the C++ Standard Library source
- Read well-written open-source C++ projects
- Learn from CppCon talks

### 6. Join the Community
- [r/cpp](https://www.reddit.com/r/cpp/)
- [Stack Overflow C++ tag](https://stackoverflow.com/questions/tagged/c++)
- [C++ Slack](https://cpplang.slack.com/)

---

## Comparing with Rust

If you're coming from Rust (or curious about Rust):

### Similarities
- Zero-cost abstractions
- No garbage collector
- Excellent performance
- Memory safety focus
- Move semantics

### Key Differences

| Aspect | C++ | Rust |
|--------|-----|------|
| **Memory Safety** | Runtime checks, RAII | Compile-time borrow checker |
| **Package Manager** | vcpkg, Conan (separate) | Cargo (integrated) |
| **Build System** | CMake, Make, etc. | Cargo |
| **Learning Curve** | Gentler start, steeper late | Steep start, smoother later |
| **Null Safety** | Pointers can be null | Option<T>, no null by default |
| **Error Handling** | Exceptions, optional | Result<T,E>, explicit |
| **Inheritance** | Full OOP support | Composition over inheritance |

### When to Choose C++
- Existing C++ codebase
- Need specific C++ libraries (Qt, Unreal Engine, etc.)
- Game development with established engines
- Maximum platform compatibility
- Gradual adoption of modern features

### When to Choose Rust
- New projects prioritizing safety
- Fearless concurrency critical
- WebAssembly or embedded systems
- Want integrated tooling from day one

**Both are excellent languages with different trade-offs!**

---

## Next Steps

1. ✅ Install compiler and tools
2. ✅ Set up development environment
3. ✅ Run your first program
4. ✅ Understand basic workflow
5. 📚 **Begin [Phase 1, Module 1](./phase1_beginner/module-01-getting-started/)**

---

## Resources

### Official
- [C++ Reference](https://en.cppreference.com/)
- [ISO C++](https://isocpp.org/)
- [Compiler Explorer](https://godbolt.org/)

### Learning
- [LearnCpp.com](https://www.learncpp.com/)
- [C++ Primer Book](https://www.amazon.com/Primer-5th-Stanley-B-Lippman/dp/0321714113)
- [CppCon YouTube](https://www.youtube.com/user/CppCon)

### Tools
- [CMake Documentation](https://cmake.org/documentation/)
- [vcpkg](https://vcpkg.io/) - C++ package manager
- [Conan](https://conan.io/) - C++ package manager

---

**Ready to start?** Head to [Module 1: Getting Started](./phase1_beginner/module-01-getting-started/) and begin your C++ journey! 🚀
