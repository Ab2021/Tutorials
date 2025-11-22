# Module 1: Getting Started with C++

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand what C++ is and its evolution (C++98 to C++23)
- Install and configure C++ compilers (GCC, Clang, MSVC)
- Understand the compilation process and build systems
- Use CMake to manage C++ projects
- Write, compile, and run C++ programs
- Work with multi-file projects
- Understand basic project structure

---

## 📖 Theoretical Concepts

### 1.1 What is C++?

C++ is a powerful, high-performance programming language created by Bjarne Stroustrup in 1979 as an extension of C. It combines:
- **Low-level control**: Direct hardware access and manual memory management
- **High-level abstractions**: Object-oriented and generic programming
- **Zero-cost abstractions**: Performance without overhead

#### Why C++?

**Performance**
- Compiled directly to machine code
- No garbage collector overhead
- Fine-grained control over memory and resources
- Used in performance-critical applications

**Versatility**
- Systems programming (operating systems, drivers)
- Game development (Unreal Engine, Unity)
- High-frequency trading
- Scientific computing
- Embedded systems
- GUI applications

**Industry Adoption**
- Powers major software: Windows, Linux kernel components, Chrome, Firefox
- Game engines: Unreal, Unity (parts), CryEngine
- Databases: MySQL, PostgreSQL, MongoDB
- Financial systems: Bloomberg, trading platforms

---

### 1.2 C++ Evolution

| Standard | Year | Major Features |
|----------|------|----------------|
| C++98 | 1998 | First ISO standard, STL |
| C++03 | 2003 | Bug fixes, minor improvements |
| **C++11** | 2011 | auto, range-for, lambdas, move semantics, smart pointers |
| **C++14** | 2014 | Generic lambdas, relaxed constexpr |
| **C++17** | 2017 | std::optional, structured bindings, filesystem |
| **C++20** | 2020 | Concepts, ranges, coroutines, modules |
| **C++23** | 2023 | std::expected, ranges improvements, more constexpr |

**This course focuses on Modern C++ (C++11 onwards).**

---

### 1.3 Compilers

C++ requires a compiler to translate source code to machine code:

#### GCC (GNU Compiler Collection)
- **Platforms:** Linux, Windows (MinGW), macOS
- **Version:** Use 10.0+ for C++20 support
- **Command:** `g++`

#### Clang/LLVM
- **Platforms:** macOS (default), Linux, Windows
- **Version:** Use 10.0+ for C++20 support
- **Command:** `clang++`
- **Benefits:** Fast compilation, excellent error messages

#### MSVC (Microsoft Visual C++)
- **Platforms:** Windows only
- **Version:** Visual Studio 2019+ for C++20
- **Command:** `cl`
- **Benefits:** Integrated with Visual Studio

---

### 1.4 The Compilation Process

```
Source Code (.cpp, .h) → Preprocessor → Compiler → Assembler → Linker → Executable
```

1. **Preprocessing**: Handle #include, #define, etc.
2. **Compilation**: Convert to assembly code
3. **Assembly**: Convert to machine code (object files .o/.obj)
4. **Linking**: Combine object files and libraries into executable

**Advantages of Ahead-of-Time Compilation:**
- Fast execution (no runtime interpreter)
- Optimization during compilation
- Platform-specific binaries

---

### 1.5 Build Systems

#### Make
Traditional Unix build system, uses Makefiles:
```makefile
main: main.cpp
    g++ -std=c++20 main.cpp -o main
```

#### CMake (Recommended)
Cross-platform, meta-build system:
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)
set(CMAKE_CXX_STANDARD 20)
add_executable(main main.cpp)
```

**Benefits:**
- Cross-platform (Windows, Linux, macOS)
- Handles dependencies automatically
- Generates native build files (Make, Ninja, Visual Studio)
- Industry standard

---

### 1.6 Your First C++ Program

```cpp
#include <iostream>  // Include standard I/O library

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}
```

**Breakdown:**
- `#include <iostream>`: Brings in input/output functions
- `int main()`: Program entry point, returns integer
- `std::cout`: Standard output stream
- `<<`: Stream insertion operator
- `std::endl`: End line and flush buffer
- `return 0`: Success (0 = no errors)

---

### 1.7 Compilation Flags

Important compiler flags to know:

```bash
# Specify C++ standard
g++ -std=c++20 file.cpp -o output

# Enable warnings
g++ -Wall -Wextra -Wpedantic file.cpp -o output

# Debug symbols
g++ -g file.cpp -o output

# Optimization levels
g++ -O0  # No optimization (debug)
g++ -O2  # Moderate optimization
g++ -O3  # Aggressive optimization

# Include directories
g++ -I./include file.cpp -o output

# Link libraries
g++ file.cpp -o output -lpthread
```

---

### 1.8 Multi-File Projects

**Project Structure:**
```
project/
├── CMakeLists.txt
├── include/
│   └── utils.h
└── src/
    ├── main.cpp
    └── utils.cpp
```

**Header File (utils.h):**
```cpp
#ifndef UTILS_H
#define UTILS_H

int add(int a, int b);

#endif
```

**Implementation (utils.cpp):**
```cpp
#include "utils.h"

int add(int a, int b) {
    return a + b;
}
```

**Main File (main.cpp):**
```cpp
#include <iostream>
#include "utils.h"

int main() {
    std::cout << add(3, 4) << std::endl;
    return 0;
}
```

---

### 1.9 Namespaces

Namespaces prevent name collisions:

```cpp
#include <iostream>

// Using full qualification
int main() {
    std::cout << "Hello!" << std::endl;
}

// Using namespace (be careful in headers!)
using namespace std;
int main() {
    cout << "Hello!" << endl;
}

// Using specific names (better practice)
using std::cout;
using std::endl;
int main() {
    cout << "Hello!" << endl;
}
```

---

## 🦀 Rust vs C++ Comparison

### Build Systems and Package Management

| Aspect | C++ | Rust |
|--------|-----|------|
| **Build System** | CMake, Make, Ninja (separate tools) | Cargo (integrated) |
| **Package Manager** | vcpkg, Conan (third-party) | Cargo (built-in) |
| **Dependency Management** | Manual CMakeLists.txt | Cargo.toml automatic |
| **Project Creation** | `mkdir project; cd project` | `cargo new project` |
| **Build Command** | `cmake --build build` | `cargo build` |
| **Run Command** | `./build/app` | `cargo run` |

**C++ Strengths:**
- More mature ecosystem
- Greater flexibility in build configuration
- Better support for complex build scenarios

**Rust Strengths:**
- Integrated tooling from day one
- Simpler dependency management
- Standardized project structure

### Compilation Model

**C++:**
- Separate compilation units
- Header files for declarations
- Include guards to prevent multiple inclusion
- Manual control over what gets compiled

**Rust:**
- Module system built into language
- No header files needed
- `mod` keyword for modules
- Compiler handles module dependencies

### First Program Comparison

**C++:**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

**Rust:**
```rust
fn main() {
    println!("Hello, World!");
}
```

**Observations:**
- Rust: Implicit return, macro for printing
- C++: Explicit includes, iostream library, return statement
- Both: main() as entry point

---

## 🔑 Key Takeaways

1. C++ is a compiled, high-performance language with zero-cost abstractions
2. Modern C++ (C++11/14/17/20/23) offers significant improvements over old C++
3. Compilers translate source code to machine code through multiple stages
4. CMake is the industry-standard build system for C++
5. Multi-file projects use headers (.h) and implementations (.cpp)
6. Namespaces prevent name collisions (std::)
7. Compiler flags control warnings, optimizations, and standards

---

## 📚 Additional Resources

- [C++ Reference](https://en.cppreference.com/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [Compiler Explorer](https://godbolt.org/) - See generated assembly
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

---

## ⏭️ Next Steps

Complete all 10 labs in the `labs/` directory:

1. **Lab 1.1:** Compiler installation and verification
2. **Lab 1.2:** Hello World variations
3. **Lab 1.3:** Building a simple calculator
4. **Lab 1.4:** Understanding compilation flags
5. **Lab 1.5:** Multi-file programs
6. **Lab 1.6:** Basic CMake project
7. **Lab 1.7:** Working with header files
8. **Lab 1.8:** Namespace basics
9. **Lab 1.9:** Command-line arguments
10. **Lab 1.10:** Build configurations (Debug vs Release)

After completing the labs, move on to **Module 2: Variables and Types**.

---

**Happy Coding!** 🚀
