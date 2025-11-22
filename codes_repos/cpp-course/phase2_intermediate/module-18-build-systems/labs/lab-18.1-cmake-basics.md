# Lab 18.1: Introduction to CMake

## Objective
Learn CMake basics for cross-platform C++ project configuration.

## Instructions

### Step 1: Basic CMakeLists.txt
Create a simple project structure:
```
my_project/
├── CMakeLists.txt
└── main.cpp
```

Create `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(myapp main.cpp)
```

Create `main.cpp`:
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CMake!\n";
    return 0;
}
```

### Step 2: Building the Project
```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Run
./myapp  # or myapp.exe on Windows
```

### Step 3: Out-of-Source Builds
```bash
# Always build in a separate directory
mkdir build && cd build
cmake ..
cmake --build .
```

### Step 4: Build Types
```cmake
# Set build type
set(CMAKE_BUILD_TYPE Release)

# Or specify at configure time:
# cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Challenges

### Challenge 1: Multiple Source Files
Create a project with multiple source files.

### Challenge 2: Compiler Flags
Add custom compiler flags for warnings.

## Solution

<details>
<summary>Click to reveal solution</summary>

**Project Structure:**
```
my_project/
├── CMakeLists.txt
├── main.cpp
├── math.cpp
└── math.h
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Challenge 2: Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Challenge 1: Multiple sources
add_executable(myapp 
    main.cpp
    math.cpp
)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
```

**math.h:**
```cpp
#pragma once

int add(int a, int b);
int multiply(int a, int b);
```

**math.cpp:**
```cpp
#include "math.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

**main.cpp:**
```cpp
#include <iostream>
#include "math.h"

int main() {
    std::cout << "5 + 3 = " << add(5, 3) << "\n";
    std::cout << "5 * 3 = " << multiply(5, 3) << "\n";
    return 0;
}
```

**Build Commands:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```
</details>

## Success Criteria
✅ Created basic CMakeLists.txt
✅ Built project with CMake
✅ Used out-of-source builds
✅ Added multiple source files (Challenge 1)
✅ Configured compiler flags (Challenge 2)

## Key Learnings
- CMake is a cross-platform build system generator
- `CMakeLists.txt` defines project configuration
- Always use out-of-source builds
- `add_executable` creates build targets
- Set C++ standard with `CMAKE_CXX_STANDARD`

## Next Steps
Proceed to **Lab 18.2: Libraries and Linking**.
