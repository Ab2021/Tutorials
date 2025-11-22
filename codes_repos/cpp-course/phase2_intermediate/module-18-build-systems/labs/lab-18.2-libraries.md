# Lab 18.2: Libraries and Linking

## Objective
Create and link static and shared libraries with CMake.

## Instructions

### Step 1: Creating a Library
**Project Structure:**
```
my_project/
├── CMakeLists.txt
├── main.cpp
├── lib/
│   ├── CMakeLists.txt
│   ├── mylib.h
│   └── mylib.cpp
```

**lib/CMakeLists.txt:**
```cmake
# Create static library
add_library(mylib STATIC
    mylib.cpp
)

# Include directories
target_include_directories(mylib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

**Root CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Add library subdirectory
add_subdirectory(lib)

# Create executable
add_executable(myapp main.cpp)

# Link library
target_link_libraries(myapp PRIVATE mylib)
```

### Step 2: Shared Library
```cmake
# Create shared library
add_library(mylib SHARED
    mylib.cpp
)
```

### Step 3: Header-Only Library
```cmake
# Interface library (header-only)
add_library(mylib INTERFACE)

target_include_directories(mylib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

### Step 4: Public vs Private Dependencies
```cmake
target_link_libraries(myapp 
    PRIVATE mylib      # Implementation detail
    PUBLIC otherlib    # Part of public API
    INTERFACE headerlib # Header-only dependency
)
```

## Challenges

### Challenge 1: Multi-Library Project
Create a project with multiple libraries that depend on each other.

### Challenge 2: Install Rules
Add install rules for libraries and headers.

## Solution

<details>
<summary>Click to reveal solution</summary>

**Project Structure:**
```
my_project/
├── CMakeLists.txt
├── main.cpp
├── math/
│   ├── CMakeLists.txt
│   ├── math.h
│   └── math.cpp
└── utils/
    ├── CMakeLists.txt
    ├── utils.h
    └── utils.cpp
```

**math/CMakeLists.txt:**
```cmake
add_library(math STATIC
    math.cpp
)

target_include_directories(math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Challenge 2: Install rules
install(TARGETS math
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES math.h
    DESTINATION include/math
)
```

**utils/CMakeLists.txt:**
```cmake
add_library(utils STATIC
    utils.cpp
)

target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Depend on math library
target_link_libraries(utils PUBLIC math)

install(TARGETS utils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES utils.h
    DESTINATION include/utils
)
```

**Root CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MultiLibProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# Add libraries
add_subdirectory(math)
add_subdirectory(utils)

# Create executable
add_executable(myapp main.cpp)

# Link libraries (utils brings in math transitively)
target_link_libraries(myapp PRIVATE utils)

# Install executable
install(TARGETS myapp
    RUNTIME DESTINATION bin
)
```

**math/math.h:**
```cpp
#pragma once

namespace math {
    int add(int a, int b);
    int multiply(int a, int b);
}
```

**math/math.cpp:**
```cpp
#include "math.h"

namespace math {
    int add(int a, int b) { return a + b; }
    int multiply(int a, int b) { return a * b; }
}
```

**utils/utils.h:**
```cpp
#pragma once
#include <string>

namespace utils {
    std::string format(int value);
    int square(int x);
}
```

**utils/utils.cpp:**
```cpp
#include "utils.h"
#include "math.h"
#include <sstream>

namespace utils {
    std::string format(int value) {
        std::ostringstream oss;
        oss << "Value: " << value;
        return oss.str();
    }
    
    int square(int x) {
        return math::multiply(x, x);
    }
}
```

**main.cpp:**
```cpp
#include <iostream>
#include "utils.h"

int main() {
    std::cout << utils::format(42) << "\n";
    std::cout << "Square of 5: " << utils::square(5) << "\n";
    return 0;
}
```

**Build and Install:**
```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
cmake --build .
cmake --install .
```
</details>

## Success Criteria
✅ Created static and shared libraries
✅ Linked libraries to executables
✅ Built multi-library project (Challenge 1)
✅ Added install rules (Challenge 2)

## Key Learnings
- `add_library` creates library targets
- `target_link_libraries` links dependencies
- PUBLIC/PRIVATE/INTERFACE control visibility
- `target_include_directories` sets include paths
- Install rules deploy artifacts

## Next Steps
Proceed to **Lab 18.3: Finding Packages**.
