# Lab 18.3: Finding Packages

## Objective
Use `find_package` to locate and link external dependencies.

## Instructions

### Step 1: Finding Packages
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Find package
find_package(Threads REQUIRED)

add_executable(myapp main.cpp)

# Link found package
target_link_libraries(myapp PRIVATE Threads::Threads)
```

### Step 2: Common Packages
```cmake
# Find Boost
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem system)
target_link_libraries(myapp PRIVATE Boost::filesystem Boost::system)

# Find OpenSSL
find_package(OpenSSL REQUIRED)
target_link_libraries(myapp PRIVATE OpenSSL::SSL OpenSSL::Crypto)

# Find Google Test
find_package(GTest REQUIRED)
target_link_libraries(mytest PRIVATE GTest::gtest GTest::gtest_main)
```

### Step 3: Optional Dependencies
```cmake
find_package(SomeLib)

if(SomeLib_FOUND)
    target_link_libraries(myapp PRIVATE SomeLib::SomeLib)
    target_compile_definitions(myapp PRIVATE HAVE_SOMELIB)
endif()
```

### Step 4: Custom Find Modules
Create `cmake/FindMyLib.cmake`:
```cmake
find_path(MyLib_INCLUDE_DIR mylib.h
    PATHS /usr/local/include /usr/include
)

find_library(MyLib_LIBRARY
    NAMES mylib
    PATHS /usr/local/lib /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MyLib
    REQUIRED_VARS MyLib_LIBRARY MyLib_INCLUDE_DIR
)

if(MyLib_FOUND AND NOT TARGET MyLib::MyLib)
    add_library(MyLib::MyLib UNKNOWN IMPORTED)
    set_target_properties(MyLib::MyLib PROPERTIES
        IMPORTED_LOCATION "${MyLib_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MyLib_INCLUDE_DIR}"
    )
endif()
```

Use it:
```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(MyLib REQUIRED)
target_link_libraries(myapp PRIVATE MyLib::MyLib)
```

## Challenges

### Challenge 1: Multi-Package Project
Create a project that uses multiple external packages.

### Challenge 2: Version Requirements
Specify version requirements for packages.

## Solution

<details>
<summary>Click to reveal solution</summary>

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MultiPackageProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# Challenge 2: Version requirements
find_package(Threads REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)

# Optional package
find_package(OpenSSL)

add_executable(myapp main.cpp)

# Link required packages
target_link_libraries(myapp PRIVATE
    Threads::Threads
    Boost::filesystem
)

# Conditional linking
if(OpenSSL_FOUND)
    target_link_libraries(myapp PRIVATE OpenSSL::SSL)
    target_compile_definitions(myapp PRIVATE HAVE_OPENSSL)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
else()
    message(STATUS "OpenSSL not found, building without SSL support")
endif()

# Print package information
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
```

**main.cpp:**
```cpp
#include <iostream>
#include <thread>
#include <boost/filesystem.hpp>

#ifdef HAVE_OPENSSL
#include <openssl/ssl.h>
#endif

void threadFunc() {
    std::cout << "Thread running\n";
}

int main() {
    // Use threads
    std::thread t(threadFunc);
    t.join();
    
    // Use Boost filesystem
    namespace fs = boost::filesystem;
    fs::path p = fs::current_path();
    std::cout << "Current path: " << p << "\n";
    
#ifdef HAVE_OPENSSL
    std::cout << "OpenSSL version: " << OPENSSL_VERSION_TEXT << "\n";
#else
    std::cout << "Built without OpenSSL\n";
#endif
    
    return 0;
}
```

**Build:**
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
</details>

## Success Criteria
✅ Used `find_package` to locate dependencies
✅ Linked external packages
✅ Handled optional dependencies
✅ Used multiple packages (Challenge 1)
✅ Specified version requirements (Challenge 2)

## Key Learnings
- `find_package` locates external dependencies
- REQUIRED makes package mandatory
- Use imported targets (e.g., `Boost::filesystem`)
- Check `_FOUND` variable for optional packages
- Specify version requirements for compatibility

## Next Steps
Proceed to **Lab 18.4: Conan Package Manager**.
