# Lab 18.6: Testing with CTest

## Objective
Integrate unit tests with CMake using CTest.

## Instructions

### Step 1: Enable Testing
**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_math test_math.cpp math.cpp)

# Register test
add_test(NAME MathTests COMMAND test_math)
```

### Step 2: Simple Test
**test_math.cpp:**
```cpp
#include <cassert>
#include "math.h"

int main() {
    assert(add(2, 3) == 5);
    assert(multiply(4, 5) == 20);
    return 0;
}
```

Run tests:
```bash
cd build
ctest
# Or: ctest --output-on-failure
```

### Step 3: Google Test Integration
```cmake
find_package(GTest REQUIRED)

add_executable(gtest_math gtest_math.cpp math.cpp)
target_link_libraries(gtest_math PRIVATE GTest::gtest GTest::gtest_main)

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(gtest_math)
```

**gtest_math.cpp:**
```cpp
#include <gtest/gtest.h>
#include "math.h"

TEST(MathTest, Addition) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
}

TEST(MathTest, Multiplication) {
    EXPECT_EQ(multiply(4, 5), 20);
    EXPECT_EQ(multiply(0, 100), 0);
}
```

### Step 4: Catch2 Integration
```cmake
find_package(Catch2 REQUIRED)

add_executable(catch_math catch_math.cpp math.cpp)
target_link_libraries(catch_math PRIVATE Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(catch_math)
```

## Challenges

### Challenge 1: Test Coverage
Set up code coverage reporting.

### Challenge 2: Test Organization
Organize tests into multiple executables by module.

## Solution

<details>
<summary>Click to reveal solution</summary>

**Project Structure:**
```
project/
├── CMakeLists.txt
├── src/
│   ├── math.h
│   ├── math.cpp
│   ├── string_utils.h
│   └── string_utils.cpp
└── tests/
    ├── CMakeLists.txt
    ├── test_math.cpp
    └── test_string.cpp
```

**Root CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(TestOrganization)

set(CMAKE_CXX_STANDARD 17)

# Challenge 1: Coverage support
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# Library
add_library(mylib
    src/math.cpp
    src/string_utils.cpp
)

target_include_directories(mylib PUBLIC src)

# Enable testing
enable_testing()

# Add tests subdirectory
add_subdirectory(tests)
```

**tests/CMakeLists.txt:**
```cmake
find_package(Catch2 REQUIRED)

# Math tests
add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE mylib Catch2::Catch2WithMain)

# String tests
add_executable(test_string test_string.cpp)
target_link_libraries(test_string PRIVATE mylib Catch2::Catch2WithMain)

# Discover tests
include(Catch)
catch_discover_tests(test_math)
catch_discover_tests(test_string)
```

**src/math.h:**
```cpp
#pragma once

int add(int a, int b);
int multiply(int a, int b);
int divide(int a, int b);
```

**src/math.cpp:**
```cpp
#include "math.h"
#include <stdexcept>

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int divide(int a, int b) {
    if (b == 0) throw std::invalid_argument("Division by zero");
    return a / b;
}
```

**src/string_utils.h:**
```cpp
#pragma once
#include <string>

std::string toUpper(const std::string& s);
std::string toLower(const std::string& s);
bool startsWith(const std::string& s, const std::string& prefix);
```

**src/string_utils.cpp:**
```cpp
#include "string_utils.h"
#include <algorithm>

std::string toUpper(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

std::string toLower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

bool startsWith(const std::string& s, const std::string& prefix) {
    return s.find(prefix) == 0;
}
```

**tests/test_math.cpp:**
```cpp
#include <catch2/catch_test_macros.hpp>
#include "math.h"

TEST_CASE("Addition", "[math]") {
    REQUIRE(add(2, 3) == 5);
    REQUIRE(add(-1, 1) == 0);
    REQUIRE(add(0, 0) == 0);
}

TEST_CASE("Multiplication", "[math]") {
    REQUIRE(multiply(4, 5) == 20);
    REQUIRE(multiply(0, 100) == 0);
    REQUIRE(multiply(-2, 3) == -6);
}

TEST_CASE("Division", "[math]") {
    REQUIRE(divide(10, 2) == 5);
    REQUIRE(divide(7, 2) == 3);
    REQUIRE_THROWS_AS(divide(1, 0), std::invalid_argument);
}
```

**tests/test_string.cpp:**
```cpp
#include <catch2/catch_test_macros.hpp>
#include "string_utils.h"

TEST_CASE("To upper", "[string]") {
    REQUIRE(toUpper("hello") == "HELLO");
    REQUIRE(toUpper("World") == "WORLD");
}

TEST_CASE("To lower", "[string]") {
    REQUIRE(toLower("HELLO") == "hello");
    REQUIRE(toLower("World") == "world");
}

TEST_CASE("Starts with", "[string]") {
    REQUIRE(startsWith("hello world", "hello"));
    REQUIRE_FALSE(startsWith("hello world", "world"));
}
```

**Build and test:**
```bash
# Normal build
cmake -B build -S .
cmake --build build
cd build && ctest --output-on-failure

# With coverage
cmake -B build -S . -DENABLE_COVERAGE=ON
cmake --build build
cd build && ctest
gcov ../src/*.cpp
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```
</details>

## Success Criteria
✅ Enabled CTest in CMake
✅ Created and ran tests
✅ Integrated Google Test or Catch2
✅ Set up coverage reporting (Challenge 1)
✅ Organized tests by module (Challenge 2)

## Key Learnings
- `enable_testing()` enables CTest
- `add_test()` registers tests
- `ctest` runs all tests
- Test frameworks integrate seamlessly
- Coverage requires compiler flags

## Next Steps
Proceed to **Lab 18.7: Custom CMake Functions**.
