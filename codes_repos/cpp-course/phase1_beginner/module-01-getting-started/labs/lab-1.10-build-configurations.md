# Lab 1.10: Build Configurations (Debug vs Release)

## Objective
Understand the difference between Debug and Release builds and how to configure them using CMake.

## Instructions

### Step 1: Create Source with Debug Code
Create `config_demo.cpp`:

```cpp
#include <iostream>

int main() {
#ifdef NDEBUG
    std::cout << "Release Build" << std::endl;
#else
    std::cout << "Debug Build" << std::endl;
#endif

    // Simulate some work
    long long sum = 0;
    for (int i = 0; i < 100000000; ++i) {
        sum += i;
    }
    std::cout << "Work done. Sum: " << sum << std::endl;

    return 0;
}
```
*Note: `NDEBUG` is a standard macro defined automatically in Release mode.*

### Step 2: CMake Setup
Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(ConfigDemo)
set(CMAKE_CXX_STANDARD 20)
add_executable(demo config_demo.cpp)
```

### Step 3: Build Debug Version
```bash
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
time ./demo  # Measure execution time
```
*Output should say "Debug Build".*

### Step 4: Build Release Version
```bash
cd ..
mkdir build_release
cd build_release
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
time ./demo  # Measure execution time
```
*Output should say "Release Build".*

### Step 5: Compare Performance
Compare the time taken by both versions. The Release version should be significantly faster due to compiler optimizations (`-O3`).

## Challenges

### Challenge 1: Custom Macros
Add a custom macro definition in `CMakeLists.txt`:
```cmake
add_compile_definitions(MY_FEATURE_ENABLED)
```
Check for it in C++:
```cpp
#ifdef MY_FEATURE_ENABLED
    std::cout << "Feature is ON" << std::endl;
#endif
```

### Challenge 2: Conditional Compilation
Modify `CMakeLists.txt` to only define the macro in Debug mode:
```cmake
target_compile_definitions(demo PRIVATE $<$<CONFIG:Debug>:DEBUG_MODE_ACTIVE>)
```

## Solution

<details>
<summary>Click to reveal solution</summary>

**CMakeLists.txt (Challenge 1)**
```cmake
cmake_minimum_required(VERSION 3.15)
project(ConfigDemo)
set(CMAKE_CXX_STANDARD 20)

add_executable(demo config_demo.cpp)

# Challenge 1
add_compile_definitions(MY_FEATURE_ENABLED)
```
</details>

## Success Criteria
✅ Built both Debug and Release versions
✅ Observed output difference (`#ifdef NDEBUG`)
✅ Observed performance difference
✅ Used custom compile definitions

## Key Learnings
- `CMAKE_BUILD_TYPE` variable
- `Debug` (symbols, no opt) vs `Release` (opt, no symbols)
- `NDEBUG` macro
- Conditional compilation with `#ifdef`

## Next Steps
Congratulations! You've completed Module 1. You now have a solid environment and understanding of the build process.

Proceed to **Module 2: Variables and Types** to start learning the C++ language itself!
