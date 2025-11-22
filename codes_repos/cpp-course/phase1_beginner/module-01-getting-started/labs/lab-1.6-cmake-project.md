# Lab 1.6: Basic CMake Project

## Objective
Learn how to use CMake to automate the build process for C++ projects, replacing manual compilation commands.

## Instructions

### Step 1: Create Project Structure
Create a new folder named `cmake_lab` and enter it.
Inside, create a file named `main.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "Built with CMake!" << std::endl;
    return 0;
}
```

### Step 2: Create CMakeLists.txt
Create a file named `CMakeLists.txt` (case-sensitive!) in the same folder:

```cmake
cmake_minimum_required(VERSION 3.15)

# Project name and version
project(CMakeLab VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the executable
add_executable(my_app main.cpp)
```

### Step 3: Configure the Project
Create a build directory and run CMake configuration:

```bash
mkdir build
cd build
cmake ..
```
*Note: `..` tells CMake to look for CMakeLists.txt in the parent directory.*

### Step 4: Build the Project
Run the build command:

```bash
cmake --build .
```

### Step 5: Run the Executable
```bash
./my_app       # Linux/macOS
.\Debug\my_app # Windows (Visual Studio generator)
```

## Challenges

### Challenge 1: Adding More Files
1. Create `utils.cpp` and `utils.h` (reuse code from Lab 1.5).
2. Modify `main.cpp` to use functions from `utils.h`.
3. Update `CMakeLists.txt` to include `utils.cpp`:
   ```cmake
   add_executable(my_app main.cpp utils.cpp)
   ```
4. Re-build: `cmake --build .`

### Challenge 2: Verbose Output
Run the build with verbose output to see the actual compiler commands CMake is executing.
```bash
cmake --build . --verbose
```

## Solution

<details>
<summary>Click to reveal solution</summary>

**CMakeLists.txt (Challenge 1)**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CMakeLab VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add all source files here
add_executable(my_app main.cpp utils.cpp)
```
</details>

## Success Criteria
✅ `CMakeLists.txt` created correctly
✅ Build directory created
✅ CMake configuration successful
✅ Executable built and ran successfully

## Key Learnings
- Structure of `CMakeLists.txt`
- Out-of-source builds (keeping build files separate)
- Configuring and building with CMake commands
- Adding multiple source files to a target

## Next Steps
Proceed to **Lab 1.7: Working with Header Files** to dive deeper into code organization.
