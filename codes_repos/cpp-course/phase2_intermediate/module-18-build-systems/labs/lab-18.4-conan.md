# Lab 18.4: Conan Package Manager

## Objective
Use Conan to manage C++ dependencies easily.

## Instructions

### Step 1: Install Conan
```bash
pip install conan
```

### Step 2: Create conanfile.txt
```ini
[requires]
fmt/9.1.0
spdlog/1.11.0

[generators]
CMakeDeps
CMakeToolchain

[options]
fmt:shared=False
```

### Step 3: Integrate with CMake
**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(ConanProject)

set(CMAKE_CXX_STANDARD 17)

# Find Conan-installed packages
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE
    fmt::fmt
    spdlog::spdlog
)
```

### Step 4: Build with Conan
```bash
# Install dependencies
conan install . --output-folder=build --build=missing

# Configure CMake
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake

# Build
cmake --build .
```

### Step 5: conanfile.py (Advanced)
```python
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class MyProjectConan(ConanFile):
    name = "myproject"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    
    def requirements(self):
        self.requires("fmt/9.1.0")
        self.requires("spdlog/1.11.0")
    
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()
```

## Challenges

### Challenge 1: Multiple Dependencies
Create a project with 5+ Conan dependencies.

### Challenge 2: Custom Conan Profile
Create a custom Conan profile for your build configuration.

## Solution

<details>
<summary>Click to reveal solution</summary>

**conanfile.txt (Challenge 1):**
```ini
[requires]
fmt/9.1.0
spdlog/1.11.0
nlohmann_json/3.11.2
boost/1.81.0
catch2/3.3.2

[generators]
CMakeDeps
CMakeToolchain

[options]
boost:shared=False
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MultiDepProject)

set(CMAKE_CXX_STANDARD 17)

# Find all packages
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Catch2 REQUIRED)

# Main executable
add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::filesystem
)

# Test executable
add_executable(tests test.cpp)

target_link_libraries(tests PRIVATE
    Catch2::Catch2WithMain
    fmt::fmt
)
```

**main.cpp:**
```cpp
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

using json = nlohmann::json;
namespace fs = boost::filesystem;

int main() {
    // Use fmt
    fmt::print("Hello from fmt!\n");
    
    // Use spdlog
    spdlog::info("Hello from spdlog!");
    
    // Use JSON
    json j = {{"name", "John"}, {"age", 30}};
    fmt::print("JSON: {}\n", j.dump());
    
    // Use Boost
    fs::path p = fs::current_path();
    fmt::print("Current path: {}\n", p.string());
    
    return 0;
}
```

**test.cpp:**
```cpp
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>

TEST_CASE("Basic test", "[example]") {
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("String formatting", "[fmt]") {
    std::string result = fmt::format("Hello, {}!", "World");
    REQUIRE(result == "Hello, World!");
}
```

**Challenge 2: Custom Conan Profile**

Create `~/.conan2/profiles/myprofile`:
```ini
[settings]
os=Windows
arch=x86_64
compiler=msvc
compiler.version=193
compiler.runtime=dynamic
build_type=Release

[conf]
tools.cmake.cmaketoolchain:generator=Ninja

[options]
*:shared=False
```

Use it:
```bash
conan install . --profile=myprofile --output-folder=build --build=missing
```

**Build script (build.sh):**
```bash
#!/bin/bash

# Install dependencies
conan install . --output-folder=build --build=missing

# Configure
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Run tests
ctest -C Release
```
</details>

## Success Criteria
✅ Installed Conan package manager
✅ Created conanfile.txt
✅ Integrated Conan with CMake
✅ Used multiple dependencies (Challenge 1)
✅ Created custom profile (Challenge 2)

## Key Learnings
- Conan simplifies C++ dependency management
- `conanfile.txt` declares dependencies
- CMakeDeps/CMakeToolchain integrate with CMake
- Conan profiles customize build settings
- `--build=missing` builds dependencies from source

## Next Steps
Proceed to **Lab 18.5: vcpkg Package Manager**.
