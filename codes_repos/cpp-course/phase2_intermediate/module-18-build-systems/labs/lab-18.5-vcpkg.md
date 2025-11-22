# Lab 18.5: vcpkg Package Manager

## Objective
Use vcpkg to manage C++ libraries across platforms.

## Instructions

### Step 1: Install vcpkg
```bash
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap
./bootstrap-vcpkg.sh  # Linux/Mac
./bootstrap-vcpkg.bat # Windows
```

### Step 2: Install Packages
```bash
# Install a package
./vcpkg install fmt

# Install multiple packages
./vcpkg install fmt spdlog nlohmann-json

# Install for specific triplet
./vcpkg install fmt:x64-windows
```

### Step 3: Integrate with CMake
```bash
# Integrate with CMake (one-time setup)
./vcpkg integrate install
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(VcpkgProject)

set(CMAKE_CXX_STANDARD 17)

# Find vcpkg-installed packages
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)
```

### Step 4: vcpkg.json (Manifest Mode)
Create `vcpkg.json`:
```json
{
  "name": "myproject",
  "version": "1.0.0",
  "dependencies": [
    "fmt",
    "spdlog",
    "nlohmann-json",
    "boost-filesystem"
  ]
}
```

Build with manifest:
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

### Step 5: Versioning
**vcpkg.json with versions:**
```json
{
  "name": "myproject",
  "version": "1.0.0",
  "dependencies": [
    {
      "name": "fmt",
      "version>=": "9.0.0"
    },
    {
      "name": "spdlog",
      "version>=": "1.10.0"
    }
  ]
}
```

## Challenges

### Challenge 1: Cross-Platform Build
Set up vcpkg for both Windows and Linux builds.

### Challenge 2: Custom Triplet
Create a custom vcpkg triplet for static linking.

## Solution

<details>
<summary>Click to reveal solution</summary>

**vcpkg.json (Challenge 1):**
```json
{
  "name": "crossplatform",
  "version": "1.0.0",
  "dependencies": [
    "fmt",
    "spdlog",
    "nlohmann-json",
    "boost-filesystem",
    {
      "name": "openssl",
      "platform": "!windows"
    }
  ],
  "builtin-baseline": "2023.04.15"
}
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CrossPlatform)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

if(UNIX)
    find_package(OpenSSL REQUIRED)
endif()

add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::filesystem
)

if(UNIX)
    target_link_libraries(myapp PRIVATE OpenSSL::SSL)
    target_compile_definitions(myapp PRIVATE HAVE_OPENSSL)
endif()
```

**main.cpp:**
```cpp
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

#ifdef HAVE_OPENSSL
#include <openssl/ssl.h>
#endif

using json = nlohmann::json;
namespace fs = boost::filesystem;

int main() {
    fmt::print("Cross-platform application\n");
    
    spdlog::info("Running on: {}", 
#ifdef _WIN32
        "Windows"
#elif __linux__
        "Linux"
#elif __APPLE__
        "macOS"
#else
        "Unknown"
#endif
    );
    
    json config = {
        {"app", "myapp"},
        {"version", "1.0"}
    };
    
    fmt::print("Config: {}\n", config.dump(2));
    
    fs::path p = fs::current_path();
    fmt::print("Working directory: {}\n", p.string());
    
#ifdef HAVE_OPENSSL
    fmt::print("OpenSSL version: {}\n", OPENSSL_VERSION_TEXT);
#endif
    
    return 0;
}
```

**Challenge 2: Custom Triplet**

Create `triplets/x64-windows-static.cmake`:
```cmake
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)
```

Use it:
```bash
./vcpkg install fmt:x64-windows-static
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=x64-windows-static
```

**Build script (build.ps1 for Windows):**
```powershell
# Set vcpkg root
$env:VCPKG_ROOT = "C:\vcpkg"

# Configure
cmake -B build -S . `
    -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows-static

# Build
cmake --build build --config Release
```

**Build script (build.sh for Linux):**
```bash
#!/bin/bash

export VCPKG_ROOT="$HOME/vcpkg"

cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build
```
</details>

## Success Criteria
✅ Installed vcpkg package manager
✅ Installed packages with vcpkg
✅ Integrated vcpkg with CMake
✅ Used manifest mode (vcpkg.json)
✅ Created cross-platform build (Challenge 1)
✅ Created custom triplet (Challenge 2)

## Key Learnings
- vcpkg is Microsoft's C++ package manager
- Manifest mode (`vcpkg.json`) declares dependencies
- Triplets specify platform/linkage combinations
- Integrates seamlessly with CMake
- Supports versioning and baselines

## Next Steps
Proceed to **Lab 18.6: Testing with CTest**.
