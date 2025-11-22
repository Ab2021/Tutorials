# Lab 18.8: Cross-Compilation

## Objective
Configure CMake for cross-platform and cross-architecture builds.

## Instructions

### Step 1: Toolchain File
Create `toolchain-arm.cmake`:
```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

Use it:
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-arm.cmake ..
```

### Step 2: Platform Detection
```cmake
if(WIN32)
    message(STATUS "Building for Windows")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Building for Linux")
elseif(APPLE)
    message(STATUS "Building for macOS")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64-bit build")
else()
    message(STATUS "32-bit build")
endif()
```

### Step 3: Conditional Compilation
```cmake
if(WIN32)
    target_sources(myapp PRIVATE windows_impl.cpp)
    target_link_libraries(myapp PRIVATE ws2_32)
elseif(UNIX)
    target_sources(myapp PRIVATE unix_impl.cpp)
    target_link_libraries(myapp PRIVATE pthread)
endif()
```

### Step 4: Cross-Platform Paths
```cmake
# Use generator expressions for cross-platform paths
target_compile_definitions(myapp PRIVATE
    DATA_DIR="$<IF:$<PLATFORM_ID:Windows>,C:/data,/usr/share/data>"
)
```

## Challenges

### Challenge 1: Multi-Platform Build
Set up builds for Windows, Linux, and macOS from the same source.

### Challenge 2: Android Build
Create a toolchain for Android cross-compilation.

## Solution

<details>
<summary>Click to reveal solution</summary>

**Project Structure:**
```
project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── platform.h
│   ├── windows_platform.cpp
│   ├── linux_platform.cpp
│   └── macos_platform.cpp
└── toolchains/
    ├── android.cmake
    └── raspberry-pi.cmake
```

**CMakeLists.txt (Challenge 1):**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CrossPlatform)

set(CMAKE_CXX_STANDARD 17)

# Common sources
set(COMMON_SOURCES
    src/main.cpp
)

# Platform-specific sources
if(WIN32)
    list(APPEND PLATFORM_SOURCES src/windows_platform.cpp)
    set(PLATFORM_LIBS ws2_32)
elseif(APPLE)
    list(APPEND PLATFORM_SOURCES src/macos_platform.cpp)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    set(PLATFORM_LIBS ${COREFOUNDATION_LIBRARY})
elseif(UNIX)
    list(APPEND PLATFORM_SOURCES src/linux_platform.cpp)
    set(PLATFORM_LIBS pthread dl)
endif()

add_executable(myapp ${COMMON_SOURCES} ${PLATFORM_SOURCES})

target_link_libraries(myapp PRIVATE ${PLATFORM_LIBS})

# Platform-specific compile definitions
target_compile_definitions(myapp PRIVATE
    $<$<PLATFORM_ID:Windows>:PLATFORM_WINDOWS>
    $<$<PLATFORM_ID:Linux>:PLATFORM_LINUX>
    $<$<PLATFORM_ID:Darwin>:PLATFORM_MACOS>
)

# Install
install(TARGETS myapp
    RUNTIME DESTINATION bin
)
```

**src/platform.h:**
```cpp
#pragma once
#include <string>

std::string getPlatformName();
std::string getArchitecture();
void platformSpecificInit();
```

**src/windows_platform.cpp:**
```cpp
#include "platform.h"
#include <windows.h>

std::string getPlatformName() {
    return "Windows";
}

std::string getArchitecture() {
#ifdef _WIN64
    return "x64";
#else
    return "x86";
#endif
}

void platformSpecificInit() {
    // Windows-specific initialization
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
}
```

**src/linux_platform.cpp:**
```cpp
#include "platform.h"
#include <sys/utsname.h>

std::string getPlatformName() {
    return "Linux";
}

std::string getArchitecture() {
    struct utsname info;
    uname(&info);
    return info.machine;
}

void platformSpecificInit() {
    // Linux-specific initialization
}
```

**src/macos_platform.cpp:**
```cpp
#include "platform.h"
#include <sys/utsname.h>

std::string getPlatformName() {
    return "macOS";
}

std::string getArchitecture() {
    struct utsname info;
    uname(&info);
    return info.machine;
}

void platformSpecificInit() {
    // macOS-specific initialization
}
```

**src/main.cpp:**
```cpp
#include <iostream>
#include "platform.h"

int main() {
    platformSpecificInit();
    
    std::cout << "Platform: " << getPlatformName() << "\n";
    std::cout << "Architecture: " << getArchitecture() << "\n";
    
#ifdef PLATFORM_WINDOWS
    std::cout << "Running on Windows\n";
#elif defined(PLATFORM_LINUX)
    std::cout << "Running on Linux\n";
#elif defined(PLATFORM_MACOS)
    std::cout << "Running on macOS\n";
#endif
    
    return 0;
}
```

**toolchains/android.cmake (Challenge 2):**
```cmake
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_HOME})
set(CMAKE_ANDROID_STL_TYPE c++_shared)

set(CMAKE_C_COMPILER ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang)
set(CMAKE_CXX_COMPILER ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++)
```

**toolchains/raspberry-pi.cmake:**
```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(TOOLCHAIN_PREFIX arm-linux-gnueabihf)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)

set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

**Build scripts:**

**build-windows.bat:**
```batch
mkdir build-windows
cd build-windows
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**build-linux.sh:**
```bash
#!/bin/bash
mkdir -p build-linux
cd build-linux
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**build-android.sh:**
```bash
#!/bin/bash
mkdir -p build-android
cd build-android
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/android.cmake
cmake --build .
```
</details>

## Success Criteria
✅ Created toolchain files
✅ Detected platform at build time
✅ Built for multiple platforms (Challenge 1)
✅ Created Android toolchain (Challenge 2)

## Key Learnings
- Toolchain files configure cross-compilation
- Platform detection enables conditional builds
- Generator expressions provide cross-platform paths
- Each platform may need specific sources/libraries
- Android/embedded require special toolchains

## Next Steps
Proceed to **Lab 18.9: Continuous Integration**.
