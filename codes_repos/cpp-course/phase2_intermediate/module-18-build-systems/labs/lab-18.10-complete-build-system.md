# Lab 18.10: Complete Build System (Capstone)

## Objective
Build a complete, production-ready build system with all best practices.

## Instructions

### Step 1: Project Structure
```
project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── Sanitizers.cmake
│   └── CodeCoverage.cmake
├── include/
│   └── myproject/
│       ├── core.h
│       └── utils.h
├── src/
│   ├── core.cpp
│   └── utils.cpp
├── tests/
│   ├── test_core.cpp
│   └── test_utils.cpp
├── examples/
│   └── example.cpp
├── docs/
│   └── Doxyfile
├── CMakeLists.txt
├── vcpkg.json
├── conanfile.txt
├── README.md
└── LICENSE
```

### Step 2: Root CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)

project(MyProject
    VERSION 1.0.0
    DESCRIPTION "A complete C++ project"
    LANGUAGES CXX
)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(CodeCoverage)

# Dependencies
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Library
add_library(myproject
    src/core.cpp
    src/utils.cpp
)

add_library(MyProject::myproject ALIAS myproject)

target_include_directories(myproject
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(myproject
    PUBLIC
        fmt::fmt
        spdlog::spdlog
)

set_project_warnings(myproject)

if(ENABLE_SANITIZERS)
    enable_sanitizers(myproject)
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        doxygen_add_docs(docs
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
        )
    endif()
endif()

# Install
include(GNUInstallDirs)

install(TARGETS myproject
    EXPORT MyProjectTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT MyProjectTargets
    FILE MyProjectTargets.cmake
    NAMESPACE MyProject::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProject
)

# Package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/MyProjectConfig.cmake.in
    ${CMAKE_BINARY_DIR}/MyProjectConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProject
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/MyProjectConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/MyProjectConfig.cmake
    ${CMAKE_BINARY_DIR}/MyProjectConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyProject
)
```

## Challenges

### Challenge 1: Complete All Components
Implement all cmake modules, tests, examples, and CI.

### Challenge 2: Package Distribution
Create packages for distribution (DEB, RPM, installer).

### Challenge 3: Documentation
Generate API documentation with Doxygen.

## Solution

<details>
<summary>Click to reveal solution</summary>

**cmake/CompilerWarnings.cmake:**
```cmake
function(set_project_warnings target)
    set(MSVC_WARNINGS
        /W4
        /WX
        /permissive-
    )
    
    set(GCC_CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wdouble-promotion
        -Wformat=2
    )
    
    if(MSVC)
        set(WARNINGS ${MSVC_WARNINGS})
    else()
        set(WARNINGS ${GCC_CLANG_WARNINGS})
    endif()
    
    target_compile_options(${target} PRIVATE ${WARNINGS})
endfunction()
```

**cmake/Sanitizers.cmake:**
```cmake
function(enable_sanitizers target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SANITIZERS "")
        
        option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ON)
        if(ENABLE_SANITIZER_ADDRESS)
            list(APPEND SANITIZERS "address")
        endif()
        
        option(ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ON)
        if(ENABLE_SANITIZER_UNDEFINED)
            list(APPEND SANITIZERS "undefined")
        endif()
        
        option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        if(ENABLE_SANITIZER_THREAD)
            list(APPEND SANITIZERS "thread")
        endif()
        
        list(JOIN SANITIZERS "," SANITIZER_LIST)
        
        if(SANITIZER_LIST)
            target_compile_options(${target} PRIVATE
                -fsanitize=${SANITIZER_LIST}
            )
            target_link_options(${target} PRIVATE
                -fsanitize=${SANITIZER_LIST}
            )
        endif()
    endif()
endfunction()
```

**cmake/CodeCoverage.cmake:**
```cmake
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
    endif()
endif()

function(add_coverage_target target)
    if(ENABLE_COVERAGE)
        find_program(LCOV lcov)
        find_program(GENHTML genhtml)
        
        if(LCOV AND GENHTML)
            add_custom_target(coverage
                COMMAND ${LCOV} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${GENHTML} coverage.info --output-directory coverage
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating coverage report"
            )
        endif()
    endif()
endfunction()
```

**vcpkg.json:**
```json
{
  "name": "myproject",
  "version": "1.0.0",
  "dependencies": [
    "fmt",
    "spdlog",
    "catch2"
  ],
  "builtin-baseline": "2023.04.15"
}
```

**tests/CMakeLists.txt:**
```cmake
find_package(Catch2 CONFIG REQUIRED)

add_executable(tests
    test_core.cpp
    test_utils.cpp
)

target_link_libraries(tests PRIVATE
    MyProject::myproject
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(tests)

if(ENABLE_COVERAGE)
    add_coverage_target(tests)
endif()
```

**examples/CMakeLists.txt:**
```cmake
add_executable(example example.cpp)
target_link_libraries(example PRIVATE MyProject::myproject)

install(TARGETS example
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

**cmake/MyProjectConfig.cmake.in:**
```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

find_dependency(fmt)
find_dependency(spdlog)

include("${CMAKE_CURRENT_LIST_DIR}/MyProjectTargets.cmake")

check_required_components(MyProject)
```

**CMakeLists.txt (CPack for packaging):**
```cmake
# Add at the end of root CMakeLists.txt

include(CPack)

set(CPACK_PACKAGE_NAME "MyProject")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "My awesome C++ project")
set(CPACK_PACKAGE_CONTACT "your@email.com")

# DEB package
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libfmt-dev, libspdlog-dev")

# RPM package
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_REQUIRES "fmt-devel, spdlog-devel")

# NSIS installer (Windows)
set(CPACK_NSIS_DISPLAY_NAME "MyProject")
set(CPACK_NSIS_PACKAGE_NAME "MyProject")
```

**Build and package:**
```bash
# Build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Test
cd build && ctest

# Install
cmake --install build --prefix /usr/local

# Create packages
cd build
cpack -G DEB      # Debian package
cpack -G RPM      # RPM package
cpack -G NSIS     # Windows installer
cpack -G TGZ      # Tarball
```

**README.md:**
```markdown
# MyProject

A complete C++ project with modern build system.

## Features
- Modern CMake build system
- vcpkg/Conan package management
- Comprehensive testing with Catch2
- CI/CD with GitHub Actions
- Code coverage reporting
- Cross-platform support

## Building

### Prerequisites
- CMake 3.15+
- C++17 compiler
- vcpkg or Conan

### Build Steps
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

### Testing
```bash
cd build && ctest
```

### Installing
```bash
cmake --install build --prefix /usr/local
```

## License
MIT
```
</details>

## Success Criteria
✅ Created complete project structure
✅ Implemented all CMake modules
✅ Set up testing and examples
✅ Configured CI/CD
✅ Created distribution packages (Challenge 2)
✅ Generated documentation (Challenge 3)

## Key Learnings
- Modern CMake projects are modular and maintainable
- Package managers simplify dependency management
- CI/CD ensures code quality
- Proper install rules enable distribution
- Documentation is essential for libraries

## Congratulations!
You've completed Module 18 and all of Phase 2 (Intermediate C++)! You now have comprehensive knowledge of C++ build systems, package management, and modern development workflows.

## Next Steps
Proceed to **Phase 3: Advanced C++** to learn about advanced topics like metaprogramming, design patterns, and performance optimization.
