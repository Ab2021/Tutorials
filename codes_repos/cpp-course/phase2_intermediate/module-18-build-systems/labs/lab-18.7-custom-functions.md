# Lab 18.7: Custom CMake Functions

## Objective
Create reusable CMake functions and macros for build automation.

## Instructions

### Step 1: Basic Function
```cmake
function(add_my_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    )
endfunction()

# Usage
add_my_executable(myapp main.cpp utils.cpp)
```

### Step 2: Function with Named Arguments
```cmake
function(add_my_library)
    set(options SHARED STATIC)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPENDENCIES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(ARG_SHARED)
        add_library(${ARG_NAME} SHARED ${ARG_SOURCES})
    else()
        add_library(${ARG_NAME} STATIC ${ARG_SOURCES})
    endif()
    
    if(ARG_DEPENDENCIES)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_DEPENDENCIES})
    endif()
endfunction()

# Usage
add_my_library(
    NAME mylib
    SHARED
    SOURCES lib.cpp utils.cpp
    DEPENDENCIES fmt::fmt
)
```

### Step 3: Macro vs Function
```cmake
# Macro: operates in caller's scope
macro(set_warning_flags)
    set(WARNING_FLAGS "-Wall -Wextra")
endmacro()

# Function: has own scope
function(print_message msg)
    message(STATUS "${msg}")
endfunction()
```

### Step 4: Include Custom Modules
Create `cmake/MyFunctions.cmake`:
```cmake
function(add_test_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE Catch2::Catch2WithMain)
    
    include(Catch)
    catch_discover_tests(${target_name})
endfunction()
```

Use it:
```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(MyFunctions)

add_test_executable(test_math test_math.cpp)
```

## Challenges

### Challenge 1: Library Builder
Create a function that builds a library with tests and install rules.

### Challenge 2: Code Generation
Create a function that generates C++ code from templates.

## Solution

<details>
<summary>Click to reveal solution</summary>

**cmake/LibraryBuilder.cmake:**
```cmake
# Challenge 1: Complete library builder
function(build_library)
    set(options HEADER_ONLY)
    set(oneValueArgs NAME NAMESPACE)
    set(multiValueArgs 
        SOURCES 
        HEADERS 
        DEPENDENCIES 
        TEST_SOURCES
    )
    
    cmake_parse_arguments(LIB 
        "${options}" 
        "${oneValueArgs}" 
        "${multiValueArgs}" 
        ${ARGN}
    )
    
    # Create library
    if(LIB_HEADER_ONLY)
        add_library(${LIB_NAME} INTERFACE)
        target_include_directories(${LIB_NAME} INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )
    else()
        add_library(${LIB_NAME} ${LIB_SOURCES})
        target_include_directories(${LIB_NAME} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )
    endif()
    
    # Add alias
    if(LIB_NAMESPACE)
        add_library(${LIB_NAMESPACE}::${LIB_NAME} ALIAS ${LIB_NAME})
    endif()
    
    # Link dependencies
    if(LIB_DEPENDENCIES)
        target_link_libraries(${LIB_NAME} PUBLIC ${LIB_DEPENDENCIES})
    endif()
    
    # Add tests
    if(LIB_TEST_SOURCES)
        find_package(Catch2 REQUIRED)
        
        add_executable(test_${LIB_NAME} ${LIB_TEST_SOURCES})
        target_link_libraries(test_${LIB_NAME} PRIVATE 
            ${LIB_NAME}
            Catch2::Catch2WithMain
        )
        
        include(Catch)
        catch_discover_tests(test_${LIB_NAME})
    endif()
    
    # Install rules
    install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    if(LIB_HEADERS)
        install(FILES ${LIB_HEADERS}
            DESTINATION include/${LIB_NAME}
        )
    endif()
    
    # Export targets
    install(EXPORT ${LIB_NAME}Targets
        FILE ${LIB_NAME}Targets.cmake
        NAMESPACE ${LIB_NAMESPACE}::
        DESTINATION lib/cmake/${LIB_NAME}
    )
endfunction()
```

**cmake/CodeGenerator.cmake:**
```cmake
# Challenge 2: Code generation
function(generate_version_header target_name)
    set(VERSION_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/version.h.in")
    set(VERSION_OUTPUT "${CMAKE_BINARY_DIR}/generated/version.h")
    
    configure_file(${VERSION_TEMPLATE} ${VERSION_OUTPUT} @ONLY)
    
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_BINARY_DIR}/generated
    )
endfunction()

function(generate_config_class)
    set(oneValueArgs NAME NAMESPACE OUTPUT_DIR)
    set(multiValueArgs OPTIONS)
    
    cmake_parse_arguments(GEN "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    set(CLASS_NAME "${GEN_NAME}")
    set(NAMESPACE "${GEN_NAMESPACE}")
    
    # Generate header
    set(HEADER_CONTENT "#pragma once\n\nnamespace ${NAMESPACE} {\n")
    set(HEADER_CONTENT "${HEADER_CONTENT}class ${CLASS_NAME} {\npublic:\n")
    
    foreach(opt ${GEN_OPTIONS})
        set(HEADER_CONTENT "${HEADER_CONTENT}    static constexpr bool ${opt} = true;\n")
    endforeach()
    
    set(HEADER_CONTENT "${HEADER_CONTENT}};\n}\n")
    
    file(WRITE "${GEN_OUTPUT_DIR}/${CLASS_NAME}.h" "${HEADER_CONTENT}")
endfunction()
```

**cmake/version.h.in:**
```cpp
#pragma once

#define PROJECT_NAME "@PROJECT_NAME@"
#define PROJECT_VERSION "@PROJECT_VERSION@"
#define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
```

**CMakeLists.txt (using custom functions):**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CustomFunctions VERSION 1.2.3)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(LibraryBuilder)
include(CodeGenerator)

# Use library builder
build_library(
    NAME math
    NAMESPACE MyProject
    SOURCES src/math.cpp
    HEADERS include/math.h
    TEST_SOURCES tests/test_math.cpp
)

# Generate version header
add_executable(myapp src/main.cpp)
generate_version_header(myapp)

# Generate config class
generate_config_class(
    NAME Config
    NAMESPACE myapp
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated
    OPTIONS ENABLE_LOGGING ENABLE_METRICS
)

target_include_directories(myapp PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(myapp PRIVATE MyProject::math)
```

**src/main.cpp:**
```cpp
#include <iostream>
#include "version.h"
#include "Config.h"
#include "math.h"

int main() {
    std::cout << PROJECT_NAME << " v" << PROJECT_VERSION << "\n";
    
    if (myapp::Config::ENABLE_LOGGING) {
        std::cout << "Logging enabled\n";
    }
    
    std::cout << "5 + 3 = " << add(5, 3) << "\n";
    
    return 0;
}
```
</details>

## Success Criteria
✅ Created custom CMake functions
✅ Used named arguments
✅ Understood macro vs function
✅ Built library builder (Challenge 1)
✅ Implemented code generation (Challenge 2)

## Key Learnings
- Functions create reusable build logic
- `cmake_parse_arguments` enables named arguments
- Macros operate in caller's scope
- Functions have their own scope
- Code generation automates boilerplate

## Next Steps
Proceed to **Lab 18.8: Cross-Compilation**.
