# Module 18: Build Systems and Package Managers

## 🎯 Learning Objectives

- Master CMake for cross-platform builds
- Understand Makefiles
- Use package managers (vcpkg, Conan)
- Configure build types (Debug, Release)
- Manage dependencies
- Create libraries (static, dynamic)
- Use precompiled headers
- Understand compilation units
- Set up CI/CD pipelines
- Profile and optimize builds

## 📖 Key Concepts

### CMake
```cmake
cmake_minimum_required(VERSION 3.20)
project(MyProject)
add_executable(app main.cpp)
```

### Package Management
```bash
vcpkg install fmt
conan install .
```

## 🦀 Rust vs C++ Comparison

**C++:** CMake, Make, vcpkg, Conan (fragmented ecosystem).
**Rust:** Cargo (unified build system and package manager).

## Labs

1. CMake Basics
2. Multi-File Projects
3. Libraries (Static/Dynamic)
4. Package Managers
5. Build Configurations
6. Precompiled Headers
7. Cross-Platform Builds
8. Testing Integration
9. CI/CD Setup
10. Complete Project (Capstone)

## Completion
Congratulations! You've completed Phase 2 (Intermediate).

Proceed to **Phase 3: Advanced C++**.
