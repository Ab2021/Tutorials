# Lab 18.9: Continuous Integration

## Objective
Set up CI/CD pipelines for automated building and testing.

## Instructions

### Step 1: GitHub Actions
Create `.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        build_type: [Debug, Release]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies (Ubuntu)
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake ninja-build
    
    - name: Configure
      run: |
        cmake -B build -S . \
          -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
          -G Ninja
    
    - name: Build
      run: cmake --build build
    
    - name: Test
      run: cd build && ctest --output-on-failure
```

### Step 2: GitLab CI
Create `.gitlab-ci.yml`:
```yaml
stages:
  - build
  - test

build:linux:
  stage: build
  image: gcc:latest
  script:
    - cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    - cmake --build build
  artifacts:
    paths:
      - build/

test:linux:
  stage: test
  image: gcc:latest
  dependencies:
    - build:linux
  script:
    - cd build && ctest --output-on-failure
```

### Step 3: Azure Pipelines
Create `azure-pipelines.yml`:
```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: CMake@1
  inputs:
    workingDirectory: 'build'
    cmakeArgs: '.. -DCMAKE_BUILD_TYPE=Release'

- script: |
    cmake --build build
  displayName: 'Build'

- script: |
    cd build && ctest --output-on-failure
  displayName: 'Test'
```

### Step 4: Docker Build
Create `Dockerfile`:
```dockerfile
FROM gcc:latest

WORKDIR /app

COPY . .

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build

CMD ["./build/myapp"]
```

## Challenges

### Challenge 1: Multi-Platform CI
Set up CI for Windows, Linux, and macOS with matrix builds.

### Challenge 2: Code Coverage
Integrate code coverage reporting in CI.

## Solution

<details>
<summary>Click to reveal solution</summary>

**.github/workflows/ci.yml (Challenge 1 & 2):**
```yaml
name: Comprehensive CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    name: ${{ matrix.os }}-${{ matrix.build_type }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        build_type: [Debug, Release]
        include:
          - os: ubuntu-latest
            triplet: x64-linux
          - os: windows-latest
            triplet: x64-windows
          - os: macos-latest
            triplet: x64-osx
    
    steps:
    - uses: actions/checkout@v3
      with:
        submodules: recursive
    
    - name: Install vcpkg
      uses: lukka/run-vcpkg@v11
      with:
        vcpkgGitCommitId: 'a42af01b72c28a8e1d7b48107b33e4f286a55ef6'
    
    - name: Install dependencies (Ubuntu)
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake ninja-build lcov
    
    - name: Install dependencies (macOS)
      if: runner.os == 'macOS'
      run: |
        brew install cmake ninja lcov
    
    - name: Install dependencies (Windows)
      if: runner.os == 'Windows'
      run: |
        choco install cmake ninja
    
    - name: Configure CMake
      run: |
        cmake -B build -S . \
          -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
          -DCMAKE_TOOLCHAIN_FILE=${{ github.workspace }}/vcpkg/scripts/buildsystems/vcpkg.cmake \
          -DENABLE_COVERAGE=${{ matrix.os == 'ubuntu-latest' && matrix.build_type == 'Debug' }} \
          -G Ninja
    
    - name: Build
      run: cmake --build build --config ${{ matrix.build_type }}
    
    - name: Test
      run: |
        cd build
        ctest -C ${{ matrix.build_type }} --output-on-failure
    
    - name: Generate Coverage (Ubuntu Debug only)
      if: matrix.os == 'ubuntu-latest' && matrix.build_type == 'Debug'
      run: |
        cd build
        lcov --capture --directory . --output-file coverage.info
        lcov --remove coverage.info '/usr/*' --output-file coverage.info
        lcov --list coverage.info
    
    - name: Upload Coverage
      if: matrix.os == 'ubuntu-latest' && matrix.build_type == 'Debug'
      uses: codecov/codecov-action@v3
      with:
        files: ./build/coverage.info
        fail_ci_if_error: true

  static-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install clang-tidy
      run: sudo apt-get install -y clang-tidy
    
    - name: Run clang-tidy
      run: |
        cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        clang-tidy -p build src/*.cpp

  sanitizers:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        sanitizer: [address, undefined, thread]
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure with ${{ matrix.sanitizer }} sanitizer
      run: |
        cmake -B build -S . \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_CXX_FLAGS="-fsanitize=${{ matrix.sanitizer }}"
    
    - name: Build
      run: cmake --build build
    
    - name: Test
      run: cd build && ctest --output-on-failure
```

**CMakeLists.txt (with coverage support):**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CIProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
    endif()
endif()

# Find packages
find_package(Catch2 CONFIG REQUIRED)

# Library
add_library(mylib src/math.cpp)
target_include_directories(mylib PUBLIC include)

# Executable
add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE mylib)

# Tests
enable_testing()
add_executable(tests tests/test_math.cpp)
target_link_libraries(tests PRIVATE mylib Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(tests)
```

**Dockerfile (multi-stage):**
```dockerfile
# Build stage
FROM gcc:latest AS builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . .

# Build
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G Ninja && \
    cmake --build build

# Runtime stage
FROM debian:bullseye-slim

WORKDIR /app

# Copy only the executable
COPY --from=builder /app/build/myapp .

CMD ["./myapp"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    volumes:
      - ./data:/app/data
    environment:
      - LOG_LEVEL=info
  
  test:
    build:
      context: .
      target: builder
    command: ["ctest", "--test-dir", "build", "--output-on-failure"]
```
</details>

## Success Criteria
✅ Set up GitHub Actions CI
✅ Created multi-platform builds
✅ Integrated testing in CI
✅ Added coverage reporting (Challenge 2)
✅ Configured Docker builds

## Key Learnings
- CI automates building and testing
- Matrix builds test multiple configurations
- Coverage reports track test quality
- Docker ensures reproducible builds
- Static analysis catches bugs early

## Next Steps
Proceed to **Lab 18.10: Complete Build System (Capstone)**.
