# Lab 1.1: Compiler Installation and Verification

## Objective
Verify that your C++ development environment is correctly set up with a compiler and build tools.

## Instructions

### Step 1: Verify C++ Compiler

Open your terminal or command prompt and run the following commands to check if a C++ compiler is installed.

**For GCC (Linux/Windows MinGW):**
```bash
g++ --version
```

**For Clang (macOS/Linux):**
```bash
clang++ --version
```

**For MSVC (Windows Visual Studio):**
Open "Developer Command Prompt for VS" and run:
```cmd
cl
```

### Step 2: Verify CMake

Check if CMake is installed and accessible:

```bash
cmake --version
```

You should see version 3.15 or higher.

### Step 3: Verify Debugger

Check if a debugger is available:

**GDB (Linux/MinGW):**
```bash
gdb --version
```

**LLDB (macOS):**
```bash
lldb --version
```

### Step 4: Your Task

Create a text file named `environment_check.txt` and record the versions of your tools.

**Example Content:**
```text
Compiler: g++ 11.2.0
CMake: 3.22.1
Debugger: GDB 11.1
OS: Windows 10
```

## Challenges

### Challenge 1: Locate Compiler Path
Find where your compiler is installed on your system.
- **Linux/macOS:** Run `which g++` or `which clang++`
- **Windows:** Run `where g++` or `where cl`

Add this path to your `environment_check.txt`.

### Challenge 2: Check C++ Standard Support
Try to find out which C++ standards your compiler supports by default.
- **GCC/Clang:** `g++ -dM -E -x c++ /dev/null | grep __cplusplus` (Linux/macOS)
- **MSVC:** Check the documentation for your Visual Studio version.

## Solution

<details>
<summary>Click to reveal solution</summary>

There is no code solution for this lab as it involves environment verification. 

**Expected `environment_check.txt`:**
```text
Compiler: g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Path: /usr/bin/g++
CMake: cmake version 3.22.1
Debugger: GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 12.1
Standard: __cplusplus 201703L (Default C++17)
```
</details>

## Success Criteria
✅ Compiler version command returns valid output
✅ CMake version command returns valid output
✅ `environment_check.txt` created with tool details

## Key Learnings
- How to verify toolchain installation
- Understanding which compiler is being used
- Locating build tools on the system

## Next Steps
Proceed to **Lab 1.2: Hello World Variations** to write your first program.
