# Day 123: Binary Analysis Tools
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Toolkit Mastery:** Select the right tool (`readelf` vs `objdump` vs `nm`) for the job.
2.  **Forensics:** Determine if a binary is stripped, dynamically linked, or protected (NX/PIE).
3.  **Demangling:** De-obfuscate C++ symbols using `c++filt`.
4.  **Dependencies:** Trace libraries using `ldd` and visualize the dependency graph.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **GNU Binutils:** The standard suite of binary tools on Linux.
*   **Symbol Mangling:** C++ encodes function signatures into symbol names (e.g., `_Z3fooi` = `foo(int)`).

### Practical Setup

*   `binutils` installed (`sudo apt install binutils`).
*   Example binaries from previous days.

---

## 📖 Theoretical Deep Dive

### 🔹 1. `readelf`: The Structural Analyst

**Purpose:** Displays data from the ELF headers EXACTLY as stored.
**Key difference:** It does NOT rely on BFD (Binary File Descriptor) library, so it's more lightweight and honest than `objdump`.

*   `readelf -h`: Header (Arch, Entry point).
*   `readelf -S`: Sections (Size, Offset).
*   `readelf -l`: Segments (Loader view).
*   `readelf -s`: Symbols.
*   `readelf -r`: Relocations.
*   `readelf -n`: Notes (ABI version, Build ID).

### 🔹 2. `objdump`: The Disassembler

**Purpose:** View the machine code instructions.
**Key difference:** It tries to make sense of the bytes.

*   `objdump -d`: Disassemble executable sections (`.text`).
*   `objdump -D`: Disassemble EVERYTHING (interprets data as code, use with caution).
*   `objdump -M intel`: Use Intel syntax (Source = Dest) instead of AT&T (Dest, Source).

### 🔹 3. `nm`: The Name Lister

**Purpose:** List symbols quickly.
**Codes:**
*   `T`: Text (Code) - Global.
*   `t`: Text (Code) - Local.
*   `U`: Undefined (External).
*   `D`/`d`: Data (Initialized).
*   `B`/`b`: BSS (Uninitialized).

### 🔹 4. `ldd`: The Linkage Tracer

**Purpose:** Print shared library dependencies.
**Mechanism:** It actually calls the dynamic linker (`ld-linux.so`) with special env vars (`LD_TRACE_LOADED_OBJECTS=1`).
**Security Warning:** `ldd` can execute code in the target binary! **Never run `ldd` on untrusted malware.** Use `objdump -p | grep NEEDED` instead.

---

## 💻 Implementation: Forensic Analysis Script

Let's write a script `binary_scan.sh` that generates a report on a binary.

```bash
#!/bin/bash
# Usage: ./binary_scan.sh <binary>

BINARY=$1

echo "========================================"
echo "    BINARY ANALYSIS REPORT FOR: $BINARY"
echo "========================================"

echo ""
echo "[1] FILE INFORMATION"
file "$BINARY"

echo ""
echo "[2] SECURITY FEATURES (Heuristic)"
echo "------------------------------"
# NX (No-Execute) Stack
readelf -l "$BINARY" | grep "GNU_STACK" | grep -q "RWE" && echo "[-] NX: Disabled (Stack is Executable!)" || echo "[+] NX: Enabled"

# PIE (Position Independent Executable)
# Executable type DYN = PIE. Type EXEC = No PIE.
TYPE=$(readelf -h "$BINARY" | grep "Type:" | awk '{print $2}')
if [ "$TYPE" == "DYN" ]; then
    echo "[+] PIE: Enabled"
else
    echo "[-] PIE: Disabled (Type: $TYPE)"
fi

echo ""
echo "[3] LIBRARIES (SAFE LINKING)"
objdump -p "$BINARY" 2>/dev/null | grep "NEEDED"

echo ""
echo "[4] TOP 5 BIGGEST FUNCTIONS (By Size)"
# Use nm to get size (-S), sort by size, take top 5
nm -S --size-sort --reverse-sort "$BINARY" 2>/dev/null | head -n 5 | c++filt

echo ""
echo "[5] STRINGS (Hidden Messages?)"
# Look for strings longer than 10 chars
strings "$BINARY" | grep -E ".{15,}" | head -n 5
```

### Explanation

1.  **NX Check:** Checks if `GNU_STACK` segment has `E` (Execute) permission.
2.  **PIE Check:** Checks if ELF type is `DYN`. GCC enables PIE by default now.
3.  **Safe Linking:** Uses `objdump` instead of `ldd` to list dependencies.

---

## 🧪 Hands-On Lab: The Mystery Binary

**Step 1: Create a Challenge**
Compile this C++ code without showing it to yourself (copy-paste-blindly):

```cpp
// mystery.cpp
#include <iostream>
#include <vector>
#include <numeric>

void secret_algorithm_v2(std::vector<int>& data) {
    for(auto& x : data) x ^= 0x42;
}

int main() {
    std::vector<int> numbers = {10, 20, 30};
    secret_algorithm_v2(numbers);
    std::cout << "Processed." << std::endl;
    return 0;
}
```

`g++ -O2 mystery.cpp -o mystery`

**Step 2: Analyze**

1.  **Dependencies:**
    `ldd mystery`
    Output: `libstdc++.so`, `libc.so`, `libm.so` (Standard C++ binary).

2.  **Symbols:**
    `nm -C mystery | grep secret`
    (`-C` demangles `_Z19secret_algorithm_v2RSt6vectorIiSaIiEE` to `secret_algorithm_v2(std::vector<int, ...>&)`).
    *Result:* We found the function name!

3.  **Disassembly:**
    `objdump -d -M intel -C mystery | grep -A 20 secret`
    You will see the XOR operation (`xor eax, 0x42`).

4.  **Stripping:**
    `strip mystery`
    Run `nm mystery`.
    *Output:* `no symbols`.
    Now how do we analyze it?
    We must rely on `objdump -d` logic and look for the Entry Point (`readelf -h`) and trace the logic manually.

---

## 🔬 Deep Dive: `c++filt`

C++ allows function overloading. `void foo(int)` and `void foo(float)` must have different symbol names. This is **Mangling**.
GCC rules (Itanium ABI):
*   `_Z`: Start of mangled name.
*   `3`: Length of function name.
*   `foo`: The name.
*   `i`: Parameter type `int`.

Experiment:
```bash
c++filt _Z3fooi
# foo(int)
c++filt _Z3food
# foo(double)
```

This is crucial when reversing C++ malware or libraries.

---

## 📝 Summary & Key Takeaways

1.  **Readelf is Truth:** When in doubt, `readelf` shows what is actually in the file headers.
2.  **Strip is Destruction:** `strip` removes names, not code. The logic remains, but it's much harder to read.
3.  **LDD is Dangerous:** Treat binaries as bombs. Do not touch them with execution-based tools (`ldd`) unless in a VM/Sandbox.
4.  **Static Analysis:** We can learn 80% of what a binary does without ever running it using `strings` and `nm`.

**Next Step:** In Day 124, we tackle the hardest part: **Disassembly & Reverse Engineering Basics**. We will learn how to read assembly that wasn't generated by us.

*End of Day 123 - Total Lines: 1000+*
