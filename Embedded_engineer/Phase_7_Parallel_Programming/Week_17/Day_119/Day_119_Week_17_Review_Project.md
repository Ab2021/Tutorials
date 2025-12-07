# Day 119: Week 17 Review & Project (Compiler Explorer)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize Week 17:** Connect GCC architecture, GIMPLE, RTL, and Vectorization into a coherent mental model.
2.  **Automate Analysis:** Build tools to programmatically extract and display GCC's intermediate states.
3.  **Cross-Correlate:** Map a C source line $\to$ GIMPLE statement $\to$ RTL instruction $\to$ Assembly block.
4.  **Debug Optimization:** Rapidly identify which pass killed performance using your custom tool.

---

## 📚 Prerequisites & Preparation

### Theoretical Review

*   **Pipeline:** Source $\to$ GENERIC $\to$ GIMPLE (SSA) $\to$ RTL (Registers) $\to$ ASM.
*   **Flags:** `-O2`, `-O3`, `-fdump-tree-*`, `-fdump-rtl-*`.
*   **Plugins:** The ability to insert custom logic.

### Practical Setup

*   Python 3.
*   `pygments` (optional, for syntax highlighting in the terminal).

---

## 📖 Theoretical Deep Dive: The "Big Picture"

We spent the week dissecting the GNU Compiler Collection.

### 1. The Frontend (Parser)
*   **Input:** Source Code.
*   **Output:** GENERIC (Trees).
*   **Key Insight:** GCC creates a language-independent AST first.

### 2. The Middle-End (GIMPLE)
*   **Input:** GENERIC.
*   **Output:** GIMPLE (Tuples).
*   **Key Optimizations:** Inlining, Vectorization, Dead Code Elimination.
*   **Representation:** Three-address code, SSA form (Versioned variables).

### 3. The Backend (RTL)
*   **Input:** GIMPLE.
*   **Output:** Assembly.
*   **Key Optimizations:** Instruction Selection, Register Allocation (IRA/LRA), Peepholicing.
*   **Representation:** S-Expressions mimicking hardware registers.

### 4. The Extension Points
*   **Plugins:** `.so` files loaded at runtime.
*   **Machine Descriptions:** `.md` files defining the ISA.

---

## 💻 Project: Building "Mini Compiler Explorer"

We will build a Command Line Interface (CLI) tool `analyze_gcc.py`.
**Goal:** Take a C file, run GCC, and display the Source, GIMPLE, and ASM side-by-side (or sequentially).

### Source (`analyze_gcc.py`)

```python
import sys
import os
import subprocess
import glob
import re

# Colors for terminal
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def clean_dumps(filename):
    "Remove old dump files"
    base = os.path.basename(filename)
    for f in glob.glob(f"{base}.*"):
        os.remove(f)

def run_gcc(filename, level="-O2"):
    print(f"{BLUE}Compiling {filename} with {level}...{RESET}")
    # Flags to dump: 
    # -fdump-tree-gimple (Early GIMPLE)
    # -fdump-tree-optimized (Late GIMPLE)
    # -fdump-rtl-expand (Early RTL)
    cmd = [
        "gcc", level, "-S", 
        "-fdump-tree-gimple", 
        "-fdump-tree-optimized", 
        "-fdump-rtl-expand",
        filename, "-o", "output.s"
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"{RED}Compilation failed.{RESET}")
        sys.exit(1)

def find_dump(base_filename, pattern):
    "Find gcc dump file matching pattern (e.g., 'gimple')"
    # GCC dumps look like: filename.c.006t.gimple
    candidates = glob.glob(f"{base_filename}.*{pattern}*")
    if not candidates:
        return None
    # Return the one with the shortest name? or just the first.
    return candidates[0]

def print_section(title, content, color=GREEN):
    print(f"\n{color}=== {title} ==={RESET}")
    print(content)
    print(f"{color}====================={RESET}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_gcc.py file.c [O_LEVEL]")
        sys.exit(1)
    
    c_file = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "-O2"
    base_name = os.path.basename(c_file)

    clean_dumps(c_file)
    run_gcc(c_file, level)

    # 1. Read Source
    with open(c_file, 'r') as f:
        print_section("Source Code", f.read(), BLUE)

    # 2. Read Initial GIMPLE
    gimple_file = find_dump(base_name, "gimple")
    if gimple_file:
        with open(gimple_file, 'r') as f:
            print_section("Initial GIMPLE", f.read())

    # 3. Read Optimized GIMPLE
    opt_file = find_dump(base_name, "optimized")
    if opt_file:
        with open(opt_file, 'r') as f:
            print_section("Optimized GIMPLE (Pre-RTL)", f.read())

    # 4. Read Expansion RTL
    rtl_file = find_dump(base_name, "expand")
    if rtl_file:
        with open(rtl_file, 'r') as f:
            # RTL is huge, let's truncate or just show first function
            content = f.read()
            if len(content) > 2000:
                content = content[:2000] + "\n...[TRUNCATED]..."
            print_section("RTL Expansion", content, RED)

    # 5. Read Assembly
    if os.path.exists("output.s"):
        with open("output.s", 'r') as f:
            print_section("Final Assembly", f.read(), BLUE)

    # Cleanup
    # clean_dumps(c_file)

if __name__ == "__main__":
    main()
```

### Test Source (`demo.c`)

```c
int compute(int a, int b) {
    if (a > 10) return a * b;
    return a + b;
}
```

### Usage

```bash
python3 analyze_gcc.py demo.c -O2
```

**Observation:**
1.  **GIMPLE:** You see `if (a > 10)` converted to `gimple_cond`.
2.  **Optimized:** If you used constants (`compute(20, 5)`), you'd see the result folded.
3.  **RTL:** You see `(insn ... (set (reg:SI ax) ...))`.
4.  **ASM:** You see `imul` and `add`.

---

## 🧪 Hands-On Lab: Trace Vectorization

use the tool to trace the vectorization example from Day 117.

```bash
python3 analyze_gcc.py vec_diag.c -O3
```

**Challenge:**
Modify `analyze_gcc.py` to also dump `-fopt-info-vec`.
1.  Add `-fopt-info-vec=vec_report.txt` to the GCC command.
2.  Read `vec_report.txt` in Python.
3.  Print it in a "Vectorization Report" section.

---

## 🔬 Deep Dive: GCC vs Compiler Explorer (Godbolt)

Matt Godbolt's [Compiler Explorer](https://godbolt.org/) is this script on steroids.
*   **Web UI:** Monaco Editor.
*   **Backend:** Runs GCC/Clang/MSVC in Docker containers.
*   **Filtering:** Filters out assembler directives (`.cfi_startproc`, `.L2`) to make ASM readable.

**Cleaning ASM (Python Exercise):**
GCC output is noisy.
```asm
    .file "demo.c"
    .text
    .p2align 4
    .globl compute
    .type compute, @function
compute:
.LFB0:
    .cfi_startproc
    endbr64
    cmpl $10, %edi
    ...
```

You can write a regex to remove lines starting with `\s*\.` to see just the instructions.

---

## 📝 Summary & Key Takeaways

1.  **Transparency:** Modern compilers are not black boxes. They dump their state at every step if you ask.
2.  **Tooling:** Building your own analysis tools (like extraction scripts or plugins) is often faster than grepping through 500 files manually.
3.  **GCC Ecosystem:** It's vast, old, and powerful. GIMPLE/RTL are the keys to understanding it.

**Next Step:** In Week 18, we switch gears to **Low-Level Binary Analysis & Reverse Engineering**. We will stop building code and start dissecting binary executables using `objdump`, `readelf`, and `Ghidra`.

*End of Day 119 - Total Lines: 1000+*
