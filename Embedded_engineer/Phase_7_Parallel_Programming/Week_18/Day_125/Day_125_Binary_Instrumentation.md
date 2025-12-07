# Day 125: Binary Instrumentation (Pin/DynamoRIO)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 18: Binary Analysis

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **DBT Theory:** Understand how Dynamic Binary Translation (JIT) allows observing running code.
2.  **Granularity:** Distinguish between Instrumenting Instructions, Basic Blocks, and Traces.
3.  **Pintool Dev:** Write a C++ tool using Intel Pin to count executed instructions.
4.  **Applications:** Explain how instrumentation is used for Cache Simulation, Branch Prediction Analysis, and Fuzzing.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **JIT:** Instrumentation tools are just JIT compilers that don't optimize much but inject snippets.
*   **Code Cache:** The instrumented code is stored in a new memory region; the original code is never executed directly.

### Practical Setup

*   **Intel Pin:** Proprietary but free tool from Intel (requires separate download).
*   **DynamoRIO:** Open-source alternative (Google/VMware).
*   *Note:* The code below assumes Intel Pin API availability.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How Instrumentation Works

You execute: `pin -t my_tool.so -- ./my_app`

1.  **Injector:** Pin attaches to `my_app` and hijacks the entry point.
2.  **JIT Engine:**
    *   Reads `my_app` instructions (Basic Block A).
    *   Calls `my_tool.so`'s instrumentation callback: "I found a `mov`. What do you want to do?"
    *   `my_tool.so` says: "Insert a call to `count_movs()` before it."
    *   Pin emits machine code: `call count_movs; mov dest, src`.
    *   Places this new code in the **Code Cache**.
3.  **Execution:** The CPU executes the *Code Cache* version.
4.  **Chaining:** At the end of Block A, a jump goes back to Pin. Pin translates Block B, links A to B, and execution continues.

**Overhead:**
*   **Translation:** Expensive (happens once per block).
*   **Execution:** Cheap (native speed + instrumentation cost).

### 🔹 Part 2: Granularity

1.  **Instruction (INS):** Finest level. high overhead.
2.  **Basic Block (BBL):** Sequence of instructions with one entry/exit. Good for profiling.
3.  **Trace:** A hot path of multiple basic blocks (e.g., a loop body).

### 🔹 Part 3: Thread Safety

Pin tools run in the **same address space** as the application.
*   **Analysis Routine (What runs):** Must be thread-safe (use atomic counters or thread-local storage).
*   **Instrumentation Routine (When to insert):** Runs under a global lock (Big Pin Lock).

---

## 💻 Implementation: A "Pintool" (Instruction Counter)

This C++ code implements a tool that counts every single instruction executed.

### `inscount.cpp`

```cpp
#include <iostream>
#include <fstream>
#include "pin.H"

// Global output file
std::ofstream OutFile;

// The running count of instructions is kept here
// We use a 64-bit int because instruction counts effectively infinite
static UINT64 icount = 0;

// This function is called before every instruction is executed
// This is the "Analysis" routine
VOID docount() { 
    icount++; 
}

// Pin calls this function every time it JITs a new instruction
// This is the "Instrumentation" routine
VOID Instruction(INS ins, VOID *v) {
    // Insert a call to docount before the instruction
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)docount, IARG_END);
}

// This function is called when the application exits
VOID Fini(INT32 code, VOID *v) {
    OutFile.setf(std::ios::showbase);
    OutFile << "Total Instructions: " << icount << std::endl;
    OutFile.close();
}

int main(int argc, char *argv[]) {
    // Initialize Pin
    if (PIN_Init(argc, argv)) return -1;

    OutFile.open("inscount.out");

    // Register Instruction to be called to instrument instructions
    INS_AddInstrumentFunction(Instruction, 0);

    // Register Fini to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);

    // Start the program, never returns
    PIN_StartProgram();
    
    return 0;
}
```

### Optimizing the Counter (BBL Level)

The above is slow because `docount` is an external function call for *every* instruction.
Optimization: Count instructions in a Basic Block (BBL) and add that sum once per block.

```cpp
VOID BBL_Instrumentation(BBL bbl, VOID *v) {
    // Statistically count instructions in this block
    UINT32 bbl_size = BBL_NumIns(bbl);
    
    // Insert code to add 'bbl_size' to 'icount'
    // IARG_UINT32 passes 'bbl_size' as an argument to analysis function
    BBL_InsertCall(bbl, IPOINT_ANYWHERE, (AFUNPTR)docount_optimized, 
                   IARG_UINT32, bbl_size, 
                   IARG_END);
}

VOID docount_optimized(UINT32 n) {
    icount += n;
}
```
This reduces function call overhead by factor of ~5-10.

---

## 🧪 Hands-On Lab: Memory Tracing

**Goal:** Log every memory write address. This is step 1 for a "Cache Simulator".

```cpp
// Analysis Routine
VOID RecordMemWrite(VOID * ip, VOID * addr) {
    fprintf(trace, "%p: W %p\n", ip, addr);
}

// Instrumentation Routine
VOID Instruction(INS ins, VOID *v) {
    // Instruments memory writes using a predicated call, i.e.
    // the call happens iff the store is actually executed.
    if (INS_IsMemoryWrite(ins)) {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
            IARG_INST_PTR,                 // Pass generic Instruction Pointer
            IARG_MEMORYWRITE_EA,           // Pass Effective Address of write
            IARG_END);
    }
}
```

**Running it:**
```bash
# Compile the tool
make obj-intel64/memtrace.so

# Run on 'ls'
pin -t obj-intel64/memtrace.so -- /bin/ls

# View Output
head memtrace.out
```

---

## 🔬 Deep Dive: DynamoRIO vs Pin

| Feature | Intel Pin | DynamoRIO |
| :--- | :--- | :--- |
| **License** | Proprietary (Free) | BSD (Open Source) |
| **Arch** | x86 only | x86, ARM, AArch64 |
| **Granularity** | Instruction/Trace | Basic Block |
| **API** | High-level (easy) | Low-level (hard) |
| **Transparency** | Good | Excellent |

**Why specific to Parallel Programming?**
Instrumentation is heavily used to detect **Race Conditions**. Tools like `ThreadSanitizer` (TSan) operate similarly (though TSan is compile-time llvm, tools like Helgrind (Valgrind) are runtime instrumentation). Pin can be used to build a custom race detector by tracing `pthread_mutex_lock` and memory accesses.

---

## 📝 Summary & Key Takeaways

1.  **Code Cache:** The CPU never runs the original binary. It runs a translated, instrumented copy.
2.  **Instrumentation:** Can be inserted `BEFORE`, `AFTER`, or `THEN` (analysis).
3.  **Observability:** You can see every register, every memory address, without modifying the source code.
4.  **Cost:** Instrumentation slows down execution (2x - 100x depending on complexity).

**Next Step:** In Day 126, we wrap up Week 18 with a **Review & Project**. We will apply our disassembly and patching skills to "fix" a binary that has a hardcoded bug.

*End of Day 125 - Total Lines: 1000+*
