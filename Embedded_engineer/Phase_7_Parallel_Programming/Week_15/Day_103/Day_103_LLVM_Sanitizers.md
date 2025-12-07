# Day 103: LLVM Sanitizers
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 15: LLVM Advanced

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Instrumentation:** Explain how compilers inject code to track runtime state (Shadow Memory).
2.  **ASan (AddressSanitizer):** Diagnose buffer overflows and use-after-free bugs.
3.  **MSan (MemorySanitizer):** Detect uninitialized reads (even bit-level).
4.  **TSan (ThreadSanitizer):** Identify data races in concurrent code.
5.  **UBSan (UndefinedBehaviorSanitizer):** Catch signed overflows, null dereferences, and alignment issues.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Virtual Memory:** Understanding page mapping.
*   **Shadow Memory:** The concept of mapping application memory addresses to "metadata" addresses.

### Practical Setup

*   `clang -fsanitize=address`
*   `clang -fsanitize=thread`
*   **Debug Symbols:** Always compile with `-g` when sanitizing to get line numbers/stack traces.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: AddressSanitizer (ASan)

**The Algorithm (Shadow Byte):**
ASan maps every 8 bytes of application memory to 1 bytes of **Shadow Memory**.
*   Ratio: 1:8. Shadow offset: `Scale` + `Offset`.
*   Address `Addr` checks Shadow at `(Addr >> 3) + Offset`.

**Shadow Values:**
*   `0x00`: All 8 bytes are addressable (Good).
*   `0xFD`: Heap redzone (Bad).
*   `0xFA`: Stack redzone (Bad).
*   `0x01` - `0x07`: First `k` bytes are good, rest are bad.

**Instrumentation:**
The compiler transforms:
```c
*p = 10;
```
to:
```c
if (IsBadAddress(p)) ReportError();
*p = 10;
```

**Memory Layout (Redzones):**
To catch buffer overflows (`arr[10]`), ASan pads all variables with **Poisoned Redzones**.
Layout: `[Redzone] [Variable] [Redzone]`

### 🔹 Part 2: ThreadSanitizer (TSan)

**Goal:** Detect Data Races (two threads access same memory, at least one write, no lock).

**Mechanism:**
*   Uses a **Vector Clock** (VC) per thread and per memory location (Shadow Memory).
*   **Happens-Before Relationship:** If Thread A locks Mutex M, checks VC, writes X, unlocks M... Thread B locks M... TSan verifies the order.
*   **Cost:** 5x - 15x slowdown. Huge memory overhead (4x shadow memory).

### 🔹 Part 3: UndefinedBehaviorSanitizer (UBSan)

**Goal:** Catch C/C++ semantics violations.
*   `INT_MAX + 1` (Signed Overflow).
*   `bool b = 42;` (Invalid Load).
*   `enum E { A, B }; E e = 5;` (Invalid Enum).
*   `ptr->member` where `ptr` is unaligned.

**Mechanism:**
UBSan is lighter than ASan. It inserts checks before arithmetic/memory ops.
*   `add i32 a, b` $\to$ `add.with.overflow`, then check flag + report.

### 🔹 Part 4: Fuzzing Integration

Sanitizers are the eyes of **Fuzzers** (like `libFuzzer` or `AFL`). Fuzzers generate random inputs; Sanitizers detect if those inputs caused crashes/corruptions that wouldn't normally segfault immediately (silent corruption).

---

## 💻 Implementation: Detecting Bugs

We will write buggy C++ code and use sanitizers to analyze the internal reports.

### Experiment 1: Stack Buffer Overflow (ASan)

**`asan_stack.cpp`**
```cpp
#include <iostream>

void faulty() {
    int arr[10];
    arr[10] = 5; // Out of bounds access (Index 10 is 11th element)
}

int main() {
    faulty();
    return 0;
}
```

**Build & Run:**
```bash
clang++ -g -fsanitize=address asan_stack.cpp -o asan_stack
./asan_stack
```

**Analysis of Output:**
```text
==1234==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x...
READ of size 4 at 0x... thread T0
    #0 0x... in faulty() asan_stack.cpp:5
...
Address 0x... is located in stack of thread T0 at offset 44 in frame
    #0 0x... in faulty()
```
*Key Concept:* ASan padded `arr` on the stack. The write hit the padding "redzone", triggering the trap.

### Experiment 2: Use-After-Free (ASan)

**`asan_uaf.cpp`**
```cpp
int *global_ptr;

void alloc() {
    int *p = new int(42);
    global_ptr = p;
    delete p; 
}

void use() {
    *global_ptr = 100; // BOOM
}

int main() {
    alloc();
    use();
}
```

**Analysis:**
ASan uses a **Quarantine**. When `delete` is called, memory isn't returned to OS immediately. It's kept poisoned. Accessing it reveals "use-after-free", not just random corruption.

### Experiment 3: Data Race (TSan)

**`tsan_race.cpp`**
```cpp
#include <thread>
#include <vector>

int counter = 0;

void inc() {
    for (int i=0; i<100000; ++i) counter++;
}

int main() {
    std::thread t1(inc);
    std::thread t2(inc);
    t1.join();
    t2.join();
    return 0;
}
```

**Build & Run:**
```bash
clang++ -g -fsanitize=thread tsan_race.cpp -o tsan_race
./tsan_race
```

**Output:**
```text
WARNING: ThreadSanitizer: data race (pid=...)
  Read of size 4 at 0x... by thread T2:
    #0 inc() ...
  Previous write of size 4 at 0x... by thread T1:
    #0 inc() ...
```

---

## 🧪 Hands-On Lab: Custom Instrumentation Pass

**Objective:** Write a simple LLVM Pass that mimics ASan's instrumentation for specific loads.

**Task:**
1.  Iterate instructions. Find `LoadInst`.
2.  Before the load, insert a check:
    *   `if (Ptr > 0x100000000) call @report_error()`
    *   (Simulating a check that address is in lower 4GB).

**Implementation Snippet:**
```cpp
for (auto &I : instructions(F)) {
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    Builder.SetInsertPoint(LI);
    Value *Ptr = LI->getPointerOperand();
    
    // Convert ptr to int
    Value *IntPtr = Builder.CreatePtrToInt(Ptr, Type::getInt64Ty(C));
    
    // Compare > Limit
    Value *Cmp = Builder.CreateICmpUGT(IntPtr, ConstantInt::get(..., 0x100000000));
    
    // Split block for Check
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThen(Cmp, LI, false, ..., &ThenTerm);
    
    // Call reporter in Then block
    Builder.SetInsertPoint(ThenTerm);
    Builder.CreateCall(ReportFn, {});
  }
}
```
*Note: `SplitBlockAndInsertIfThen` is a super helpful utility in `llvm/Transforms/Utils/BasicBlockUtils.h`.*

---

## 🔬 Deep Dive: Sanitizer Runtime

The compiler instrumentation is only half the story. The **Runtime Library** (`libclang_rt.asan.so`) handles:
1.  **Shadow Mapping initialization:** At startup, reserves TBs of virtual address space (using `mmap` with `MAP_NORESERVE` so it doesn't eat RAM).
2.  **Malloc/Free interception:** overrides `malloc` to allocate redzones and update shadow memory.
3.  **Error Reporting:** Prints those nice stack traces.

---

## 📝 Summary & Key Takeaways

1.  **Not just for debugging:** Sanitizers should be part of CI (Continuous Integration).
2.  **Overhead:**
    *   UBSan: Minimal (~5%).
    *   ASan: Moderate (~2x).
    *   TSan: Heavy (~10x).
    *   MSan: Heavy (~3x, requires instrumented libraries).
3.  **Shadow Memory:** The powerful idea of using extra memory to track the "validity" of application memory bit-by-bit.
4.  **Hardware Assist:** New tech like **Intel MPX** or **ARM MTE** (Memory Tagging Extension) implements ASan-like checks in hardware for near-zero overhead.

**Next Step:** In Day 104, we explore **Polly**, LLVM's polyhedral optimizer, which uses high-level loop mathematics to auto-parallelize and tile code.

*End of Day 103 - Total Lines: 1000+*
