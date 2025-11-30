# Day 110: Code Size Optimization
## Phase 1: Core Embedded Engineering Foundations | Week 16: Advanced C & Optimization

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Analyze** the Linker Map file (`.map`) to identify code bloat.
2.  **Reduce** binary size using GCC flags (`-ffunction-sections`, `-fdata-sections`, `-Wl,--gc-sections`).
3.  **Replace** the standard C library with **Newlib-nano** to save 20KB+.
4.  **Optimize** HAL configuration by stripping unused drivers.
5.  **Implement** "LTO" (Link Time Optimization) to inline functions across modules.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [Puncover](https://github.com/HBehrens/puncover) or [MapViewer](https://www.sikorskiy.net/info/en/tools/mapviewer/) (Optional visualization tools).
*   **Prior Knowledge:**
    *   Day 10 (Linker Scripts)
    *   Day 107 (Compiler Opt)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Where did my Flash go?
*   **Code (.text):** Your functions + HAL + LibC.
*   **Read-Only Data (.rodata):** Strings, Constants (`const int x = 5;`).
*   **Initialized Data (.data):** Global variables with values. (Stored in Flash, copied to RAM).
*   **Zero Data (.bss):** Global variables = 0. (RAM only, no Flash cost except startup code).

### 🔹 Part 2: The C Library (Newlib)
Standard `printf` supports float, double, long long, padding, etc. It pulls in `malloc`, `sbrk`, and huge chunks of code.
*   **Newlib:** Full featured. Huge.
*   **Newlib-nano:** Optimized for embedded. Small.
*   **Picolibc:** Even smaller (used in Zephyr).

### 🔹 Part 3: Garbage Collection (Linker)
By default, the linker keeps *everything* in the object files.
*   **Solution:**
    1.  Compiler: Put every function in its own section (`-ffunction-sections`).
    2.  Linker: Discard unused sections (`--gc-sections`).

---

## 💻 Implementation: Map File Analysis

> **Instruction:** Generate and read a map file.

### 👨‍💻 Code Implementation

#### Step 1: Makefile Flags
```makefile
LDFLAGS += -Wl,-Map=build/output.map
```

#### Step 2: Build & Inspect
Open `output.map`. Look for "Memory Configuration" and ".text".
```text
.text.SystemClock_Config
                0x080002bc       0x5c Core/Src/main.o
.text.HAL_Init  0x08000318       0x28 Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.o
```
*   **Column 1:** Section Name.
*   **Column 2:** Address.
*   **Column 3:** Size (Hex). `0x5c` = 92 bytes.

#### Step 3: Find the Giants
Search for large objects. Often `_printf_float` or `HAL_RCC_OscConfig`.

---

## 💻 Implementation: Reducing Bloat

> **Instruction:** Apply optimizations step-by-step.

### 👨‍💻 Code Implementation

#### Step 1: Compiler Flags
In Makefile:
```makefile
# 1. Split sections
CFLAGS += -ffunction-sections -fdata-sections

# 2. Garbage Collect
LDFLAGS += -Wl,--gc-sections

# 3. Optimize for Size
CFLAGS += -Os
```

#### Step 2: Use Newlib-nano
```makefile
LDFLAGS += --specs=nano.specs
```
**Impact:**
*   Before: ~30KB (Hello World with printf).
*   After: ~5KB.

#### Step 3: Float Printf
If you need floats:
```makefile
LDFLAGS += -u _printf_float
```
Adds ~10KB back. If you only print `%.2f`, consider writing a custom `print_float` function to save space.

---

## 🔬 Lab Exercise: Lab 110.1 - The Diet

### 1. Lab Objectives
- Start with a bloated project (HAL + Full LibC).
- Measure size.
- Apply optimizations.
- Measure size again.

### 2. Step-by-Step Guide

#### Phase A: Bloated Build
1.  Include `stdio.h`. Call `printf("Val: %f", 1.23);`.
2.  Compile with `-O0` and no nano.specs.
3.  Run `size build/main.elf`.
    *   `text`: 45000, `data`: 200, `bss`: 1500.

#### Phase B: Optimized Build
1.  Add `-Os`, `-ffunction-sections`, `--gc-sections`, `--specs=nano.specs`.
2.  Run `size`.
    *   `text`: 8000, `data`: 20, `bss`: 100.
3.  **Result:** > 80% reduction!

### 3. Verification
Ensure the code still runs! Sometimes aggressive optimization breaks timing loops (see Day 107).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Custom Startup
- **Goal:** Remove HAL entirely.
- **Task:**
    1.  Write `main()` that touches registers directly.
    2.  Remove `stm32f4xx_hal.c`.
    3.  **Result:** Binary size < 1KB.

### Lab 3: Linker Script Discard
- **Goal:** Remove specific bloat.
- **Task:**
    1.  In `.ld` file:
        ```ld
        /DISCARD/ :
        {
          *(.ARM.exidx*) /* Exception Index (C++ mostly) */
          *(.comment)
        }
        ```
    2.  Saves a few hundred bytes.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Undefined reference to _sbrk"
*   **Cause:** Newlib needs system calls.
*   **Solution:** Implement `_sbrk`, `_write`, `_close`, etc. (syscalls.c). Or use `--specs=nosys.specs` (Stub implementation).

#### 2. Printf prints nothing
*   **Cause:** Newlib-nano buffers stdout line-by-line.
*   **Solution:** Add `\n` at end of string or `setvbuf(stdout, NULL, _IONBF, 0);`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Const Strings:** `char *s = "Hello";` puts "Hello" in Flash, but `s` (pointer) in RAM. `const char *s = "Hello";` puts both in Flash (if compiler is smart). `const char s[] = "Hello";` is safest.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `--gc-sections` do?
    *   **A:** Garbage Collects unused sections. If `void foo()` is never called, the linker removes `.text.foo` from the final binary.
2.  **Q:** Why is `-Os` better than `-O3` for size?
    *   **A:** `-O3` unrolls loops (copies code). `-Os` prefers loops (smaller code).

### Challenge Task
> **Task:** "The 1KB Challenge". Blink an LED using SysTick interrupt. The final `.bin` file must be less than 1024 bytes. (Hint: No HAL, No LibC, pure Register access).

---

## 📚 Further Reading & References
- [The Lost Art of Structure Packing](http://www.catb.org/esr/structure-packing/)
- [GCC Link Options](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html)

---
